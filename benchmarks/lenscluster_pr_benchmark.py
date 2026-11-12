"""Likelihood and gradient timings on the FF-SIMS Hera model, for the PR benchmark comment.

``run`` times one installed kappaccino and writes ``benchmark.json`` and a markdown ``output``; ``compare`` renders the
base-versus-PR table that .github/workflows/publish-pr-benchmarks.yml posts on the pull request.  The workflow runs this
script from the PR checkout against both installs, with ``--catalog-root`` on the PR checkout so the inputs are the same.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from kappaccino.config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                                LikelihoodConfig, MemberPopulationConfig, ReferenceFrameConfig, RunPathsConfig, ShearHaloConfig)
from kappaccino.likelihood.evaluator import Evaluator
from kappaccino.inference.search import Objective
from kappaccino.core.state import build_state

REPO_ROOT = Path(__file__).resolve().parents[1]
MARKER = "<!-- lenscluster benchmark -->"
# Name -> (likelihood_mode, gp_kernel): the library default and the grf_gp model of the cluster run.yaml files.
LIKELIHOODS = {"smoothed": ("smoothed", "image_plane"), "grf_gp": ("gp_image_plane", "knee")}


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def _stderr(values: list[float]) -> float:
    return float(statistics.stdev(values) / np.sqrt(len(values))) if len(values) > 1 else 0.0


def _percent_delta(candidate: float, baseline: float) -> float:
    return float(100.0 * (candidate - baseline) / baseline) if baseline else float("nan")


def _percent_delta_stderr(candidate: float, baseline: float, candidate_stderr: float, baseline_stderr: float) -> float:
    if baseline == 0.0:
        return float("nan")
    return float(100.0 * np.hypot(candidate_stderr / baseline, candidate * baseline_stderr / baseline**2))


def _workflow_url() -> str:
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com")
    repo = os.getenv("GITHUB_REPOSITORY", "")
    run_id = os.getenv("GITHUB_RUN_ID", "")
    return f"{server}/{repo}/actions/runs/{run_id}" if repo and run_id else "local"


def hera_config(catalog_root: Path, likelihood: str, output_dir: str) -> LensClusterSolverConfig:
    """The Hera model of catalogs/hera/priors.yaml with library defaults elsewhere."""
    mode, kernel = LIKELIHOODS[likelihood]
    return LensClusterSolverConfig(
        model=LensModelConfig(
            reference=ReferenceFrameConfig(reference=3, ra0_deg=0.0, dec0_deg=0.0),
            cosmology=CosmologyConfig(H0=72.0, Om0=0.24, Ode0=0.76),
            large_halos=(DPIEHaloConfig.cluster_scale("1", 19.54080009, 2.37820005, 0.507, v_disp=800.0),
                         DPIEHaloConfig.cluster_scale("2", 0.001, 0.003, 0.507, v_disp=700.0),
                         ShearHaloConfig.free("S1", 0.507)),
            member_populations=(MemberPopulationConfig.bergamini2019("potfile_1", str(catalog_root / "catalogs/hera/members.yaml"), 0.507,
                                                                     prior_mag0=19.82),),
            image_constraints=ImageConstraintsConfig(catalog_path=str(catalog_root / "catalogs/hera/images.yaml"))),
        likelihood=LikelihoodConfig(likelihood_mode=mode, gp_kernel=kernel),
        paths=RunPathsConfig(output_dir=output_dir, run_name="benchmark")).validate()


def _bench_jitted(name: str, fn: Callable[[], Any], repeats: int, trials: int) -> dict[str, Any]:
    """Compile time of the first call, then ``trials`` blocks of ``repeats`` calls."""
    start = time.perf_counter()
    jax.block_until_ready(fn())
    compile_seconds = time.perf_counter() - start
    trial_ms = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(repeats):
            out = fn()
        jax.block_until_ready(out)
        trial_ms.append(1.0e3 * (time.perf_counter() - start) / repeats)
    return {"name": name, "repeats": repeats, "trials": trials, "compile_seconds": float(compile_seconds),
            "trial_ms_per_eval": trial_ms, "ms_per_eval": _mean(trial_ms), "ms_per_eval_stderr": _stderr(trial_ms)}


def run_case(catalog_root: Path, likelihood: str, *, repeats: int, trials: int, lbfgs_iters: int) -> dict[str, Any]:
    phases: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as output_dir:
        start = time.perf_counter()
        config = hera_config(catalog_root, likelihood, output_dir)
        state = build_state(config)
        phases["build_state_seconds"] = time.perf_counter() - start

        start = time.perf_counter()
        evaluator = Evaluator(state, config.likelihood.likelihood_mode)
        objective = Objective(state, evaluator, state.initial_theta, log_likelihood=evaluator.log_likelihood)
        phases["evaluator_setup_seconds"] = time.perf_counter() - start

        theta = jnp.asarray(state.initial_theta)
        u0 = objective.base
        log_likelihood = jax.jit(evaluator.log_likelihood)
        rows = [_bench_jitted("log-likelihood", lambda: log_likelihood(theta), repeats, trials),
                _bench_jitted("log-posterior value+grad", lambda: objective.value_and_grad(u0), repeats, trials)]

        # A fixed-length L-BFGS run from the initial model: the per-iteration cost the init search pays.
        def fun(x):
            value, grad = objective.value_and_grad(jnp.asarray(x))
            return float(value), np.asarray(grad, dtype=float)

        start = time.perf_counter()
        result = minimize(fun, np.asarray(u0, dtype=float), jac=True, method="L-BFGS-B",
                          options={"maxiter": lbfgs_iters, "gtol": 0.0, "ftol": 0.0})
        phases["lbfgs_seconds"] = time.perf_counter() - start
    return {"likelihood": likelihood, "n_members": len(state.members.member_ids), "n_systems": len(state.family_data),
            "n_images": int(sum(family.n_images for family in state.family_data)),
            "n_parameters": int(np.size(state.initial_theta)), "n_unconstrained": int(np.size(u0)), "phases": phases,
            "timings": rows, "lbfgs_iterations": int(result.nit), "lbfgs_evaluations": int(result.nfev)}


def run_benchmark(*, label: str, sha: str, catalog_root: Path, likelihoods: list[str], repeats: int, trials: int,
                  lbfgs_iters: int) -> dict[str, Any]:
    start = time.perf_counter()
    cases = [run_case(catalog_root, name, repeats=repeats, trials=trials, lbfgs_iters=lbfgs_iters) for name in likelihoods]
    return {"label": label, "sha": sha, "repeats": repeats, "trials": trials, "lbfgs_iters": lbfgs_iters,
            "jax_version": jax.__version__, "python": platform.python_version(), "machine": platform.machine(),
            "cases": cases, "total_seconds": time.perf_counter() - start}


def _fmt_ms(row: dict[str, Any]) -> str:
    return f"{row['ms_per_eval']:.3f} +/- {row['ms_per_eval_stderr']:.3f} ms"


def _fmt_delta(candidate: float, baseline: float, candidate_stderr: float = 0.0, baseline_stderr: float = 0.0) -> str:
    delta = _percent_delta(candidate, baseline)
    stderr = _percent_delta_stderr(candidate, baseline, candidate_stderr, baseline_stderr)
    flag = " :warning:" if delta > 10.0 and delta > 2.0 * stderr else ""
    if not np.isfinite(stderr) or stderr <= 0.0:
        return f"{delta:+.1f}%{flag}"
    return f"{delta:+.1f}% +/- {stderr:.1f}%{flag}"


def _header(result: dict[str, Any]) -> list[str]:
    case = result["cases"][0]
    return [MARKER, "### kappaccino PR benchmark", "",
            f"Model: FF-SIMS Hera ({case['n_images']} images in {case['n_systems']} systems, {case['n_members']} members, {case['n_unconstrained']} free parameters), "
            f"`ubuntu-latest` CPU, {result['trials']} trials x {result['repeats']} evaluations, L-BFGS for {result['lbfgs_iters']} iterations.", ""]


def render_markdown(result: dict[str, Any], *, workflow_url: str) -> str:
    lines = _header(result) + [f"Commit `{result['sha'][:12]}`. No base result to compare against.", "",
                               "| likelihood | metric | value |", "| --- | --- | ---: |"]
    for case in result["cases"]:
        for row in case["timings"]:
            lines.append(f"| {case['likelihood']} | {row['name']} | {_fmt_ms(row)} |")
            lines.append(f"| {case['likelihood']} | {row['name']} compile | {row['compile_seconds']:.2f} s |")
        for name, seconds in case["phases"].items():
            lines.append(f"| {case['likelihood']} | `{name}` | {seconds:.2f} s |")
    lines.extend(["", f"Run: {workflow_url}", ""])
    return "\n".join(lines)


def render_comparison_markdown(baseline: dict[str, Any], candidate: dict[str, Any], *, workflow_url: str) -> str:
    lines = _header(candidate) + ["| likelihood | metric | base | PR | delta |", "| --- | --- | ---: | ---: | ---: |",
                                  f"| | commit | `{baseline['sha'][:12]}` | `{candidate['sha'][:12]}` | |"]
    base_cases = {case["likelihood"]: case for case in baseline["cases"]}
    for case in candidate["cases"]:
        base = base_cases.get(case["likelihood"])
        if base is None:
            continue
        base_rows = {row["name"]: row for row in base["timings"]}
        for row in case["timings"]:
            b = base_rows.get(row["name"])
            if b is None:
                continue
            lines.append(f"| {case['likelihood']} | {row['name']} | {_fmt_ms(b)} | {_fmt_ms(row)} | "
                         f"{_fmt_delta(row['ms_per_eval'], b['ms_per_eval'], row['ms_per_eval_stderr'], b['ms_per_eval_stderr'])} |")
            lines.append(f"| {case['likelihood']} | {row['name']} compile | {b['compile_seconds']:.2f} s | {row['compile_seconds']:.2f} s | "
                         f"{_fmt_delta(row['compile_seconds'], b['compile_seconds'])} |")
        for name, seconds in case["phases"].items():
            if name in base["phases"]:
                lines.append(f"| {case['likelihood']} | `{name}` | {base['phases'][name]:.2f} s | {seconds:.2f} s | "
                             f"{_fmt_delta(seconds, base['phases'][name])} |")
    lines.extend(["", f"| | total benchmark | {baseline['total_seconds']:.1f} s | {candidate['total_seconds']:.1f} s | "
                      f"{_fmt_delta(candidate['total_seconds'], baseline['total_seconds'])} |",
                  "", ":warning: marks a slowdown above 10% and twice its standard error. Timings on shared runners vary by a few percent.",
                  "", f"Run: {workflow_url}", ""])
    return "\n".join(lines)


def _run_command(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_benchmark(label=args.label, sha=args.sha, catalog_root=args.catalog_root, likelihoods=args.likelihoods,
                           repeats=args.repeats, trials=args.trials, lbfgs_iters=args.lbfgs_iters)
    (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "output").write_text(render_markdown(result, workflow_url=_workflow_url()), encoding="utf-8")


def _compare_command(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    if args.baseline_json.exists():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        text = render_comparison_markdown(baseline, candidate, workflow_url=_workflow_url())
    else:
        text = render_markdown(candidate, workflow_url=_workflow_url())
    (args.output_dir / "output").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--label", default="benchmark")
    run_parser.add_argument("--sha", default=os.getenv("GITHUB_SHA", "local"))
    run_parser.add_argument("--catalog-root", type=Path, default=REPO_ROOT)
    run_parser.add_argument("--likelihoods", nargs="+", choices=sorted(LIKELIHOODS), default=["smoothed", "grf_gp"])
    run_parser.add_argument("--repeats", type=int, default=int(os.getenv("LENSCLUSTER_BENCH_REPEATS", "50")))
    run_parser.add_argument("--trials", type=int, default=int(os.getenv("LENSCLUSTER_BENCH_TRIALS", "3")))
    run_parser.add_argument("--lbfgs-iters", type=int, default=int(os.getenv("LENSCLUSTER_BENCH_LBFGS_ITERS", "25")))
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline-json", type=Path, required=True)
    compare_parser.add_argument("--candidate-json", type=Path, required=True)
    compare_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _compare_command(args) if args.command == "compare" else _run_command(args)


if __name__ == "__main__":
    sys.exit(main())
