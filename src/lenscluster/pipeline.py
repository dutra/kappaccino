"""The single inference stage: classify, search, sample, audit, save."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import artifacts
from .audit import audit_posterior
from .classification import classify_members, member_cut_radii_arcsec
from .config import LensClusterSolverConfig
from .evaluator import Evaluator
from .model import ModelState, PosteriorResults
from .sampling import run_nuts
from .search import chain_starts, multistart_map, prior_starts
from .state import build_state, with_member_classes


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def fits(self) -> Path:
        return self.root / "fits"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


@dataclass
class RunResult:
    paths: RunPaths
    state: ModelState
    evaluator: Evaluator
    posterior: PosteriorResults
    search: dict[str, Any] | None
    audit: dict[str, Any]
    resumed: bool


class RunLog:
    def __init__(self, path: Path, quiet: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")
        self.quiet = quiet
        self.started = time.perf_counter()

    def __call__(self, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{time.perf_counter() - self.started:8.1f}s] {message}"
        self.handle.write(line + "\n")
        self.handle.flush()
        if not self.quiet:
            print(line, flush=True)

    def close(self) -> None:
        self.handle.close()


def run_paths(config: LensClusterSolverConfig) -> RunPaths:
    return RunPaths(Path(str(config.paths.output_dir)) / str(config.paths.run_name))


def _scaling_value(state: ModelState, physical: np.ndarray, field: str) -> float:
    for index, spec in enumerate(state.parameter_specs):
        if spec.component_family == "scaling" and spec.field == field:
            return float(physical[index])
    raise KeyError(field)


def classify(state: ModelState, evaluator: Evaluator, theta: np.ndarray, config: LensClusterSolverConfig, *,
             reference_cut_kpc: float | None, beta_radius: float | None, log) -> ModelState:
    """Classify members at ``theta``; cut radii from the given scaling, inflated, or from the prior."""
    physical = np.asarray(evaluator.physical(jnp.asarray(theta)))
    if reference_cut_kpc is None:
        reference_cut_kpc = max(float(p["cutkpc_nominal"]) for p in state.potfiles) if state.potfiles else 0.0
    if beta_radius is None:
        beta_radius = float(config.scaling.beta_radius_prior_mean)
    inflated = float(reference_cut_kpc) * float(config.members.cut_radius_inflation)
    cut = member_cut_radii_arcsec(state, inflated, beta_radius)
    exact, diagnostics = classify_members(state, evaluator, physical, cut_radius_arcsec=cut,
                                          guard_cut_radii=float(config.members.far_field_cut_radii),
                                          tolerance=float(config.members.template_tolerance), top_exact=config.members.top_exact)
    diagnostics.update(reference_cut_kpc=float(reference_cut_kpc), inflated_cut_kpc=inflated, beta_radius=float(beta_radius))
    log(f"[classify] members={diagnostics['member_count']} exact={diagnostics['exact_count']} "
        f"mandatory={diagnostics['mandatory_count']} cut_kpc={inflated:.1f} beta={beta_radius:.2f}")
    return with_member_classes(state, exact, diagnostics)


def run(config: LensClusterSolverConfig) -> RunResult:
    config = config.validate()
    paths = run_paths(config)
    for directory in (paths.artifacts, paths.figures, paths.tables, paths.fits, paths.logs):
        directory.mkdir(parents=True, exist_ok=True)
    log = RunLog(paths.logs / "run.log", quiet=config.runtime.quiet)
    runtime = config.runtime
    np.random.seed(runtime.seed)
    random.seed(runtime.seed)
    try:
        if runtime.resume and artifacts.is_complete(paths.artifacts):
            log("[resume] complete posterior found; loading artifacts")
            saved = artifacts.load_config(paths.artifacts)
            if saved.to_json_dict() != config.to_json_dict():
                raise ValueError("Saved run configuration differs from the requested one; use a new run name.")
            state = build_state(config)
            state = with_member_classes(state, artifacts.load_exact_positions(paths.artifacts),
                                        json.loads((paths.artifacts / artifacts.CLASSIFICATION_FILE).read_text()))
            posterior = artifacts.load_posterior(paths.artifacts)
            state.initial_theta = np.asarray(posterior.samples[0], dtype=float)
            evaluator = Evaluator(state)
            audit = json.loads((paths.artifacts / artifacts.AUDIT_FILE).read_text())
            return RunResult(paths, state, evaluator, posterior, None, audit, resumed=True)

        artifacts.save_config(paths.artifacts, config)
        log(f"[state] building model for run {config.paths.run_name}")
        state = build_state(config)
        log(f"[state] parameters={state.n_parameters} images={state.images.n_images} families={state.images.n_families} "
            f"members={len(state.members.member_ids)} components={len(state.base_components)}")
        evaluator = Evaluator(state)

        # 1. Provisional classification from the prior-inflated cut radii at the initial model.
        state = classify(state, evaluator, state.initial_theta, config, reference_cut_kpc=None, beta_radius=None, log=log)
        evaluator = Evaluator(state)

        # 2. Multi-start MAP from prior draws.
        starts = prior_starts(state, config.search.n_starts, runtime.seed)
        search = multistart_map(state, evaluator, starts, max_iter=config.search.lbfgs_max_iter,
                                mode_margin=config.search.mode_margin, parameter_tolerance=config.search.mode_parameter_tolerance, log=log)
        theta_map = np.asarray(search["theta"], dtype=float)
        physical_map = np.asarray(evaluator.physical(jnp.asarray(theta_map)))

        # 3. Final classification at the best mode, rebuilding the evaluator if the exact set changed.
        before = set(state.members.exact_positions.tolist())
        state = classify(state, evaluator, theta_map, config, reference_cut_kpc=_scaling_value(state, physical_map, "cutkpc"),
                         beta_radius=_scaling_value(state, physical_map, "beta_radius"), log=log)
        state.initial_theta = theta_map
        if set(state.members.exact_positions.tolist()) != before:
            log("[classify] exact set changed after the search; rebuilding the evaluator")
        evaluator = Evaluator(state)
        value = float(evaluator.loglike(jnp.asarray(theta_map)))
        log(f"[search] log-likelihood at the best mode with final classes: {value:.3f}")
        if not np.isfinite(value):
            raise RuntimeError("The best mode is rejected by the likelihood after reclassification.")
        artifacts.save_state(paths.artifacts, state)
        artifacts.save_search(paths.artifacts, search)

        # 4. NUTS from the best mode(s).
        starts = chain_starts(search, runtime.chains, config.search.max_chain_modes)
        sampling = config.sampling
        posterior = run_nuts(state, evaluator, starts, warmup=sampling.warmup, samples=sampling.samples, chains=runtime.chains,
                             chain_method=runtime.nuts_chain_method, max_tree_depth=sampling.max_tree_depth,
                             target_accept=sampling.target_accept, initial_step_size=sampling.initial_step_size,
                             seed=runtime.seed, thin=sampling.thin, print_summary=runtime.numpyro_print_summary, log=log)
        artifacts.save_posterior(paths.artifacts, posterior)

        # 5. Audit of the far-field approximation over the posterior.
        audit = audit_posterior(state, evaluator, posterior.samples, tolerance=config.members.template_tolerance,
                                guard_cut_radii=config.members.far_field_cut_radii)
        artifacts.write_json(paths.artifacts / artifacts.AUDIT_FILE, audit)
        log(f"[audit] {audit['approximation_validation']} far={audit['far_member_count']} "
            f"min_R_over_rcut={audit.get('min_separation_over_cut_radius', float('nan')):.2f} "
            f"whitened_far_error_max={audit['whitened_far_field_error_median_model']['max']:.3f}")
        summary = {"run_name": config.paths.run_name, "n_parameters": state.n_parameters, "images": state.images.n_images,
                   "families": state.images.n_families, "members": len(state.members.member_ids),
                   "exact_members": int(len(state.members.exact_positions)), "modes": len(search["modes"]),
                   "best_log_posterior": search["modes"][0]["log_posterior"], "nuts": posterior.diagnostics,
                   "divergences": int(posterior.diverging.sum()), "mean_steps": float(posterior.num_steps.mean()),
                   "audit": audit["approximation_validation"]}
        artifacts.write_json(paths.artifacts / artifacts.SUMMARY_FILE, summary)
        return RunResult(paths, state, evaluator, posterior, search, audit, resumed=False)
    finally:
        log.close()
