"""The uncertain-image test on a simulated cluster: best fits (MAP) along a ladder of trusted families.

For every number of trusted (gold) families in ``--trusted`` and every seed in ``--seeds``, this degrades the image
catalog (scripts/degrade_image_catalog.py: the other families' images become uncertain images with a noisy photo-z)
and fits three models.

- ``full``: every family trusted (once per cluster).
- ``<case>_ignore``: the trusted families only.
- ``<case>_groups``: the trusted families plus the uncertain images, summed over their partitions
  (``kappaccino.partition.groups``).  The library of groups is built at the ``ignore`` optimum, the lens is refit with
  the library summed and the library rebuilt (``PartitionConfig.library_refits`` times), then the lens is fit once more.

Each fit is the init's stage A: the large-scale halos, shear, sources and the member scaling relation from ``--starts``
prior starts by L-BFGS, every member on the relation (offsets zero) and the error scales held at their starts.  A best
fit has no uncertainties, so it answers "is the lens less biased" and "is the partition recovered", not "are the errors
calibrated".  At each optimum this writes what scripts/score_uncertain_images.py reads, under
``results/<campaign>/<cluster>/<fit>/final``: ``artifacts/config.json``, ``artifacts/theta_map.npy``,
``artifacts/uncertain_images.h5`` (the partition at the optimum, with joint draws) and the truth-recovery maps.

Example (Hera, the ladder of the paper figure, then score and plot):

    python scripts/uncertain_images_ladder.py HERA --trusted 15,12,8,5,3,2 --seeds 0,1,2
    python scripts/score_uncertain_images.py results/uncertain_images/hera
    python scripts/plot_uncertain_images.py results/uncertain_images/hera
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "uncertain_images"


def load_run_module():
    """run.xsh as a module (xonsh's import hook), for the cluster entries and build_config."""
    from xonsh.main import setup
    setup()
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import run
    return run


def lens_fit(config, *, starts: int, seed: int, log) -> dict:
    """Stage A from prior starts: the state, evaluator, optimum ``theta``, its log posterior and the active sites."""
    from kappaccino.core.state import build_state
    from kappaccino.inference.search import multistart_map, prior_starts
    from kappaccino.inference.search_split import STAGE_A_INACTIVE
    from kappaccino.likelihood.evaluator import Evaluator
    from kappaccino.pipeline import classify

    mode = config.likelihood.likelihood_mode
    state = build_state(config)
    state = classify(state, Evaluator(state, mode), state.initial_theta, config, reference_cut_kpc=None, beta_radius=None, log=log)
    evaluator = Evaluator(state, mode)
    theta0 = prior_starts(state, int(starts), seed)
    theta0[0] = np.asarray(state.initial_theta, dtype=float)
    sites = {s.sample_site_name or s.sample_name for s in state.parameter_specs if s.component_family not in STAGE_A_INACTIVE}
    result = multistart_map(state, evaluator, theta0, config.search, method="lbfgs", seed=seed, log=log,
                            active_sites=sites, polish_sites=sites, label="lens")
    return {"state": state, "evaluator": evaluator, "theta": np.asarray(result["theta"], dtype=float),
            "log_posterior": float(result["modes"][0]["log_posterior"]), "sites": sites}


def groups_fit(config, base: dict, *, seed: int, log) -> dict:
    """The uncertain images added at the trusted-only optimum ``base``, the library refit and rebuilt, and a last fit of
    the lens with the final library."""
    from kappaccino.inference.search import multistart_map
    from kappaccino.partition import add_uncertain_images

    def fit(state, evaluator, theta):
        return multistart_map(state, evaluator, np.asarray(theta, dtype=float)[None], config.search, method="lbfgs", seed=seed,
                              log=log, active_sites=base["sites"], polish_sites=base["sites"], label="lens + uncertain images")

    state, evaluator, start = add_uncertain_images(config, base["state"], base["evaluator"], base["theta"],
                                                   refit=lambda s, e, t: fit(s, e, t)["theta"], log=log)
    result = fit(state, evaluator, start)
    return {"state": state, "evaluator": evaluator, "theta": np.asarray(result["theta"], dtype=float),
            "log_posterior": float(result["modes"][0]["log_posterior"])}


def write_fit(config, result: dict, final: Path, *, draws: int, seed: int, seconds: float, log) -> None:
    """The artifacts of one best fit: the configuration, the optimum, the partition at the optimum and the truth maps."""
    from kappaccino.diagnostics.truth_recovery import compute_truth_recovery
    from kappaccino.io import artifacts
    from kappaccino.partition import write_partition_outputs
    stage = SimpleNamespace(artifacts=final / "artifacts", fits=final / "fits", tables=final / "tables")
    for directory in vars(stage).values():
        directory.mkdir(parents=True, exist_ok=True)
    artifacts.save_config(stage.artifacts, config)
    state, evaluator, theta = result["state"], result["evaluator"], result["theta"]
    np.save(stage.artifacts / "theta_map.npy", theta)
    if state.uncertain_images is not None:
        write_partition_outputs(state, evaluator, theta[None], stage, draws=1, seed=seed, draws_per_sample=draws, log=log)
    if config.truth.kappa_true_fits:
        compute_truth_recovery(state, evaluator, theta[None], config.truth, stage.fits, stage.tables, seed=seed, log=log)
    artifacts.write_json(stage.artifacts / "fast_fit.json", {"log_posterior": result["log_posterior"], "seconds": seconds,
                                                             "n_parameters": state.n_parameters})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cluster", choices=("HERA", "ARES"), type=str.upper)
    parser.add_argument("--trusted", default="15,12,8,5,3,2", help="comma-separated numbers of trusted families (default 15,12,8,5,3,2)")
    parser.add_argument("--seeds", default="0,1,2", help="comma-separated seeds of the trusted choice and the noise (default 0,1,2)")
    parser.add_argument("--counterimages", type=int, default=0, help="trusted families that lose one image each (default 0)")
    parser.add_argument("--starts", type=int, default=16, help="prior starts of the lens fit (default 16)")
    parser.add_argument("--likelihood", default="smoothed", help="run.xsh likelihood name (default smoothed)")
    parser.add_argument("--draws", type=int, default=200, help="joint draws of the partition at the optimum (default 200)")
    parser.add_argument("--campaign", default=CAMPAIGN, help=f"results/<campaign>/<cluster> (default {CAMPAIGN})")
    parser.add_argument("--redo", action="store_true", help="refit cases that already have results")
    args = parser.parse_args(argv)

    sys.path[:0] = [str(ROOT / "src")]
    from kappaccino.cpu_runtime import configure_cpu_runtime
    configure_cpu_runtime(1, "sequential", None)             # before run.xsh builds its priors (and starts JAX)
    run = load_run_module()
    from kappaccino.pipeline import RunLog
    root = ROOT / "results" / args.campaign / args.cluster.lower()
    root.mkdir(parents=True, exist_ok=True)

    def config_for(label: str, image_catalog: str | None = None, uncertain: str | None = None):
        config = run.build_config(args.cluster, cores=1, campaign_name=args.campaign, run_label=label, likelihood=args.likelihood, fast=True,
                                  image_catalog=image_catalog, uncertain_images=uncertain)
        return replace(config, runtime=replace(config.runtime, chains=1, nuts_chain_method="sequential", cpu_threads_per_chain=None,
                                               skip_plots=True)).validate()

    def done(label: str) -> bool:
        return not args.redo and (root / label / "final" / "artifacts" / "fast_fit.json").exists()

    def run_fit(label: str, config, fitter, seed: int = 0) -> dict:
        final = root / label / "final"
        final.mkdir(parents=True, exist_ok=True)
        log = RunLog(final / "fast_fit.log", quiet=True)
        try:
            started = time.perf_counter()
            result = fitter(config, log)
            seconds = time.perf_counter() - started
            write_fit(config, result, final, draws=args.draws, seed=seed, seconds=seconds, log=log)
            print(f"[ladder] {label}: log posterior {result['log_posterior']:.2f}, fit {seconds:.0f} s", flush=True)
            return result
        finally:
            log.close()

    lens = lambda seed: (lambda config, log: lens_fit(config, starts=args.starts, seed=seed, log=log))
    if not done("full"):
        run_fit("full", config_for("full"), lens(0))
    for n in (int(s) for s in args.trusted.split(",")):
        for seed in (int(s) for s in args.seeds.split(",")):
            subprocess.run([sys.executable, str(ROOT / "scripts" / "degrade_image_catalog.py"), args.cluster, "--trusted", str(n),
                            "--seed", str(seed), "--counterimages", str(args.counterimages), "--output-root", f"results/{args.campaign}"],
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            case = f"n{n}_c{args.counterimages}_s{seed}"
            if done(f"{case}_ignore") and done(f"{case}_groups"):
                continue
            inputs = (root / "inputs" / case).relative_to(ROOT)
            base = run_fit(f"{case}_ignore", config_for(f"{case}_ignore", str(inputs / "trusted.yaml")), lens(seed), seed)
            run_fit(f"{case}_groups", config_for(f"{case}_groups", str(inputs / "trusted.yaml"), str(inputs / "uncertain.csv")),
                    lambda config, log: groups_fit(config, base, seed=seed, log=log), seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
