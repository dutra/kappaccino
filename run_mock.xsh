$XONSH_SHOW_TRACEBACK = True
$MPLBACKEND = "Agg"

import json
import os
os.environ["MPLBACKEND"] = "Agg"
import sys
import time
from pathlib import Path
import numpy as np

cores = 4  # NUTS chains; one JAX CPU device per chain (same budget as run.xsh).
seed_default = 12345
output_dir_default = "validation_runs"
# The mock injects Gaussian image-position noise of this size and the solver uses it as the catalog error (the floor
# under the sampled sigma_obs_extra), so the coverage test checks both the error model and the mass model.
pos_sigma_arcsec = 0.05
# Injected member scatter in ln(sigma), at the median of the solver's scatter prior (LogNormal(0.10, 0.3)).
scatter_ln = 0.10
lens_redshift_min = 0.30
lens_redshift_max = 0.50

$MPLCONFIGDIR = "/tmp/matplotlib-lenscluster-mock"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-lenscluster-mock"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lenscluster.config import (
    ImageDiagnosticsConfig,
    LensClusterSolverConfig,
    LikelihoodConfig,
    RuntimeConfig,
    SamplingConfig,
    SearchConfig,
    TruthRecoveryConfig,
)
from lenscluster.cpu_runtime import configure_cpu_runtime, resolve_cpu_plan

VALID_PRESETS = ("smoke", "production")


def _usage() -> str:
    return ("Usage: xonsh run_mock.xsh --campaign NAME [--preset smoke|production] [--seed INT] "
            "[--realizations INT] [--output-dir PATH] [--aggregate-only] [--dry-run]")


def _parse_args(argv: list[str]) -> dict:
    values = {"campaign": None, "preset": "smoke", "seed": str(seed_default), "realizations": "1",
              "output_dir": output_dir_default, "aggregate_only": False, "dry_run": False}
    flag_to_key = {"--campaign": "campaign", "--preset": "preset", "--seed": "seed", "--realizations": "realizations", "--output-dir": "output_dir"}
    index = 1
    while index < len(argv):
        flag = argv[index]
        if flag in {"--help", "-h"}:
            print(_usage())
            raise SystemExit(0)
        if flag in {"--aggregate-only", "--dry-run"}:
            values[flag.lstrip("-").replace("-", "_")] = True
            index += 1
            continue
        if flag not in flag_to_key or index + 1 >= len(argv):
            print(f"Unknown argument or missing value: {flag!r}\n{_usage()}")
            raise SystemExit(2)
        values[flag_to_key[flag]] = argv[index + 1]
        index += 2
    if values["campaign"] is None:
        print(f"Missing required --campaign.\n{_usage()}")
        raise SystemExit(2)
    preset = str(values["preset"]).strip().lower()
    if preset not in VALID_PRESETS:
        print(f"Preset must be 'smoke' or 'production'; got {values['preset']!r}.\n{_usage()}")
        raise SystemExit(2)
    seed, realizations = int(values["seed"]), int(values["realizations"])
    if seed < 0 or realizations <= 0:
        print("Seed must be nonnegative and realizations positive.")
        raise SystemExit(2)
    return dict(preset=preset, seed=seed, realizations=realizations, campaign=str(values["campaign"]).strip(),
                output_dir=str(values["output_dir"]), aggregate_only=values["aggregate_only"], dry_run=values["dry_run"])


def build_runtime(seed: int) -> RuntimeConfig:
    return RuntimeConfig(seed=seed, chains=cores, resume=True, numpyro_print_summary=True, nuts_chain_method="parallel",
                         cpu_threads_per_chain=2)   # same CPU budget as run.xsh: chains * 2 cores, chosen to avoid other runs


def _solver(*, production: bool, seed: int) -> LensClusterSolverConfig:
    """The solver template: library defaults (the tuned Hera settings) plus what the mock needs.  The lens model, with
    priors centred on the mock's truth, is attached per realization by the mock-validation pipeline."""
    return LensClusterSolverConfig(
        runtime=build_runtime(seed),
        search=SearchConfig() if production else SearchConfig(n_starts=8),
        sampling=(SamplingConfig(warmup=250, samples=200, max_tree_depth=8, target_accept=0.85) if production
                  else SamplingConfig(warmup=100, samples=100, max_tree_depth=6, target_accept=0.85)),
        likelihood=LikelihoodConfig(pos_sigma_arcsec=pos_sigma_arcsec),
        image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=64 if production else 8, match_tolerance_arcsec=3.0),
        truth=TruthRecoveryConfig(posterior_truth_recovery_draws=0),
    )


def build_config(preset: str, *, seed: int, realizations: int, campaign_name: str, output_dir: str):
    from lenscluster.mock_validation import MockValidationConfig, MockValidationPathsConfig, MockValidationRuntimeConfig, SingleBCGMockConfig
    production = preset == "production"
    n_primary_families = 15 if production else 5
    n_subhalo_families = 5 if production else 1
    mock = SingleBCGMockConfig(
        seed=seed,
        n_primary_families=n_primary_families,
        n_subhalo_families=n_subhalo_families,
        n_subhalos=50 if production else 10,
        subhalo_spatial_distribution="dpie",
        subhalo_spatial_core_radius_arcsec=25.0,
        subhalo_spatial_cut_radius_arcsec=180.0,
        subhalo_field_radius_arcsec=250.0,
        subhalo_force_core_count=0,
        subhalo_sigma_scatter_dex=float(scatter_ln) / np.log(10.0),
        subhalo_cut_scatter_dex=0.0,
        lens_redshift_min=lens_redshift_min,
        lens_redshift_max=lens_redshift_max,
        min_images_per_family=3,
        max_images_per_family=None,
        primary_source_redshifts=np.linspace(1.5, 9.0, n_primary_families),
        subhalo_source_redshifts=np.linspace(1.5, 9.0, n_subhalo_families),
        pos_sigma_arcsec=pos_sigma_arcsec,
        mock_image_candidate_batch_size=1024,
        mock_image_seed_cap=64,
        mock_image_search_window_arcsec=80.0,
        primary_seed_grid_pixel_arcsec=1.0,
        subhalo_seed_grid_pixel_arcsec=1.0,
        max_sources_to_try=400,
        mock_generation_workers=cores if production else 1,
    )
    return MockValidationConfig(
        mock=mock,
        paths=MockValidationPathsConfig(output_dir=output_dir, campaign_name=campaign_name, run_name=f"single_bcg_{preset}"),
        runtime=MockValidationRuntimeConfig(realizations=realizations, seed=seed, resume=True, quiet=False),
        solver=_solver(production=production, seed=seed),
    )


def main(argv=None):
    options = _parse_args(sys.argv if argv is None else argv)
    runtime = build_runtime(options["seed"])
    cpu_plan = resolve_cpu_plan(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    if not options["dry_run"]:
        configure_cpu_runtime(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)
    config = build_config(options["preset"], seed=options["seed"], realizations=options["realizations"],
                          campaign_name=options["campaign"], output_dir=options["output_dir"]).validate()
    campaign_dir = Path(config.paths.output_dir) / config.paths.campaign_name / config.paths.run_name
    if options["dry_run"]:
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), "campaign_dir": str(campaign_dir), **config.to_json_dict()}, indent=2, default=str))
        return 0
    from lenscluster.mock_validation import run_mock_validation, write_aggregate_plots
    print(f"[config] preset={options['preset']} seed={options['seed']} realizations={options['realizations']} "
          f"scatter_ln={scatter_ln} pos_sigma_arcsec={pos_sigma_arcsec} n_subhalos={config.mock.n_subhalos} "
          f"n_starts={config.solver.search.n_starts} chains={cores} cpus={cpu_plan.selected_cpu_ids} campaign_dir={campaign_dir}")
    start = time.monotonic()
    try:
        if not options["aggregate_only"]:
            run_mock_validation(config)
        for path in write_aggregate_plots(campaign_dir):
            print(f"[aggregate] {path}")
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
