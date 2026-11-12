"""Run a mock-validation campaign: ``kappaccino-mock --campaign NAME`` (or ``python -m kappaccino.mock_validation``).

Each realization draws an analytic cluster, lenses its sources, fits it with the solver and scores the recovery
(:func:`kappaccino.mock_validation.run_mock_validation`)::

    kappaccino-mock --campaign nov11_smoke --dry-run
    kappaccino-mock --campaign nov11 --preset production --realizations 4 --fast
    kappaccino-mock --campaign nov11 --preset production --set likelihood.gp_blocks=system

Outputs go to ``<output dir>/<campaign>/analytic_<mode>_<preset>[_fast][_<likelihood>]/seed_<seed>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

CHAINS = 4                      # NUTS chains; one JAX CPU device per chain (the cluster runs' budget)
SEED = 12345
OUTPUT_DIR = "validation_runs"

# What the mock injects.  The measurement noise is the truth of the sampled independent scatter images.sigma_meas (every
# likelihood); the solver assumes no fixed astrometric error.  The source-plane scatter is the truth of sigma_def (smoothed
# likelihood only; under the GP likelihoods it is unmodelled scatter that the image-plane terms must absorb); the member
# scatters are the truths of the two population scatters (0.10 is the median of the solver's dispersion-scatter prior).
# Unmodelled members are subhalos present in the truth but missing from the catalog, the misspecification experiment.
# 0.015" per axis: the scatter between the CANUCS (JWST) and Bergamini et al. (HST) positions of M0416, 0.014-0.015" per
# axis over 207 matched images, and the measurement scatter the M0416 fit settles at (0.012-0.015").
MEASUREMENT_NOISE_ARCSEC = 0.015
# The injected white scatter varies between realizations (2026-11-06): log-uniform over 5-20 mas, JWST centroids of
# compact knots, inside the solver's sigma_meas prior (median 0.015", log-width 0.5, cap 0.1"); None injects
# MEASUREMENT_NOISE_ARCSEC everywhere.
MEASUREMENT_NOISE_RANGE_ARCSEC = (0.005, 0.02)
SIGMA_DEF_ARCSEC = 0.0           # astrometric error only (2026-11-05); 0.05 adds source-plane scatter the GP likelihoods do not model
SCATTER_LN = 0.10
CUT_SCATTER_LN = 0.0
UNMODELLED_MEMBERS = 0
# Correlated errors, drawn per realization (mock_validation.field): the mass field of unmodelled mass and line-of-sight
# structure with sigma_grf 0.04-0.10" at z = 2 (the M0416 fits give 0.06-0.11"), knee 3-12" (log-uniform) and slope 3-5;
# the image-plane displacement with sigma 0.02-0.06" and ell 0.5-2" (log-uniform).  Raised back from the JWST amplitudes
# (sigma_grf 0.02-0.05", sigma_gp 0.01-0.04", nov6b_correlated_v1) on 2026-11-06: at those the free lens absorbed a
# quarter of the field (sigma_grf 0.67-0.89x truth, 0/4 covered; .agents/nov6-mock-sigma-alpha-free-lens.md) and ell was
# measurable in one seed of four (it needs sigma_gp well above the 5-20 mas white noise).  The solver samples the slope
# (uniform 2-6) since the truth varies it, and its sigma_grf and sigma_gp priors are centred on these ranges (likelihood).
FIELD_SIGMA_GRF_RANGE_ARCSEC = (0.04, 0.10)
FIELD_KNEE_RANGE_ARCSEC = (3.0, 12.0)
FIELD_GAMMA_RANGE = (3.0, 5.0)
IMAGE_PLANE_SIGMA_RANGE_ARCSEC = (0.02, 0.06)
IMAGE_PLANE_ELL_RANGE_ARCSEC = (0.5, 2.0)
# Sources with knots (2026-11-06): this many primary families are galaxies with 2-3 knots 0.05-1.0" apart in the source
# plane (log-uniform), each knot its own family of one system, so images come in groups that sample the image-plane GP's
# ell.  Offsets above ~0.3" rarely survive the image-count check (they cross a caustic), so knot groups stay under 2"; the
# same-system pairs at 2-15" that pin the field's knee are the fold pairs of the knotted families, duplicated once per
# knot: five knotted families give 100-145 such pairs per mock against 70-80 with three (nov6b).  All-image blocks see the
# 600-1400 cross-system pairs at those separations as well (Fisher knee contraction 80-92% against 76-87% with system
# blocks, ell 58% against 30-34%, seeds 12345-12346).  --knotted-families overrides.
KNOTTED_FAMILIES = {"production": 5, "smoke": 1}
KNOTS_PER_FAMILY = (2, 3)
KNOT_OFFSET_RANGE_ARCSEC = (0.05, 1.0)

# Likelihood names -> (LikelihoodConfig.likelihood_mode, gp_kernel), as in the cluster run.yaml files.
LIKELIHOODS = {"grf_gp": ("gp_image_plane", "knee"), "gp_image_plane": ("gp_image_plane", "image_plane"),
               "smoothed": ("smoothed", "image_plane")}
# Named options: the solver fields each sets, after the preset and --fast and before --set (so --set wins).
FIELD_OPTIONS = {"chains": ("runtime.chains",), "starts": ("search.n_starts",), "warmup": ("sampling.warmup",),
                 "samples": ("sampling.samples",), "tree_depth": ("sampling.max_tree_depth",),
                 "target_accept": ("sampling.target_accept",), "draws": ("reporting.draws", "reporting.image_draws")}


def likelihood(name: str, gp_blocks: str):
    """The likelihood of the cluster run.yaml files (keep them in step): no selection term; the GRF of unmodelled mass with
    a knee spectrum (slope sampled on 2-6, knee 2.5-15" around 6") and the image-plane GP (ell 0.4-3" around 1"), with the
    priors centred on the mock's injection ranges (sigma_grf 0.06", sigma_gp 0.035" median); the shared field's
    translation, convergence and shear projected out on all blocks (system blocks, the --fast default, have no projection)."""
    from ..config import LikelihoodConfig
    from ..priors import truncated_lognormal
    mode, kernel = LIKELIHOODS[name]
    return LikelihoodConfig(
        likelihood_mode=mode, sigma_sel_prior=None, gp_kernel=kernel, gp_blocks=gp_blocks,
        gp_knee_project_modes=(("translation", "convergence", "shear") if name == "grf_gp" and gp_blocks == "all" else ()),
        gp_knee_gamma_prior=(2.0, 6.0), gp_knee_gamma=4.0,     # sampled: the injected field varies the slope
        gp_knee_scales_arcsec=(0.2, 100.0),
        lambda_b_prior=truncated_lognormal(6.0, 0.6, low=2.5, high=15.0),
        lambda_b_start_arcsec=6.0,
        # Mock only (the cluster runs keep median 0.08", cap 0.5", and the library sigma_gp prior, median 0.10", cap 0.3"):
        # the geometric centres of the mock's injection ranges, each range inside about one prior width of the median.
        sigma_grf_prior=truncated_lognormal(0.06, 0.5, high=0.3),
        sigma_grf_start_arcsec=0.06,
        gp_knee_sigma_gp_prior=truncated_lognormal(0.035, 0.6, high=0.15),
        gp_knee_sigma_gp_start_arcsec=0.035,
        gp_knee_ell_gp_prior=truncated_lognormal(1.0, 0.4, low=0.4, high=3.0),
        gp_knee_ell_gp_start_arcsec=1.0,
    )


def solver(*, production: bool, fast: bool, seed: int, likelihood_name: str, gp_blocks: str | None):
    """The solver template: the settings of the cluster run.yaml files (search, sampling, members, likelihood) plus what the
    mock needs.  The lens model, with priors centred on the mock's truth, is attached per realization by the
    mock-validation pipeline.  The smoke preset keeps its small search and chains under --fast; the member and GP settings
    of --fast apply to both presets."""
    from ..config import (ImageDiagnosticsConfig, LensClusterSolverConfig, MemberConfig, ReportingConfig, RuntimeConfig,
                          SamplingConfig, ScalingModelConfig, SearchConfig)
    from ..cpu_runtime import threads_per_chain_if_supported
    # All-image blocks with the projection by default (2026-11-06), as full cluster runs: under system blocks the free lens
    # absorbed a quarter of the field (nov6b, .agents/nov6-mock-sigma-alpha-free-lens.md) and the fit lost the 600-1400
    # cross-system pairs at 2-15" that pin the knee and ell.  --fast keeps system blocks, about 5x cheaper per gradient.
    blocks = gp_blocks or ("system" if fast else "all")
    config = LensClusterSolverConfig(
        # Chains x 2 cores on Linux, no pinning on macOS; plots on, because the recovery reads the solver's output tables.
        runtime=RuntimeConfig(seed=seed, chains=CHAINS, resume=True, numpyro_print_summary=True, nuts_chain_method="parallel",
                              cpu_threads_per_chain=threads_per_chain_if_supported(2), skip_plots=False),
        search=SearchConfig(n_starts=128) if production else SearchConfig(n_starts=8),
        # Far-field member offsets pinned at the relation: their radius offsets never reach the likelihood and their velocity
        # offsets barely do, so they were sampled from the prior alone and set the orbit length (the softest directions,
        # 400-700 leapfrog steps per half orbit against the 255 cap, astrometric_bcg1as seed 12345).  Target accept 0.8 as
        # the cluster runs (the chains accepted 0.92-0.97 at 0.85, steps at 0.3-0.5 of the stiffness limit).
        sampling=(SamplingConfig(warmup=2000, samples=500, max_tree_depth=10, target_accept=0.8, far_field_offsets="fixed") if production
                  else SamplingConfig(warmup=100, samples=100, max_tree_depth=6, target_accept=0.8, far_field_offsets="fixed")),
        likelihood=likelihood(likelihood_name, blocks),
        # The 0.15 kpc member core is not a lensing observable; fixed at the truth value.  Normal member deviations: the
        # mock's scatter is Gaussian, and the Student-t tails of the cluster runs soften the posterior.
        scaling=ScalingModelConfig(fit_corekpc=False, member_deviation_tail="normal"),
        # Centered deviations for the members the data pin, unit offsets for the rest, classification at the current model
        # (reclassified at the MAP), as for the observed clusters.
        members=MemberConfig(centering="auto", classification_prior_quantile=None),
        image_diagnostics=ImageDiagnosticsConfig(match_tolerance_arcsec=3.0),
        reporting=ReportingConfig(draws=64, image_draws=64, extra_images=True) if production
                  else ReportingConfig(draws=32, image_draws=32, extra_images=False),
    )
    if fast:
        config = replace(
            config,
            sampling=SamplingConfig(warmup=500, samples=250, max_tree_depth=8, target_accept=0.8, far_field_offsets="fixed") if production else config.sampling,
            image_diagnostics=replace(config.image_diagnostics, fold_pair_max_separation_arcsec=3.0),
            reporting=ReportingConfig(draws=32, image_draws=32, extra_images=False),
            members=replace(config.members, far_field_cut_radii=2.0, classification_prior_quantile=None, min_deflection_arcsec=None,
                            cut_radius_inflation=3.0),
            scaling=replace(config.scaling, member_shape_scatter=None),
        )
    return config


def _overridden(config, overrides):
    """``config`` with ``section.field=value`` assignments merged in (values as in run.yaml: priors, arithmetic, YAML)."""
    from .. import config as _config
    from ..run_files import _dotted, _merge, parse_value
    if not overrides:
        return config
    payload = _merge(config.to_json_dict(), _config._jsonable(parse_value(_dotted(overrides))))
    return type(config).from_json_dict(payload)


def build_config(preset: str = "smoke", *, seed: int = SEED, realizations: int = 1, campaign: str, output_dir: str = OUTPUT_DIR,
                 mode: str = "production", fast: bool = False, likelihood_name: str = "grf_gp", gp_blocks: str | None = None,
                 knotted: int | None = None, overrides=()):
    """The campaign configuration: the mock of the preset and the solver template with ``overrides`` applied.

    Returns
    -------
    MockValidationConfig
        Not validated.
    """
    from . import MockConfig, MockValidationConfig, MockValidationPathsConfig, MockValidationRuntimeConfig
    production = preset == "production"
    mock = MockConfig(
        seed=seed,
        mode=mode,
        n_members=150 if production else 10,   # Hera has 332 members, AS1063 222; far-field approximation for the distant ones
        n_primary_families=15 if production else 4,
        n_member_families=5 if production else 1,   # galaxy-galaxy families are what constrain sigma* and r_cut*
        measurement_noise_arcsec=MEASUREMENT_NOISE_ARCSEC,
        measurement_noise_range_arcsec=MEASUREMENT_NOISE_RANGE_ARCSEC,
        n_knotted_families=KNOTTED_FAMILIES[preset] if knotted is None else int(knotted),
        knots_per_family=KNOTS_PER_FAMILY,
        knot_offset_range_arcsec=KNOT_OFFSET_RANGE_ARCSEC,
        sigma_def_arcsec=SIGMA_DEF_ARCSEC,
        sigma_scatter_ln=SCATTER_LN,
        cut_scatter_ln=CUT_SCATTER_LN,
        unmodelled_members=UNMODELLED_MEMBERS,
        field_sigma_grf_range_arcsec=FIELD_SIGMA_GRF_RANGE_ARCSEC,
        field_knee_range_arcsec=FIELD_KNEE_RANGE_ARCSEC,
        field_gamma_range=FIELD_GAMMA_RANGE,
        image_plane_sigma_range_arcsec=IMAGE_PLANE_SIGMA_RANGE_ARCSEC,
        image_plane_ell_range_arcsec=IMAGE_PLANE_ELL_RANGE_ARCSEC,
    )
    template = _overridden(solver(production=production, fast=fast, seed=seed, likelihood_name=likelihood_name,
                                  gp_blocks=gp_blocks), overrides)
    run_name = f"analytic_{mode}_{preset}" + ("_fast" if fast else "") + ("" if likelihood_name == "grf_gp" else f"_{likelihood_name}")
    return MockValidationConfig(
        mock=mock,
        paths=MockValidationPathsConfig(output_dir=output_dir, campaign_name=campaign, run_name=run_name),
        runtime=MockValidationRuntimeConfig(realizations=realizations, seed=seed, resume=True, quiet=False),
        solver=template,
    )


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def _accept(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must lie in (0, 1)")
    return value


def parser() -> argparse.ArgumentParser:
    """The command-line options."""
    p = argparse.ArgumentParser(prog="kappaccino-mock", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--campaign", required=True, help="the campaign folder under the output dir")
    p.add_argument("--preset", choices=("smoke", "production"), default="smoke",
                   help="production: the solver settings of a full cluster run (128 search starts, 2000+500 NUTS draws at "
                        "tree depth 10) on 150 members and 15+5 families; smoke (default): 10 members and 4+1 families, 8 "
                        "starts, 100+100 draws at tree depth 6, for tests of the pipeline itself")
    p.add_argument("--mode", choices=("production", "idealized"), default="production",
                   help="production (default): the priors of the real clusters (halo on the BCG light, the Bergamini et al. "
                        "2019 member relation); idealized: priors centred on each realization's truth, which isolates the "
                        "sampler and the likelihood")
    p.add_argument("--seed", type=_nonnegative_int, default=SEED, help=f"seed of the first realization (default {SEED})")
    p.add_argument("--realizations", type=_positive_int, default=1, help="number of mock clusters (default 1)")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help=f"default {OUTPUT_DIR}")
    p.add_argument("--likelihood", choices=sorted(LIKELIHOODS), default="grf_gp",
                   help="grf_gp (default; the mass-field GRF with a knee spectrum plus the image-plane GP, as the cluster "
                        "runs), gp_image_plane (the image-plane GP alone, per family) or smoothed (the per-image "
                        "source-plane floor sigma_def and sigma_meas)")
    p.add_argument("--gp-blocks", choices=("all", "system", "family"),
                   help="grf_gp only: all (default: one shared field with translation, convergence and shear projected out), "
                        "system (an independent field per system, no projection; the --fast default) or family")
    p.add_argument("--knotted-families", type=_nonnegative_int, metavar="N",
                   help="primary families that are galaxies with knots (default 5 for production, 1 for smoke)")
    p.add_argument("--fast", action="store_true",
                   help="the exploratory settings of the cluster fast profile: far-field members down to 2 cut radii from "
                        "an image, classification at the current model, 500+250 draws at tree depth 8 (production), system "
                        "GP blocks, member shapes fixed, 32 reporting draws")
    p.add_argument("--chains", type=_positive_int, metavar="N", help=f"NUTS chains, one CPU device each (default {CHAINS})")
    p.add_argument("--starts", type=_positive_int, metavar="N", help="starts of the initial search (search.n_starts)")
    p.add_argument("--warmup", type=_positive_int, metavar="N", help="NUTS warmup draws per chain (sampling.warmup)")
    p.add_argument("--samples", type=_positive_int, metavar="N", help="NUTS draws per chain after warmup (sampling.samples)")
    p.add_argument("--tree-depth", type=_positive_int, metavar="N", help="NUTS maximum tree depth (sampling.max_tree_depth)")
    p.add_argument("--target-accept", type=_accept, metavar="P", help="NUTS target acceptance in (0, 1) (sampling.target_accept)")
    p.add_argument("--draws", type=_positive_int, metavar="N",
                   help="posterior draws behind every reported product (reporting.draws and reporting.image_draws)")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.FIELD=VALUE",
                   help="override one field of the solver template, after the options above, e.g. "
                        "likelihood.gp_blocks=system; repeatable")
    p.add_argument("--fisher-only", action="store_true",
                   help="generate (or reuse) the mocks and forecast how well the GP hyperparameters are constrained (a "
                        "Fisher matrix at the truth), without running the solver; writes fisher.json per mock and "
                        "fisher_precheck.csv")
    p.add_argument("--aggregate-only", action="store_true", help="only redraw the campaign's aggregate plots")
    p.add_argument("--rescore", action="store_true", help="rescore the finished realizations, then redraw the aggregate plots")
    p.add_argument("--dry-run", action="store_true", help="print the configuration; write nothing")
    return p


def _fast_warning(template) -> None:
    """A banner that cannot be missed: a --fast run uses exploratory settings."""
    from rich.console import Console
    from rich.panel import Panel
    members = template.members
    text = ("This is a FAST run.  Its settings are exploratory:\n"
            f"  - far-field members are allowed down to {members.far_field_cut_radii:g} cut radii from an image (default 3)\n"
            f"  - {template.search.n_starts} search starts, {template.sampling.warmup} warmup + {template.sampling.samples} draws "
            f"per chain at tree depth {template.sampling.max_tree_depth}, target accept {template.sampling.target_accept:g}\n"
            f"  - GP covariance blocks: {template.likelihood.gp_blocks}, projection "
            f"{', '.join(template.likelihood.gp_knee_project_modes) or 'off'} (full runs use all blocks with the projection)")
    Console().print(Panel(text, title="[bold]WARNING: --fast[/bold]", border_style="yellow", style="yellow", expand=False))


def main(argv=None) -> int:
    """Run (or dry-run) one mock-validation campaign; returns the exit code."""
    args = parser().parse_args(argv)
    if args.gp_blocks is not None and args.likelihood != "grf_gp":
        parser().error("--gp-blocks applies to the grf_gp likelihood only")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kappaccino-mock")
    from ..cpu_runtime import configure_cpu_runtime, resolve_cpu_plan, threads_per_chain_if_supported
    overrides = [f"{field}={getattr(args, name)}" for name, targets in FIELD_OPTIONS.items() if getattr(args, name) is not None
                 for field in targets] + args.overrides
    # The CPU devices are set before anything starts JAX (building a prior does).
    from ..run_files import _dotted
    runtime = _dotted(overrides).get("runtime") or {}
    chains, threads = int(runtime.get("chains", CHAINS)), runtime.get("cpu_threads_per_chain", threads_per_chain_if_supported(2))
    method = runtime.get("nuts_chain_method", "parallel")
    cpu_plan = resolve_cpu_plan(chains, method, threads)
    if not args.dry_run:
        configure_cpu_runtime(chains, method, threads)
    config = build_config(args.preset, seed=args.seed, realizations=args.realizations, campaign=args.campaign,
                          output_dir=args.output_dir, mode=args.mode, fast=args.fast, likelihood_name=args.likelihood,
                          gp_blocks=args.gp_blocks, knotted=args.knotted_families, overrides=overrides).validate()
    campaign_dir = Path(config.paths.output_dir) / config.paths.campaign_name / config.paths.run_name
    if args.dry_run:
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), "campaign_dir": str(campaign_dir),
                          **config.to_json_dict()}, indent=2, default=str))
        return 0
    from . import fisher_precheck, rescore_mock_validation, run_mock_validation, write_aggregate_plots
    if args.fisher_only:
        start = time.monotonic()
        try:
            fisher_precheck(config)
        finally:
            print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
        return 0
    template = config.solver
    if args.fast:
        _fast_warning(template)
    print(f"[config] preset={args.preset} mode={args.mode} fast={args.fast} seed={args.seed} realizations={args.realizations} "
          f"scatter_ln={SCATTER_LN} injected_noise_arcsec={MEASUREMENT_NOISE_RANGE_ARCSEC or MEASUREMENT_NOISE_ARCSEC} "
          f"knotted_families={config.mock.n_knotted_families} sigma_def_arcsec={SIGMA_DEF_ARCSEC} members={config.mock.n_members} "
          f"n_starts={template.search.n_starts} warmup={template.sampling.warmup} samples={template.sampling.samples} "
          f"likelihood={args.likelihood} gp_blocks={template.likelihood.gp_blocks} chains={template.runtime.chains} "
          f"cpus={cpu_plan.selected_cpu_ids} campaign_dir={campaign_dir}", flush=True)
    start = time.monotonic()
    try:
        if args.rescore:
            rescore_mock_validation(config)
        elif not args.aggregate_only:
            run_mock_validation(config)
        if args.rescore or args.aggregate_only:
            for path in write_aggregate_plots(campaign_dir):
                print(f"[aggregate] {path}")
        if args.fast:
            _fast_warning(template)
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0
