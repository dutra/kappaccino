"""Run a cluster declared in a catalog folder: ``kappaccino <folder>`` (or ``python -m kappaccino <folder>``).

The folder holds ``run.yaml`` and the ``priors.yaml`` it names (:mod:`kappaccino.run_files`).  Paths in them are
resolved against the working directory, so run from the repository root::

    kappaccino catalogs/hera --dry-run
    kappaccino catalogs/m0416/canucs2025 --profile fast
    kappaccino catalogs/tngc014/large --set sampling.warmup=1000 --set runtime.chains=8

Outputs go to ``results/<name>/<description>-<hash>``: :func:`describe` of the run (``w500s250t7_allblocks``,
``w500s250t8_systemblocks_fast``) and six characters of a hash of its configuration (:func:`config_hash`), so every
configuration has its own folder and running the same command again, on any day, resumes it.  ``--run-label`` names
the folder by hand instead, and ``--results`` moves the tree.  The files the run read and the command are copied to
the run's ``inputs/`` folder the first time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

# Likelihood (mode, kernel) -> its tag in the run label; the GRF with a knee spectrum (the cluster runs) has none.
LIKELIHOOD_TAGS = {("gp_image_plane", "knee"): "", ("gp_image_plane", "image_plane"): "imagegp", ("smoothed", "image_plane"): "smoothed"}


def describe(config, profiles=()) -> str:
    """The default run label: NUTS warmup, samples and tree depth, then the GP blocks (GRF likelihood) or the likelihood,
    the image catalog when it is not the one of ``priors.yaml``'s folder, ``uncertain`` with uncertain images,
    ``powerlaw`` with cored power-law members (and their slope when it is held fixed), and the profiles.  E.g. ``w500s250t7_allblocks``, ``w500s250t8_systemblocks_fast``, ``w500s250t8_smoothed_trusted_uncertain_fast``."""
    sampling, likelihood = config.sampling, config.likelihood
    parts = [f"w{sampling.warmup}s{sampling.samples}t{sampling.max_tree_depth}"]
    tag = LIKELIHOOD_TAGS.get((likelihood.likelihood_mode, likelihood.gp_kernel), likelihood.likelihood_mode.replace("_", ""))
    parts.append(f"{likelihood.gp_blocks}blocks" if not tag else tag)
    images = Path(config.model.image_constraints.catalog_path)
    if images.name != "images.yaml":
        parts.append(images.stem)
    if config.partition is not None:
        parts.append("uncertain")
    if any(int(population.profile_type) == 102 for population in config.model.member_populations):
        fixed = "slope" in config.scaling.fixed_relation_fields
        parts.append("powerlaw" + (f"{config.model.member_populations[0].slope:g}" if fixed else ""))
    return "_".join([*parts, *profiles])


def config_hash(config) -> str:
    """Six hex characters identifying a configuration: a hash of everything the pipeline compares when it resumes a
    run (the whole configuration but ``paths`` and the WBIC settings, which a finished run may add).  Equal hashes are
    resumable runs; any change, a prior set with ``--set`` included, gives a new folder."""
    from .pipeline import _without_wbic
    payload = {key: value for key, value in _without_wbic(config.to_json_dict()).items() if key != "paths"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:6]


def _segment(text: str) -> str:
    if not text.strip() or text in {".", ".."} or "/" in text or "\\" in text:
        raise argparse.ArgumentTypeError("must be a nonempty single path segment")
    return text


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _accept(text: str) -> float:
    value = float(text)
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError("must lie in (0, 1)")
    return value


# Named options: the fields each sets, after the profiles and before --set (so --set wins).
FIELD_OPTIONS = {"chains": ("runtime.chains",), "seed": ("runtime.seed",), "starts": ("search.n_starts",),
                 "warm_start": ("search.warm_start",), "warmup": ("sampling.warmup",), "samples": ("sampling.samples",),
                 "tree_depth": ("sampling.max_tree_depth",), "target_accept": ("sampling.target_accept",),
                 "draws": ("reporting.draws", "reporting.image_draws")}


def _option_overrides(args) -> list[str]:
    """The ``section.field=value`` assignments of the named options that were given."""
    out = [f"{field}={getattr(args, name)}" for name, targets in FIELD_OPTIONS.items() if getattr(args, name) is not None
           for field in targets]
    return out + (["runtime.skip_plots=true"] if args.skip_plots else [])


def _slope(text: str) -> float:
    value = float(text)
    if not 1.0 < value < 3.0:
        raise argparse.ArgumentTypeError("must lie in (1, 3)")
    return value


def _folder(text: str) -> str:
    if not (Path(text) / "run.yaml").is_file():
        raise argparse.ArgumentTypeError(f"no run.yaml in {text!r}")
    return text


def parser() -> argparse.ArgumentParser:
    """The command-line options."""
    p = argparse.ArgumentParser(prog="kappaccino", description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cluster", type=_folder, help="the cluster folder, e.g. catalogs/hera")
    p.add_argument("--profile", action="append", default=[], metavar="NAME",
                   help="apply a profile of run.yaml (fast; jwst or hst for the figures' imaging), in order; repeatable")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.FIELD=VALUE",
                   help="override one field after the profiles, e.g. sampling.warmup=1000, runtime.skip_plots=true, "
                        "search.warm_start=results/<name>/<run>, sampling.wbic=true; repeatable")
    p.add_argument("--run-label", type=_segment, default=None,
                   help="the run folder (default <description>-<hash> of the run, e.g. w500s250t7_allblocks-3f9a1c)")
    p.add_argument("--results", default="results", metavar="DIR", help="the results tree, DIR/<name>/<run> (default results)")
    p.add_argument("--chains", type=_positive_int, metavar="N", help="NUTS chains, one CPU device each (runtime.chains)")
    p.add_argument("--seed", type=int, metavar="N", help="random seed of the search, the sampler and the reporting draws (runtime.seed)")
    p.add_argument("--starts", type=_positive_int, metavar="N", help="starts of the initial search (search.n_starts)")
    p.add_argument("--warm-start", metavar="RUN_DIR", help="start the search from a previous run's saved start (search.warm_start)")
    p.add_argument("--warmup", type=_positive_int, metavar="N", help="NUTS warmup draws per chain (sampling.warmup)")
    p.add_argument("--samples", type=_positive_int, metavar="N", help="NUTS draws per chain after warmup (sampling.samples)")
    p.add_argument("--tree-depth", type=_positive_int, metavar="N", help="NUTS maximum tree depth (sampling.max_tree_depth)")
    p.add_argument("--target-accept", type=_accept, metavar="P", help="NUTS target acceptance in (0, 1) (sampling.target_accept)")
    p.add_argument("--draws", type=_positive_int, metavar="N",
                   help="posterior draws behind every reported product (reporting.draws and reporting.image_draws)")
    p.add_argument("--skip-plots", action="store_true", help="no figures or tables after sampling (runtime.skip_plots)")
    p.add_argument("--member-profile", choices=("dpie", "power-law"), default=None,
                   help="the member halos: dpie, or power-law, the cored power law (the dPIE with a free 3D slope gamma, "
                        "rho ~ r^-gamma between core and cut, at the dPIE's mass inside the cut), one sampled slope per "
                        "population.  Default: what priors.yaml declares")
    p.add_argument("--member-slope", type=_slope, default=None, metavar="G",
                   help="power-law members: hold the slope at G, in (1, 3), instead of sampling it")
    p.add_argument("--images", default=None, metavar="YAML",
                   help="fit this image catalog instead of the one priors.yaml names (e.g. a degraded catalog). Images of "
                        "unknown family: --set partition.catalog_path=CSV (null for the untrusted images of the catalog), "
                        "partition.redshift_grid=[Z_MIN,Z_MAX,POINTS]")
    p.add_argument("--stage", choices=("init", "final", "both"), default="both")
    p.add_argument("--dry-run", action="store_true", help="classify the members and print the model's size; write nothing")
    return p


def build_config(cluster, *, profiles=(), overrides=(), run_label=None, images=None, results="results",
                 member_profile=None, member_slope=None):
    """The configuration of a cluster folder with the profiles and overrides, writing to
    ``<results>/<name>/<run_label>`` (run_label None: ``<description>-<hash>``, :func:`describe` and :func:`config_hash`), with ``images`` in place of the image
    catalog of ``priors.yaml``.  ``member_profile`` ("dpie" or "power-law") sets the profile of every member population
    (None keeps what ``priors.yaml`` declares), and ``member_slope`` holds the power-law slope fixed.

    Returns
    -------
    LensClusterSolverConfig
        Not validated.
    """
    from .run_files import load_run
    config = load_run(cluster, profiles, overrides)
    if images is not None:
        config = replace(config, model=replace(config.model, image_constraints=replace(config.model.image_constraints, catalog_path=images)))
    if member_profile is not None or member_slope is not None:
        # Profile 102 is the cored power law (MemberPopulationConfig): one 3D slope per population, sampled or held fixed.
        code = {"dpie": 81, "power-law": 102, None: None}[member_profile]
        populations = tuple(replace(p, **({} if code is None else {"profile_type": code}),
                                    **({} if member_slope is None else {"slope": float(member_slope)}))
                            for p in config.model.member_populations)
        if member_slope is not None and any(int(p.profile_type) != 102 for p in populations):
            raise ValueError("--member-slope needs power-law members (--member-profile power-law, or profile_type 102 in priors.yaml).")
        config = replace(config, model=replace(config.model, member_populations=populations))
        if member_slope is not None and "slope" not in config.scaling.fixed_relation_fields:
            config = replace(config, scaling=replace(config.scaling, fixed_relation_fields=(*config.scaling.fixed_relation_fields, "slope")))
    label = run_label or f"{describe(config, profiles)}-{config_hash(config)}"
    return replace(config, paths=replace(config.paths, output_dir=str(Path(results) / config.paths.figure_prefix), run_name=label))


def _fast_warning(config) -> None:
    """A banner that cannot be missed: a fast run uses exploratory settings."""
    from rich.console import Console
    from rich.panel import Panel
    members = config.members
    cut = members.min_deflection_arcsec
    text = ("This is a FAST run.  Its settings are exploratory:\n"
            + (f"  - members deflecting every image by less than {cut:g}\" are dropped from the model\n" if cut is not None
               else "  - every catalog member is kept (no impact cut)\n")
            + f"  - far-field members are allowed down to {members.far_field_cut_radii:g} cut radii from an image (default 3)\n"
            f"  - {config.search.n_starts} search starts, {config.sampling.warmup} warmup + {config.sampling.samples} draws per chain "
            f"at tree depth {config.sampling.max_tree_depth}, target accept {config.sampling.target_accept:g}\n"
            f"  - GP covariance blocks: {config.likelihood.gp_blocks} (full runs default to all blocks)")
    Console().print(Panel(text, title="[bold]WARNING: fast profile[/bold]", border_style="yellow", style="yellow", expand=False))


def main(argv=None) -> int:
    """Run (or dry-run) one cluster folder; returns the exit code."""
    args = parser().parse_args(argv)
    args.overrides = [*_option_overrides(args), *args.overrides]
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kappaccino")
    from .cpu_runtime import configure_cpu_runtime, resolve_cpu_plan
    from .run_files import input_files, read_run
    # The CPU devices are set before anything starts JAX (building a prior does).
    runtime = read_run(args.cluster, args.profile, args.overrides).get("runtime") or {}
    chains, method = int(runtime.get("chains", 1)), runtime.get("nuts_chain_method", "parallel")
    cpu_plan = resolve_cpu_plan(chains, method)
    if not args.dry_run:
        configure_cpu_runtime(chains, method)
    config = build_config(args.cluster, profiles=args.profile, overrides=args.overrides, results=args.results,
                          member_profile=args.member_profile, member_slope=args.member_slope,
                          run_label=args.run_label, images=args.images).validate()
    if args.dry_run:
        from .pipeline import dry_run
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), **dry_run(config)}, indent=2, default=str))
        return 0
    # The files the run was declared with, kept as first written: a resumed run must have the same configuration (the
    # pipeline refuses a changed one), so a later call never overwrites them.
    inputs = Path(config.paths.output_dir) / config.paths.run_name / "inputs"
    if not inputs.exists():
        inputs.mkdir(parents=True)
        for index, path in enumerate(input_files(args.cluster, args.profile)):
            shutil.copy2(path, inputs / f"{index}_{path.name}")
        (inputs / "command.json").write_text(json.dumps(
            {"cluster": args.cluster, "profiles": args.profile, "overrides": args.overrides, "images": args.images,
             "member_profile": args.member_profile, "member_slope": args.member_slope}, indent=2))
    from .pipeline import run
    fast = "fast" in args.profile
    if fast:
        _fast_warning(config)
    print(f"[config] cluster={args.cluster} profiles={','.join(args.profile) or '-'} run_name={config.paths.run_name} "
          f"output_dir={config.paths.output_dir} stage={args.stage} chains={config.runtime.chains} "
          f"search={config.search.halo_method}+{config.search.member_method} n_starts={config.search.n_starts} "
          f"likelihood={config.likelihood.likelihood_mode}/{config.likelihood.gp_kernel} gp_blocks={config.likelihood.gp_blocks}",
          flush=True)
    start = time.monotonic()
    try:
        run(config, stage=args.stage)
        if fast:
            _fast_warning(config)
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0
