"""Run a cluster declared in a catalog folder: ``kappaccino <folder>`` (or ``python -m kappaccino <folder>``).

The folder holds ``run.yaml`` and the ``priors.yaml`` it names (:mod:`kappaccino.run_files`).  Paths in them are
resolved against the working directory, so run from the repository root::

    kappaccino catalogs/hera --dry-run
    kappaccino catalogs/m0416/canucs2025 --profile fast --campaign nov11 --run-label fast
    kappaccino catalogs/tngc014 --set sampling.warmup=1000 --set runtime.chains=8

Outputs go to ``results/<campaign>/<name>/<run label>``; the campaign defaults to today's date (``nov11``), so pass
``--campaign`` to resume a run on a later day.  The files the run read are copied to its ``inputs/`` folder the first
time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

RUN_LABEL = "smoothed-local-image-plane"


def today() -> str:
    """The default campaign: today's date as month and day, e.g. ``nov11``."""
    day = date.today()
    return f"{day.strftime('%b').lower()}{day.day}"


def _segment(text: str) -> str:
    if not text.strip() or text in {".", ".."} or "/" in text or "\\" in text:
        raise argparse.ArgumentTypeError("must be a nonempty single path segment")
    return text


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
                        "search.warm_start=results/<campaign>/<name>/<run>, sampling.wbic=true; repeatable")
    p.add_argument("--campaign", type=_segment, default=None, help="results/<campaign>/ (default today's date, e.g. nov11)")
    p.add_argument("--run-label", type=_segment, default=RUN_LABEL, help=f"the run folder (default {RUN_LABEL})")
    p.add_argument("--images", default=None, metavar="YAML",
                   help="fit this image catalog instead of the one priors.yaml names (e.g. a degraded catalog). Images of "
                        "unknown family: --set partition.catalog_path=CSV (null for the untrusted images of the catalog), "
                        "partition.redshift_grid=[Z_MIN,Z_MAX,POINTS]")
    p.add_argument("--stage", choices=("init", "final", "both"), default="both")
    p.add_argument("--dry-run", action="store_true", help="classify the members and print the model's size; write nothing")
    return p


def build_config(cluster, *, profiles=(), overrides=(), campaign=None, run_label=RUN_LABEL, images=None):
    """The configuration of a cluster folder with the profiles and overrides, writing to
    ``results/<campaign>/<name>/<run_label>`` (campaign None: :func:`today`), with ``images`` in place of the image
    catalog of ``priors.yaml``.  CPU pinning follows the platform unless the files or ``--set`` give it.

    Returns
    -------
    LensClusterSolverConfig
        Not validated.
    """
    from .cpu_runtime import threads_per_chain_if_supported
    from .run_files import load_run, read_run
    threads = (read_run(cluster, profiles, overrides).get("runtime") or {}).get("cpu_threads_per_chain")
    config = load_run(cluster, profiles, overrides)
    if images is not None:
        config = replace(config, model=replace(config.model, image_constraints=replace(config.model.image_constraints, catalog_path=images)))
    campaign = campaign or today()
    return replace(config,
                   paths=replace(config.paths, output_dir=f"results/{campaign}/{config.paths.figure_prefix}", run_name=run_label),
                   runtime=replace(config.runtime, cpu_threads_per_chain=threads if threads is not None else threads_per_chain_if_supported(2)))


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
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-kappaccino")
    from .cpu_runtime import configure_cpu_runtime, resolve_cpu_plan, threads_per_chain_if_supported
    from .run_files import input_files, read_run
    # The CPU devices are set before anything starts JAX (building a prior does).
    runtime = read_run(args.cluster, args.profile, args.overrides).get("runtime") or {}
    chains, method = int(runtime.get("chains", 1)), runtime.get("nuts_chain_method", "parallel")
    threads = runtime.get("cpu_threads_per_chain") or threads_per_chain_if_supported(2)
    cpu_plan = resolve_cpu_plan(chains, method, threads)
    if not args.dry_run:
        configure_cpu_runtime(chains, method, threads)
    config = build_config(args.cluster, profiles=args.profile, overrides=args.overrides, campaign=args.campaign,
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
            {"cluster": args.cluster, "profiles": args.profile, "overrides": args.overrides, "images": args.images}, indent=2))
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
