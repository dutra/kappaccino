$XONSH_SHOW_TRACEBACK = True

import json
import os
import shutil
import sys
import time
from pathlib import Path

$MPLCONFIGDIR = "/tmp/matplotlib-kappaccino"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib-kappaccino"
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataclasses import replace

from kappaccino.cpu_runtime import configure_cpu_runtime, resolve_cpu_plan, threads_per_chain_if_supported
from kappaccino.run_files import input_files, load_run, read_run

OUTPUT_DIR_LABEL = "oct22_single_stage"
RUN_NAME = "smoothed-local-image-plane"
CLUSTERS = ("catalogs/hera", "catalogs/ares", "catalogs/as1063/bergamini2019", "catalogs/as1063/diego2026",
            "catalogs/m0416/bergamini2023", "catalogs/m0416/canucs2025")


def _usage() -> str:
    return (
        "Usage: xonsh run.xsh <cluster folder> [--profile NAME]... [--set section.field=value]... [--campaign NAME]\n"
        "                     [--run-label NAME] [--images YAML] [--stage init|final|both] [--dry-run]\n"
        "  The cluster folder holds run.yaml and the priors.yaml it names (src/kappaccino/run_files.py).\n"
        "  --profile NAME: apply a profile of run.yaml (fast; jwst or hst for the figures' imaging), in order\n"
        "  --set section.field=value: override one field after the profiles, e.g. sampling.warmup=1000,\n"
        "          search.warm_start=results/<campaign>/<cluster>/<run>, runtime.skip_plots=true, sampling.wbic=true,\n"
        "          likelihood.lambda_b_prior='lognormal(20, 1) in [0.5, 80]'\n"
        "  --images YAML: fit this image catalog instead of the one priors.yaml names (e.g. a degraded catalog)\n"
        "  Uncertain images of unknown family: --set partition.catalog_path=CSV, or partition.catalog_path=null for the\n"
        "          untrusted images of the image catalog; partition.redshift_grid=[Z_MIN,Z_MAX,POINTS] for their quadrature\n"
        "  Outputs go to results/<campaign>/<name>/<run label>, with the input files copied to its inputs/ folder.\n"
        f"Clusters: {', '.join(CLUSTERS)}"
    )


def _parse_args(argv: list[str]) -> dict:
    if len(argv) < 2 or argv[1] in {"--help", "-h"}:
        print(_usage())
        raise SystemExit(0 if len(argv) == 2 else 2)
    values = {"cluster": argv[1], "profiles": [], "overrides": [], "campaign": OUTPUT_DIR_LABEL, "run_label": RUN_NAME,
              "stage": "both", "dry_run": False, "images": None}
    flags = {"--profile": "profiles", "--set": "overrides", "--campaign": "campaign", "--run-label": "run_label",
             "--stage": "stage", "--images": "images"}
    index = 2
    while index < len(argv):
        flag = argv[index]
        if flag == "--dry-run":
            values["dry_run"] = True
            index += 1
            continue
        if flag not in flags or index + 1 >= len(argv):
            print(f"Unknown argument or missing value: {flag!r}\n{_usage()}")
            raise SystemExit(2)
        key = flags[flag]
        if isinstance(values[key], list):
            values[key].append(argv[index + 1])
        else:
            values[key] = argv[index + 1]
        index += 2
    for key in ("campaign", "run_label"):
        value = str(values[key])
        if not value.strip() or value in {".", ".."} or "/" in value or "\\" in value:
            print(f"--{key.replace('_', '-')} must be a nonempty single path segment.")
            raise SystemExit(2)
    if values["stage"] not in {"init", "final", "both"}:
        print("--stage must be init, final or both.")
        raise SystemExit(2)
    if not (Path(values["cluster"]) / "run.yaml").is_file():
        print(f"No run.yaml in {values['cluster']!r}\n{_usage()}")
        raise SystemExit(2)
    return values


def build_config(cluster: str, *, profiles=(), overrides=(), campaign: str = OUTPUT_DIR_LABEL, run_label: str = RUN_NAME,
                 images: str | None = None):
    """The configuration of a cluster folder: its run.yaml and priors.yaml with the profiles and overrides, written to
    results/<campaign>/<name>/<run_label>, with ``images`` in place of the image catalog of priors.yaml.  CPU pinning
    follows the platform unless run.yaml or --set gives it."""
    run = read_run(cluster, profiles, overrides)
    threads = (run.get("runtime") or {}).get("cpu_threads_per_chain")
    config = load_run(cluster, profiles, overrides)
    if images is not None:
        config = replace(config, model=replace(config.model, image_constraints=replace(config.model.image_constraints, catalog_path=images)))
    return replace(config, paths=replace(config.paths, output_dir=f"results/{campaign}/{config.paths.figure_prefix}", run_name=run_label),
                   runtime=replace(config.runtime, cpu_threads_per_chain=threads if threads is not None else threads_per_chain_if_supported(2)))


def _fast_mode_warning(config) -> None:
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
    Console(stderr=False).print(Panel(text, title="[bold]WARNING: fast profile[/bold]", border_style="yellow", style="yellow", expand=False))


def main(argv=None):
    options = _parse_args(sys.argv if argv is None else argv)
    cluster, profiles, overrides = options["cluster"], options["profiles"], options["overrides"]
    # The CPU devices are set before anything starts JAX (building a prior does).
    runtime = read_run(cluster, profiles, overrides).get("runtime") or {}
    chains, method = int(runtime.get("chains", 1)), runtime.get("nuts_chain_method", "parallel")
    threads = runtime.get("cpu_threads_per_chain") or threads_per_chain_if_supported(2)
    cpu_plan = resolve_cpu_plan(chains, method, threads)
    if not options["dry_run"]:
        configure_cpu_runtime(chains, method, threads)
    config = build_config(cluster, profiles=profiles, overrides=overrides, campaign=options["campaign"],
                          run_label=options["run_label"], images=options["images"]).validate()
    if options["dry_run"]:
        from kappaccino.pipeline import dry_run
        print(json.dumps({"dry_run": True, "cpu_plan": cpu_plan.to_dict(), **dry_run(config)}, indent=2, default=str))
        return 0
    # The files the run was declared with, kept as first written: a resumed run must have the same configuration (the
    # pipeline refuses a changed one), so a later call never overwrites them.
    inputs = Path(config.paths.output_dir) / config.paths.run_name / "inputs"
    if not inputs.exists():
        inputs.mkdir(parents=True)
        for index, path in enumerate(input_files(cluster, profiles)):
            shutil.copy2(path, inputs / f"{index}_{path.name}")
        (inputs / "command.json").write_text(json.dumps({"cluster": cluster, "profiles": profiles, "overrides": overrides, "images": options["images"]}, indent=2))
    from kappaccino.pipeline import run
    fast = "fast" in profiles
    if fast:
        _fast_mode_warning(config)
    print(f"[config] cluster={cluster} profiles={','.join(profiles) or '-'} run_name={config.paths.run_name} "
          f"output_dir={config.paths.output_dir} stage={options['stage']} chains={config.runtime.chains} "
          f"search={config.search.halo_method}+{config.search.member_method} n_starts={config.search.n_starts} "
          f"likelihood={config.likelihood.likelihood_mode}/{config.likelihood.gp_kernel} gp_blocks={config.likelihood.gp_blocks}")
    start = time.monotonic()
    try:
        run(config, stage=options["stage"])
        if fast:
            _fast_mode_warning(config)
    finally:
        print(f"[timing] elapsed={time.monotonic() - start:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
