"""Regenerate only the cluster mass overview from a saved init/final run; no inference."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, help="Directory containing init/ and final/")
    parser.add_argument("--imaging", choices=("saved", "hst", "jwst"), default="saved",
                        help="Imaging for the background: the run's saved configuration, the HST (BUFFALO) mosaics, or the JWST NIRCam mosaics (suffix _jwst)")
    sys.path.insert(0, str(ROOT / "scripts"))
    from _covariance_plot_cli import add_catalogs_argument, with_catalogs, with_ffsims_dir
    add_catalogs_argument(parser)
    args = parser.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from kappaccino.io import artifacts
    from kappaccino.plotting import plots
    from kappaccino.imaging.cutouts import imaging_variants, load_cluster_imaging
    from kappaccino.likelihood.evaluator import Evaluator
    from kappaccino.plotting.mass_figure import render_cluster_mass_figure
    from kappaccino.core.state import build_state, with_member_classes

    run = args.run_directory.resolve()
    init, final = run / "init" / "artifacts", run / "final"
    config = with_ffsims_dir(with_catalogs(artifacts.load_config(final / "artifacts"), args.catalogs), args.ffsims_dir)
    if args.imaging != "saved":
        from dataclasses import replace
        from kappaccino.imaging.rgb import DEFAULT_HFF_RGB_BANDS, DEFAULT_JWST_RGB_BANDS
        jwst = args.imaging == "jwst"
        from kappaccino.config import RGBDisplayConfig
        from kappaccino.imaging.cutouts import cluster_from_config
        cluster = cluster_from_config(config.model, config.paths.run_name)
        jwst_dir = f"data/imaging/{cluster}/{'canucs_dr1' if cluster == 'm0416' else 'dja_v7.5'}"
        hst_dir = f"data/imaging/{cluster}/buffalo"
        # The run's saved display numbers belong to the imaging it was run with; the library defaults carry the balance.
        config = replace(config, image_catalog=replace(config.image_catalog, image_dir=jwst_dir if jwst else hst_dir,
                                                       image_scale="auto", bands=tuple(DEFAULT_JWST_RGB_BANDS if jwst else DEFAULT_HFF_RGB_BANDS),
                                                       rgb=RGBDisplayConfig(), figure_suffix="_jwst" if jwst else ""))
    from dataclasses import replace
    config = replace(config, image_catalog=imaging_variants(config, overview_only=True)[0])
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get("dropped_members", ()))
    centered_ids = set(classes.get("centered_members", []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered_ids for m in state.members.member_ids]))
    state.initial_theta = artifacts.load_start(init)
    saved_state = json.loads((init / artifacts.STATE_FILE).read_text())
    if [p.name for p in state.parameter_specs] != [p["name"] for p in saved_state["parameter_specs"]]:
        raise ValueError("Rebuilt parameter order differs from saved run")
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    posterior = artifacts.load_posterior(final / "artifacts")
    from kappaccino.reporting.draws import reporting_draws
    draws = reporting_draws(config, evaluator, posterior)
    log = lambda x: print(x, flush=True)
    catalog = pd.read_csv(final / "tables" / "image_fit_quality.csv", dtype={"image_label": str})
    try:
        extra = pd.read_csv(final / "tables" / "image_recovery_extra_images.csv", dtype={"family_id": str})
    except pd.errors.EmptyDataError:
        extra = pd.DataFrame()  # No extra images were recovered in this saved run.
    position_contours = None
    if args.imaging == 'jwst' or config.image_catalog.figure_suffix == '_jwst':
        from kappaccino.diagnostics.position_contours import load_or_compute_contours
        position_contours = load_or_compute_contours(config, state, draws, extra, final / 'tables', log=log)
    imaging = load_cluster_imaging(config.image_catalog, config.model, config.paths.run_name)
    plt.style.use(str(ROOT / "src" / "kappaccino" / "plotting" / "style.mplstyle"))
    output = plots.figure_path(final / "figures", f"image_catalog_family_cluster_mass{config.image_catalog.figure_suffix}.pdf", config.paths.figure_prefix)
    print(f"Rendering the median over {len(draws)} posterior draws: {output}", flush=True)
    provenance = render_cluster_mass_figure(draws, catalog, extra, imaging,
        z_source=float(config.truth.caustic_source_redshift), kappa_true_fits=config.truth.kappa_true_fits,
        figure_path=output, tables_dir=final / "tables", fits_dir=final / "fits", artifact_suffix=config.image_catalog.figure_suffix,
        position_contours=position_contours, log=log)
    print(json.dumps(provenance["arc_directions"], indent=2))
    # The RGB critical-curve overview, as in write_outputs: the simulated HST frames of Hera and Ares, or the JWST mosaics of M0416.
    cluster_name = imaging[3].lower()
    simulated_hst = cluster_name in {"hera", "ares"} and config.image_catalog.figure_suffix != "_jwst"
    if simulated_hst or (config.image_catalog.figure_suffix == "_jwst" and cluster_name == "m0416"):
        from kappaccino.plotting.jwst_rgb import render_rgb_critical
        critical_name = "hst_rgb_critical_z9.pdf" if simulated_hst else "jwst_rgb_critical_z9.pdf"
        render_rgb_critical(config, state, draws, catalog, extra, imaging,
                            plots.figure_path(final / "figures", critical_name, config.paths.figure_prefix), final / "tables", log=log)


if __name__ == "__main__":
    main()
