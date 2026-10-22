"""Render the publication overview from saved stage-matched image diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lenscluster import plotting
from lenscluster.output_paths import output_paths, stage_name


def render(stage: Path, paper_output: Path | None = None, *, mass_panel: bool = False) -> Path:
    paths = output_paths(stage)
    with h5py.File(paths.artifacts / "plot_bundle.h5", "r") as handle:
        args = json.loads(handle["runtime_args_json"][()])
        meta = json.loads(handle["state/build_state_meta_json"][()])
    summary = json.loads((paths.tables / "run_summary.json").read_text())
    if args["sample_likelihood_mode"] != summary["sample_likelihood_mode"]:
        raise ValueError("Saved model and image diagnostics use different likelihoods")
    if summary.get("stage_semantic_name") != stage_name(stage):
        raise ValueError("Saved image diagnostics belong to a different stage")
    catalog = pd.read_csv(paths.tables / "image_fit_quality.csv")
    if "arc_recovery_p_arc_threshold" in summary:
        catalog["arc_recovery_p_arc_threshold"] = summary["arc_recovery_p_arc_threshold"]
    np.testing.assert_allclose(
        np.hypot(catalog.x_model_arcsec - catalog.x_obs_arcsec,
                 catalog.y_model_arcsec - catalog.y_obs_arcsec),
        catalog.point_image_residual_arcsec, rtol=1e-7, atol=1e-9,
    )
    try:
        extra = pd.read_csv(paths.tables / "image_recovery_extra_images.csv")
    except pd.errors.EmptyDataError:
        extra = pd.DataFrame()
    helpers = plotting._load_image_catalog_cutout_helpers()
    bands = tuple(args["image_catalog_family_cutout_bands"])
    images_dir = ROOT / args["image_catalog_family_cutout_image_dir"]
    from types import SimpleNamespace
    cluster = str(paths.figure_prefix or plotting._infer_image_catalog_cutout_cluster(
        SimpleNamespace(run_name=meta.get("run_name", "")), model_config=args.get("model_config"),
    )).lower()
    band_paths = helpers.find_rgb_band_paths(images_dir, cluster=cluster, bands=bands,
                                            image_scale=args["image_catalog_family_cutout_image_scale"])
    images = helpers.load_rgb_metadata(band_paths, bands=bands)
    rgb_kwargs = {key: args[f"image_catalog_family_cutout_rgb_{key}"]
                  for key in ("q", "stretch", "minimum")
                  if args.get(f"image_catalog_family_cutout_rgb_{key}") is not None}
    rgb_kwargs["channel_gains"] = {key: args[f"image_catalog_family_cutout_rgb_{key}_gain"]
                                    for key in ("red", "green", "blue")}
    rgb = helpers.build_rgb_display(images, bands=bands, **rgb_kwargs)
    config = helpers.build_family_cutout_render_config(mode="full", dpi=300,
                                                        max_side_pixels=2400, critical_lines="off")
    if mass_panel:
        from lenscluster.image_catalog_mass import load_saved_model, render_mass_companion
        if args.get("kappa_true_fits"):
            args["kappa_true_fits"] = str(ROOT / args["kappa_true_fits"])
        evaluator, best_fit = load_saved_model(paths.artifacts / "plot_bundle.h5", summary)
        reference_band = getattr(helpers, "DEFAULT_HFF_RGB_REFERENCE_BAND", bands[-1])
        display_image = images.get(reference_band, images[bands[-1]])
        output = render_mass_companion(
            stage, evaluator, best_fit, SimpleNamespace(**args), helpers, images, bands, rgb,
            display_image, catalog, extra, config,
            sample_index=summary["fit_quality_reference_sample_index"],
            sample_kind=summary["fit_quality_reference_sample_kind"],
        )
        if paper_output is not None:
            shutil.copy2(output, paper_output)
        return output
    fig = plotting._image_catalog_point_overview_figure(
        helpers, images, bands, rgb, images[bands[-1]], catalog, extra,
        tuple(meta["reference"]), render_config=config,
    )
    stem = "image_catalog_family_cluster"
    output = plotting._plot_path(stage, stem + ".pdf")
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plotting.plt.close(fig)
    if paper_output is not None:
        shutil.copy2(output, paper_output)
    provenance = {"stage": str(stage.resolve()), "sample_likelihood_mode": args["sample_likelihood_mode"],
                  "reference_sample_kind": summary["fit_quality_reference_sample_kind"],
                  "reference_sample_index": summary["fit_quality_reference_sample_index"],
                  "images": len(catalog), "sources": int(catalog.family_id.nunique()),
                  "point_image_rms_arcsec": summary["point_image_rms_arcsec"],
                  "bands": bands}
    provenance_name = "image_catalog_publication_provenance.json"
    (paths.tables / provenance_name).write_text(json.dumps(provenance, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--paper-output", type=Path)
    parser.add_argument("--mass-panel", action="store_true", help="Add model contours and integrated profiles, with truth comparisons when available")
    options = parser.parse_args()
    print(render(options.stage, options.paper_output, mass_panel=options.mass_panel))
