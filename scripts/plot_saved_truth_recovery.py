"""Render recovery from saved median grids or a saved best-fit model, without inference.

Run with the lenstronomy interpreter from any working directory. Relative paths
stored in the original run are resolved against the repository root.
"""
from __future__ import annotations

import argparse
import json
import inspect
import time
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from lenscluster import plotting
from lenscluster.output_paths import output_paths, resolve_artifacts_dir


def _best_fit_grids(artifact, wcs, shape, z_source):
    from lenscluster import cluster_solver as cs

    state, saved, arrays, diagnostics = cs._rebuild_state_from_h5(artifact)
    arrays, _ = cs._maybe_convert_loaded_posterior_arrays_to_physical(
        arrays, state.parameter_specs, diagnostics,
    )
    parameters = inspect.signature(cs.ClusterJAXEvaluator).parameters
    kwargs = {key: value for key, value in saved.items() if key in parameters and key != "state"}
    kwargs.setdefault("match_tolerance_arcsec", cs.DEFAULT_MATCH_TOLERANCE)
    kwargs["cab_likelihood_weight"] = cs._effective_cab_likelihood_weight(saved.get("cab_likelihood_weight"), state)
    kwargs["image_presence_penalty_weight"] = cs._effective_image_presence_penalty_weight(
        saved.get("image_presence_penalty_weight"),
        sample_likelihood_mode=saved.get("sample_likelihood_mode", cs.SAMPLE_LIKELIHOOD_SOURCE),
        fit_mode=saved.get("fit_mode", cs.FIT_MODE_SEQUENTIAL),
        image_plane_mode=saved.get("image_plane_mode", cs.IMAGE_PLANE_MODE_NONE),
    )
    kwargs.update(cs._normalized_saved_likelihood_stabilizer_args(saved))
    evaluator = cs.ClusterJAXEvaluator(state=state, **kwargs)
    best_fit = np.asarray(arrays["best_fit"], dtype=float)
    if not np.isfinite(best_fit).all():
        raise ValueError("Saved best_fit contains nonfinite values")
    started = time.perf_counter()
    models, _, _ = plotting._best_fit_truth_recovery_grids(evaluator, best_fit, wcs, shape, z_source)
    provenance = {
        "model_summary": "saved_best_fit", "best_value": diagnostics.get("best_value_selected", saved.get("best_value")),
        "grid_shape": list(shape), "z_source": z_source, "artifact": str(artifact.resolve()),
        "best_fit": best_fit.tolist(), "grid_seconds": time.perf_counter() - started,
    }
    return models, provenance


def render_saved_recovery(run_dir: Path, paper_output: Path | None = None, *, best_fit=False, grid_size=None) -> Path:
    if grid_size is not None and grid_size <= 0:
        raise ValueError("grid_size must be positive")
    stage = resolve_artifacts_dir(run_dir).parent
    paths = output_paths(stage)
    with h5py.File(paths.artifacts / "plot_bundle.h5", "r") as handle:
        args = json.loads(handle["runtime_args_json"][()])
        meta = json.loads(handle["state/build_state_meta_json"][()])
    native = {}
    for quantity, key in (("kappa", "kappa_true_fits"), ("gamma1", "gammax_true_fits"), ("gamma2", "gammay_true_fits")):
        source = Path(args[key])
        native[quantity] = plotting._load_kappa_true_fits(source if source.is_absolute() else ROOT / source)
    native_kappa, native_wcs = native["kappa"]
    expected_wcs, shape, _ = plotting._truth_recovery_diagnostic_grid(
        native_wcs, native_kappa.shape, args["truth_grid_size"] if grid_size is None else grid_size,
    )
    if grid_size is not None and not best_fit:
        raise ValueError("A new grid size requires --best-fit; saved median FITS have a fixed grid")
    if best_fit:
        models, provenance = _best_fit_grids(
            paths.artifacts / "plot_bundle.h5", expected_wcs, shape, args["caustic_source_redshift"],
        )
        for quantity, values in models.items():
            header = expected_wcs.to_header()
            header["MODSTAT"] = "BEST_FIT"
            header["ZSOURCE"] = args["caustic_source_redshift"]
            fits.writeto(paths.fits / f"truth_recovery_{quantity}_model_best_fit.fits", values, header, overwrite=True)
        (paths.tables / "truth_recovery_best_fit_grid.json").write_text(json.dumps(provenance, indent=2) + "\n")
    else:
        models = {}
        for quantity in ("kappa", "abs_mu", "detA"):
            values, header = fits.getdata(paths.fits / f"truth_recovery_{quantity}_model_median.fits", header=True)
            if values.shape != shape:
                raise ValueError(f"Saved {quantity} shape differs from the configured truth grid")
            wcs = WCS(header)
            y, x = np.indices(shape, dtype=float)
            for saved, expected in zip(wcs.pixel_to_world_values(x, y), expected_wcs.pixel_to_world_values(x, y), strict=True):
                np.testing.assert_allclose(saved, expected, atol=1e-9, rtol=0)
            models[quantity] = values
    truth = {}
    for quantity, (values, wcs) in native.items():
        if values.shape != native_kappa.shape:
            raise ValueError(f"Truth {quantity} shape differs from the convergence grid")
        plotting._validate_matching_truth_wcs(native_wcs, wcs, native_kappa.shape, label=quantity)
        truth[quantity] = plotting._truth_recovery_sample_truth_image(
            values, wcs, expected_wcs, shape, native_shape=values.shape,
        )
    determinant = plotting._critical_determinant_from_kappa_gamma(truth["kappa"], truth["gamma1"], truth["gamma2"])
    abs_mu = np.abs(plotting._signed_magnification_from_kappa_gamma(truth["kappa"], truth["gamma1"], truth["gamma2"]))
    evaluator = SimpleNamespace(state=SimpleNamespace(reference=meta["reference"]))
    x, y = plotting._truth_grid_arcsec_coordinates(evaluator, expected_wcs, shape)
    image_df = pd.read_csv(paths.tables / "image_fit_quality.csv")
    plotting._plot_combined_truth_recovery_from_grids(
        stage, truth["kappa"], models["kappa"], abs_mu, models["abs_mu"],
        determinant, models["detA"], x, y, args["caustic_source_redshift"], image_df=image_df,
    )
    output = plotting.plot_path(stage, "truth_recovery_residuals.pdf")
    if paper_output is not None:
        paper_output.parent.mkdir(parents=True, exist_ok=True)
        if paper_output.resolve() != output.resolve():
            shutil.copy2(output, paper_output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Saved run root or stage directory")
    parser.add_argument("--paper-output", type=Path, help="Optional identical PDF copy for the manuscript")
    parser.add_argument("--best-fit", action="store_true", help="Evaluate the saved best-fit realization")
    parser.add_argument("--grid-size", type=int, help="Diagnostic grid side length for --best-fit")
    options = parser.parse_args()
    print(render_saved_recovery(options.run_dir, options.paper_output, best_fit=options.best_fit, grid_size=options.grid_size))
