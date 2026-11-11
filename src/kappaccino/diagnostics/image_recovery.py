"""Image-recovery tables from exact lens-equation solutions at the reporting posterior draws."""
from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..config import ImageDiagnosticsConfig
from ..likelihood.evaluator import Evaluator
from .exact_images import ExactImageSolver
from ..core.model import ModelState, PosteriorResults
from ..core.utils import progress


def predictions_at(solver: ExactImageSolver, state: ModelState, evaluator: Evaluator, theta: np.ndarray, *,
                   adaptive: bool | None = None, include_field: bool = True) -> dict[str, Any]:
    """Per-image model positions, residuals and magnifications for one latent vector.  Under the knee kernel the lens
    carries the GRF conditional-mean deflection of the draw; ``include_field=False`` solves with the parametric lens alone
    (the lens-only residual, which shows how much of the misfit the field takes)."""
    theta = np.asarray(theta, dtype=float)
    physical = evaluator.physical(jnp.asarray(theta))
    sources = evaluator.source_positions(jnp.asarray(theta))
    images = state.images
    family_by_id = {f.family_id: f for f in state.family_data}
    families = [family_by_id[family_id] for family_id in images.family_ids]
    # Potential kernel: the draw's source scatters by the deflection error, so the lens gets the GP's mean deflection.
    correction = evaluator.gp_deflection_correction(theta) if include_field and getattr(evaluator, "gp_knee", False) else None
    extra_kwargs = {} if correction is None else dict(correction=correction)
    if adaptive is False:        # posterior draws: all families in one batched local solve
        batched = solver.predict_all_local(images, families, evaluator.packed_state(physical)[0], np.asarray(sources), **extra_kwargs)
    elif solver.config.exact_image_finder == "global-jax":   # a global search that also reports extra images
        batched, adaptive = solver.predict_all_global(images, families, np.asarray(physical), np.asarray(sources), **extra_kwargs), False
    elif correction is not None:
        raise ValueError("under gp_kernel='potential' only the global-jax and batched local image finders apply the GP mean deflection; "
                         f"image_diagnostics.exact_image_finder is {solver.config.exact_image_finder!r}")
    rows, extra_rows, count_rows = [], [], []
    for family_index, family in enumerate(families):
        family_id = family.family_id
        if adaptive is False:
            details = batched[family_index]
        else:
            packed = evaluator.packed_state_for_z(physical, family.z_source)
            details = solver.predict_family(family, packed, float(sources[family_index, 0]), float(sources[family_index, 1]), adaptive=adaptive)
        recovered = np.asarray(details["recovered_image_mask"], dtype=bool)
        mx, my = details["matched_model_x_arcsec"], details["matched_model_y_arcsec"]
        mu = np.asarray(details["magnification"], dtype=float)
        mu_model = np.asarray(details["magnification_at_model"], dtype=float)
        count_rows.append({"family_id": family_id, "z_source": family.z_source, "observed_image_count": family.n_images,
                           "produced_image_count": details["produced_image_count"], "recovered_image_count": details["recovered_image_count"],
                           "missing_image_count": details["missing_image_count"], "extra_image_count": details["extra_image_count"],
                           "multiplicity_failed": bool(details["multiplicity_failed"]),
                           "multiplicity_failure_reason": str(details["multiplicity_failure_reason"]),
                           "exact_image_rms_arcsec": float(details["exact_image_rms"]),
                           "source_x_arcsec": float(sources[family_index, 0]), "source_y_arcsec": float(sources[family_index, 1])})
        for k, label in enumerate(family.image_labels):
            x_obs, y_obs = float(family.x_obs[k]), float(family.y_obs[k])
            residual = math.hypot(mx[k] - x_obs, my[k] - y_obs) if recovered[k] else np.nan
            rows.append({"family_id": family_id, "image_label": label, "x_obs_arcsec": x_obs, "y_obs_arcsec": y_obs,
                         "z_source": family.z_source,
                         "radius_arcsec": math.hypot(x_obs, y_obs), "angle_deg": math.degrees(math.atan2(y_obs, x_obs)),
                         "image_recovery_status": "recovered" if recovered[k] else "not_recovered",
                         "x_model_arcsec": float(mx[k]), "y_model_arcsec": float(my[k]), "image_residual_arcsec": float(residual),
                         "magnification_model": float(mu[k]), "magnification_model_at_model_position": float(mu_model[k]),
                         "model_produced_image_count": details["produced_image_count"], "model_recovered_image_count": details["recovered_image_count"],
                         "model_missing_image_count": details["missing_image_count"], "model_extra_image_count": details["extra_image_count"],
                         "model_multiplicity_failed": bool(details["multiplicity_failed"]),
                         "model_multiplicity_failure_reason": str(details["multiplicity_failure_reason"])})
        for ex, ey in zip(details["extra_model_x_arcsec"], details["extra_model_y_arcsec"]):
            extra_rows.append({"family_id": family_id, "x_model_arcsec": float(ex), "y_model_arcsec": float(ey)})
    return {"image_rows": rows, "extra_rows": extra_rows, "count_rows": count_rows}


LENS_ONLY_COLUMNS = ["family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "image_recovery_status", "recovered_fraction",
                     "x_model_arcsec", "y_model_arcsec", "image_residual_arcsec", "image_residual_q16", "image_residual_q84"]
# Extra model images of different draws within this distance (arcsec) are one image.
EXTRA_IMAGE_MATCH_ARCSEC = 1.0


def lens_only_summary(draw_rows: pd.DataFrame | None) -> dict[str, Any]:
    """Headline numbers of the lens-only draws (the parametric lens without the GRF conditional mean), for
    run_summary.json and the results macros: median and 16-84% over draws of the recovered count, RMS and median
    residual."""
    if draw_rows is None or draw_rows.empty:
        return {}
    return {f"lens_only_{key}": value for key, value in draw_summary(draw_rows).items()}


def draw_summary(draw_rows: pd.DataFrame) -> dict[str, Any]:
    """Per-draw recovered count, RMS and median residual of ``draw_rows`` (one row per draw and image), as
    ``{"recovered_image_count": {"median", "q16", "q84"}, "image_rms_arcsec": ..., "image_median_residual_arcsec": ...}``."""
    per_draw = []
    for _, group in draw_rows.groupby("draw_index", sort=True):
        residual = pd.to_numeric(group["image_residual_arcsec"], errors="coerce").to_numpy(float)
        valid = group["recovered"].astype(bool).to_numpy() & np.isfinite(residual)
        per_draw.append((int(valid.sum()), float(np.sqrt(np.mean(residual[valid] ** 2))) if valid.any() else np.nan,
                         float(np.median(residual[valid])) if valid.any() else np.nan))
    values = np.asarray(per_draw, dtype=float)
    out = {}
    for column, name in enumerate(("recovered_image_count", "image_rms_arcsec", "image_median_residual_arcsec")):
        q16, q50, q84 = (float(v) for v in np.nanpercentile(values[:, column], [16, 50, 84]))
        out[name] = {"median": q50, "q16": q16, "q84": q84}
    return out


def draw_residual_rows(solver: ExactImageSolver, state: ModelState, evaluator: Evaluator, draws: np.ndarray, *,
                       include_field: bool) -> list[dict[str, Any]]:
    """Per-image rows of every draw (``draw_index``, ``family_id``, ``image_label``, ``recovered``,
    ``image_residual_arcsec``, ``x_model_arcsec``, ``y_model_arcsec``), solved with the batched local solver from the
    observed positions, with or without the GRF conditional mean of the draw."""
    rows = []
    for draw_index, theta in enumerate(progress(draws, desc="exact images" if include_field else "exact images, lens only")):
        prediction = predictions_at(solver, state, evaluator, theta, adaptive=False, include_field=include_field)
        rows.extend({"draw_index": draw_index, "family_id": row["family_id"], "image_label": row["image_label"],
                     "recovered": row["image_recovery_status"] == "recovered", "image_residual_arcsec": row["image_residual_arcsec"],
                     "x_model_arcsec": row["x_model_arcsec"], "y_model_arcsec": row["y_model_arcsec"]}
                    for row in prediction["image_rows"])
    return rows


def has_field(evaluator: Evaluator, theta: np.ndarray) -> bool:
    """Whether the run has a GRF conditional-mean deflection (so lens-only residuals differ from the quoted ones)."""
    return bool(getattr(evaluator, "gp_knee", False) and evaluator.gp_deflection_correction(theta) is not None)


def per_image_table(draw_rows: pd.DataFrame, observed: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """One row per observed image from the per-draw rows: the fraction of draws that recover it, its status
    (``"recovered"`` when at least half the draws recover it) and, over the draws that recover it, the median of each
    of ``columns`` (same name) with its 16th, 50th and 84th percentiles (``<stem>_q16`` and so on, ``<stem>`` the
    column name without ``_arcsec``)."""
    rows = []
    for label, group in draw_rows.groupby("image_label", sort=False):
        recovered = group["recovered"].astype(bool).to_numpy()
        row = {"image_label": label, "recovered_fraction": float(recovered.mean()),
               "image_recovery_status": "recovered" if recovered.mean() >= .5 else "not_recovered",
               "posterior_valid_draws": int(recovered.sum()), "posterior_failed_draws": int((~recovered).sum())}
        for column in columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)[recovered]
            values = values[np.isfinite(values)]
            q16, q50, q84 = np.percentile(values, [16, 50, 84]) if values.size else (np.nan,) * 3
            stem = column.removesuffix("_arcsec")
            row.update({column: float(q50), f"{stem}_q16": float(q16), f"{stem}_q50": float(q50), f"{stem}_q84": float(q84)})
        rows.append(row)
    return observed.merge(pd.DataFrame(rows), on="image_label", how="left")


def common_extra_images(extra: pd.DataFrame, n_draws: int) -> pd.DataFrame:
    """Extra model images found in at least half of the draws: the per-draw extras of a family within
    ``EXTRA_IMAGE_MATCH_ARCSEC`` of each other are one image, reported at the median position with the fraction of draws
    that produce it."""
    rows = []
    if extra.empty:
        return pd.DataFrame(columns=["family_id", "x_model_arcsec", "y_model_arcsec", "x_model_q16", "x_model_q84",
                                     "y_model_q16", "y_model_q84", "draw_fraction"])
    for family_id, group in extra.groupby("family_id", sort=False):
        xy = group[["x_model_arcsec", "y_model_arcsec"]].to_numpy(float)
        draw = group["draw_index"].to_numpy(int)
        unassigned = np.ones(len(xy), bool)
        while unassigned.any():
            seed = np.flatnonzero(unassigned)[0]
            centre = xy[seed]
            for _ in range(5):          # recentre on the median of the members
                member = unassigned & (np.hypot(*(xy - centre).T) <= EXTRA_IMAGE_MATCH_ARCSEC)
                centre = np.median(xy[member], axis=0)
            member = unassigned & (np.hypot(*(xy - centre).T) <= EXTRA_IMAGE_MATCH_ARCSEC)
            member[seed] = True
            unassigned &= ~member
            fraction = len(np.unique(draw[member])) / n_draws
            if fraction >= .5:
                (x16, x50, x84), (y16, y50, y84) = (np.percentile(xy[member, i], [16, 50, 84]) for i in range(2))
                rows.append({"family_id": family_id, "x_model_arcsec": x50, "y_model_arcsec": y50, "x_model_q16": x16,
                             "x_model_q84": x84, "y_model_q16": y16, "y_model_q84": y84, "draw_fraction": fraction})
    return pd.DataFrame(rows)


def image_recovery_tables(state: ModelState, evaluator: Evaluator, draws, config: ImageDiagnosticsConfig, *,
                          extra_images: bool = True, log=print) -> dict[str, pd.DataFrame]:
    """Image-recovery tables over the reporting draws (``reporting.draws.ReportingDraws``); nothing at a single sample.

    The observed images of every image draw (``draws.image_samples``) are solved with the batched local solver from the
    observed positions, with and without the draw's GRF conditional mean (it agrees with the global search to 1e-4" on
    the recovered images).  Per image: the fraction of draws that recover it, the median model position, residual and
    magnifications over those draws with their 16-84% range.  Per family: the median and 16-84% of the per-draw counts.
    With ``extra_images``, the global search (``config.exact_image_finder``) runs on the product draws
    (``draws.samples``) to find extra model images; those found in at least half of them are reported.  Every row
    carries its ``posterior_index``."""
    solver = ExactImageSolver(evaluator, config)
    draw_rows, count_rows, rms_rows = [], [], []
    observed = None
    for draw_index, (index, theta) in enumerate(zip(progress(draws.image_indices, desc="exact images"), draws.image_samples)):
        prediction = predictions_at(solver, state, evaluator, theta, adaptive=False)
        if observed is None:
            observed = pd.DataFrame([{key: row[key] for key in ("family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "z_source",
                                                                  "radius_arcsec", "angle_deg")} for row in prediction["image_rows"]])
        residuals = np.asarray([r["image_residual_arcsec"] for r in prediction["image_rows"]], dtype=float)
        rms_rows.append({"draw_index": draw_index, "posterior_index": int(index),
                         "recovered_image_count": int(np.isfinite(residuals).sum()),
                         "image_rms_arcsec": float(np.sqrt(np.nanmean(residuals ** 2))) if np.isfinite(residuals).any() else np.nan})
        for row in prediction["image_rows"]:
            draw_rows.append({"draw_index": draw_index, "posterior_index": int(index), "family_id": row["family_id"],
                              "image_label": row["image_label"], "recovered": row["image_recovery_status"] == "recovered",
                              "image_residual_arcsec": row["image_residual_arcsec"],
                              "x_model_arcsec": row["x_model_arcsec"], "y_model_arcsec": row["y_model_arcsec"],
                              "magnification_model": row["magnification_model"],
                              "magnification_model_at_model_position": row["magnification_model_at_model_position"]})
        count_rows.extend({"draw_index": draw_index, **row} for row in prediction["count_rows"])
    log(f"[diagnostics] observed images at {len(draws.image_indices)} posterior draws (local solver), "
        f"{solver.timing['exact_solver']:.1f}s of solver time")
    extra_rows = []
    if extra_images:
        for draw_index, (index, theta) in enumerate(zip(progress(draws.indices, desc="extra images"), draws.samples)):
            prediction = predictions_at(solver, state, evaluator, theta)
            extra_rows.extend({"draw_index": draw_index, "posterior_index": int(index), **row} for row in prediction["extra_rows"])
        log(f"[diagnostics] global image search for extra images at {len(draws.indices)} posterior draws")
    draw_df, extra_df, counts = pd.DataFrame(draw_rows), pd.DataFrame(extra_rows), pd.DataFrame(count_rows)
    image_df = per_image_table(draw_df, observed, ("x_model_arcsec", "y_model_arcsec", "image_residual_arcsec", "magnification_model",
                                                   "magnification_model_at_model_position"))
    count_df = family_count_table(counts.drop(columns=["extra_image_count"]))
    if extra_images:
        per_draw = (extra_df.groupby(["family_id", "draw_index"]).size() if not extra_df.empty else pd.Series(dtype=int))
        for name, q in (("extra_image_count", 50), ("extra_image_count_q16", 16), ("extra_image_count_q84", 84)):
            count_df[name] = [float(np.percentile([per_draw.get((f, d), 0) for d in range(len(draws.indices))], q))
                              for f in count_df.family_id]
    magnification_columns = ["family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec", "z_source", "image_recovery_status", "x_model_arcsec", "y_model_arcsec",
                             "magnification_model", "magnification_model_q16", "magnification_model_q50", "magnification_model_q84",
                             "magnification_model_at_model_position", "magnification_model_at_model_position_q16",
                             "magnification_model_at_model_position_q50", "magnification_model_at_model_position_q84"]
    magnification_df = image_df[[c for c in magnification_columns if c in image_df.columns]].copy()
    extra_tables = {}
    if has_field(evaluator, draws.image_samples[0]):
        lens_rows = pd.DataFrame(draw_residual_rows(solver, state, evaluator, draws.image_samples, include_field=False))
        lens_only = per_image_table(lens_rows, observed[["family_id", "image_label", "x_obs_arcsec", "y_obs_arcsec"]],
                                    ("x_model_arcsec", "y_model_arcsec", "image_residual_arcsec"))
        extra_tables = {"image_fit_quality_lens_only": lens_only[[c for c in LENS_ONLY_COLUMNS if c in lens_only.columns]],
                        "posterior_image_residual_draws_lens_only": lens_rows}
    return {**extra_tables, "image_fit_quality": image_df, "image_count_recovery": count_df, "model_magnification": magnification_df,
            "image_recovery_extra_images": common_extra_images(extra_df, len(draws.indices)) if extra_images else common_extra_images(pd.DataFrame(), 1),
            "posterior_extra_images": extra_df, "posterior_image_rms": pd.DataFrame(rms_rows), "posterior_image_residual_draws": draw_df}


def family_count_table(counts: pd.DataFrame) -> pd.DataFrame:
    """Per family: the median and 16-84% over draws of the produced, recovered, missing and extra image counts, and the
    fraction of draws whose multiplicity fails."""
    rows = []
    for family_id, group in counts.groupby("family_id", sort=False):
        row = {"family_id": family_id, "z_source": float(group["z_source"].iloc[0]), "observed_image_count": int(group["observed_image_count"].iloc[0]),
               "multiplicity_failed_fraction": float(group["multiplicity_failed"].astype(bool).mean()),
               "multiplicity_failed": bool(group["multiplicity_failed"].astype(bool).mean() >= .5)}
        for name in [n for n in ("produced_image_count", "recovered_image_count", "missing_image_count", "extra_image_count") if n in group]:
            q16, q50, q84 = np.percentile(group[name].to_numpy(float), [16, 50, 84])
            row.update({name: float(q50), f"{name}_q16": float(q16), f"{name}_q84": float(q84)})
        rows.append(row)
    return pd.DataFrame(rows)


def fit_summary(state: ModelState, recovery: dict[str, pd.DataFrame], *, extra_image_draws: int = 0) -> dict[str, Any]:
    """Headline recovery numbers over the reporting draws: the median and 16-84% of the per-draw recovered count, RMS
    and median residual (with and without the GRF), the median extra-image count and the families whose multiplicity
    holds in at least half the draws (``extra_image_draws``: how many draws had the global search, 0 if none).  The effective parameter count follows the lenstool convention,
    2 N_images - 2 N_families - N_global, where the global parameters are the halo, relation and error-model
    parameters: the member deviations are prior-dominated and add nothing to the effective count."""
    n_images = int(sum(f.n_images for f in state.family_data))
    n_global = int(sum(spec.component_family not in {"source_position", "member_scatter_offset", "member_shape_offset"} for spec in state.parameter_specs))
    k = n_global + 2 * len(state.family_data)
    out: dict[str, Any] = {"observed_image_count": n_images, "n_effective_parameters": k, "n_global_parameters": n_global}
    draws = recovery.get("posterior_image_residual_draws")
    if draws is None or draws.empty:
        return out
    out.update(draw_summary(draws))
    out["n_draws"] = int(draws["draw_index"].nunique())
    counts = recovery["image_count_recovery"]
    if extra_image_draws:
        extra = recovery["posterior_extra_images"]
        per_draw_extra = np.array([int((extra["draw_index"] == d).sum()) if not extra.empty else 0 for d in range(extra_image_draws)])
        q16, q50, q84 = np.percentile(per_draw_extra, [16, 50, 84])
        out["extra_image_count"] = {"median": float(q50), "q16": float(q16), "q84": float(q84)}
    out["families_fully_recovered"] = int((~counts["multiplicity_failed"].astype(bool)).sum()) if not counts.empty else 0
    out.update(lens_only_summary(recovery.get("posterior_image_residual_draws_lens_only")))
    return out
