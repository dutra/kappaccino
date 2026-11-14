"""Contours of the image positions the model predicts on the sky, from the image recovery of the reporting draws.

For an observed image the contour is of its predicted observed position, where the model expects a catalog entry:
where the product draw's own lens (parametric plus its GRF realization) puts the image, plus one fresh draw of the
image-plane GP and of the measurement error (``x_predicted_arcsec``, ``y_predicted_arcsec`` of
``posterior_image_residual_draws.csv``), so the catalog position can be compared with it (:func:`position_coverage`).
For an extra image, which has no catalog position, the contour is of its true position, where to look for it: the lens
position plus the GP displacement that the observed images near it imply (``x_true_arcsec``, ``y_true_arcsec`` of
``posterior_extra_images.csv``).  Nothing is solved again.  Each observed image gets the positions of the draws that
recover it; each extra image found in at least half the draws (``image_recovery_extra_images.csv``) gets the per-draw
extra image of its family whose lens position is nearest to it, within ``image_recovery.EXTRA_IMAGE_MATCH_ARCSEC``.
"""
import json
from pathlib import Path

import contourpy
import numpy as np
import pandas as pd
from scipy.stats import chi2, gaussian_kde

from .image_recovery import EXTRA_IMAGE_MATCH_ARCSEC
from ..io.hdf5 import save_arrays

PROBABILITIES = (.682689492, .954499736)
# Fewer finite draws than this give Gaussian ellipses instead of a density estimate.
MIN_KDE_DRAWS = 10


def position_draws(indices, tables) -> dict:
    """Per-draw positions from the saved image recovery at the posterior rows ``indices``.  ``recovered`` (draw,
    observed image, 2) and ``extra`` (draw, common extra image, 2) are the positions the contours are drawn of: the
    predicted observed position of an observed image and the true position of an extra one, NaN where a draw has none
    (or had no global search).  ``recovered_lens``, ``recovered_true``, ``recovered_predicted`` and ``extra_lens``,
    ``extra_true``, ``extra_predicted`` are the three positions of the same draws: the lens position, the lens
    position plus the GP displacement given the catalog, and the lens position plus a fresh draw of the image-plane
    scatter.  Also ``image_labels``, ``extra_family_ids``, ``extra_anchors`` (the common extra images' lens positions)
    and ``draw_indices``."""
    tables = Path(tables)
    indices = np.asarray(indices, int)
    slot = {int(index): k for k, index in enumerate(indices)}
    rows = pd.read_csv(tables / 'posterior_image_residual_draws.csv', dtype={'family_id': str, 'image_label': str})
    labels = list(dict.fromkeys(rows['image_label']))
    n = len(indices)
    recovered = np.full((n, len(labels), 2), np.nan)
    recovered_lens, recovered_true = recovered.copy(), recovered.copy()
    index = {label: i for i, label in enumerate(labels)}
    rows = rows[rows['recovered'].astype(bool) & rows['posterior_index'].isin(indices)]
    for row in rows.itertuples(index=False):
        recovered[slot[int(row.posterior_index)], index[row.image_label]] = (row.x_predicted_arcsec, row.y_predicted_arcsec)
        recovered_lens[slot[int(row.posterior_index)], index[row.image_label]] = (row.x_model_arcsec, row.y_model_arcsec)
        recovered_true[slot[int(row.posterior_index)], index[row.image_label]] = (row.x_true_arcsec, row.y_true_arcsec)
    common = _read(tables / 'image_recovery_extra_images.csv')
    per_draw = _read(tables / 'posterior_extra_images.csv')
    anchors = common[['x_model_arcsec', 'y_model_arcsec']].to_numpy(float) if len(common) else np.empty((0, 2))
    families = common['family_id'].astype(str).to_numpy() if len(common) else np.empty(0, str)
    extra = np.full((n, len(anchors), 2), np.nan)
    extra_lens, extra_predicted = extra.copy(), extra.copy()
    if len(anchors) and len(per_draw):
        per_draw = per_draw[per_draw['posterior_index'].isin(indices)]
        for posterior_index, group in per_draw.groupby('posterior_index'):
            for j, (anchor, family) in enumerate(zip(anchors, families)):
                candidates = group[group['family_id'].astype(str) == family]
                if len(candidates):
                    lens = candidates[['x_model_arcsec', 'y_model_arcsec']].to_numpy(float)
                    distance = np.hypot(*(lens - anchor).T)
                    if distance.min() <= EXTRA_IMAGE_MATCH_ARCSEC:
                        nearest = int(np.argmin(distance))
                        extra[slot[int(posterior_index)], j] = candidates[['x_true_arcsec', 'y_true_arcsec']].to_numpy(float)[nearest]
                        extra_lens[slot[int(posterior_index)], j] = lens[nearest]
                        extra_predicted[slot[int(posterior_index)], j] = candidates[['x_predicted_arcsec', 'y_predicted_arcsec']].to_numpy(float)[nearest]
    return dict(recovered=recovered, extra=extra, recovered_lens=recovered_lens, recovered_true=recovered_true,
                recovered_predicted=recovered.copy(), extra_lens=extra_lens, extra_true=extra.copy(), extra_predicted=extra_predicted,
                image_labels=np.asarray(labels), extra_family_ids=families,
                extra_anchors=anchors, draw_indices=indices)


def _read(path):
    try:
        return pd.read_csv(path, dtype={'family_id': str})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def gaussian_contours(points):
    """68.3/95.4% ellipses of the mean and covariance of ``points`` (for few draws)."""
    mean, covariance = points.mean(0), np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    angle = np.linspace(0, 2 * np.pi, 181)
    circle = np.stack((np.cos(angle), np.sin(angle)), -1)
    return [dict(probability=p, segments=[mean + np.sqrt(chi2.ppf(p, 2)) * (circle * np.sqrt(np.maximum(values, 0))) @ vectors.T])
            for p in PROBABILITIES]


def position_contours(points, *, grid_size=256):
    """68.3/95.4% regions of the draws that recover the image: highest-density regions of a KDE with at least
    ``MIN_KDE_DRAWS`` draws, Gaussian ellipses with three to that many; none with fewer, a singular distribution, or a
    density grid that does not hold the unit probability to 1%."""
    points = np.asarray(points); points = points[np.isfinite(points).all(1)]
    if len(points) < 3 or np.linalg.matrix_rank(np.cov(points.T)) < 2:
        return [], 'Fewer than 3 finite draws or singular position distribution'
    if len(points) < MIN_KDE_DRAWS:
        return gaussian_contours(points), None
    kde = gaussian_kde(points.T)
    lo, hi = points.min(0)-3*points.std(0), points.max(0)+3*points.std(0)
    x = np.linspace(lo[0], hi[0], grid_size); y = np.linspace(lo[1], hi[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ranked = np.sort(density.ravel())[::-1]
    mass = np.cumsum(ranked)*(x[1]-x[0])*(y[1]-y[0])
    if abs(mass[-1]-1) > .01:
        return [], f'Density grid ({grid_size} pixels) did not resolve unit probability'
    generator = contourpy.contour_generator(x=x, y=y, z=density)
    contours = []
    for probability in PROBABILITIES:
        threshold = ranked[min(np.searchsorted(mass, probability), len(ranked)-1)]
        contours.append(dict(probability=probability, segments=generator.lines(float(threshold))))
    return contours, None


def credible_level(points, target):
    """Probability enclosed by the highest-density region of ``points`` (draw, 2) on whose edge ``target`` lies: 0 at
    the peak, near 1 far outside.  With at least ``MIN_KDE_DRAWS`` draws it is the fraction of the draws whose density
    (a KDE without the draw's own kernel) exceeds the density at ``target``; with three to that many, the Gaussian
    value of the sample mean and covariance; NaN with fewer or a singular distribution."""
    points, target = np.asarray(points, dtype=float), np.asarray(target, dtype=float)
    points = points[np.isfinite(points).all(1)]
    if len(points) < 3 or not np.isfinite(target).all() or np.linalg.matrix_rank(np.cov(points.T)) < 2:
        return np.nan
    if len(points) < MIN_KDE_DRAWS:
        offset = target - points.mean(0)
        return float(chi2.cdf(offset @ np.linalg.solve(np.cov(points.T), offset), 2))
    kde = gaussian_kde(points.T)
    own = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(kde.covariance)))          # a draw's own kernel at its centre
    density = (len(points) * kde(points.T) - own) / (len(points) - 1)
    return float(np.mean(density > kde(target)[0]))


def position_coverage(draw_rows: pd.DataFrame) -> pd.DataFrame:
    """Where each catalog position falls in the distribution of its predicted observed positions.

    Parameters
    ----------
    draw_rows : pandas.DataFrame
        ``posterior_image_residual_draws``: one row per draw and image with ``recovered``, the catalog position
        (``x_obs_arcsec``, ``y_obs_arcsec``) and the predicted observed position (``x_predicted_arcsec``,
        ``y_predicted_arcsec``).

    Returns
    -------
    pandas.DataFrame
        One row per image: ``family_id``, ``image_label``, ``valid_draws``, ``credible_level``
        (:func:`credible_level` of the catalog position) and ``inside_68``, ``inside_95``.  If the model's scatter is
        the right size, about 68% and 95% of the images are inside.  The catalog was used in the fit, so this is an
        in-sample check.
    """
    rows = []
    for label, group in draw_rows.groupby('image_label', sort=False):
        found = group[group['recovered'].astype(bool)]
        level = credible_level(found[['x_predicted_arcsec', 'y_predicted_arcsec']].to_numpy(float),
                               group[['x_obs_arcsec', 'y_obs_arcsec']].to_numpy(float)[0])
        rows.append(dict(family_id=group['family_id'].iloc[0], image_label=label, valid_draws=len(found), credible_level=level,
                         inside_68=bool(level <= PROBABILITIES[0]) if np.isfinite(level) else np.nan,
                         inside_95=bool(level <= PROBABILITIES[1]) if np.isfinite(level) else np.nan))
    return pd.DataFrame(rows, columns=['family_id', 'image_label', 'valid_draws', 'credible_level', 'inside_68', 'inside_95'])


def prepare_contours(data, *, grid_size=256, log=print):
    curves = []; summaries = []
    for kind in ('recovered', 'extra'):
        for i in range(data[kind].shape[1]):
            points = data[kind][:, i]; valid = np.isfinite(points).all(1)
            levels, reason = position_contours(points, grid_size=grid_size)
            identity = str(data['image_labels'][i]) if kind == 'recovered' else f"{data['extra_family_ids'][i]} extra {i+1}"
            summaries.append(dict(kind=kind, identity=identity, valid_draws=int(valid.sum()), total_draws=len(points), skip_reason=reason))
            for level in levels:
                curves.append(dict(kind=kind, identity=identity, **level))
    unresolved = sum(1 for s in summaries if s['skip_reason'] and s['skip_reason'].startswith('Density grid'))
    log(f'[outputs] posterior contours: {len(curves)} probability levels'
        + (f'; {unresolved} positions not resolved on the {grid_size}-pixel grid' if unresolved else ''))
    return dict(curves=curves, summaries=summaries)


def load_or_compute_contours(config, state, draws, extra, tables, *, grid_size=256, log=print):
    """Position contours of the reporting draws (``reporting.draws.ReportingDraws``) from the saved image recovery in
    ``tables``; ``extra`` (the common extra images) is read from there too and kept in the signature for the callers."""
    tables = Path(tables)
    data = position_draws(draws.indices, tables)
    save_arrays(tables/'posterior_overview_positions.h5', **data)
    contours = prepare_contours(data, grid_size=grid_size, log=log)
    contours['position_data'] = data
    contours['fingerprint'] = draws.key
    (tables/'posterior_overview_positions.json').write_text(json.dumps(dict(
        fingerprint=draws.key, draws=len(draws.indices), selected_draw_indices=np.asarray(draws.indices).tolist(),
        solver="the image recovery of the reporting draws (no new solve); each draw's lens is parametric plus its GRF realization",
        positions='recovered (= recovered_predicted): predicted observed positions of the observed images, the lens position plus '
                  'one draw of the image-plane GP and measurement error; extra (= extra_true): true positions of the extra images, '
                  'the lens position plus the GP displacement given the catalog; *_lens, *_true, *_predicted: the three positions',
        extras='Extra images found in at least half the draws; per draw the nearest extra of the family',
        contours=f'KDE highest-density regions (at least {MIN_KDE_DRAWS} draws) or Gaussian ellipses, 68.3% and 95.4%',
        summaries=contours['summaries']), indent=2))
    return contours
