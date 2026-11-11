"""Posterior image-position contours from the image recovery of the reporting draws.

The positions are the ones the image recovery solved for every reporting draw (``posterior_image_residual_draws.csv``
and ``posterior_extra_images.csv``): nothing is solved again.  Each observed image gets the positions of the draws that
recover it; each extra image found in at least half the draws (``image_recovery_extra_images.csv``) gets the per-draw
extra image of its family nearest to it, within ``image_recovery.EXTRA_IMAGE_MATCH_ARCSEC``.
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
    """Per-draw model positions from the saved image recovery at the posterior rows ``indices``: ``recovered`` (draw,
    observed image, 2) and ``extra`` (draw, common extra image, 2), NaN where a draw has none (or had no global search),
    with ``image_labels``, ``extra_family_ids``, ``extra_anchors`` (the common extra positions) and ``draw_indices``."""
    tables = Path(tables)
    indices = np.asarray(indices, int)
    slot = {int(index): k for k, index in enumerate(indices)}
    rows = pd.read_csv(tables / 'posterior_image_residual_draws.csv', dtype={'family_id': str, 'image_label': str})
    labels = list(dict.fromkeys(rows['image_label']))
    n = len(indices)
    recovered = np.full((n, len(labels), 2), np.nan)
    index = {label: i for i, label in enumerate(labels)}
    rows = rows[rows['recovered'].astype(bool) & rows['posterior_index'].isin(indices)]
    for row in rows.itertuples(index=False):
        recovered[slot[int(row.posterior_index)], index[row.image_label]] = (row.x_model_arcsec, row.y_model_arcsec)
    common = _read(tables / 'image_recovery_extra_images.csv')
    per_draw = _read(tables / 'posterior_extra_images.csv')
    anchors = common[['x_model_arcsec', 'y_model_arcsec']].to_numpy(float) if len(common) else np.empty((0, 2))
    families = common['family_id'].astype(str).to_numpy() if len(common) else np.empty(0, str)
    extra = np.full((n, len(anchors), 2), np.nan)
    if len(anchors) and len(per_draw):
        per_draw = per_draw[per_draw['posterior_index'].isin(indices)]
        for posterior_index, group in per_draw.groupby('posterior_index'):
            for j, (anchor, family) in enumerate(zip(anchors, families)):
                candidates = group[group['family_id'].astype(str) == family][['x_model_arcsec', 'y_model_arcsec']].to_numpy(float)
                if len(candidates):
                    distance = np.hypot(*(candidates - anchor).T)
                    if distance.min() <= EXTRA_IMAGE_MATCH_ARCSEC:
                        extra[slot[int(posterior_index)], j] = candidates[np.argmin(distance)]
    return dict(recovered=recovered, extra=extra, image_labels=np.asarray(labels), extra_family_ids=families,
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
    data = position_draws(draws.image_indices, tables)
    save_arrays(tables/'posterior_overview_positions.h5', **data)
    contours = prepare_contours(data, grid_size=grid_size, log=log)
    contours['position_data'] = data
    contours['fingerprint'] = draws.key
    (tables/'posterior_overview_positions.json').write_text(json.dumps(dict(
        fingerprint=draws.key, draws=len(draws.image_indices), selected_draw_indices=np.asarray(draws.image_indices).tolist(),
        solver='the image recovery of the reporting draws (no new solve)',
        extras='Extra images found in at least half the draws; per draw the nearest extra of the family',
        contours=f'KDE highest-density regions (at least {MIN_KDE_DRAWS} draws) or Gaussian ellipses, 68.3% and 95.4%',
        summaries=contours['summaries']), indent=2))
    return contours
