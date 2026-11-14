"""Scatter of the model magnification about the truth at fixed reference magnifications, for the simulated clusters,
beside the scatter between the lens models of different teams.

The statistic follows Raney et al. (2020, MNRAS 494, 4771): select the map pixels where a reference model has
:math:`\\mu_{\\rm ref}`, and take the distribution of the magnification at those pixels over the realizations of the
other models.  Here the reference is the truth and the other models are the posterior draws.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

from .style import APJ_COLUMN_WIDTH_IN, BLUE, FIGURE_STYLE, ORANGE, save_page

STYLE = Path(__file__).with_name('style.mplstyle')
MU_REF = (3.0, 10.0)
BIN_FACTOR = 1.1                       # pixels with true |mu| within this factor of the reference
MIN_PIXELS = 8
# Raney et al. (2020): scatter between teams averaged over the six HFF clusters, and the range over the clusters.
BETWEEN_TEAMS = {3.0: dict(scatter=.40, low=.30, high=.50), 10.0: dict(scatter=.65, low=.45, high=.82)}
BETWEEN_TEAMS_LABEL = 'Between teams (HFF)'
MODELS = (('parametric', 'parametric', 'detA_parametric', ORANGE), ('model', 'parametric + GRF', 'detA', BLUE))


def conditional_scatter(true_abs_mu, draw_abs_mu, mu_ref, factor=BIN_FACTOR):
    """Bias and scatter of the model magnification where the truth has ``mu_ref``.

    Parameters
    ----------
    true_abs_mu : numpy.ndarray
        True :math:`|\\mu|` of the selected map pixels, shape (pixel,).
    draw_abs_mu : numpy.ndarray
        Model :math:`|\\mu|` of each posterior draw at those pixels, shape (draw, pixel).
    mu_ref : float
        Reference magnification; pixels with true :math:`|\\mu|` within ``factor`` of it are used.
    factor : float, optional
        Width of the selection.

    Returns
    -------
    dict
        ``"bias"`` (median of model/true minus one), ``"scatter"`` (half the 16th to 84th percentile range of model/true
        over all draws and pixels, divided by its median) and ``"n_pixels"``.  NaN with fewer than ``MIN_PIXELS`` pixels.
    """
    true_abs_mu, draw_abs_mu = np.asarray(true_abs_mu, float), np.asarray(draw_abs_mu, float)
    selected = (true_abs_mu >= mu_ref / factor) & (true_abs_mu < mu_ref * factor)
    if selected.sum() < MIN_PIXELS:
        return dict(bias=np.nan, scatter=np.nan, n_pixels=int(selected.sum()))
    q16, median, q84 = np.nanpercentile(draw_abs_mu[:, selected] / true_abs_mu[selected], [16, 50, 84])
    return dict(bias=float(median - 1), scatter=float(.5 * (q84 - q16) / median), n_pixels=int(selected.sum()))


def run_scatter_table(truth, draws, image_xy, *, log=print):
    """Scatter of one run at each reference magnification, inside the convex hull of its images.

    Parameters
    ----------
    truth : dict
        ``"x"``, ``"y"`` (arcsec, shape (ny, nx)), ``"z_source"`` and ``"truth"`` with the true ``"abs_mu"`` on that grid.
    draws : ReportingDraws
        The reporting draws of the run.
    image_xy : numpy.ndarray
        Observed image positions in arcsec, shape (image, 2).

    Returns
    -------
    pandas.DataFrame
        One row per model (``parametric``, and ``model`` = parametric + the draw's GRF realization when the run has a shared
        field) and reference magnification: ``model``, ``mu_ref``, ``bias``, ``scatter``, ``n_pixels``.
    """
    x, y = truth['x'], truth['y']
    true_mu = np.asarray(truth['truth']['abs_mu'], float)
    xy = np.unique(np.asarray(image_xy, float)[np.isfinite(image_xy).all(axis=1)], axis=0)
    inside = (Delaunay(xy).find_simplex(np.c_[x.ravel(), y.ravel()]) >= 0).reshape(x.shape) & np.isfinite(true_mu)
    maps = draws.grid_maps('truth', x, y, float(truth['z_source']), log=log)
    rows = []
    for name, _, key, _ in MODELS:
        if name == 'model' and not draws.has_field:
            continue
        with np.errstate(divide='ignore'):
            model_mu = 1.0 / np.abs(np.asarray(maps[key][:, inside], float))
        rows += [dict(model=name, mu_ref=mu_ref, **conditional_scatter(true_mu[inside], model_mu, mu_ref)) for mu_ref in MU_REF]
    return pd.DataFrame(rows)


def write_magnification_scatter(truth, draws, image_df, tables, *, log=print):
    """Write ``magnification_scatter.csv`` (:func:`run_scatter_table`); only simulations with shear truth qualify.

    Parameters
    ----------
    truth : dict or None
        Result of :func:`kappaccino.diagnostics.truth_recovery.compute_truth_recovery`.
    draws : ReportingDraws
        The reporting draws; their maps on the truth grid are the ones the truth recovery already evaluated.
    image_df : pandas.DataFrame
        Image table with ``x_obs_arcsec`` and ``y_obs_arcsec``.
    tables : pathlib.Path
        Output directory.

    Returns
    -------
    pathlib.Path or None
        The table, or None when the truth has no magnification or there are too few images for a hull.
    """
    if truth is None or 'abs_mu' not in truth['truth'] or len(image_df) < 3:
        log('[outputs] magnification scatter skipped: kappa/shear truth and at least three images required')
        return None
    table = run_scatter_table(truth, draws, image_df[['x_obs_arcsec', 'y_obs_arcsec']].to_numpy(float), log=log)
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / 'magnification_scatter.csv'
    table.to_csv(path, index=False)
    log(f'[outputs] magnification scatter about the truth: {path}')
    return path


@plt.rc_context(fname=STYLE, rc=FIGURE_STYLE)
def render_scatter_summary(tables, path):
    """One row per cluster and model, one ApJ column wide: open markers at the lower reference magnification, filled at
    the higher, joined by a line.  The first row is the scatter between teams with the range over the HFF clusters.

    Parameters
    ----------
    tables : dict
        Cluster name to its :func:`run_scatter_table`, in the order drawn.
    path : pathlib.Path
        Output PDF.
    """
    from matplotlib.lines import Line2D
    rows = [(BETWEEN_TEAMS_LABEL, 'k', [100 * BETWEEN_TEAMS[m]['scatter'] for m in MU_REF])]
    for cluster, table in tables.items():
        for name, label, _, color in MODELS:
            values = table[table.model == name].set_index('mu_ref').scatter.reindex(MU_REF)
            if values.notna().any():
                rows.append((f'{cluster}, {label}', color, list(100 * values.to_numpy())))
    left, right, bottom, top = 1.62, .1, .42, .08                       # inches
    height = bottom + top + .29 * len(rows) + .1
    fig = plt.figure(figsize=(APJ_COLUMN_WIDTH_IN, height))
    ax = fig.add_axes([left / APJ_COLUMN_WIDTH_IN, bottom / height, 1 - (left + right) / APJ_COLUMN_WIDTH_IN, 1 - (bottom + top) / height])
    for i, (_, color, values) in enumerate(rows):
        ax.plot(values, [i, i], color=color, lw=.7, zorder=2)
        if i == 0:
            team = [BETWEEN_TEAMS[m] for m in MU_REF]
            ax.errorbar(values, [i, i], xerr=[[100 * (t['scatter'] - t['low']) for t in team], [100 * (t['high'] - t['scatter']) for t in team]],
                        fmt='none', ecolor='k', elinewidth=.8, capsize=2.5, zorder=2)
        ax.plot(values[0], i, 'o', color=color, ms=5, mfc='w', mew=1.1, zorder=3)
        ax.plot(values[-1], i, 'o', color=color, ms=5, zorder=3)
    ax.set_yticks(range(len(rows)), labels=[row[0] for row in rows])
    ax.tick_params(axis='y', which='both', length=0)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    finite = [v for row in rows for v in row[2] if np.isfinite(v)] + [100 * BETWEEN_TEAMS[MU_REF[-1]]['high']]
    ax.set(ylim=(len(rows) - .45, -.55), xlim=(0, 1.08 * max(finite)), xlabel=r'Scatter in $\mu$ at fixed $\mu_{\rm ref}$ (%)')
    ax.legend(handles=[Line2D([], [], color='.3', marker='o', ms=5, mfc='w', mew=1.1, ls='none', label=rf'$\mu_{{\rm ref}}={MU_REF[0]:g}$'),
                       Line2D([], [], color='.3', marker='o', ms=5, ls='none', label=rf'$\mu_{{\rm ref}}={MU_REF[-1]:g}$')],
              loc='lower right', frameon=False, borderaxespad=.4)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_page(fig, path)
    return path
