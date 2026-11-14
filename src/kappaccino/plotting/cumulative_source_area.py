"""Cumulative source-plane area above a magnification, for a simulated cluster: truth, parametric model, and
parametric model plus the GRF."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import APJ_COLUMN_WIDTH_IN, BLUE, DASHES, FIGURE_STYLE, ORANGE, save_page

STYLE = Path(__file__).with_name('style.mplstyle')
MU = np.geomspace(2, 100, 60)          # thresholds; the truth-map pixels do not resolve the area above ~100
FIELD_LIMIT = 3.0                      # below this the area depends on the size of the map
SHELL_HALF_WIDTH = 0.5                 # redshift half-width of the shell behind the volume axis
MODELS = (('parametric', 'Parametric model', ORANGE), ('model', 'Parametric model + GRF', BLUE))


def cumulative_area(det, pixel_area, mu=MU):
    """Source-plane area magnified above each threshold, :math:`A(>\\mu)=\\sum_{|\\mu_i|>\\mu} A_{\\rm pix}/|\\mu_i|`.

    Parameters
    ----------
    det : numpy.ndarray
        :math:`\\det A = 1/\\mu` on the image-plane pixels, shape (..., ny, nx).  Nonfinite pixels are left out.
    pixel_area : float
        Image-plane area of one pixel.
    mu : numpy.ndarray, optional
        Magnification thresholds.

    Returns
    -------
    numpy.ndarray
        Areas in the units of ``pixel_area``, shape (..., len(mu)).
    """
    det = np.asarray(det, dtype=float)
    inverse = np.abs(det).reshape(det.shape[:-2] + (-1,))
    inverse = np.where(np.isfinite(inverse), inverse, np.inf)
    return np.stack([np.where(inverse < 1.0 / m, inverse, 0.0).sum(axis=-1) for m in np.atleast_1d(mu)], axis=-1) * pixel_area


def shell_volume_per_arcsec2(cosmology, z_source, half_width=SHELL_HALF_WIDTH):
    """Comoving volume (Mpc^3) behind one square arcsecond between ``z_source`` -+ ``half_width``."""
    from astropy import units as u
    shell = cosmology.comoving_volume(z_source + half_width) - cosmology.comoving_volume(z_source - half_width)
    return float((shell / (4 * np.pi * u.sr)).to_value(u.Mpc**3 / u.arcsec**2))


def area_table(truth_det, maps, pixel_area, *, with_grf, mu=MU):
    """One row per threshold: the true area and the median and 16th to 84th percentiles over the posterior draws of the
    parametric model (``parametric``) and, with a shared field, of parametric plus GRF (``model``)."""
    table = pd.DataFrame({'mu': mu, 'area_true_arcsec2': cumulative_area(truth_det, pixel_area, mu)})
    for name, key in (('parametric', 'detA_parametric'), ('model', 'detA')):
        if name == 'model' and not with_grf:
            continue
        q16, median, q84 = np.percentile(cumulative_area(maps[key], pixel_area, mu), [16, 50, 84], axis=0)
        table[f'area_{name}_q16_arcsec2'], table[f'area_{name}_median_arcsec2'], table[f'area_{name}_q84_arcsec2'] = q16, median, q84
    return table


@plt.rc_context(fname=STYLE, rc=FIGURE_STYLE)
def render_cumulative_area(table, z_source, volume_per_arcsec2, path):
    """Area against threshold, one ApJ column wide, with the comoving volume of the shell on the right axis.  Dashed
    black: truth.  Lines and bands: median and 16-84 per cent range over the posterior draws.  The lower panel is the
    ratio to the truth; the shaded thresholds depend on the size of the map."""
    from matplotlib.ticker import FuncFormatter
    # A square main panel over the ratio panel; lengths in inches.
    left, right, bottom, top, gap, ratio_height = .55, .67, .4, .05, .05, .72
    side = APJ_COLUMN_WIDTH_IN - left - right
    height = bottom + ratio_height + gap + side + top
    fig = plt.figure(figsize=(APJ_COLUMN_WIDTH_IN, height))
    ratio_ax = fig.add_axes([left / APJ_COLUMN_WIDTH_IN, bottom / height, side / APJ_COLUMN_WIDTH_IN, ratio_height / height])
    ax = fig.add_axes([left / APJ_COLUMN_WIDTH_IN, (bottom + ratio_height + gap) / height, side / APJ_COLUMN_WIDTH_IN, side / height],
                      sharex=ratio_ax)
    ax.tick_params(labelbottom=False)
    mu, truth = table.mu.to_numpy(), table.area_true_arcsec2.to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        for axis in (ax, ratio_ax):
            axis.axvspan(mu[0], FIELD_LIMIT, color='.93', lw=0)
            axis.axvline(FIELD_LIMIT, color='.35', lw=.8, ls='-.')
        ax.text(np.sqrt(mu[0] * FIELD_LIMIT), .03, 'field-of-view limited', rotation=90, ha='center', va='bottom', color='.45',
                fontsize=7, transform=ax.get_xaxis_transform())
        handles, labels, spread = [], [], [.2]
        exact = dict(color='0.1', linestyle=DASHES, linewidth=.7, zorder=3)          # as in the other truth-recovery figures
        true_line, = ax.plot(mu, truth, **exact)
        ratio_ax.axhline(1, **exact)
        handles.append(true_line), labels.append('Truth')
        for name, label, color in MODELS:
            if f'area_{name}_median_arcsec2' not in table:
                continue
            q16, median, q84 = (table[f'area_{name}_{q}_arcsec2'].to_numpy() for q in ('q16', 'median', 'q84'))
            band = ax.fill_between(mu, q16, q84, color=color, alpha=.22, lw=0)
            line, = ax.plot(mu, median, color=color, lw=1.5)
            ratio_ax.fill_between(mu, q16 / truth, q84 / truth, color=color, alpha=.22, lw=0)
            ratio_ax.plot(mu, median / truth, color=color, lw=1.5)
            handles.append((band, line)), labels.append(label)
            shown = (mu >= FIELD_LIMIT) & (truth > 0)
            spread.append(1.15 * np.nanmax(np.abs(np.r_[q16[shown], q84[shown]] / np.r_[truth[shown], truth[shown]] - 1), initial=0))
    positive = truth[truth > 0]
    if positive.size:
        anchor = int(np.argmin(np.abs(mu - 30)))
        slope, = ax.plot(mu, .4 * truth[anchor] * (mu / mu[anchor])**-2., color='.4', lw=.9, ls=':')
        handles.append(slope), labels.append(r'$\propto\mu^{-2}$')
        ax.set_ylim(.2 * positive.min(), 4 * positive.max())
    ax.set(xscale='log', yscale='log', xlim=(mu[0], mu[-1]), ylabel=r'Source area $A(\!>\!\mu)$ (arcsec$^2$)')
    half = float(min(max(spread), .95))
    ratio_ax.set(ylim=(1 - half, 1 + half), xlabel=r'Magnification $\mu$')
    ratio_ax.set_ylabel('Model / true', fontsize=9)
    ratio_ax.set_xticks([3, 10, 30, 100])
    ratio_ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    volume = ax.twinx()
    volume.set(yscale='log', ylim=np.array(ax.get_ylim()) * volume_per_arcsec2)
    volume.set_ylabel(f'Comoving volume (Mpc$^3$)\n${z_source - SHELL_HALF_WIDTH:g}<z<{z_source + SHELL_HALF_WIDTH:g}$', linespacing=1.3)
    ax.tick_params(which='both', right=False)
    ax.legend(handles, labels, frameon=False, loc='upper right', borderaxespad=.5)
    fig.align_ylabels((ax, ratio_ax))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_page(fig, path)


def write_cumulative_source_area(truth, draws, cosmology, figures, tables, prefix, *, log=print):
    """Write ``cumulative_source_area.pdf`` with its table and provenance; only simulations with shear truth qualify.

    Parameters
    ----------
    truth : dict or None
        Result of :func:`kappaccino.diagnostics.truth_recovery.compute_truth_recovery`.
    draws : ReportingDraws
        The reporting draws; their maps on the truth grid are the ones the truth recovery already evaluated.
    cosmology : astropy.cosmology.FLRW
        Cosmology of the volume axis.
    figures, tables : pathlib.Path
        Output directories.
    prefix : str
        Figure prefix of the run.

    Returns
    -------
    pathlib.Path or None
        The figure, or None when the truth has no magnification.
    """
    from ..diagnostics.truth_recovery import pixel_scale_arcsec
    if truth is None or 'detA' not in truth['truth']:
        log('[outputs] cumulative source area skipped: kappa/shear truth required')
        return None
    x, y, z_source = truth['x'], truth['y'], float(truth['z_source'])
    pixel_area = pixel_scale_arcsec(x, y)**2
    with_grf = bool(draws.has_field)
    table = area_table(truth['truth']['detA'], draws.grid_maps('truth', x, y, z_source, log=log), pixel_area, with_grf=with_grf)
    volume = shell_volume_per_arcsec2(cosmology, z_source)
    tables.mkdir(parents=True, exist_ok=True)
    table.to_csv(tables / 'cumulative_source_area.csv', index=False)
    metadata = dict(source_redshift=z_source, definition='sum of pixel_area/|mu| over map pixels with |mu| above the threshold',
                    region='the whole truth map', pixel_arcsec=float(np.sqrt(pixel_area)), map_area_arcsec2=float(pixel_area * np.isfinite(x).sum()),
                    field_of_view_limited_below=FIELD_LIMIT, posterior_draws=len(draws), bands='16th to 84th percentiles over the posterior draws',
                    grf='conditional realization of each draw' if with_grf else 'none',
                    comoving_volume_shell=[z_source - SHELL_HALF_WIDTH, z_source + SHELL_HALF_WIDTH], comoving_mpc3_per_arcsec2=volume)
    (tables / 'cumulative_source_area_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    path = figures / f'{prefix + "_" if prefix else ""}cumulative_source_area.pdf'
    render_cumulative_area(table, z_source, volume, path)
    log(f'[outputs] cumulative source area: {path}')
    return path
