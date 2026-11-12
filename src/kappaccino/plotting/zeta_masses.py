"""Galaxy-scale masses and concentrations on the true and model convergence maps of a simulated cluster, from the
zeta statistic of Fahlman et al. (1994) about each cluster member.

For an aperture of radius r1 and an annulus r1 < r < r2 about a member,

    M_zeta = M(<r1) - A1 * [M(<r2) - M(<r1)] / (A2 - A1),

the mass inside r1 above the mean surface density of the annulus (A1 and A2 are the areas inside r1 and r2), which
removes the smooth cluster to first order and is a lower bound on the projected mass of the galaxy's halo inside r1.
The concentration is the contrast C = kappa(<r1) / kappa(r1 < r < r2) - 1.  Two apertures are measured:

* ``10kpc``: r1 = 10 kpc, r2 = 20 kpc for every member (the mass comparison);
* ``re``: r1 = R_e, r2 = 2 R_e, with R_e from the fixed F160W size-magnitude relation of ``aperture_mass_histogram``
  (the concentration comparison).

The model convergence is evaluated at the centres of the truth map's pixels and both maps are integrated with the same
pixel weights, so the two are measured at the same resolution.
"""
from pathlib import Path
import json

import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator

from .aperture_mass_histogram import NORMALIZATION_KPC, inside_image_footprint
from .style import PAPER_STYLE
from ..diagnostics.truth_recovery import load_truth_fits, mass_per_kappa_pixel_msun

STYLE = Path(__file__).with_name('style.mplstyle')
# The brightest members sit at the centres of the cluster-scale halos, where the annulus is not a local background.
N_BRIGHTEST_EXCLUDED = 2
FIXED_APERTURE_KPC = 10.0
# Members whose posterior 16-84 per cent range of the plotted quantity is wider than this many times its |median|
# (consistent with zero at about 1 sigma) are left out of the figures, for truth and model alike; the tables keep them.
MAX_RELATIVE_WIDTH = 2.0
SUBPIXELS = 8
# The model and the truth are measured on the true map binned by this factor: the mean of each block of the true map
# is its bilinear value at the block centre, where the model is evaluated.  2 cuts the model evaluations 4x
# (Hera: 0.11" -> 0.22" pixels; the median R_e aperture is 0.6").
REBIN = 2
MODELS = ('parametric', 'full')
# The figures show the full model (parametric + GRF) only; the tables keep both.
PLOTTED = ('full',)
COLORS = {'truth': '0.15', 'parametric': '#D55E00', 'full': '#2878A5'}
LABELS = {'truth': 'Truth', 'parametric': 'Parametric', 'full': 'This work'}
# Binned points, as in the mock parameter-recovery figure.
BINNED_COLOR = 'tab:orange'


def rebinned(values, wcs, factor=REBIN):
    """``values`` averaged over ``factor`` x ``factor`` blocks (the map cropped to whole blocks) and the WCS of the
    binned grid: old 1-based pixel 2 n - 1/2 is new pixel n for factor 2, i.e. new = (old + (factor - 1)/2) / factor."""
    if factor == 1:
        return values, wcs
    ny, nx = (np.asarray(values.shape) // factor) * factor
    binned = values[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))
    coarse = wcs.deepcopy()
    coarse.wcs.crpix = (np.asarray(wcs.wcs.crpix) + (factor - 1) / 2) / factor
    if coarse.wcs.has_cd():
        coarse.wcs.cd = np.asarray(wcs.wcs.cd) * factor
    else:
        coarse.wcs.cdelt = np.asarray(wcs.wcs.cdelt) * factor
    return binned, coarse


def pixel_weights(center_px, radius_px, shape):
    """Fraction of every pixel of the box around ``center_px`` (x, y in pixel units) inside a circle of ``radius_px``,
    from ``SUBPIXELS``^2 sub-pixels.  Returns the flat pixel indices and their weights."""
    cx, cy = center_px
    x0, x1 = int(np.floor(cx - radius_px - 1)), int(np.ceil(cx + radius_px + 1))
    y0, y1 = int(np.floor(cy - radius_px - 1)), int(np.ceil(cy + radius_px + 1))
    xs, ys = np.arange(x0, x1 + 1), np.arange(y0, y1 + 1)
    offsets = (np.arange(SUBPIXELS) + 0.5) / SUBPIXELS - 0.5
    dx2 = (xs[:, None] + offsets[None, :] - cx) ** 2
    dy2 = (ys[:, None] + offsets[None, :] - cy) ** 2
    weight = ((dy2[:, None, :, None] + dx2[None, :, None, :]) <= radius_px ** 2).mean(axis=(2, 3))
    keep = weight > 0
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    return (yy[keep] * shape[1] + xx[keep]).astype(np.int64), weight[keep]


def member_apertures(members, wcs, reference, shape, kpc_per_arcsec, pixel_arcsec):
    """Apertures of the members, by aperture name: every member but the ``N_BRIGHTEST_EXCLUDED`` brightest, and their
    (r1, r2) pixel weights.  Raises if an outer circle leaves the truth map."""
    _, ra0, dec0 = reference
    candidates = members.loc[members.sort_values('catalog_mag').index[N_BRIGHTEST_EXCLUDED:]].copy()
    candidates['re_arcsec'] = NORMALIZATION_KPC * 10 ** (-0.24 * (candidates.catalog_mag.to_numpy(float) - 20.0)) / kpc_per_arcsec
    px, py = wcs.world_to_pixel_values(ra0 - candidates.x_centre.to_numpy(float) / (3600 * np.cos(np.deg2rad(dec0))),
                                       dec0 + candidates.y_centre.to_numpy(float) / 3600)
    radii = {'10kpc': np.full(len(candidates), FIXED_APERTURE_KPC / kpc_per_arcsec / pixel_arcsec),
             're': candidates.re_arcsec.to_numpy(float) / pixel_arcsec}
    result = {}
    for name, r1 in radii.items():
        off = ~((px - 2 * r1 > 1) & (py - 2 * r1 > 1) & (px + 2 * r1 < shape[1] - 2) & (py + 2 * r1 < shape[0] - 2))
        if off.any():
            raise ValueError(f'{name} apertures of {candidates.member_id[off].tolist()} leave the truth map')
        table = candidates.reset_index(drop=True)
        table['r1_arcsec'] = r1 * pixel_arcsec
        weights = [(pixel_weights((x, y), r, shape), pixel_weights((x, y), 2 * r, shape)) for x, y, r in zip(px, py, r1)]
        result[name] = (table, weights)
    return result


def aperture_sums(values, weights):
    """Inner and outer sums of ``values`` and the inner and outer areas (pixels) of every member, shape (4, member)."""
    return np.array([[values[i] @ wi, values[o] @ wo, wi.sum(), wo.sum()] for (i, wi), (o, wo) in weights]).T


def zeta_and_contrast(sums):
    """Zeta statistic (same units as the sums) and concentration kappa(<r1)/kappa(annulus) - 1 from ``aperture_sums``."""
    inner, outer, a1, a2 = sums
    ring = (outer - inner) / (a2 - a1)
    return inner - a1 * ring, inner / a1 / ring - 1.0


def compute_zeta_masses(config, state, draws, members, image_xy, *, log=print):
    """Zeta masses and contrasts of ``members`` (``member_diagnostics.csv``) for both apertures, on the true map and,
    for the reporting draws (``reporting.draws.ReportingDraws``), on the parametric and parametric + GRF model maps.  Returns {aperture: (member table with ``truth_zeta_msun``,
    ``truth_contrast`` and ``inside_image_hull``, {model: {"zeta": (draw, member) in Msun, "contrast": (draw, member)}})}."""
    native, wcs = rebinned(*load_truth_fits(config.truth.kappa_true_fits))
    z_source = float(config.truth.caustic_source_redshift)
    pixel_arcsec = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600
    apertures = member_apertures(members, wcs, state.reference, native.shape, state.kpc_per_arcsec, pixel_arcsec)
    factor = mass_per_kappa_pixel_msun(state, wcs, z_source)
    truth = native.ravel()
    for table, weights in apertures.values():
        zeta, contrast = zeta_and_contrast(aperture_sums(truth, weights))
        table['truth_zeta_msun'], table['truth_contrast'] = zeta * factor, contrast
        table['inside_image_hull'] = inside_image_footprint(table[['x_centre', 'y_centre']].to_numpy(float), image_xy)
    pixels = np.unique(np.concatenate([idx for _, weights in apertures.values() for pair in weights for idx, _ in pair]))
    rows, cols = np.divmod(pixels, native.shape[1])
    _, ra0, dec0 = state.reference
    ra, dec = wcs.pixel_to_world_values(cols, rows)
    # The simulated maps are centred at RA = 0: wrap the RA difference into [-180, 180) degrees.
    d_ra = (ra - ra0 + 180.0) % 360.0 - 180.0
    xy = np.column_stack((-d_ra * 3600 * np.cos(np.deg2rad(dec0)), (dec - dec0) * 3600))
    n_draws = len(draws)
    log(f'[zeta] {", ".join(f"{k}: {len(t)} members" for k, (t, _) in apertures.items())}; {len(pixels)} truth-map pixels, '
        f'{n_draws} posterior draws')
    model = {name: {m: {'zeta': np.empty((n_draws, len(t))), 'contrast': np.empty((n_draws, len(t)))} for m in MODELS}
             for name, (t, _) in apertures.items()}
    values = np.zeros(native.size)
    maps = draws.grid_maps('zeta truth pixels', xy[:, 0], xy[:, 1], z_source, log=log)
    for d in range(n_draws):
        for m, kappa in zip(MODELS, (maps['kappa_parametric'][d].astype(float), maps['kappa'][d].astype(float))):
            values[pixels] = kappa
            for name, (_, weights) in apertures.items():
                zeta, contrast = zeta_and_contrast(aperture_sums(values, weights))
                model[name][m]['zeta'][d], model[name][m]['contrast'][d] = zeta * factor, contrast
    return {name: (table, model[name]) for name, (table, _) in apertures.items()}


CACHE_NAME = 'zeta_masses_cache.h5'


def save_cache(path, results, key):
    """Per-member tables and per-draw model arrays of ``compute_zeta_masses`` for redrawing without recomputing;
    ``key`` is the fingerprint of the reporting draws."""
    from ..io.hdf5 import save_arrays
    arrays = {'draws_key': key}
    for name, (table, model) in results.items():
        for column in table.columns:
            values = table[column].to_numpy()
            arrays[f'{name}__table__{column}'] = values.astype(str) if values.dtype == object else values
        for m, quantities in model.items():
            for quantity, values in quantities.items():
                arrays[f'{name}__model__{m}__{quantity}'] = values
    save_arrays(path, **arrays)


def load_cache(path, key):
    """The results saved by ``save_cache``, or None when there is no cache or it was made with other draws."""
    from ..io.hdf5 import load_arrays
    if not Path(path).exists():
        return None
    arrays = load_arrays(path)
    if str(arrays.get('draws_key', '')) != key:
        return None
    results = {}
    for key, values in arrays.items():
        parts = key.split('__')
        if len(parts) < 3:
            continue
        name = parts[0]
        table, model = results.setdefault(name, ({}, {}))
        if parts[1] == 'table':
            table[parts[2]] = values
        else:
            model.setdefault(parts[2], {})[parts[3]] = values
    return {name: (pd.DataFrame(table), model) for name, (table, model) in results.items()}


def mass_function(masses, bins, floor):
    """Members per bin of log10 mass (``bins``), with masses at or below ``floor`` (zeta <= 0 included) counted in the
    lowest bin; masses has shape (member,) or (draw, member)."""
    return np.array([np.histogram(np.log10(np.maximum(m, floor)), bins)[0] for m in np.atleast_2d(masses)])


def equal_count_bins(x, n_bins):
    """Indices of ``x`` split into ``n_bins`` groups of (nearly) equal size, in increasing ``x``."""
    return np.array_split(np.argsort(x), n_bins)


def log_floor(values):
    """Lower edge of a log axis: half the smallest positive value.  Values at or below it are drawn at it."""
    values = np.asarray(values)
    return 0.5 * values[values > 0].min()


def scatter_with_floor(ax, x, y, low, high, floor, color):
    """Model (``y``, error bars ``low``-``high``) against truth (``x``) on log axes, in the style of the mock
    parameter-recovery figure (faint points, light error bars, no caps); a member with x or y at or below ``floor`` is
    drawn at the floor with an arrow marker (left for x, down for y) and no error bar."""
    below_x, below_y = x <= floor, y <= floor
    normal = ~(below_x | below_y)
    ax.errorbar(x[normal], y[normal], yerr=(np.clip(y - low, 0, None)[normal], np.clip(high - y, 0, None)[normal]), fmt='o',
                color=color, markersize=2.7, markeredgewidth=0, ecolor=matplotlib.colors.to_rgba(color, 0.2), elinewidth=0.55,
                capsize=0, alpha=0.5, zorder=2, label='Subhalos')
    for mask, marker in ((below_y & ~below_x, 'v'), (below_x & ~below_y, '<'), (below_x & below_y, 'D')):
        if mask.any():
            ax.scatter(np.maximum(x[mask], floor), np.maximum(y[mask], floor), marker=marker, s=9, color=color, alpha=0.6,
                       linewidths=0, zorder=2)


def binned_points(ax, x, y, floor, color, n_bins=5):
    """Mean and standard error of log10 ``y`` in ``n_bins`` bins of equal membership in ``x``, at the mean log10 ``x``
    (values at or below ``floor`` taken at the floor; ``floor`` 0 for quantities that are always positive), as larger
    points with error bars."""
    lx, ly = (np.log10(np.maximum(v, floor)) if floor > 0 else np.log10(v) for v in (x, y))
    groups = equal_count_bins(lx, n_bins)
    mx = np.array([lx[g].mean() for g in groups])
    my = np.array([ly[g].mean() for g in groups])
    se = np.array([ly[g].std(ddof=1) / np.sqrt(len(g)) for g in groups])
    return ax.errorbar(10 ** mx, 10 ** my, yerr=(10 ** my - 10 ** (my - se), 10 ** (my + se) - 10 ** my), fmt='o', color=color,
                       markersize=4, elinewidth=0.8, capsize=1.5, zorder=4, label='Binned')


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def render_subhalo_figure(mass_table, mass_model, concentration_model, path):
    """Mass (10 kpc zeta; first two panels) and concentration (kappa(<R_e)/kappa(R_e-2R_e) - 1; last panel) of the
    members on the true and model maps, in one row of square panels in the style of the mock parameter-recovery figure:
    the number of subhalos per 0.2 dex of zeta mass (the lowest bin also holds the masses at or below the axis floor;
    model as posterior median and 16-84 per cent band), model against true mass, and model against true concentration.
    Per-member points carry posterior 16-84 per cent error bars, and orange points the mean and standard error of log10
    model in five bins of equal membership in the truth; values at or below a log axis floor are drawn at it as
    arrows.  Returns the mass-function table."""
    m = 'full'
    truth_mass = mass_table.truth_zeta_msun.to_numpy(float)
    truth_conc = mass_table.truth_contrast.to_numpy(float)
    q16, q50, q84 = np.percentile(mass_model[m]['zeta'], [16, 50, 84], axis=0)
    c16, c50, c84 = np.percentile(concentration_model[m]['contrast'], [16, 50, 84], axis=0)
    mass_floor = log_floor(np.concatenate([truth_mass, q50]))
    mass_top = max(truth_mass.max(), q50.max()) * 1.5
    conc_floor = log_floor(np.concatenate([truth_conc, c50]))
    conc_top = max(truth_conc.max(), c50.max()) * 1.5
    bins = np.arange(np.log10(mass_floor), np.log10(mass_top) + 0.2, 0.2)
    fig, (ax_n, ax_m, ax_c) = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw={'wspace': 0.45})
    counts = {'bin_left_log10_msun': bins[:-1], 'bin_right_log10_msun': bins[1:],
              'truth': mass_function(truth_mass, bins, mass_floor)[0]}
    ax_n.stairs(counts['truth'], 10 ** bins, color=COLORS['truth'], linewidth=1.2, label=LABELS['truth'], zorder=3)
    low, mid, high = np.percentile(mass_function(mass_model[m]['zeta'], bins, mass_floor), [16, 50, 84], axis=0)
    counts.update({f'{m}_median': mid, f'{m}_q16': low, f'{m}_q84': high})
    ax_n.stairs(high, 10 ** bins, baseline=low, fill=True, color=COLORS[m], alpha=0.2, linewidth=0)
    ax_n.stairs(mid, 10 ** bins, color=COLORS[m], linewidth=1.0, label=LABELS[m])
    scatter_with_floor(ax_m, truth_mass, q50, q16, q84, mass_floor, COLORS[m])
    binned_points(ax_m, truth_mass, q50, mass_floor, BINNED_COLOR)
    scatter_with_floor(ax_c, truth_conc, c50, c16, c84, conc_floor, COLORS[m])
    binned_points(ax_c, truth_conc, c50, conc_floor, BINNED_COLOR)
    # About 0.3 dex of room beyond the floor and the largest value on both axes.
    for ax, floor, top in ((ax_m, mass_floor, mass_top), (ax_c, conc_floor, conc_top)):
        limits = (floor / 2.0, top * 1.5)
        ax.plot(limits, limits, color='0.25', linewidth=0.7, linestyle='--', alpha=0.7, zorder=1)
        ax.set(xscale='log', yscale='log', xlim=limits, ylim=limits)
    ax_n.set(xscale='log', yscale='log', xlim=(10 ** (bins[0] - 0.3), 10 ** (bins[-1] + 0.3)),
             ylim=(0.6, 1.5 * max(counts['truth'].max(), high.max())))
    for ax, xlabel, ylabel in ((ax_n, r'$M_\zeta$ ($M_\odot$)', '$N$ subhalos'),
                               (ax_m, r'True $M_\zeta$ ($M_\odot$)', r'Model $M_\zeta$ ($M_\odot$)'),
                               (ax_c, 'True concentration', 'Model concentration')):
        ax.set_xlabel(xlabel, fontsize=8.5, labelpad=2)
        ax.set_ylabel(ylabel, fontsize=8.5, labelpad=2)
        ax.tick_params(labelsize=7)
        ax.set_box_aspect(1)
        # A major tick at every decade and minor ticks between them, on all four sides (the stylesheet's tick placement).
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, numticks=15))
            axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=15))
    # Legends on a mostly opaque white box drawn above the data.
    box = dict(fontsize=7.5, frameon=True, framealpha=0.8, facecolor='white', edgecolor='none')
    ax_n.legend(loc='lower right', **box).set_zorder(10)
    ax_m.legend(handles=ax_m.containers[:2], loc='lower right', **box).set_zorder(10)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)
    return pd.DataFrame(counts)


# Absolute limits on the posterior 16-84 per cent range of a member in the figure: 1 dex in zeta mass, 1.0 in
# concentration (the widest few per cent of the error bars).
MAX_MASS_WIDTH_DEX = 1.0
MAX_CONCENTRATION_WIDTH = 1.0


def narrow_enough(zeta, contrast):
    """Members whose 16-84 per cent posterior range is within ``MAX_MASS_WIDTH_DEX`` in zeta mass (both ends positive)
    and ``MAX_CONCENTRATION_WIDTH`` in concentration; arrays have shape (draw, member)."""
    z16, z84 = np.percentile(zeta, [16, 84], axis=0)
    c16, c84 = np.percentile(contrast, [16, 84], axis=0)
    mass_width = np.where((z16 > 0) & (z84 > 0), np.log10(np.where(z84 > 0, z84, 1.0) / np.where(z16 > 0, z16, 1.0)), np.inf)
    return (mass_width <= MAX_MASS_WIDTH_DEX) & (c84 - c16 <= MAX_CONCENTRATION_WIDTH)


def well_measured(values):
    """Members whose posterior 16-84 per cent range of ``values`` (draw, member) is at most ``MAX_RELATIVE_WIDTH``
    times the absolute posterior median."""
    q16, q50, q84 = np.percentile(values, [16, 50, 84], axis=0)
    return (q84 - q16) <= MAX_RELATIVE_WIDTH * np.abs(q50)


def select_members(table, model, keep):
    """``table`` and ``model`` restricted to the members in ``keep``."""
    return (table[keep].reset_index(drop=True),
            {m: {q: values[:, keep] for q, values in quantities.items()} for m, quantities in model.items()})


def write_zeta_masses(config, state, draws, paths, image_df, *, recompute=False, log=print):
    """Zeta masses (10 kpc aperture) and concentrations (R_e aperture) of a Hera or Ares run with a truth map:
    ``<prefix>_subhalo_mass_concentration.pdf`` (``render_subhalo_figure``) with the members ``well_measured`` in both.
    The masses come from ``zeta_masses_cache.h5`` in the run's tables when it was made with the same reporting draws and ``recompute`` is False, and are otherwise computed and cached there.  Also writes, in the run's tables,
    ``member_zeta_masses.csv`` (one row per member and aperture: truth and the model median, 16th and 84th
    percentiles), ``zeta_mass_function.csv`` and ``zeta_masses.json`` (summary).  Made for every run with a true
    convergence map and cluster members; returns None otherwise."""
    prefix = str(config.paths.figure_prefix or '')
    if not config.truth.kappa_true_fits or not len(state.members.member_ids):
        log('[zeta] skipped: no true convergence map or no cluster members')
        return None
    members = pd.read_csv(paths.tables / 'member_diagnostics.csv', dtype={'member_id': str, 'catalog_id': str})
    if members.member_id.tolist() != list(state.members.member_ids):
        raise ValueError('Zeta-mass catalog identities differ from saved model')
    cache = paths.tables / CACHE_NAME
    key = f'{draws.key}-rebin{REBIN}'
    results = None if recompute else load_cache(cache, key)
    if results is None:
        results = compute_zeta_masses(config, state, draws, members,
                                      image_df[['x_obs_arcsec', 'y_obs_arcsec']].to_numpy(float), log=log)
        save_cache(cache, results, key)
    else:
        log(f'[zeta] using the cached masses in {cache}')
    rows, summary = [], {'n_draws': int(next(iter(results.values()))[1]['full']['zeta'].shape[0]),
                         'excluded': f'the {N_BRIGHTEST_EXCLUDED} brightest members (BCGs); the figure also leaves out members whose 16-84% range of zeta mass or concentration exceeds {MAX_RELATIVE_WIDTH:g} |median| (well_measured)',
                         'definition': 'zeta = M(<r1) - A1 [M(<r2) - M(<r1)]/(A2 - A1), concentration = kappa(<r1)/kappa(r1-r2) - 1, '
                                       f'r2 = 2 r1; 10kpc: r1 = {FIXED_APERTURE_KPC:g} kpc; re: r1 = R_e = '
                                       f'{NORMALIZATION_KPC:g} kpc * 10**(-0.24*(m_F160W-20))'}
    plotted = {'10kpc': 'zeta', 're': 'contrast'}
    for name, (table, model) in results.items():
        out = table[['member_id', 'catalog_mag', 're_arcsec', 'r1_arcsec', 'x_centre', 'y_centre', 'inside_image_hull',
                     'truth_zeta_msun', 'truth_contrast']].copy()
        out.insert(1, 'aperture', name)
        out['well_measured'] = well_measured(model['full'][plotted[name]])
        entry = {'n_members': len(out), 'n_well_measured': int(out.well_measured.sum()),
                 'n_nonpositive_truth_zeta': int((out.truth_zeta_msun <= 0).sum()),
                 'truth_contrast_median': float(out.truth_contrast.median())}
        for m in MODELS:
            for quantity, unit in (('zeta', '_msun'), ('contrast', '')):
                q16, q50, q84 = np.percentile(model[m][quantity], [16, 50, 84], axis=0)
                out[f'{m}_{quantity}_median{unit}'], out[f'{m}_{quantity}_q16{unit}'], out[f'{m}_{quantity}_q84{unit}'] = q50, q16, q84
            entry[f'{m}_contrast_median'] = float(np.median(out[f'{m}_contrast_median']))
            ratio = out[f'{m}_zeta_median_msun'] / out.truth_zeta_msun
            for sample, mask in (('all', out.truth_zeta_msun > 0), ('image_hull', (out.truth_zeta_msun > 0) & out.inside_image_hull)):
                values = ratio[mask].to_numpy(float)
                low, median, high = np.percentile(values, [25, 50, 75]) if len(values) else (np.nan,) * 3
                entry[f'{m}_zeta_ratio_{sample}'] = {'n': int(len(values)), 'median': float(median), 'p25': float(low), 'p75': float(high)}
        summary[name] = entry
        rows.append(out)
    pd.concat(rows, ignore_index=True).to_csv(paths.tables / 'member_zeta_masses.csv', index=False)
    path = paths.figures / f'{prefix}_subhalo_mass_concentration.pdf'
    mass_table, mass_model = results['10kpc']
    concentration_table, concentration_model = results['re']
    if mass_table.member_id.tolist() != concentration_table.member_id.tolist():
        raise ValueError('The two apertures list different members')
    keep = np.logical_and.reduce([well_measured(results[name][1]['full'][quantity]) for name, quantity in plotted.items()])
    keep &= narrow_enough(mass_model['full']['zeta'], concentration_model['full']['contrast'])
    table = mass_table.assign(truth_contrast=concentration_table.truth_contrast)
    table, mass_model = select_members(table, mass_model, keep)
    _, concentration_model = select_members(concentration_table, concentration_model, keep)
    render_subhalo_figure(table, mass_model, concentration_model, path).to_csv(paths.tables / 'zeta_mass_function.csv', index=False)
    (paths.tables / 'zeta_masses.json').write_text(json.dumps(summary, indent=2) + '\n')
    log(f'[zeta] wrote {path}; {int(keep.sum())} of {len(keep)} members shown (well measured in both, error bars within '
        f'{MAX_MASS_WIDTH_DEX:g} dex and {MAX_CONCENTRATION_WIDTH:g})')
    return path
