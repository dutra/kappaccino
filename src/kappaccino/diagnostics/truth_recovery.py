"""Truth recovery for simulated clusters: posterior convergence and magnification grids on the truth WCS.

The WCS helpers that place a model grid on a truth map (:func:`load_truth_fits`, :func:`diagnostic_grid`,
:func:`sample_on_grid`, :func:`grid_arcsec`), and the run-level comparison against the truth convergence and shear maps
of a simulated cluster (:func:`compute_truth_recovery`, configured by ``TruthRecoveryConfig``).  Model positions are
arcsec offsets from the state's reference, :math:`x = (\\alpha_0 - \\alpha)\\cos\\delta_0 \\cdot 3600` (positive west) and
:math:`y = (\\delta - \\delta_0) \\cdot 3600`.  The model maps come from the reporting draws
(``ReportingDraws.grid_maps``): from :math:`A = I - \\partial\\alpha/\\partial\\theta`,
:math:`\\kappa = 1 - (A_{11} + A_{22})/2`, :math:`\\gamma_1 = (A_{22} - A_{11})/2`,
:math:`\\gamma_2 = -(A_{12} + A_{21})/2`, :math:`\\mu = 1/\\det A`.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import gaussian_filter, map_coordinates

from ..config import TruthRecoveryConfig
from ..core.model import ModelState
from ..likelihood.grf_fields import shared_field_evaluator, shared_grf_label

QUANTITIES = ("kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu")
APERTURE_CENTER_SMOOTHING_SIGMA_PIX = 2.0


def load_truth_fits(path: str | Path) -> tuple[np.ndarray, WCS]:
    """Read the first two-dimensional image with a celestial WCS from a FITS file.

    Parameters
    ----------
    path : str or pathlib.Path
        FITS file (for example a truth convergence or shear map).

    Returns
    -------
    image : numpy.ndarray
        The image as float, shape (row, column) = (y, x) pixels; length-one axes are squeezed.
    wcs : astropy.wcs.WCS
        Its celestial WCS.

    Raises
    ------
    ValueError
        If no HDU holds a two-dimensional image with a celestial WCS.
    """
    with fits.open(Path(path), memmap=True) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            image = np.squeeze(np.asarray(data, dtype=float))
            if image.ndim != 2:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FITSFixedWarning)
                wcs = WCS(hdu.header).celestial
            if wcs.has_celestial:
                return image, wcs
    raise ValueError(f"No 2D celestial WCS image found in {path}")


def diagnostic_grid(truth_wcs: WCS, native_shape: tuple[int, int], grid_size: int) -> tuple[WCS, tuple[int, int]]:
    """A reduced WCS grid of at most grid_size pixels per side, sharing the native footprint.

    The pixel scale grows by ``width / grid_size`` along x and ``height / grid_size`` along y, so the reduced grid is
    ``grid_size`` x ``grid_size`` pixels over the same sky area (pixels need not stay square).

    Parameters
    ----------
    truth_wcs : astropy.wcs.WCS
        Celestial WCS of the native image.
    native_shape : tuple of int
        Native image shape ``(height, width)`` in pixels.
    grid_size : int
        Maximum pixels per side; 0 or None, or a native image no larger than this, keeps the native grid.

    Returns
    -------
    wcs : astropy.wcs.WCS
        WCS of the reduced grid (a copy of ``truth_wcs`` when unchanged).
    shape : tuple of int
        Its shape ``(height, width)`` in pixels.
    """
    height, width = int(native_shape[0]), int(native_shape[1])
    size = int(grid_size or 0)
    if size <= 0 or (height <= size and width <= size):
        return truth_wcs.deepcopy(), (height, width)
    scale_x, scale_y = width / size, height / size
    wcs = truth_wcs.deepcopy()
    crpix = np.asarray(wcs.wcs.crpix, dtype=float).copy()
    crpix[0] = 1.0 + (crpix[0] - (0.5 * scale_x - 0.5) - 1.0) / scale_x
    crpix[1] = 1.0 + (crpix[1] - (0.5 * scale_y - 0.5) - 1.0) / scale_y
    wcs.wcs.crpix = crpix
    matrix = np.asarray(truth_wcs.pixel_scale_matrix, dtype=float)
    if matrix.shape == (2, 2):
        wcs.wcs.cd = matrix @ np.diag([scale_x, scale_y])
    else:
        wcs.wcs.cdelt = np.asarray(wcs.wcs.cdelt, dtype=float) * np.asarray([scale_x, scale_y])
    return wcs, (size, size)


def sample_on_grid(values: np.ndarray, source_wcs: WCS, target_wcs: WCS, shape: tuple[int, int]) -> np.ndarray:
    """Resample an image onto another WCS grid by bilinear interpolation in sky coordinates.

    Parameters
    ----------
    values : array_like
        Image on the ``source_wcs`` grid, shape (row, column).
    source_wcs : astropy.wcs.WCS
        WCS of ``values``.
    target_wcs : astropy.wcs.WCS
        WCS of the output grid.
    shape : tuple of int
        Output shape ``(height, width)`` in pixels.

    Returns
    -------
    numpy.ndarray
        Resampled image, shape ``shape``, in the units of ``values``; NaN where the target pixel falls outside the
        source image.
    """
    y, x = np.indices(shape, dtype=float)
    ra, dec = target_wcs.pixel_to_world_values(x, y)
    px, py = source_wcs.world_to_pixel_values(ra, dec)
    return map_coordinates(np.asarray(values, dtype=float), [py, px], order=1, mode="constant", cval=np.nan)


def grid_arcsec(state: ModelState, wcs: WCS, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Model-frame coordinates of every pixel center of a WCS grid.

    Parameters
    ----------
    state : ModelState
        The model state; its ``reference`` gives the origin ``(ra0, dec0)``.
    wcs : astropy.wcs.WCS
        Celestial WCS of the grid.
    shape : tuple of int
        Grid shape ``(height, width)`` in pixels.

    Returns
    -------
    x : numpy.ndarray
        :math:`(\\alpha_0 - \\alpha)\\cos\\delta_0 \\cdot 3600` in arcsec (positive west; RA differences wrapped
        to [-180, 180) degrees), shape ``shape``.
    y : numpy.ndarray
        :math:`(\\delta - \\delta_0) \\cdot 3600` in arcsec, shape ``shape``.
    """
    y, x = np.indices(shape, dtype=float)
    ra, dec = wcs.pixel_to_world_values(x, y)
    _, ra0, dec0 = state.reference
    cos_dec0 = math.cos(math.radians(dec0))
    delta_ra = (ra0 - np.asarray(ra, dtype=float) + 180.0) % 360.0 - 180.0
    return delta_ra * cos_dec0 * 3600.0, (np.asarray(dec, dtype=float) - dec0) * 3600.0


def smoothed_peak_center(kappa_true: np.ndarray, x: np.ndarray, y: np.ndarray, sigma_pix: float = APERTURE_CENTER_SMOOTHING_SIGMA_PIX) -> dict[str, Any] | None:
    finite = np.isfinite(kappa_true)
    if not finite.any():
        return None
    weights = gaussian_filter(finite.astype(float), sigma=sigma_pix, mode="nearest")
    smoothed = gaussian_filter(np.where(finite, kappa_true, 0.0), sigma=sigma_pix, mode="nearest")
    smoothed = np.where(weights > 0, smoothed / np.where(weights > 0, weights, 1.0), np.nan)
    score = np.where(finite & np.isfinite(smoothed), smoothed, -np.inf)
    row, col = np.unravel_index(int(np.argmax(score)), score.shape)
    return {"center_mode": "smoothed_truth_kappa_peak", "center_x_arcsec": float(x[row, col]), "center_y_arcsec": float(y[row, col]),
            "center_smoothing_sigma_pix": float(sigma_pix), "center_smoothed_kappa_peak": float(smoothed[row, col])}


def annular_kappa_profile(kappa_true: np.ndarray, kappa_draws: np.ndarray, x: np.ndarray, y: np.ndarray, center: dict[str, Any],
                          n_bins: int = 30, r_min: float = 2.0, return_draws: bool = False):
    """Azimuthally averaged convergence of truth and model draws in logarithmic annuli about the centre, from ``r_min`` to
    the largest radius with complete annuli on the grid, as in Figs. 12-13 of Meneghetti et al. (2017).  Per annulus: the
    geometric-mean radius, pixel count, true mean, the model mean's median and 16-84 per cent range over draws, and the
    same for the ratio model/true."""
    truth = np.asarray(kappa_true, dtype=float).ravel()
    radius = np.hypot(x.ravel() - center["center_x_arcsec"], y.ravel() - center["center_y_arcsec"])
    valid = np.isfinite(truth) & np.isfinite(radius)
    edge = min(abs(float(np.nanmax(x) - center["center_x_arcsec"])), abs(float(np.nanmin(x) - center["center_x_arcsec"])),
               abs(float(np.nanmax(y) - center["center_y_arcsec"])), abs(float(np.nanmin(y) - center["center_y_arcsec"])))
    edges = np.geomspace(r_min, edge, n_bins + 1)
    draws = np.asarray(kappa_draws, dtype=float).reshape(len(kappa_draws), -1)
    rows, ratio_draws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = valid & (radius >= lo) & (radius < hi)
        if not inside.any():
            continue
        true_mean = float(truth[inside].mean())
        model_means = np.nanmean(draws[:, inside], axis=1)
        ratio_draws.append(model_means / true_mean)
        q = (lambda v: np.nanpercentile(v, [16, 50, 84]) if len(draws) > 1 else (np.nan, v[0], np.nan))
        m16, m50, m84 = q(model_means)
        r16, r50, r84 = q(model_means / true_mean)
        rows.append({"radius_arcsec": float(np.sqrt(lo * hi)), "radius_lo_arcsec": float(lo), "radius_hi_arcsec": float(hi),
                     "pixel_count": int(inside.sum()), "kappa_true_mean": true_mean, "kappa_model_mean": float(m50),
                     "kappa_model_mean_q16": float(m16), "kappa_model_mean_q84": float(m84), "kappa_ratio_median": float(r50),
                     "kappa_ratio_q16": float(r16), "kappa_ratio_q84": float(r84)})
    table = pd.DataFrame(rows)
    return (table, ratio_draws_table(table["radius_arcsec"], ratio_draws)) if return_draws else table


def pixel_scale_arcsec(x: np.ndarray, y: np.ndarray) -> float:
    steps = []
    for axis in (0, 1):
        if x.shape[axis] > 1:
            d = np.hypot(np.diff(x, axis=axis), np.diff(y, axis=axis))
            d = d[np.isfinite(d) & (d > 0)]
            if d.size:
                steps.append(float(np.median(d)))
    return min(steps) if steps else 1.0


def aperture_profile(kappa_true: np.ndarray, kappa_draws: np.ndarray, x: np.ndarray, y: np.ndarray, center: dict[str, Any],
                     n_radii: int = 40, return_draws: bool = False):
    """Enclosed convergence sums of truth and model draws within circular apertures about the centre.  With
    ``return_draws`` also the model/true ratio of every draw (``ratio_draws_table``)."""
    truth = np.asarray(kappa_true, dtype=float).ravel()
    radius = np.hypot(x.ravel() - center["center_x_arcsec"], y.ravel() - center["center_y_arcsec"])
    valid = np.isfinite(truth) & np.isfinite(radius)
    r_max = float(radius[valid].max())
    r_min = min(pixel_scale_arcsec(x, y), r_max)
    radii = np.linspace(r_min, r_max, n_radii)
    draws = np.asarray(kappa_draws, dtype=float).reshape(len(kappa_draws), -1)
    rows, ratio_draws = [], []
    for r in radii:
        inside = valid & (radius <= r)
        truth_sum = float(np.nansum(truth[inside]))
        model_sums = np.nansum(np.where(inside[None, :] & np.isfinite(draws), draws, 0.0), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = model_sums / truth_sum
        ratio_draws.append(ratios)
        q16, q50, q84 = np.nanpercentile(model_sums, [16, 50, 84]) if len(draws) > 1 else (np.nan, model_sums[0], np.nan)
        r16, r50, r84 = np.nanpercentile(ratios, [16, 50, 84]) if len(draws) > 1 else (np.nan, ratios[0], np.nan)
        rows.append({"radius_arcsec": float(r), "pixel_count": int(inside.sum()), "kappa_true_sum": truth_sum,
                     "kappa_model_sum": float(q50), "kappa_model_sum_q16": float(q16), "kappa_model_sum_q84": float(q84),
                     "m2d_ratio": float(r50), "m2d_ratio_q16": float(r16), "m2d_ratio_median": float(r50), "m2d_ratio_q84": float(r84),
                     **{k: center[k] for k in ("center_mode", "center_x_arcsec", "center_y_arcsec", "center_smoothing_sigma_pix", "center_smoothed_kappa_peak")}})
    table = pd.DataFrame(rows)
    return (table, ratio_draws_table(table["radius_arcsec"], ratio_draws)) if return_draws else table


def ratio_draws_table(radius, ratio_draws) -> pd.DataFrame:
    """Model/true ratio of every posterior draw at every radius: columns ``radius_arcsec``, ``draw`` (index among the
    draws) and ``ratio``."""
    ratios = np.asarray(ratio_draws, dtype=float)
    radius = np.repeat(np.asarray(radius, dtype=float), ratios.shape[1])
    return pd.DataFrame({"radius_arcsec": radius, "draw": np.tile(np.arange(ratios.shape[1]), ratios.shape[0]),
                         "ratio": ratios.ravel()})


LITERATURE_RATIOS = "meneghetti2017_ratios.csv"


def literature_ratios(config: TruthRecoveryConfig, quantity: str = "mass") -> pd.DataFrame | None:
    """Model-to-true ratios of the published FF-SIMS reconstructions, digitized from Figs. 12 (Ares) and 13 (Hera) of
    Meneghetti et al. (2017): the ``quantity`` rows (``"mass"``, enclosed projected mass; ``"kappa"``, azimuthally averaged
    convergence) of ``meneghetti2017_ratios.csv`` beside the truth convergence map, or None when there is no such file.
    Columns ``approach`` (parametric or free-form), ``model``, ``r_arcsec``, ``ratio``, ``width_px`` (thickness of the
    curve at that radius) and ``flag`` (0 clean, 1
    partly hidden by another curve, 2 at the panel edge, 3 steep segment)."""
    if not config.kappa_true_fits:
        return None
    path = Path(config.kappa_true_fits).parent / LITERATURE_RATIOS
    if not path.exists():
        return None
    table = pd.read_csv(path, comment="#")
    return table[table["quantity"] == quantity].drop(columns="quantity").reset_index(drop=True)


# Inner radius of the profile figures and of the footprint mean ratios: inside it the annuli hold few pixels.
PROFILE_INNER_ARCSEC = 5.0


def smoothed_digitized_curve(radius, value, n_grid: int = 240, median_window: int = 9, sigma_grid: float = 3.0):
    """One continuous curve through the digitized samples of a published ratio: a running median over ``median_window``
    samples removes single-pixel outliers, the result is interpolated linearly in log radius on ``n_grid`` points (which
    bridges the stretches where the published curve is hidden behind another one), and a Gaussian of ``sigma_grid`` grid
    points smooths the pixel steps.  Returns the grid radii and the smoothed values."""
    from scipy.ndimage import gaussian_filter1d
    order = np.argsort(radius)
    radius, value = np.asarray(radius, dtype=float)[order], np.asarray(value, dtype=float)[order]
    median = pd.Series(value).rolling(median_window, center=True, min_periods=1).median().to_numpy()
    grid = np.geomspace(radius[0], radius[-1], n_grid)
    return grid, gaussian_filter1d(np.interp(np.log(grid), np.log(radius), median), sigma_grid, mode="nearest")


def mean_ratio(radius, ratio, r_inner: float, r_outer: float, n_grid: int = 200) -> tuple[float, float, float] | None:
    """Mean of ``ratio``, mean of ``|ratio - 1|`` and root mean square of ``ratio - 1`` over [``r_inner``, ``r_outer``]
    (arcsec), uniform in log radius, on the part of that range the curve covers; None when it covers none of it."""
    radius, ratio = np.asarray(radius, dtype=float), np.asarray(ratio, dtype=float)
    keep = np.isfinite(radius) & np.isfinite(ratio) & (radius > 0)
    radius, ratio = radius[keep], ratio[keep]
    grid = np.geomspace(r_inner, r_outer, n_grid)
    grid = grid[(grid >= radius.min()) & (grid <= radius.max())] if radius.size else grid[:0]
    if not grid.size:
        return None
    values = np.interp(np.log(grid), np.log(radius), ratio)
    return float(np.mean(values)), float(np.mean(np.abs(values - 1.0))), float(np.sqrt(np.mean((values - 1.0) ** 2)))


def footprint_mean_ratios(mass_profile: pd.DataFrame, kappa_profile: pd.DataFrame, image_radii: np.ndarray,
                          literature_mass: pd.DataFrame | None, literature_kappa: pd.DataFrame | None,
                          mass_draws: pd.DataFrame | None = None, kappa_draws: pd.DataFrame | None = None) -> dict[str, Any]:
    """Mean model/true ratio, mean |ratio - 1| and rms of ratio - 1 (``mean_ratio``) of the enclosed mass and the azimuthally averaged
    convergence within the image footprint,
    for this model (posterior median profile) and the published parametric reconstructions (``smoothed_digitized_curve``
    of their clean and partly hidden samples).  The footprint runs from ``PROFILE_INNER_ARCSEC`` to the outermost image
    (``image_radii``, arcsec from the profile centre), cut at the outer edge of the published figure when there is one,
    so every model is measured over the same radii; a published curve that stops short is measured over the part it
    covers.  Returns ``inner_arcsec``, ``outer_arcsec`` and ``{"mass"|"kappa": {model: (ratio, mean |ratio - 1|, rms of ratio - 1)}}`` with this model
    under ``"This work"``.  With the per-draw ratios (``mass_draws``, ``kappa_draws``: ``ratio_draws_table``), this
    model's entry is instead the median and 16th and 84th percentiles over draws of each of the three numbers, an array
    of shape (3 percentiles, 3 numbers)."""
    outer = float(np.nanmax(image_radii))
    literature = {"mass": literature_mass, "kappa": literature_kappa}
    for table in literature.values():
        if table is not None and len(table):
            outer = min(outer, float(table["r_arcsec"].max()))
    result = {"inner_arcsec": PROFILE_INNER_ARCSEC, "outer_arcsec": outer}
    for quantity, profile, column, draws in (("mass", mass_profile, "m2d_ratio_median", mass_draws),
                                             ("kappa", kappa_profile, "kappa_ratio_median", kappa_draws)):
        if draws is None:
            values = {"This work": mean_ratio(profile["radius_arcsec"], profile[column], PROFILE_INNER_ARCSEC, outer)}
        else:
            per_draw = np.array([mean_ratio(d["radius_arcsec"], d["ratio"], PROFILE_INNER_ARCSEC, outer)
                                 for _, d in draws.groupby("draw")], dtype=float)
            values = {"This work": np.percentile(per_draw, [50, 16, 84], axis=0)}
        table = literature[quantity]
        if table is not None:
            for model, curve in table[(table["approach"] == "parametric") & (table["flag"] <= 1)].groupby("model", sort=False):
                if len(curve) >= 2:
                    values[model] = mean_ratio(*smoothed_digitized_curve(curve["r_arcsec"], curve["ratio"]),
                                                         PROFILE_INNER_ARCSEC, outer)
        result[quantity] = values
    return result


def literature_m2d_ratios(config: TruthRecoveryConfig) -> pd.DataFrame | None:
    """The enclosed-mass rows of :func:`literature_ratios`."""
    return literature_ratios(config, "mass")


def mass_per_kappa_pixel_msun(state: ModelState, wcs: WCS, z_source: float) -> float:
    """Solar masses per unit convergence per grid pixel for the model cosmology."""
    from astropy import constants as const, units as u
    from astropy.wcs.utils import proj_plane_pixel_area
    from .exact_images import astropy_cosmology
    cosmo = astropy_cosmology(state.cosmo_config)
    dl = cosmo.angular_diameter_distance(state.z_lens)
    ds = cosmo.angular_diameter_distance(z_source)
    dls = cosmo.angular_diameter_distance_z1z2(state.z_lens, z_source)
    sigma_crit = (const.c ** 2 / (4 * np.pi * const.G) * ds / (dl * dls)).to(u.Msun / u.Mpc ** 2)
    pixel_solid_angle = float(proj_plane_pixel_area(wcs.celestial)) * (np.pi / 180.0) ** 2
    return float((sigma_crit * dl ** 2).to_value(u.Msun) * pixel_solid_angle)


def truth_maps_on_grid(state: ModelState, config: TruthRecoveryConfig):
    """The true lens maps on the diagnostic grid.

    Parameters
    ----------
    state : ModelState
        Model state; only ``reference`` is read.
    config : TruthRecoveryConfig
        Truth configuration with ``kappa_true_fits`` set.

    Returns
    -------
    wcs : astropy.wcs.WCS
        WCS of the diagnostic grid.
    shape : tuple of int
        Grid shape (ny, nx).
    x, y : numpy.ndarray
        Pixel positions in arcsec from the reference point, shape (ny, nx).
    truth : dict
        ``"kappa"``, and with shear truth also ``"gamma1"``, ``"gamma2"``, ``"detA"``, ``"mu"`` and ``"abs_mu"``.
    """
    native_kappa, native_wcs = load_truth_fits(config.kappa_true_fits)
    wcs, shape = diagnostic_grid(native_wcs, native_kappa.shape, int(config.truth_grid_size))
    truth = {"kappa": sample_on_grid(native_kappa, native_wcs, wcs, shape)}
    if config.gammax_true_fits and config.gammay_true_fits:
        for name, path in (("gamma1", config.gammax_true_fits), ("gamma2", config.gammay_true_fits)):
            values, source_wcs = load_truth_fits(path)
            truth[name] = sample_on_grid(values, source_wcs, wcs, shape)
        truth["detA"] = (1.0 - truth["kappa"]) ** 2 - truth["gamma1"] ** 2 - truth["gamma2"] ** 2
        truth["mu"] = 1.0 / truth["detA"]
        truth["abs_mu"] = np.abs(truth["mu"])
    x, y = grid_arcsec(state, wcs, shape)
    return wcs, shape, x, y, truth


def compute_truth_recovery(state: ModelState, draws, config: TruthRecoveryConfig,
                           fits_dir: Path, tables_dir: Path, *, log=print) -> dict[str, Any] | None:
    """Write median and 16/84 percentile grids of the lens quantities and the aperture profile. Returns grid data.

    Every grid is the parametric model plus the same draw's conditional realization of the shared GRF, when the run has one
    shared mass field.  The truth convergence (and, when both are configured, shear) maps are resampled onto a grid of at most
    ``config.truth_grid_size`` pixels per side.  The model is evaluated at the reporting draws
    (``reporting.draws.ReportingDraws``), for a source at ``config.caustic_source_redshift``, and every map is a
    per-pixel percentile over those draws: the field is never conditioned at a summary of the parameters, which is not
    a posterior state.
    Writes to ``fits_dir`` ``truth_recovery_<q>_model_median.fits``, ``truth_recovery_<q>_model_q16.fits`` and
    ``truth_recovery_<q>_model_q84.fits`` (the percentiles over the draws; q16 and q84 only with at least two) and
    ``truth_recovery_<q>_true.fits`` for each quantity ``q`` in ``kappa``, ``gamma1``, ``gamma2``, ``detA``, ``mu``,
    ``abs_mu`` that the truth provides; and ``truth_recovery_m2d_aperture_profile.csv`` (enclosed convergence of
    truth and model in circular apertures about the smoothed truth-convergence peak),
    ``truth_recovery_kappa_annulus_profile.csv`` and, with at least two draws, the model/true ratio of every draw in
    ``truth_recovery_m2d_aperture_ratio_draws.csv`` and ``truth_recovery_kappa_annulus_ratio_draws.csv``
    (``ratio_draws_table``) to ``tables_dir``.

    Parameters
    ----------
    state : ModelState
        The model state.
    evaluator : Evaluator
        Evaluator of the fitted model.
    draws : ReportingDraws
        The reporting draws and their cached lens evaluations.
    config : TruthRecoveryConfig
        Truth map paths, source redshift, grid size and number of draws.
    fits_dir : pathlib.Path
        Output directory of the FITS grids (created if missing).
    tables_dir : pathlib.Path
        Output directory of the aperture-profile table.
    log : callable, optional
        Logging function.  Default ``print``.

    Returns
    -------
    dict or None
        None when no truth convergence map is configured or the source redshift is not behind the lens.
        Otherwise: ``"wcs"`` and ``"shape"`` of the grid; ``"x"``, ``"y"`` (arcsec, shape (height, width));
        ``"truth"`` (dict of truth maps by quantity); ``"model"`` (dict by quantity of ``{"median", "q16", "q84"}``
        maps); ``"profile"`` (the aperture-profile DataFrame or None) and ``"center"`` (its center, arcsec);
        ``"grf_included"`` (bool); ``"kappa_profile"`` (:func:`annular_kappa_profile` or None);
        ``"z_source"``; ``"literature_m2d"`` and ``"literature_kappa"`` (:func:`literature_ratios`); ``"mass_per_kappa_pixel_msun"`` (Msun per unit
        convergence per grid pixel).
    """
    if not config.kappa_true_fits:
        return None
    z_source = float(config.caustic_source_redshift)
    if z_source <= state.z_lens:
        log(f"[truth] skipped: source redshift {z_source:g} is not behind the lens")
        return None
    wcs, shape, x, y, truth = truth_maps_on_grid(state, config)
    evaluator = draws.evaluator
    n_draws = len(draws)
    grf_included = draws.has_field
    if not grf_included:
        log("[truth] WARNING: no knee mass field in this run; truth maps are the parametric model only")
    else:
        log(f"[truth] parametric + GRF: {shared_grf_label(evaluator)}")
    maps = draws.grid_maps("truth", x, y, z_source, log=log)
    grids = {q: maps[q].astype(float) for q in QUANTITIES}
    fits_dir.mkdir(parents=True, exist_ok=True)
    header = wcs.to_header()
    header["ZSOURCE"] = z_source
    truth_header = header.copy()
    header["GRF"] = grf_included
    header["GRFTYPE"] = "conditional realization; same draw"
    header["GRFFIELD"] = ("run" if shared_field_evaluator(evaluator) is evaluator else "all-block twin") if grf_included else "none"
    header["MAPSTAT"] = (f"pixel percentiles of {n_draws} draws", "each draw with its own GRF")
    products: dict[str, dict[str, np.ndarray]] = {}
    for q in QUANTITIES:
        median = np.nanmedian(grids[q], axis=0)
        products[q] = {"median": median}
        fits.writeto(fits_dir / f"truth_recovery_{q}_model_median.fits", median, header, overwrite=True)
        if n_draws > 1:
            q16, q84 = np.nanpercentile(grids[q], [16, 84], axis=0)
            products[q].update(q16=q16, q84=q84)
            fits.writeto(fits_dir / f"truth_recovery_{q}_model_q16.fits", q16, header, overwrite=True)
            fits.writeto(fits_dir / f"truth_recovery_{q}_model_q84.fits", q84, header, overwrite=True)
        if q in truth:
            fits.writeto(fits_dir / f"truth_recovery_{q}_true.fits", truth[q], truth_header, overwrite=True)
    center = smoothed_peak_center(truth["kappa"], x, y)
    profile = kappa_profile = None
    if center is not None:
        profile, profile_draws = aperture_profile(truth["kappa"], grids["kappa"], x, y, center, return_draws=True)
        kappa_profile, kappa_draws = annular_kappa_profile(truth["kappa"], grids["kappa"], x, y, center, return_draws=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
        profile.to_csv(tables_dir / "truth_recovery_m2d_aperture_profile.csv", index=False)
        kappa_profile.to_csv(tables_dir / "truth_recovery_kappa_annulus_profile.csv", index=False)
        if n_draws > 1:
            profile_draws.to_csv(tables_dir / "truth_recovery_m2d_aperture_ratio_draws.csv", index=False)
            kappa_draws.to_csv(tables_dir / "truth_recovery_kappa_annulus_ratio_draws.csv", index=False)
    return {"wcs": wcs, "shape": shape, "x": x, "y": y, "truth": truth, "model": products, "profile": profile, "kappa_profile": kappa_profile, "center": center,
            "grf_included": grf_included,
            "z_source": z_source, "mass_per_kappa_pixel_msun": mass_per_kappa_pixel_msun(state, wcs, z_source),
            "literature_m2d": literature_m2d_ratios(config), "literature_kappa": literature_ratios(config, "kappa")}
