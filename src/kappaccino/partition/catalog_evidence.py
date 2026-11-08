"""Catalog evidence of a group of images: how probable their redshifts and colors are as one source drawn from the
galaxy population (``slots`` uses it for an unrelated image, for a trusted family's predictive and for seeding).

For a family F with images i, catalog redshifts zhat_i (Gaussian, sigma_i) and colors c_i (adjacent-band magnitude
differences, measurement covariance S_i), on a grid of source redshifts z_g:

    E_cat(F, z_g) = dz_g prod_i N(zhat_i | z_g, sigma_i^2) sum_k pi_k N(z_g | mu_kz, s_kz^2) I_k(F, z_g)

    I_k(F, z) = int N(theta | a_k(z), B_k(z)) prod_i N(c_i | theta, S_i + s^2 I) dtheta

where (pi_k, mu_k, Sigma_k) is a Gaussian mixture of the population in (z, colors) and a_k(z), B_k(z) the conditional
of its colors at redshift z.  The integral is closed form: with P = B^-1 + sum_i V_i^-1 and m = P^-1 (B^-1 a + sum_i
V_i^-1 c_i), I = N(m | a, B) prod_i N(m | c_i, V_i) (2 pi)^(d/2) |P|^-1/2.  Real counterimages differ in color
(differential magnification, blending, photometry systematics), so s is a same-source scatter fit on trusted families,
and a fraction f of families draws its colors with a broad scatter instead:  I = (1 - f) I(s) + f I(s_broad).

The mixture's redshift marginal is the redshift prior of a source, so a family pays the prior volume of its colors and
redshift once: grouping images is favored only when they agree better than population members do by chance.  The
evidence does not depend on the lens and is computed once, before sampling.  A missing color gets a variance of
1e4 mag^2, which marginalizes it to within a partition-independent constant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

LOG_2PI = float(np.log(2.0 * np.pi))
MISSING_VARIANCE = 1.0e4
DEFAULT_MAG_ERR = 0.1
REDSHIFT_COLUMNS = ("zspec_best", "catalog_z", "zphot_best", "image_zphot_family", "z_source")
DEFAULT_REDSHIFT_SIGMA = 0.5


def _numeric(table: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in table.columns:
        return np.full(len(table), default, dtype=float)
    return pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)


def image_redshifts(images: pd.DataFrame, *, default_sigma: float = DEFAULT_REDSHIFT_SIGMA) -> tuple[np.ndarray, np.ndarray]:
    """Best redshift (first positive of ``REDSHIFT_COLUMNS``, NaN if none) and its 1-sigma uncertainty per row.

    The uncertainty is ``catalog_z_sigma`` when given; 0.01 for a secure spectroscopic redshift and 0.05 for any other;
    half the ``pagul_zpdf_low``/``pagul_zpdf_high`` range (at least 0.05) for a photometric one; else ``default_sigma``.
    """
    mean = np.full(len(images), np.nan)
    for column in REDSHIFT_COLUMNS:
        values = _numeric(images, column)
        fill = ~np.isfinite(mean) & np.isfinite(values) & (values > 0.0)
        mean[fill] = values[fill]
    sigma = np.full(len(images), float(default_sigma))
    zspec = _numeric(images, "zspec_best")
    confidence = images.get("zspec_best_confidence", pd.Series("", index=images.index)).fillna("").astype(str).str.lower()
    secure = confidence.str.contains("secure").to_numpy() | (_numeric(images, "zspec_best_confidence_rank") >= 3.0)
    zphot = _numeric(images, "zphot_best")
    low, high = _numeric(images, "pagul_zpdf_low"), _numeric(images, "pagul_zpdf_high")
    has_pdf = np.isfinite(low) & np.isfinite(high) & (high > low)
    is_phot = np.isfinite(zphot) & (zphot > 0.0)
    sigma = np.where(is_phot & has_pdf, np.maximum(0.5 * (high - low), 0.05), sigma)
    is_spec = np.isfinite(zspec) & (zspec > 0.0)
    sigma = np.where(is_spec, np.where(secure, 0.01, 0.05), sigma)
    given = _numeric(images, "catalog_z_sigma")
    sigma = np.where(np.isfinite(given) & (given > 0.0), np.maximum(given, 1.0e-3), sigma)
    return mean, sigma


@dataclass(frozen=True)
class GaussianMixture:
    """Gaussian mixture over (z, colors): weights (k,), means (k, 1 + d), covariances (k, 1 + d, 1 + d)."""
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    @property
    def n_colors(self) -> int:
        return int(self.means.shape[1]) - 1


@dataclass(frozen=True)
class CatalogData:
    """Catalog quantities of the uncertain images used by the catalog evidence.

    Attributes
    ----------
    z_mean, z_sigma : numpy.ndarray
        Catalog redshift (NaN when unknown) and its Gaussian uncertainty, shape (image,).
    colors : numpy.ndarray
        Adjacent-band colors in ``bands`` order, 0 where missing, shape (image, color).
    color_covariance : numpy.ndarray
        Measurement covariance of the colors, ``MISSING_VARIANCE`` on missing ones, shape (image, color, color).
    bands : tuple of str
        Band names; color j is mag_<bands[j]> - mag_<bands[j + 1]>.
    """
    z_mean: np.ndarray
    z_sigma: np.ndarray
    colors: np.ndarray
    color_covariance: np.ndarray
    bands: tuple[str, ...]


def available_bands(images: pd.DataFrame, *, min_fraction: float = 0.5) -> tuple[str, ...]:
    """Bands with a ``mag_<band>`` column measured for at least ``min_fraction`` of the images, sorted by name."""
    bands = []
    for column in sorted(images.columns):
        name = str(column)
        if name.startswith("mag_") and not name.startswith("mag_err_"):
            values = pd.to_numeric(images[column], errors="coerce").to_numpy(dtype=float)
            if len(values) and np.isfinite(values).mean() >= float(min_fraction):
                bands.append(name[4:])
    return tuple(bands)


def color_table(table: pd.DataFrame, bands: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Adjacent-band colors (NaN where a band is missing), shape (row, color), and their measurement covariance from
    ``mag_err_<band>`` (``DEFAULT_MAG_ERR`` where absent), shape (row, color, color)."""
    n, d = len(table), max(len(bands) - 1, 0)
    if d == 0:
        return np.zeros((n, 0)), np.zeros((n, 0, 0))
    mags = np.column_stack([pd.to_numeric(table[f"mag_{b}"], errors="coerce").to_numpy(dtype=float) if f"mag_{b}" in table
                            else np.full(n, np.nan) for b in bands])
    errors = np.column_stack([pd.to_numeric(table[f"mag_err_{b}"], errors="coerce").to_numpy(dtype=float) if f"mag_err_{b}" in table
                              else np.full(n, np.nan) for b in bands])
    errors = np.where(np.isfinite(errors) & (errors > 0.0), errors, DEFAULT_MAG_ERR)
    difference = np.zeros((d, len(bands)))
    difference[np.arange(d), np.arange(d)], difference[np.arange(d), np.arange(1, d + 1)] = 1.0, -1.0
    colors = np.einsum("cb,nb->nc", difference, np.where(np.isfinite(mags), mags, 0.0))
    colors[~(np.isfinite(mags[:, :-1]) & np.isfinite(mags[:, 1:]))] = np.nan
    covariance = np.einsum("cb,nb,eb->nce", difference, errors ** 2, difference)
    return colors, covariance


def catalog_data(images: pd.DataFrame, bands: tuple[str, ...] | None = None) -> CatalogData:
    """Catalog redshifts and colors of the uncertain images."""
    work = images.reset_index(drop=True)
    bands = available_bands(work) if bands is None else tuple(bands)
    z_mean, z_sigma = image_redshifts(work)
    colors, covariance = color_table(work, bands)
    missing = ~np.isfinite(colors)
    covariance = np.where(missing[:, :, None] | missing[:, None, :], 0.0, covariance)
    index = np.arange(colors.shape[1])
    covariance[:, index, index] += np.where(missing, MISSING_VARIANCE, 0.0)
    return CatalogData(z_mean=z_mean, z_sigma=z_sigma, colors=np.where(missing, 0.0, colors), color_covariance=covariance, bands=bands)


# ------------------------------------------------------------------ population model
def fit_gaussian_mixture(data: np.ndarray, n_components: int = 6, *, iterations: int = 300, tolerance: float = 1.0e-6,
                         regularization: float = 1.0e-4, seed: int = 0) -> GaussianMixture:
    """Maximum-likelihood Gaussian mixture by EM with k-means++ starts; rows with a non-finite entry are dropped."""
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x).all(axis=1)]
    n, p = x.shape
    if n == 0:
        raise ValueError("No complete rows to fit the population mixture.")
    k = int(max(1, min(n_components, n // max(2 * p, 1) or 1)))
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):
        distance = np.min([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(x[rng.choice(n, p=distance / distance.sum())] if distance.sum() > 0 else x[rng.integers(n)])
    means = np.asarray(centers)
    floor = regularization * np.eye(p) * np.maximum(np.var(x, axis=0), 1.0e-6)
    covariances = np.repeat((np.cov(x.T).reshape(p, p) + floor)[None], k, axis=0)
    weights = np.full(k, 1.0 / k)
    previous = -np.inf
    for _ in range(int(iterations)):
        log_resp = np.stack([np.log(weights[j]) + _log_normal(x, means[j], covariances[j]) for j in range(k)], axis=1)
        total = logsumexp(log_resp, axis=1)
        resp = np.exp(log_resp - total[:, None])
        counts = resp.sum(axis=0) + 1.0e-12
        weights = counts / n
        means = (resp.T @ x) / counts[:, None]
        for j in range(k):
            centered = x - means[j]
            covariances[j] = (resp[:, j, None] * centered).T @ centered / counts[j] + floor
        if total.mean() - previous < tolerance:
            break
        previous = total.mean()
    return GaussianMixture(weights=weights, means=means, covariances=covariances)


def _log_normal(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    solved = np.linalg.solve(chol, (x - mean).T)
    return -0.5 * (np.sum(solved ** 2, axis=0) + x.shape[1] * LOG_2PI) - np.sum(np.log(np.diag(chol)))


def fit_population(table: pd.DataFrame, bands: tuple[str, ...], *, n_components: int = 6, seed: int = 0) -> GaussianMixture:
    """Population mixture over (redshift, adjacent-band colors) from a photometric catalog with a redshift column
    (``REDSHIFT_COLUMNS``, as for the images) and ``mag_<band>`` columns."""
    z, _ = image_redshifts(table.reset_index(drop=True))
    colors, _ = color_table(table.reset_index(drop=True), tuple(bands))
    return fit_gaussian_mixture(np.column_stack([z, colors]), n_components=n_components, seed=seed)


# ------------------------------------------------------------------ evidence
def redshift_grid_weights(z_grid: np.ndarray) -> np.ndarray:
    """Trapezoid quadrature weights of a sorted grid (one point: weight 1)."""
    z = np.asarray(z_grid, dtype=float)
    if z.size == 1:
        return np.ones(1)
    edges = np.concatenate([[z[0]], 0.5 * (z[1:] + z[:-1]), [z[-1]]])
    return np.diff(edges)


def _gaussian_product_log_integral(prior_mean, prior_cov, values, covariances, mask) -> np.ndarray:
    """log int N(theta | a, B) prod_i N(c_i | theta, V_i) dtheta per family, masked members ignored.

    prior_mean (..., d), prior_cov (..., d, d) broadcast against families; values (u, n, d); covariances (u, n, d, d);
    mask (u, n).  Returns shape (u,) (or the broadcast batch shape)."""
    d = values.shape[-1]
    eye = np.eye(d)
    member_precision = np.linalg.inv(np.where(mask[..., None, None], covariances, eye))
    member_precision = np.where(mask[..., None, None], member_precision, 0.0)
    prior_precision = np.linalg.inv(prior_cov)
    precision = prior_precision + member_precision.sum(axis=-3)
    weighted = np.einsum("...ij,...j->...i", prior_precision, prior_mean) + np.einsum("unij,unj->ui", member_precision, values)
    m = np.linalg.solve(precision, weighted[..., None])[..., 0]
    residual = values - m[:, None, :]
    quad = np.einsum("uni,unij,unj->un", residual, member_precision, residual).sum(axis=1)
    _, logdet_members = np.linalg.slogdet(np.where(mask[..., None, None], covariances, eye))
    prior_residual = m - prior_mean
    quad = quad + np.einsum("...i,...ij,...j->...", prior_residual, prior_precision, prior_residual)
    _, logdet_prior = np.linalg.slogdet(prior_cov)
    _, logdet_precision = np.linalg.slogdet(precision)
    n = mask.sum(axis=1)
    return -0.5 * (quad + np.where(mask, logdet_members, 0.0).sum(axis=1) + logdet_prior + logdet_precision + n * d * LOG_2PI)


def catalog_log_evidence(data: CatalogData, member_index: np.ndarray, member_mask: np.ndarray, z_grid: np.ndarray,
                         population: GaussianMixture | None, *, color_scatter: float = 0.1, outlier_fraction: float = 0.05,
                         outlier_scatter: float = 1.0) -> np.ndarray:
    """log E_cat(F, z_g) per family and grid redshift, shape (family, grid), quadrature weight included.

    ``member_index`` (family, n_max) holds image indices, ``member_mask`` which slots are used.  Without a population
    mixture the redshift prior is uniform over the grid and colors are not used.
    """
    index, mask = np.asarray(member_index, dtype=int), np.asarray(member_mask, dtype=bool)
    z_grid = np.asarray(z_grid, dtype=float)
    log_dz = np.log(redshift_grid_weights(z_grid))
    z_mean, z_sigma = data.z_mean[index], data.z_sigma[index]
    known = mask & np.isfinite(z_mean)
    photo_z = np.where(known[..., None], -0.5 * (((z_grid - np.where(known, z_mean, 0.0)[..., None]) / z_sigma[..., None]) ** 2
                                                 + 2.0 * np.log(z_sigma[..., None]) + LOG_2PI), 0.0).sum(axis=1)   # (u, g)
    if population is None:
        return photo_z + log_dz - np.log(np.ptp(z_grid) if z_grid.size > 1 else 1.0)
    d = data.colors.shape[1]
    if population.n_colors != d:
        raise ValueError(f"population mixture has {population.n_colors} colors; the images have {d}.")
    values, covariances = data.colors[index], data.color_covariance[index]
    out = np.full((index.shape[0], z_grid.size), -np.inf)
    for g, z in enumerate(z_grid):
        terms = []
        for weight, mean, cov in zip(population.weights, population.means, population.covariances):
            log_pz = -0.5 * ((z - mean[0]) ** 2 / cov[0, 0] + np.log(cov[0, 0]) + LOG_2PI)
            if d == 0:
                terms.append(np.full(index.shape[0], np.log(weight) + log_pz))
                continue
            gain = cov[1:, 0] / cov[0, 0]
            a, b = mean[1:] + gain * (z - mean[0]), cov[1:, 1:] - np.outer(gain, cov[0, 1:])
            parts = [np.log1p(-outlier_fraction) + _gaussian_product_log_integral(a, b, values, covariances + color_scatter ** 2 * np.eye(d), mask)]
            if outlier_fraction > 0.0:
                parts.append(np.log(outlier_fraction) + _gaussian_product_log_integral(a, b, values, covariances + outlier_scatter ** 2 * np.eye(d), mask))
            terms.append(np.log(weight) + log_pz + np.logaddexp.reduce(parts, axis=0))
        out[:, g] = logsumexp(np.stack(terms), axis=0)
    return out + photo_z + log_dz


def fit_color_scatter(data: CatalogData, member_index: np.ndarray, member_mask: np.ndarray, z_grid: np.ndarray,
                      population: GaussianMixture, *, outlier_fraction: float = 0.05, outlier_scatter: float = 1.0,
                      bounds: tuple[float, float] = (0.005, 1.0)) -> float:
    """Same-source color scatter s maximizing the catalog evidence of known families (the trusted ones):
    sum_F log sum_g E_cat(F, z_g; s)."""
    def negative(log_s):
        evidence = catalog_log_evidence(data, member_index, member_mask, z_grid, population, color_scatter=float(np.exp(log_s)),
                                        outlier_fraction=outlier_fraction, outlier_scatter=outlier_scatter)
        return -float(np.sum(logsumexp(evidence, axis=1)))
    result = minimize_scalar(negative, bounds=tuple(np.log(bounds)), method="bounded")
    return float(np.exp(result.x))
