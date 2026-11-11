"""Inputs of the uncertain-image likelihood (``partition.groups``): the image table, the catalog terms and the prior weights.

``build_inputs`` reads the uncertain images (a CSV table, or the images of the image catalog the run does not trust),
fits the population mixture q(z, c) and the same-source color scatter, and computes everything that does not depend on
the lens.  That covers the redshift grid of the quadrature, the catalog evidence of each image as its own source, the
catalog predictive of each image given each trusted family, and the prior weights of the options of each image.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from ..core.model import ModelState
from ..likelihood import smoothed as sl
from .catalog_evidence import (LOG_2PI, CatalogData, GaussianMixture, available_bands, catalog_data, catalog_log_evidence, fit_color_scatter,
                               fit_gaussian_mixture, fit_population, image_redshifts)
from .lens_evidence import LensImages, lens_images

TRUSTED_Z_SIGMA = 0.01        # redshift uncertainty given to trusted images in the catalog terms
GRID_POINTS = 96              # default points of the redshift quadrature


# ------------------------------------------------------------------ tables
def load_uncertain_images(path, state: ModelState) -> pd.DataFrame:
    """The uncertain-image table with ``x_obs``/``y_obs`` in the state frame (arcsec, x positive west) and ``parity``;
    ``path`` is a CSV file or a table already read.  The table carries no positional errors: uncertain images take the
    sampled error model of the trusted ones."""
    table = path if isinstance(path, pd.DataFrame) else pd.read_csv(Path(path))
    missing = {"image_label", "ra", "dec"} - set(table.columns)
    if missing:
        raise ValueError(f"{path if not isinstance(path, pd.DataFrame) else 'uncertain images'}: missing columns {sorted(missing)}")
    if "sigma_arcsec" in table:
        raise ValueError("uncertain images: sigma_arcsec is not used; the measurement error of every image is the sampled "
                         "images.sigma_meas.  Drop the column.")
    from ..io.sky import radec_to_offsets_arcsec
    _, ra0, dec0 = state.reference
    table = table.copy()
    table["x_obs"], table["y_obs"] = radec_to_offsets_arcsec(table["ra"].to_numpy(dtype=float), table["dec"].to_numpy(dtype=float),
                                                             float(ra0), float(dec0))
    parity = pd.to_numeric(table["parity"], errors="coerce").fillna(0).to_numpy(dtype=int) if "parity" in table else np.zeros(len(table), dtype=int)
    table["parity"] = parity
    table["image_label"] = table["image_label"].astype(str)
    return table.reset_index(drop=True)


def trusted_catalog_table(config, state: ModelState) -> pd.DataFrame:
    """The trusted images (families present in the state) with their catalog photometry and secure redshifts."""
    from ..io.image_catalog import load_image_catalog
    table = load_image_catalog(config.model.image_constraints.catalog_path)
    table = table[table["image_label"].astype(str).isin(set(state.images.image_labels))].copy()
    table["catalog_z_sigma"] = TRUSTED_Z_SIGMA
    return table.reset_index(drop=True)


def catalog_uncertain_images(config, state: ModelState) -> pd.DataFrame:
    """The images of the image catalog the run does not trust (below ``trusted_prior_min``, in a family without a
    positive redshift or left with one image), as an uncertain-image table claimed by their own families."""
    from ..io.image_catalog import load_image_catalog
    table = load_image_catalog(config.model.image_constraints.catalog_path)
    table = table[~table["image_label"].astype(str).isin(set(state.images.image_labels))].copy()
    table["catalog_z"] = table["catalog_z"].where(table["catalog_z"] > 0.0)
    table["catalog_z_sigma"] = table["catalog_z_err"]
    table["claimed_family"] = table["family_id"].astype(str)
    return table.reset_index(drop=True)


# ------------------------------------------------------------------ prior weights
def claimed_groups(table: pd.DataFrame, trusted_ids) -> list[tuple[int, ...]]:
    """Groups of two or more uncertain images claimed by the same family that is not trusted: one new source each."""
    if "claimed_family" not in table:
        return []
    trusted = set(map(str, trusted_ids))
    groups: dict[str, list[int]] = {}
    for i, family in enumerate(table["claimed_family"].fillna("").astype(str)):
        if family and family not in trusted:
            groups.setdefault(family, []).append(i)
    return [tuple(g) for g in groups.values() if len(g) > 1]


def claim_columns(table: pd.DataFrame, trusted_ids, claimed: list[tuple[int, ...]]) -> np.ndarray:
    """Column of each uncertain image's claimed option in the weight matrix (1 + trusted + new sources), -1 for none:
    its trusted family, or the new source of its claimed group."""
    columns = np.full(len(table), -1)
    if "claimed_family" not in table:
        return columns
    trusted = {str(f): k for k, f in enumerate(trusted_ids)}
    for i, family in enumerate(table["claimed_family"].fillna("").astype(str)):
        if family in trusted:
            columns[i] = 1 + trusted[family]
    for k, group in enumerate(claimed):
        columns[list(group)] = 1 + len(trusted) + k
    return columns


def association_priors(table: pd.DataFrame) -> np.ndarray:
    return (pd.to_numeric(table["association_prior"], errors="coerce").to_numpy(dtype=float) if "association_prior" in table
            else np.full(len(table), np.nan))


def association_log_weights(weights: np.ndarray, columns: np.ndarray, priors: np.ndarray, *, floor: float = 1.0e-6) -> np.ndarray:
    """Per-image log weights over the options (unrelated, trusted families, new sources), shape (image, option).

    An image with a claimed option (``columns`` >= 0) and a finite prior p puts p on the claimed option and shares
    1 - p among the other options in proportion to ``weights``.  Any other image keeps ``weights``.
    """
    weights = np.asarray(weights, dtype=float)
    out = np.broadcast_to(weights, (len(columns), weights.size)).copy()
    for i, (column, p) in enumerate(zip(columns, priors)):
        if column < 0 or not np.isfinite(p):
            continue
        p = float(np.clip(p, floor, 1.0 - floor))
        rest = weights.sum() - weights[column]
        out[i] = weights * (1.0 - p) / rest
        out[i, column] = p
    return np.log(out)


# ------------------------------------------------------------------ redshift grid
def efficiency_spaced_grid(z_min: float, z_max: float, points: int, z_lens: float, cosmo_config) -> np.ndarray:
    """``points`` redshifts from z_min to z_max spaced uniformly in the lensing efficiency D_ls / D_s.

    A family of several images fixes the efficiency to about the ratio of its position error to its deflection (a
    few percent), so the quadrature needs points that resolve that width at every redshift.  Uniform in efficiency,
    the points are dense at low redshift, where the efficiency changes fastest.
    """
    from ..lens.cosmology import dpie_sigma0_factor_from_config
    coarse = np.linspace(float(z_min), float(z_max), 40)
    efficiency = np.array([dpie_sigma0_factor_from_config(z_lens, float(z), cosmo_config) for z in coarse])
    if not np.all(np.diff(efficiency) > 0):
        raise ValueError("the lensing efficiency must increase with redshift over the grid.")
    fine = np.linspace(float(z_min), float(z_max), 4000)
    efficiency_fine = np.interp(fine, coarse, efficiency)       # the efficiency is smooth in redshift
    return np.interp(np.linspace(efficiency[0], efficiency[-1], int(points)), efficiency_fine, fine)


def redshift_grid(partition, images: pd.DataFrame, state: ModelState) -> np.ndarray:
    """The quadrature grid: ``partition.redshift_grid`` (z_min, z_max, points), or by default the catalog redshifts
    +- 2 sigma (1 to 6 when none is known, above z_lens + 0.05) with ``GRID_POINTS`` points, spaced uniformly in the
    lensing efficiency."""
    z_lens = float(state.z_lens)
    if partition.redshift_grid is not None:
        z_min, z_max, points = partition.redshift_grid
    else:
        mean, sigma = image_redshifts(images)
        known = np.isfinite(mean)
        z_min = max(float(np.min(mean[known] - 2.0 * sigma[known])) if known.any() else 1.0, z_lens + 0.05)
        z_max = max(float(np.max(mean[known] + 2.0 * sigma[known])) if known.any() else 6.0, z_min + 0.1)
        points = GRID_POINTS
    return efficiency_spaced_grid(z_min, z_max, int(points), z_lens, state.cosmo_config)


# ------------------------------------------------------------------ lens terms
def position_loglike(terms: dict[str, jnp.ndarray], source: jnp.ndarray, parity: jnp.ndarray) -> jnp.ndarray:
    """log N(beta_i | source_k, C_i) + log h_i per source and image from ``image_terms`` at the sources' redshifts,
    shape (source, image); -inf where a term is invalid."""
    residual = terms["beta"] - source[:, None, :]
    quad = jnp.einsum("kni,knij,knj->kn", residual, terms["precision"], residual)
    determinant_mean = terms["determinant"] + jnp.einsum("kni,kni->kn", terms["p"], residual)
    log_h = sl.log_expected_absolute_normal(determinant_mean, terms["determinant_variance"], jnp.broadcast_to(parity, quad.shape))
    value = -0.5 * (quad + terms["logdet"] + 2.0 * LOG_2PI) + log_h
    return jnp.where(terms["valid"] & jnp.isfinite(value), value, -jnp.inf)


# ------------------------------------------------------------------ inputs
@dataclass(frozen=True, eq=False)
class UncertainImageInputs:
    """Everything about the uncertain images that does not depend on the lens.

    Attributes
    ----------
    table : pandas.DataFrame
        The uncertain images (``load_uncertain_images``).
    images : LensImages
        Their positions in the state frame, the redshift grid and the source prior.
    trusted_ids : tuple of str
        Trusted family ids in ``ImageData`` order.
    trusted_ratio : numpy.ndarray
        Lensing efficiency of each trusted family over ``images.factor_reference``, shape (family,).
    trusted_catalog : numpy.ndarray
        log p(catalog data of image i | the family's members), shape (image, family).
    singleton_catalog : numpy.ndarray
        Catalog evidence of each image as its own source on the redshift grid, shape (image, grid).
    data, rows, population, settings
        The catalog of trusted and uncertain images, the rows of the uncertain ones, the population mixture and the
        ``catalog_log_evidence`` settings, for the catalog evidence of any group (``catalog``).
    """
    table: pd.DataFrame
    images: LensImages
    trusted_ids: tuple[str, ...]
    trusted_ratio: np.ndarray
    trusted_catalog: np.ndarray
    singleton_catalog: np.ndarray
    data: CatalogData
    rows: np.ndarray
    population: GaussianMixture
    settings: dict[str, Any]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.table["image_label"])

    @property
    def n_images(self) -> int:
        return len(self.table)

    def catalog(self, index: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Catalog evidence of padded groups of uncertain images (group, width) as one source, shape (group, grid)."""
        return catalog_log_evidence(self.data, self.rows[np.asarray(index)], np.asarray(mask, dtype=bool), np.asarray(self.images.z_grid),
                                    self.population, **self.settings)


def build_inputs(config, state: ModelState, *, log=print) -> UncertainImageInputs:
    """The ``UncertainImageInputs`` of ``config.partition`` for the trusted families of ``state``."""
    partition = config.partition
    source = partition.catalog_path if partition.catalog_path is not None else catalog_uncertain_images(config, state)
    table = load_uncertain_images(source, state)
    trusted = trusted_catalog_table(config, state)
    z_grid = redshift_grid(partition, table, state)
    bands = tuple(partition.bands) if partition.bands is not None else available_bands(table)
    combined = pd.concat([trusted, table], ignore_index=True, sort=False)
    population, bands = _population(partition, combined, bands, log=log)
    data = catalog_data(combined, bands)
    n = len(table)
    family_rows = {f: np.flatnonzero(trusted["family_id"].astype(str).to_numpy() == f) for f in state.images.family_ids}
    scatter = partition.color_scatter if partition.color_scatter is not None else _fit_scatter(data, family_rows, population, partition, log=log)
    settings = dict(color_scatter=float(scatter), outlier_fraction=float(partition.outlier_fraction),
                    outlier_scatter=float(partition.outlier_scatter), photoz_outlier_fraction=float(partition.photoz_outlier_fraction))

    rows = len(trusted) + np.arange(n)
    singleton_catalog = catalog_log_evidence(data, rows[:, None], np.ones((n, 1), dtype=bool), z_grid, population, **settings)
    z_by_family = {f.family_id: float(f.z_source) for f in state.family_data}
    trusted_catalog = np.zeros((n, len(state.images.family_ids)))
    for k, family in enumerate(state.images.family_ids):
        members = family_rows[family]
        z_family = np.asarray([z_by_family[family]])
        index = np.column_stack([np.broadcast_to(members, (n, members.size)), rows])
        joint = catalog_log_evidence(data, index, np.ones(index.shape, dtype=bool), z_family, population, **settings)[:, 0]
        alone = (catalog_log_evidence(data, members[None], np.ones((1, members.size), dtype=bool), z_family, population, **settings)[0, 0]
                 if members.size else 0.0)
        trusted_catalog[:, k] = joint - alone

    images = lens_images(state, table[["x_obs", "y_obs"]].to_numpy(dtype=float), table["parity"].to_numpy(dtype=int), z_grid,
                         prior_sigma=partition.source_prior_sigma_arcsec)
    trusted_ratio = np.asarray([state.dpie_sigma0_factor_by_z[z_by_family[f]] for f in state.images.family_ids]) / images.factor_reference
    log(f"[uncertain] {n} uncertain images; {data.colors.shape[1]} colors from bands {list(bands)}; color scatter {scatter:.3f} mag; "
        f"redshift grid {z_grid[0]:.2f}-{z_grid[-1]:.2f} ({z_grid.size} points, uniform in lensing efficiency)")
    return UncertainImageInputs(table=table, images=images, trusted_ids=tuple(state.images.family_ids), trusted_ratio=trusted_ratio,
                                trusted_catalog=trusted_catalog, singleton_catalog=singleton_catalog, data=data, rows=rows,
                                population=population, settings=settings)


def _population(partition, combined: pd.DataFrame, bands: tuple[str, ...], *, log) -> tuple[GaussianMixture, tuple[str, ...]]:
    """The population mixture over (z, colors), and the bands it uses (none when no colors can be fit)."""
    if partition.population_catalog_path is not None:
        source = pd.read_csv(Path(partition.population_catalog_path))
    else:
        source = combined
        log("[uncertain] no population catalog: the population mixture is fit on the trusted and uncertain images")
    if len(bands) >= 2:
        try:
            return fit_population(source, bands, n_components=int(partition.population_components)), bands
        except ValueError:
            log(f"[uncertain] no complete colors in bands {list(bands)}: colors are not used")
    z, _ = image_redshifts(source.reset_index(drop=True))
    z = z[np.isfinite(z)]
    if z.size >= 2:
        return fit_gaussian_mixture(z[:, None], n_components=min(int(partition.population_components), 3)), ()
    return GaussianMixture(weights=np.ones(1), means=np.asarray([[3.0]]), covariances=np.asarray([[[4.0]]])), ()


def _fit_scatter(data, family_rows: dict[str, np.ndarray], population: GaussianMixture, partition, *, log) -> float:
    """Same-source color scatter fit on the trusted families with two or more photometered images (0.1 mag without)."""
    measured = np.all(np.diagonal(data.color_covariance, axis1=1, axis2=2) < 1.0e3, axis=1)
    families = [rows[measured[rows]] for rows in family_rows.values() if measured[rows].sum() >= 2]
    if data.colors.shape[1] == 0 or not families:
        return 0.1
    width = max(rows.size for rows in families)
    index = np.zeros((len(families), width), dtype=int)
    mask = np.zeros((len(families), width), dtype=bool)
    for k, rows in enumerate(families):
        index[k, :rows.size], mask[k, :rows.size] = rows, True
    z = np.unique(np.round(data.z_mean[index[mask]], 6))
    scatter = fit_color_scatter(data, index, mask, z[np.isfinite(z)], population, outlier_fraction=partition.outlier_fraction,
                                outlier_scatter=partition.outlier_scatter)
    log(f"[uncertain] same-source color scatter {scatter:.3f} mag fit on {len(families)} trusted families")
    return scatter
