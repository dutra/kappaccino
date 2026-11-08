"""Uncertain images as a mixture over source slots, sampled jointly with the lens by NUTS.

Each uncertain image i came from one of: a trusted family's source (its slot: source beta_F sampled by the position
likelihood, redshift z_F fixed); a new slot k (source beta_k, redshift z_k and intrinsic colors theta_k, all sampled);
or its own unrelated source (label 0).  Given the lens and the slot parameters the images are independent, so each
label sums out on its own:

    log L_slots = sum_i log [ pi_0 E_i + sum_F pi_F p(D_i | F) + sum_k pi_k p(D_i | beta_k, z_k, theta_k) ]

    p(D_i | beta, z, theta) = N(beta | beta_i(z), C_i(z)) h_i  x  N(zhat_i | z, sigma_i^2)  x  p(c_i | theta)

with the per-image position term of the trusted images (``smoothed_likelihood``: source-plane Gaussian times the
smoothed |det A|), the catalog redshift, and the colors with a same-source scatter s and an outlier fraction f:
p(c | theta) = (1 - f) N(c | theta, S + s^2 I) + f N(c | theta, S + s_broad^2 I).  For a trusted slot the color term
is the predictive given the family's own members (closed form, independent of the lens).  E_i is the evidence of the
image as its own source: the same terms integrated over a source drawn from the priors (``lens_evidence``,
``catalog_evidence``; a redshift-grid quadrature).  The weights pi are fixed: pi_0 for an unrelated source, the rest
shared equally by the slots.

Expanding the product of sums and integrating each slot's parameters over its prior gives the sum over every
partition of the uncertain images of p(Z) prod_F E_F: the same model as summing over partitions, with the sources
integrated by NUTS instead of analytically.  Within a draw the labels are independent with probabilities
r_ik = softmax_k(log pi_k + log p(D_i | slot k)); sampling them gives joint draws of the lens, sources and partition.

New-slot parameters are sampled as unit offsets: beta_k = anchor_k + scale_k u, z_k = z_lens + exp(c_k + b v),
theta_k = m_k + s_c w, with u, v, w ~ N(0, 1) in the sampled model and a correction term that turns that prior into the
physical one: beta_k ~ N(mu_0, s_0^2 I) (the trusted families' source prior unless configured) and (z_k, theta_k) from
the population mixture.  Anchors and scales come from ``initial_slots``, which seeds one slot per group of uncertain
images whose pair evidence favors a common source at the start lens; a slot far from every image feels no gradient, so
the seeding decides which groupings the chain can reach.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pandas as pd
from scipy.special import logsumexp as np_logsumexp

from .. import smoothed_likelihood as sl
from ..model import ModelState, ParameterSpec
from .catalog_evidence import (LOG_2PI, GaussianMixture, available_bands, catalog_data, catalog_log_evidence, fit_color_scatter,
                               fit_gaussian_mixture, fit_population, image_redshifts)
from .lens_evidence import LensImages, error_scales, family_lens_evidence, grid_fields, image_terms, lens_images, reference_fields

SLOT_SITE = "slot_latent"
LOG_DZ_SCALE = 0.3            # b: the log(z - z_lens) step of a unit offset
COLOR_SCALE = 0.3             # s_c: the color step (mag) of a unit offset
TRUSTED_Z_SIGMA = 0.01        # redshift uncertainty given to trusted images in the catalog terms


@dataclass(frozen=True, eq=False)
class SlotModel:
    """The slot-mixture likelihood of the uncertain images (see the module docstring).

    Attributes
    ----------
    images : LensImages
        Uncertain images (state frame), the redshift grid of their unrelated-source evidence and the source prior.
    labels : tuple of str
        Uncertain image labels, length (image,).
    trusted_ids : tuple of str
        Trusted family ids in ``ImageData`` order: the trusted slots.
    trusted_ratio : numpy.ndarray
        Lensing efficiency of each trusted family over ``images.factor_reference``, shape (family,).
    trusted_catalog : numpy.ndarray
        log p(catalog data of image i | the family's members), shape (image, family).
    singleton_catalog : numpy.ndarray
        Catalog evidence of each image as its own source on the redshift grid, shape (image, grid).
    log_weights : numpy.ndarray
        log pi of label 0, the trusted slots, then the new slots, shape (1 + family + slot,).
    param_start : int
        Index of the first slot parameter in the latent vector; each new slot has 3 + color parameters
        (u_x, u_y, v, w_1..w_d).
    anchor, scale : numpy.ndarray
        Source position of a zero offset and its step per unit offset (arcsec), shape (slot, 2).
    log_dz_center : numpy.ndarray
        log(z - z_lens) at a zero offset, shape (slot,).
    color_center : numpy.ndarray
        Colors at a zero offset, shape (slot, color).
    z_lens : float
    z_table, factor_table : numpy.ndarray
        Lensing-efficiency factor tabulated in redshift, interpolated for the slot redshifts.
    z_mean, z_sigma : numpy.ndarray
        Catalog redshifts of the uncertain images (NaN when unknown), shape (image,).
    colors : numpy.ndarray
        Colors of the uncertain images (0 where missing), shape (image, color).
    color_precision, color_logdet : numpy.ndarray
        Inverse and log det of the color covariance plus the core and broad scatters, shape (2, image, color, color)
        and (2, image).
    outlier_fraction : float
    population : GaussianMixture
        Prior of (z, colors) of a source.
    """
    images: LensImages
    labels: tuple[str, ...]
    trusted_ids: tuple[str, ...]
    trusted_ratio: np.ndarray
    trusted_catalog: np.ndarray
    singleton_catalog: np.ndarray
    log_weights: np.ndarray
    param_start: int
    anchor: np.ndarray
    scale: np.ndarray
    log_dz_center: np.ndarray
    color_center: np.ndarray
    z_lens: float
    z_table: np.ndarray
    factor_table: np.ndarray
    z_mean: np.ndarray
    z_sigma: np.ndarray
    colors: np.ndarray
    color_precision: np.ndarray
    color_logdet: np.ndarray
    outlier_fraction: float
    population: GaussianMixture

    @property
    def n_slots(self) -> int:
        """Number of new slots."""
        return int(self.anchor.shape[0])

    @property
    def n_colors(self) -> int:
        return int(self.colors.shape[1])

    @property
    def n_parameters(self) -> int:
        return self.n_slots * (3 + self.n_colors)

    # ------------------------------------------------------------------ parameters
    def slot_values(self, params: jnp.ndarray) -> dict[str, jnp.ndarray]:
        """Physical slot parameters from the latent vector: ``beta`` (slot, 2), ``z`` (slot,), ``colors`` (slot, color),
        with the latent offsets and log(z - z_lens)."""
        latent = jnp.asarray(params)[self.param_start:self.param_start + self.n_parameters].reshape(self.n_slots, 3 + self.n_colors)
        log_dz = jnp.asarray(self.log_dz_center) + LOG_DZ_SCALE * latent[:, 2]
        return dict(latent=latent, beta=jnp.asarray(self.anchor) + jnp.asarray(self.scale) * latent[:, :2], log_dz=log_dz,
                    z=self.z_lens + jnp.exp(log_dz), colors=jnp.asarray(self.color_center) + COLOR_SCALE * latent[:, 3:])

    def log_prior_correction(self, values: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Physical slot priors plus the log Jacobian of the offset maps, minus the unit-normal priors of the sampled
        offsets (which the sampled model already contains)."""
        mean, sigma = self.images.prior_mean, self.images.prior_sigma
        log_beta = -0.5 * jnp.sum((values["beta"] - mean) ** 2, axis=-1) / sigma ** 2 - jnp.log(2.0 * jnp.pi * sigma ** 2)
        log_pop = population_log_density(self.population, jnp.column_stack([values["z"], values["colors"]]))
        log_jacobian = (jnp.sum(jnp.log(jnp.asarray(self.scale)), axis=-1) + np.log(LOG_DZ_SCALE) + values["log_dz"]
                        + self.n_colors * np.log(COLOR_SCALE))
        base = -0.5 * jnp.sum(values["latent"] ** 2, axis=-1) - 0.5 * values["latent"].shape[-1] * LOG_2PI
        return jnp.sum(log_beta + log_pop + log_jacobian - base)

    # ------------------------------------------------------------------ likelihood
    def log_terms(self, evaluator, params: jnp.ndarray, stats: dict[str, Any]) -> jnp.ndarray:
        """log pi_k + log p(D_i | option k) per uncertain image and option (unrelated, trusted slots, new slots), shape
        (image, 1 + family + slot)."""
        physical = evaluator.physical(params)
        reference = reference_fields(evaluator, physical, self.images)
        values = self.slot_values(params)
        # Unrelated source: the image's evidence on the redshift grid.
        grid = image_terms(grid_fields(reference, self.images), error_scales(evaluator, physical, self.images, self.images.ratio), self.images)
        n = self.images.n_images
        alone = family_lens_evidence(grid, self.images, jnp.arange(n)[:, None], jnp.ones((n, 1), dtype=bool))["log_evidence"]
        log_unrelated = jax.scipy.special.logsumexp(alone + jnp.asarray(self.singleton_catalog), axis=1)
        # Every slot, trusted then new: the per-image position term at its source and redshift.
        ratio = jnp.concatenate([jnp.asarray(self.trusted_ratio),
                                 jnp.interp(values["z"], jnp.asarray(self.z_table), jnp.asarray(self.factor_table)) / self.images.factor_reference])
        source = jnp.concatenate([stats["source_by_family"], values["beta"]])
        terms = image_terms(grid_fields(reference, self.images, ratio), error_scales(evaluator, physical, self.images, ratio), self.images)
        lens = slot_position_loglike(terms, source, self.images.parity).T                         # (image, slot)
        catalog = jnp.concatenate([jnp.asarray(self.trusted_catalog), self.new_slot_catalog(values["z"], values["colors"])], axis=1)
        return jnp.asarray(self.log_weights) + jnp.column_stack([log_unrelated, lens + catalog])

    def new_slot_catalog(self, z: jnp.ndarray, colors: jnp.ndarray) -> jnp.ndarray:
        """log N(zhat_i | z_k, sigma_i^2) + log p(c_i | theta_k), shape (image, slot)."""
        known = np.isfinite(self.z_mean)
        z_mean, z_sigma = jnp.asarray(np.where(known, self.z_mean, 0.0)), jnp.asarray(self.z_sigma)
        photo_z = jnp.where(jnp.asarray(known)[:, None],
                            -0.5 * ((z_mean[:, None] - z[None]) / z_sigma[:, None]) ** 2 - jnp.log(z_sigma)[:, None] - 0.5 * LOG_2PI, 0.0)
        if self.n_colors == 0:
            return photo_z
        residual = jnp.asarray(self.colors)[:, None, :] - colors[None]                         # (image, slot, color)
        quad = jnp.einsum("isc,micd,isd->mis", residual, jnp.asarray(self.color_precision), residual)
        log_normal = -0.5 * (quad + jnp.asarray(self.color_logdet)[:, :, None] + self.n_colors * LOG_2PI)
        f = self.outlier_fraction
        color = (jnp.logaddexp(np.log1p(-f) + log_normal[0], np.log(f) + log_normal[1]) if f > 0.0 else log_normal[0])
        return photo_z + color

    def log_likelihood(self, evaluator, params: jnp.ndarray, stats: dict[str, Any]) -> jnp.ndarray:
        """sum_i log sum_k pi_k p(D_i | k) plus the slot prior correction: the term ``Evaluator.log_likelihood`` adds."""
        terms = self.log_terms(evaluator, params, stats)
        return jnp.sum(jax.scipy.special.logsumexp(terms, axis=1)) + self.log_prior_correction(self.slot_values(params))

    # ------------------------------------------------------------------ posterior
    def option_names(self) -> list[str]:
        """Column names of ``log_terms``: ``unrelated``, the trusted family ids, ``slot<k>``."""
        return ["unrelated"] + list(self.trusted_ids) + [f"slot{k}" for k in range(self.n_slots)]

    def posterior(self, evaluator, samples: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Label probabilities and joint draws for latent posterior draws.

        Returns ``responsibilities`` (draw, image, option), sampled ``labels`` (draw, image; option index),
        ``probability`` (image, option; averaged over draws), ``pair_probability`` (image, image; same slot, the
        unrelated option never pairing), ``slot_z`` and ``slot_beta`` (draw, slot[, 2]) of the new slots.
        """
        def one(params):
            terms = self.log_terms(evaluator, params, evaluator.statistics(params))
            values = self.slot_values(params)
            return terms - jax.scipy.special.logsumexp(terms, axis=1, keepdims=True), values["z"], values["beta"]
        one = jax.jit(one)
        outputs = [one(jnp.asarray(s, dtype=jnp.float64)) for s in np.atleast_2d(samples)]
        log_r = np.stack([np.asarray(o[0]) for o in outputs])
        r = np.exp(log_r)
        labels = np.stack([[rng.choice(r.shape[2], p=row / row.sum()) for row in draw] for draw in r])
        paired = r[:, :, 1:]
        pair = np.einsum("dik,djk->ij", paired, paired) / r.shape[0]
        np.fill_diagonal(pair, 1.0)
        return dict(responsibilities=r, labels=labels, probability=r.mean(axis=0), pair_probability=pair,
                    slot_z=np.stack([np.asarray(o[1]) for o in outputs]), slot_beta=np.stack([np.asarray(o[2]) for o in outputs]))


def slot_position_loglike(terms: dict[str, jnp.ndarray], source: jnp.ndarray, parity: jnp.ndarray) -> jnp.ndarray:
    """log N(beta_i | source_k, C_i) + log h_i per slot and image from ``image_terms`` at the slots' redshifts, shape
    (slot, image); -inf where a term is invalid."""
    residual = terms["beta"] - source[:, None, :]
    quad = jnp.einsum("kni,knij,knj->kn", residual, terms["precision"], residual)
    determinant_mean = terms["determinant"] + jnp.einsum("kni,kni->kn", terms["p"], residual)
    log_h = sl.log_expected_absolute_normal(determinant_mean, terms["determinant_variance"], jnp.broadcast_to(parity, quad.shape))
    value = -0.5 * (quad + terms["logdet"] + 2.0 * LOG_2PI) + log_h
    return jnp.where(terms["valid"] & jnp.isfinite(value), value, -jnp.inf)


def population_log_density(population: GaussianMixture, x: jnp.ndarray) -> jnp.ndarray:
    """log sum_m w_m N(x | mu_m, Sigma_m) for rows of (z, colors), shape (row,)."""
    chol = np.linalg.cholesky(population.covariances)
    inverse = jnp.asarray(np.linalg.inv(chol))
    logdet = jnp.asarray(2.0 * np.sum(np.log(np.diagonal(chol, axis1=-2, axis2=-1)), axis=-1))
    whitened = jnp.einsum("mij,rmj->rmi", inverse, x[:, None, :] - jnp.asarray(population.means)[None])
    log_normal = -0.5 * (jnp.sum(whitened ** 2, axis=-1) + logdet + x.shape[-1] * LOG_2PI)
    return jax.scipy.special.logsumexp(jnp.log(jnp.asarray(population.weights)) + log_normal, axis=-1)


# ------------------------------------------------------------------ inputs
def load_uncertain_images(path, state: ModelState, *, sigma_inst_arcsec: float) -> pd.DataFrame:
    """The uncertain-image table with ``x_obs``/``y_obs`` in the state frame (arcsec, x positive west),
    ``sigma_inst_arcsec`` and ``parity``."""
    table = pd.read_csv(Path(path))
    missing = {"image_label", "ra", "dec"} - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    from ..sky import radec_to_offsets_arcsec
    _, ra0, dec0 = state.reference
    table = table.copy()
    table["x_obs"], table["y_obs"] = radec_to_offsets_arcsec(table["ra"].to_numpy(dtype=float), table["dec"].to_numpy(dtype=float),
                                                             float(ra0), float(dec0))
    sigma = pd.to_numeric(table["sigma_arcsec"], errors="coerce").to_numpy(dtype=float) if "sigma_arcsec" in table else np.full(len(table), np.nan)
    table["sigma_inst_arcsec"] = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, float(sigma_inst_arcsec))
    parity = pd.to_numeric(table["parity"], errors="coerce").fillna(0).to_numpy(dtype=int) if "parity" in table else np.zeros(len(table), dtype=int)
    table["parity"] = parity
    table["image_label"] = table["image_label"].astype(str)
    return table.reset_index(drop=True)


def trusted_catalog_table(config, state: ModelState) -> pd.DataFrame:
    """The trusted images (families present in the state) with their catalog photometry and secure redshifts."""
    from ..image_catalog import load_image_catalog
    constraints = config.model.image_constraints
    table = load_image_catalog(constraints.catalog_path, sigma_inst_arcsec=constraints.sigma_inst_arcsec,
                               sigma_floor_arcsec=constraints.sigma_floor_arcsec)
    table = table[table["family_id"].astype(str).isin(set(state.images.family_ids))].copy()
    table["catalog_z_sigma"] = TRUSTED_Z_SIGMA
    return table.reset_index(drop=True)


def default_redshift_grid(images: pd.DataFrame, z_lens: float, *, points: int = 12) -> np.ndarray:
    """Uniform grid over the catalog redshifts +- 2 sigma (1 to 6 when none is known), above z_lens + 0.05."""
    mean, sigma = image_redshifts(images)
    known = np.isfinite(mean)
    low = float(np.min(mean[known] - 2.0 * sigma[known])) if known.any() else 1.0
    high = float(np.max(mean[known] + 2.0 * sigma[known])) if known.any() else 6.0
    low = max(low, z_lens + 0.05)
    return np.linspace(low, max(high, low + 0.1), int(points))


# ------------------------------------------------------------------ setup
def initial_slots(group_evidence, n: int, pair_catalog, unrelated: np.ndarray, trusted_best: np.ndarray, *,
                  gain_threshold: float = 0.0) -> list[tuple[int, ...]]:
    """Groups of uncertain images to seed new slots with, at one lens model.

    Two images are linked when the evidence of the pair as one new source beats the best alternative of each image
    (an unrelated source or its best trusted family) by more than ``gain_threshold`` (log).  ``group_evidence`` maps
    image groups (group, size) to their lens evidence on the redshift grid, ``pair_catalog`` image pairs to their
    catalog evidence there.  Returns the connected groups of two or more images.
    """
    alternative = np.maximum(unrelated, trusted_best)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return []
    index = jnp.asarray(np.asarray(pairs, dtype=np.int32))
    lens = np.asarray(group_evidence(index)["log_evidence"])
    pair = np_logsumexp(lens + pair_catalog(np.asarray(pairs)), axis=1)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, j), value in zip(pairs, pair):
        if value - alternative[i] - alternative[j] > gain_threshold:
            parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [tuple(g) for g in groups.values() if len(g) > 1]


def add_slots(config, state: ModelState, evaluator, theta: np.ndarray, *, log=print) -> tuple[ModelState, Any]:
    """The state with the uncertain images of ``config.partition`` and their source slots, and its evaluator.

    Deterministic given the state and ``theta`` (the start of the trusted-family fit), so a resumed final stage rebuilds
    the same slots.
    """
    from ..evaluator import Evaluator
    partition = config.partition
    table = load_uncertain_images(partition.catalog_path, state, sigma_inst_arcsec=partition.sigma_inst_arcsec)
    trusted = trusted_catalog_table(config, state)
    z_lens = float(state.z_lens)
    z_grid = (np.linspace(*partition.redshift_grid[:2], int(partition.redshift_grid[2])) if partition.redshift_grid is not None
              else default_redshift_grid(table, z_lens))
    bands = tuple(partition.bands) if partition.bands is not None else available_bands(table)
    combined = pd.concat([trusted, table], ignore_index=True, sort=False)
    population, bands = _population(partition, combined, bands, log=log)
    data = catalog_data(combined, bands)
    n_trusted_rows, n = len(trusted), len(table)
    family_rows = {f: np.flatnonzero(trusted["family_id"].astype(str).to_numpy() == f) for f in state.images.family_ids}
    scatter = partition.color_scatter
    if scatter is None:
        scatter = _fit_scatter(data, family_rows, population, partition, log=log)
    settings = dict(color_scatter=float(scatter), outlier_fraction=float(partition.outlier_fraction), outlier_scatter=float(partition.outlier_scatter))

    rows = n_trusted_rows + np.arange(n)
    singleton_catalog = catalog_log_evidence(data, rows[:, None], np.ones((n, 1), dtype=bool), z_grid, population, **settings)
    z_by_family = {f.family_id: float(f.z_source) for f in state.family_data}
    trusted_catalog = np.zeros((n, len(state.images.family_ids)))
    for k, family in enumerate(state.images.family_ids):
        members = family_rows[family]
        z_family = np.asarray([z_by_family[family]])
        index = np.column_stack([np.broadcast_to(members, (n, members.size)), rows])
        joint = catalog_log_evidence(data, index, np.ones(index.shape, dtype=bool), z_family, population, **settings)[:, 0]
        alone = catalog_log_evidence(data, members[None], np.ones((1, members.size), dtype=bool), z_family, population, **settings)[0, 0] if members.size else 0.0
        trusted_catalog[:, k] = joint - alone

    images = lens_images(state, table[["x_obs", "y_obs"]].to_numpy(dtype=float), table["sigma_inst_arcsec"].to_numpy(dtype=float),
                         table["parity"].to_numpy(dtype=int), z_grid, prior_sigma=partition.source_prior_sigma_arcsec)
    trusted_ratio = np.asarray([state.dpie_sigma0_factor_by_z[z_by_family[f]] for f in state.images.family_ids]) / images.factor_reference
    z_table = np.linspace(z_lens + 0.01, max(float(z_grid[-1]) + 3.0, 12.0), 1024)
    factor_table = efficiency_factor_table(state, z_table)

    # Seed the new slots at the start lens (jitted: run eagerly, every small operation would compile on its own).
    @jax.jit
    def start_terms(params):
        physical = evaluator.physical(params)
        reference = reference_fields(evaluator, physical, images)
        grid = image_terms(grid_fields(reference, images), error_scales(evaluator, physical, images, images.ratio), images)
        alone = family_lens_evidence(grid, images, jnp.arange(n)[:, None], jnp.ones((n, 1), dtype=bool))["log_evidence"]
        ratio = jnp.asarray(trusted_ratio)
        trusted_terms = image_terms(grid_fields(reference, images, ratio), error_scales(evaluator, physical, images, ratio), images)
        trusted = slot_position_loglike(trusted_terms, evaluator.statistics(params)["source_by_family"], images.parity).T
        return grid, alone, trusted

    grid, alone, trusted_lens = start_terms(jnp.asarray(theta, dtype=jnp.float64))
    group_evidence = jax.jit(lambda index: family_lens_evidence(grid, images, index, jnp.ones(index.shape, dtype=bool)))
    unrelated = np_logsumexp(np.asarray(alone) + singleton_catalog, axis=1)
    trusted_best = np.max(np.asarray(trusted_lens) + trusted_catalog, axis=1, initial=-np.inf)

    def pair_catalog(pairs):
        return catalog_log_evidence(data, rows[pairs], np.ones(pairs.shape, dtype=bool), z_grid, population, **settings)

    groups = initial_slots(group_evidence, n, pair_catalog, unrelated, trusted_best, gain_threshold=float(partition.slot_gain_threshold))
    anchor, scale, log_dz, color_center = [], [], [], []
    for group in groups:
        index = jnp.asarray(np.asarray(group, dtype=np.int32)[None])
        out = group_evidence(index)
        catalog = catalog_log_evidence(data, rows[np.asarray(group)][None], np.ones((1, len(group)), dtype=bool), z_grid, population, **settings)
        best = int(np.argmax(np.asarray(out["log_evidence"])[0] + catalog[0]))
        anchor.append(np.asarray(out["source_mean"])[0, best])
        scale.append(np.sqrt(np.diag(np.asarray(out["source_covariance"])[0, best])))
        log_dz.append(np.log(z_grid[best] - z_lens))
        colors = np.where(np.isfinite(data.colors[rows[list(group)]]) & (np.diagonal(data.color_covariance[rows[list(group)]], axis1=1, axis2=2) < 1e3),
                          data.colors[rows[list(group)]], np.nan)
        color_center.append(np.nan_to_num(np.nanmean(colors, axis=0)) if colors.size else np.zeros(data.colors.shape[1]))
    d = data.colors.shape[1]
    n_slots = len(groups)
    weights = np.full(1 + len(state.images.family_ids) + n_slots, (1.0 - partition.unrelated_fraction) / max(len(state.images.family_ids) + n_slots, 1))
    weights[0] = partition.unrelated_fraction
    color_precision, color_logdet = [], []
    for s in (settings["color_scatter"], settings["outlier_scatter"]):
        cov = data.color_covariance[rows] + s ** 2 * np.eye(d)
        color_precision.append(np.linalg.inv(cov) if d else np.zeros((n, 0, 0)))
        color_logdet.append(np.linalg.slogdet(cov)[1] if d else np.zeros(n))
    model = SlotModel(images=images, labels=tuple(table["image_label"]), trusted_ids=tuple(state.images.family_ids), trusted_ratio=trusted_ratio,
                      trusted_catalog=trusted_catalog, singleton_catalog=singleton_catalog, log_weights=np.log(weights),
                      param_start=state.n_parameters, anchor=np.asarray(anchor).reshape(n_slots, 2), scale=np.asarray(scale).reshape(n_slots, 2),
                      log_dz_center=np.asarray(log_dz, dtype=float), color_center=np.asarray(color_center).reshape(n_slots, d), z_lens=z_lens,
                      z_table=z_table, factor_table=factor_table, z_mean=data.z_mean[rows], z_sigma=data.z_sigma[rows], colors=data.colors[rows],
                      color_precision=np.stack(color_precision), color_logdet=np.stack(color_logdet),
                      outlier_fraction=settings["outlier_fraction"], population=population)
    specs = list(state.parameter_specs)
    fields = ["u_x", "u_y", "v_log_dz"] + [f"w_{b}_{c}" for b, c in zip(bands[:-1], bands[1:])]
    for k in range(n_slots):
        for field_name in fields:
            specs.append(ParameterSpec(name=f"slot.{k}.{field_name}", sample_name=f"slot_{k}_{field_name}", potential_id=f"slot{k}", profile_type=0,
                                       field=field_name, prior=dist.Normal(0.0, 1.0), component_family="source_position",
                                       sample_site_name=SLOT_SITE, sample_site_index=len(specs) - state.n_parameters))
    initial = np.concatenate([np.asarray(state.initial_theta, dtype=float), np.zeros(model.n_parameters)])
    extended = replace(state, parameter_specs=specs, initial_theta=initial, slots=model)
    log(f"[slots] {n} uncertain images, {len(state.images.family_ids)} trusted slots, {n_slots} new slot(s) seeded from groups "
        f"{[tuple(model.labels[i] for i in g) for g in groups]}; {d} colors from bands {list(bands)}; color scatter {scatter:.3f} mag; "
        f"redshift grid {z_grid[0]:.2f}-{z_grid[-1]:.2f} ({z_grid.size} points)")
    return extended, Evaluator(extended, evaluator.likelihood_mode)


def efficiency_factor_table(state: ModelState, z: np.ndarray) -> np.ndarray:
    """The dPIE lensing-efficiency factor at many source redshifts in one vectorized evaluation."""
    from ..jax_cosmology import dpie_sigma0_factor, h0_from_config, om0_from_config, w0_from_config
    cosmo = state.cosmo_config
    return np.asarray(dpie_sigma0_factor(float(state.z_lens), jnp.asarray(z, dtype=jnp.float64), h0_from_config(cosmo), om0_from_config(cosmo),
                                         w0_from_config(cosmo)), dtype=float).reshape(np.shape(z))


def extend_theta(state: ModelState, theta: np.ndarray) -> np.ndarray:
    """A latent vector (or rows of them) of the state before ``add_slots`` padded with zero slot offsets."""
    theta = np.asarray(theta, dtype=float)
    missing = state.n_parameters - theta.shape[-1]
    return theta if missing <= 0 else np.concatenate([theta, np.zeros(theta.shape[:-1] + (missing,))], axis=-1)


def _population(partition, combined: pd.DataFrame, bands: tuple[str, ...], *, log) -> tuple[GaussianMixture, tuple[str, ...]]:
    """The population mixture over (z, colors), and the bands it uses (none when no colors can be fit)."""
    if partition.population_catalog_path is not None:
        source = pd.read_csv(Path(partition.population_catalog_path))
    else:
        source = combined
        log("[slots] no population catalog: the population mixture is fit on the trusted and uncertain images")
    if len(bands) >= 2:
        try:
            return fit_population(source, bands, n_components=int(partition.population_components)), bands
        except ValueError:
            log(f"[slots] no complete colors in bands {list(bands)}: colors are not used")
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
    log(f"[slots] same-source color scatter {scatter:.3f} mag fit on {len(families)} trusted families")
    return scatter


SLOTS_FILE = "slots.npz"


def write_slot_outputs(state: ModelState, evaluator, posterior, stage, *, draws: int, seed: int, log=print) -> dict[str, Any]:
    """Assignment probabilities and joint draws of the uncertain images from the posterior.

    Writes ``<stage>/artifacts/slots.npz`` (``options``, ``labels`` of the images, per-draw ``responsibilities`` and
    sampled ``assignment``, ``probability``, ``pair_probability``, new-slot ``slot_z`` and ``slot_beta``),
    ``<stage>/tables/uncertain_images.csv`` (per image: the probability of every option and the most probable one) and
    ``<stage>/tables/uncertain_pairs.csv`` (image pairs sharing a source with probability above 0.01).
    """
    model: SlotModel = state.slots
    samples = np.asarray(posterior.samples, dtype=float)
    samples = samples[np.isfinite(samples).all(axis=1)]
    chosen = samples[np.unique(np.linspace(0, len(samples) - 1, min(int(draws), len(samples))).astype(int))]
    result = model.posterior(evaluator, chosen, np.random.default_rng(int(seed)))
    options = model.option_names()
    np.savez(stage.artifacts / SLOTS_FILE, options=np.asarray(options), image_labels=np.asarray(model.labels), assignment=result["labels"],
             responsibilities=result["responsibilities"], probability=result["probability"], pair_probability=result["pair_probability"],
             slot_z=result["slot_z"], slot_beta=result["slot_beta"])
    table = pd.DataFrame(result["probability"], columns=[f"p_{o}" for o in options])
    best = np.argmax(result["probability"], axis=1)
    table.insert(0, "image_label", list(model.labels))
    table.insert(1, "most_probable", [options[k] for k in best])
    table.insert(2, "p_most_probable", result["probability"][np.arange(len(best)), best])
    table.to_csv(stage.tables / "uncertain_images.csv", index=False)
    i, j = np.triu_indices(len(model.labels), k=1)
    p = result["pair_probability"][i, j]
    keep = p > 0.01
    pairs = pd.DataFrame({"image_a": np.asarray(model.labels)[i[keep]], "image_b": np.asarray(model.labels)[j[keep]], "p_same_source": p[keep]})
    pairs.sort_values("p_same_source", ascending=False).to_csv(stage.tables / "uncertain_pairs.csv", index=False)
    for label, option, probability in zip(model.labels, table["most_probable"], table["p_most_probable"]):
        log(f"[slots] {label}: {option} (p = {probability:.2f})")
    return result
