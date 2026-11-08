"""Lens-guided candidate partitions from ray-shot source-plane positions.

Every image is ray-shot once per point of a redshift grid under one lens model (a ``RayShooter``).  Two images are
lens-affine when their source positions agree at redshifts both catalogs allow; families are proposed by clustering
the source positions at each grid redshift and beam-assembling disjoint clusters into full partitions.  Like the
catalog pair model these are proposal heuristics: the partition probabilities come from the family evidence.  Calling
them under several posterior lens draws covers the lens uncertainty in the candidate set.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .candidates import canonicalize_partition, dedupe_partitions, image_positions_arcsec, image_redshifts, image_reliability, logit

RayShooter = Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]   # (x, y, z_source) -> (beta_x, beta_y)


@dataclass(frozen=True)
class SourcePlaneCache:
    """Ray-shot source positions of every image on a redshift grid under one lens model.

    Attributes
    ----------
    z_grid : numpy.ndarray
        Source redshifts, shape (grid,).
    beta_x, beta_y : numpy.ndarray
        Source-plane positions in arcsec, shape (grid, image).
    image_sigma_arcsec : numpy.ndarray
        Astrometric uncertainty per image, shape (image,).
    reliability : numpy.ndarray
        Catalog family reliability per image in [0, 1], shape (image,).
    image_redshift_mean, image_redshift_sigma : numpy.ndarray
        Catalog redshift and its uncertainty per image (NaN mean when unknown), shape (image,).
    """
    z_grid: np.ndarray
    beta_x: np.ndarray
    beta_y: np.ndarray
    image_sigma_arcsec: np.ndarray
    reliability: np.ndarray
    image_redshift_mean: np.ndarray
    image_redshift_sigma: np.ndarray

    @property
    def n_images(self) -> int:
        return int(self.beta_x.shape[1])


def image_sigma_arcsec(images: pd.DataFrame, *, default: float) -> np.ndarray:
    """Astrometric uncertainty per image from ``sigma_arcsec``, ``pos_sigma_arcsec`` or ``image_sigma_arcsec``."""
    for column in ("sigma_arcsec", "pos_sigma_arcsec", "image_sigma_arcsec"):
        if column in images.columns:
            values = pd.to_numeric(images[column], errors="coerce").to_numpy(dtype=float)
            return np.where(np.isfinite(values) & (values > 0.0), values, float(default))
    return np.full(len(images), float(default))


def redshift_grid_from_images(images: pd.DataFrame, *, n_grid: int = 8, z_min: float | None = None, z_max: float | None = None,
                              sigma_padding: float = 2.0) -> np.ndarray:
    """Uniform redshift grid spanning every catalog redshift +- ``sigma_padding`` sigma (1 to 6 with none known)."""
    mean, sigma = image_redshifts(images.reset_index(drop=True))
    finite = np.isfinite(mean) & (mean > 0.0)
    low = float(z_min) if z_min is not None else (max(0.05, float(np.min(mean[finite] - sigma_padding * sigma[finite]))) if finite.any() else 1.0)
    high = float(z_max) if z_max is not None else (float(np.max(mean[finite] + sigma_padding * sigma[finite])) if finite.any() else 6.0)
    return np.linspace(low, max(low + 0.05, high), max(2, int(n_grid)))


def precompute_source_plane_cache(images: pd.DataFrame, ray_shooter: RayShooter, *, z_grid=None, n_redshift_grid: int = 8,
                                  position_sigma_arcsec: float = 0.15) -> SourcePlaneCache:
    """Ray-shoot every image once per grid redshift."""
    work = images.reset_index(drop=True)
    x, y = image_positions_arcsec(work)
    grid = redshift_grid_from_images(work, n_grid=n_redshift_grid) if z_grid is None else np.asarray(z_grid, dtype=float)
    rays = [ray_shooter(x, y, float(z)) for z in grid]
    mean, sigma = image_redshifts(work)
    return SourcePlaneCache(z_grid=grid, beta_x=np.asarray([r[0] for r in rays], dtype=float).reshape(grid.size, len(work)),
                            beta_y=np.asarray([r[1] for r in rays], dtype=float).reshape(grid.size, len(work)),
                            image_sigma_arcsec=image_sigma_arcsec(work, default=position_sigma_arcsec),
                            reliability=image_reliability(work), image_redshift_mean=mean, image_redshift_sigma=sigma)


def redshift_weights(cache: SourcePlaneCache, members) -> np.ndarray:
    """Normalized weights on the grid from the product of the members' Gaussian catalog redshifts (uniform if none)."""
    members = np.asarray(members, dtype=int)
    mean, sigma = cache.image_redshift_mean[members], cache.image_redshift_sigma[members]
    known = np.isfinite(mean) & (mean > 0.0) & np.isfinite(sigma) & (sigma > 0.0)
    if not known.any():
        return np.full(cache.z_grid.size, 1.0 / cache.z_grid.size)
    log_weights = -0.5 * np.sum(np.square((cache.z_grid[:, None] - mean[known]) / sigma[known]), axis=1)
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


def lens_pair_affinity(cache: SourcePlaneCache, *, source_scatter_arcsec: float = 0.05) -> np.ndarray:
    """Redshift-averaged Gaussian proximity of every pair's source positions, in [1e-6, 1 - 1e-6], shape (image, image)."""
    n = cache.n_images
    affinity = np.eye(n)
    for left in range(n):
        for right in range(left + 1, n):
            variance = max(cache.image_sigma_arcsec[left] ** 2 + cache.image_sigma_arcsec[right] ** 2 + source_scatter_arcsec ** 2, 1.0e-8)
            distance2 = np.square(cache.beta_x[:, left] - cache.beta_x[:, right]) + np.square(cache.beta_y[:, left] - cache.beta_y[:, right])
            affinity[left, right] = affinity[right, left] = float(np.sum(redshift_weights(cache, [left, right]) * np.exp(-0.5 * distance2 / variance)))
    return np.clip(np.nan_to_num(affinity), 1.0e-6, 1.0 - 1.0e-6)


def combine_affinities(catalog_probability, lens_affinity, *, lens_weight: float = 1.0) -> np.ndarray:
    """Catalog and lens affinities added in logit space, shape (image, image)."""
    catalog, lens = np.asarray(catalog_probability, dtype=float), np.asarray(lens_affinity, dtype=float)
    if catalog.shape != lens.shape:
        raise ValueError("catalog_probability and lens_affinity must have matching shape.")
    combined = 1.0 / (1.0 + np.exp(-(logit(catalog) + float(lens_weight) * logit(lens))))
    np.fill_diagonal(combined, 1.0)
    return np.clip(combined, 1.0e-6, 1.0 - 1.0e-6)


def _cluster_source_loglike(beta_x, beta_y, sigma, reliability, source_scatter_arcsec: float) -> float:
    """Spread of ray-shot positions about their precision-weighted centroid, with a broad outlier term per image."""
    variance = np.square(sigma) + source_scatter_arcsec ** 2
    if not (np.isfinite(beta_x).all() and np.isfinite(beta_y).all()):
        return -np.inf
    precision = 1.0 / variance
    residual2 = np.square(beta_x - np.average(beta_x, weights=precision)) + np.square(beta_y - np.average(beta_y, weights=precision))
    outlier_variance = max(1.0, 10.0 * source_scatter_arcsec) ** 2
    rel = np.clip(reliability, 1.0e-6, 1.0 - 1.0e-6)
    return float(np.sum(np.logaddexp(np.log(rel) - 0.5 * residual2 * precision, np.log1p(-rel) - 0.5 * residual2 / outlier_variance)))


def lens_source_cluster_partitions(cache: SourcePlaneCache, *, catalog_probability=None,
                                   beta_radius_grid: tuple[float, ...] = (0.03, 0.05, 0.08, 0.12, 0.18), min_cluster_size: int = 2,
                                   min_redshift_weight: float = 0.02, source_scatter_arcsec: float = 0.05, catalog_weight: float = 1.0,
                                   beam_width: int = 50, max_cluster_hypotheses: int = 400) -> np.ndarray:
    """Full partitions assembled from source-plane clusters, shape (partition, image).

    At each grid redshift and linking radius, images whose catalog redshift allows that redshift are linked when their
    source positions lie within the radius; connected components of at least ``min_cluster_size`` images are cluster
    hypotheses, ranked by source-plane spread (redshift-marginalized), summed catalog pair logits and log size.  A beam
    search assembles disjoint high-ranked clusters into partitions; images in no chosen cluster stay singletons.
    """
    n = cache.n_images
    if n == 0:
        return np.empty((0, 0), dtype=int)
    catalog_logit = None if catalog_probability is None else logit(np.asarray(catalog_probability, dtype=float))
    if catalog_logit is not None and catalog_logit.shape != (n, n):
        raise ValueError("catalog_probability must have shape (n_images, n_images).")
    image_weights = np.vstack([redshift_weights(cache, [i]) for i in range(n)])

    def score(members: tuple[int, ...]) -> float:
        index = np.asarray(members)
        weights = redshift_weights(cache, index)
        terms = [np.log(w) + _cluster_source_loglike(cache.beta_x[g, index], cache.beta_y[g, index], cache.image_sigma_arcsec[index],
                                                     cache.reliability[index], source_scatter_arcsec)
                 for g, w in enumerate(weights) if w > 0.0]
        lens = float(np.logaddexp.reduce(terms)) if terms else -np.inf
        catalog = float(np.sum(np.triu(catalog_logit[np.ix_(index, index)], k=1))) if catalog_logit is not None else 0.0
        return lens + float(catalog_weight) * catalog + float(np.log(len(members)))

    pool: dict[tuple[int, ...], float] = {}
    for radius in beta_radius_grid:
        for g in range(cache.z_grid.size):
            eligible = image_weights[:, g] >= float(min_redshift_weight)
            if eligible.sum() < int(min_cluster_size):
                continue
            coords = np.column_stack([cache.beta_x[g], cache.beta_y[g]])
            adjacency = (np.linalg.norm(coords[:, None] - coords[None], axis=-1) <= float(radius)) & eligible[:, None] & eligible[None]
            np.fill_diagonal(adjacency, True)
            labels = _components(adjacency)
            for label in np.unique(labels):
                members = tuple(np.flatnonzero(labels == label).tolist())
                if len(members) >= int(min_cluster_size) and members not in pool:
                    value = score(members)
                    if np.isfinite(value):
                        pool[members] = value

    ranked = sorted(pool.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))[:max(1, int(max_cluster_hypotheses))]
    beam: list[tuple[float, int, tuple[tuple[int, ...], ...]]] = [(0.0, 0, ())]
    for members, value in ranked:
        mask = sum(1 << i for i in members)
        additions = [(s + value, used | mask, chosen + (members,)) for s, used, chosen in beam if not used & mask]
        best: dict[int, tuple[float, int, tuple[tuple[int, ...], ...]]] = {}
        for state in beam + additions:
            if state[1] not in best or state[0] > best[state[1]][0]:
                best[state[1]] = state
        beam = sorted(best.values(), key=lambda s: (-s[0], -int.bit_count(s[1]), s[2]))[:max(1, int(beam_width))]

    partitions = []
    for _, _, chosen in beam:
        if chosen:
            assignment = np.arange(n)
            for label, members in enumerate(chosen):
                assignment[list(members)] = n + label
            partitions.append(canonicalize_partition(assignment))
    return dedupe_partitions(partitions, n_images=n) if partitions else np.empty((0, n), dtype=int)


def propose_partitions(images: pd.DataFrame, *, ray_shooters: tuple[RayShooter, ...] = (), z_grid=None, n_redshift_grid: int = 8,
                       position_sigma_arcsec: float = 0.15, source_scatter_arcsec: float = 0.05, lens_weight: float = 1.0,
                       n_thresholds: int = 25, n_noisy_partitions: int = 100, cluster_kwargs: dict | None = None,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Candidate partitions from the catalog pair model and, per lens model in ``ray_shooters``, from the lens affinity, the
    combined affinity and source-plane clusters; canonical and unique, shape (partition, image).

    Pass ray shooters built from a handful of posterior lens draws so the candidate set covers the lens uncertainty.
    """
    from .candidates import build_pair_table, catalog_pair_probability, generate_candidate_partitions
    rng = np.random.default_rng() if rng is None else rng
    work = images.reset_index(drop=True)
    n = len(work)
    pairs = build_pair_table(work)
    catalog = catalog_pair_probability(pairs, n)

    def generate(matrix):
        return generate_candidate_partitions(pairs, matrix, n_images=n, n_thresholds=n_thresholds,
                                             n_noisy_partitions=n_noisy_partitions, rng=rng)

    found = [generate(catalog)]
    for shooter in ray_shooters:
        cache = precompute_source_plane_cache(work, shooter, z_grid=z_grid, n_redshift_grid=n_redshift_grid,
                                              position_sigma_arcsec=position_sigma_arcsec)
        lens = lens_pair_affinity(cache, source_scatter_arcsec=source_scatter_arcsec)
        found += [generate(lens), generate(combine_affinities(catalog, lens, lens_weight=lens_weight)),
                  lens_source_cluster_partitions(cache, catalog_probability=catalog, source_scatter_arcsec=source_scatter_arcsec,
                                                 **(cluster_kwargs or {}))]
    return dedupe_partitions(np.vstack(found), n_images=n)


def _components(adjacency: np.ndarray) -> np.ndarray:
    """Connected-component label per node of a boolean adjacency matrix."""
    n = adjacency.shape[0]
    labels = np.full(n, -1, dtype=int)
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack, labels[start] = [start], start
        while stack:
            neighbors = np.flatnonzero(adjacency[stack.pop()] & (labels < 0))
            labels[neighbors] = start
            stack.extend(neighbors.tolist())
    return labels
