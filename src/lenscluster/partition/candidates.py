"""Candidate partitions of lensed images into families.

The partition likelihood sums the family evidence over a fixed candidate set C of partitions, so C only has to contain
every partition with appreciable posterior weight: how a candidate was proposed never enters its probability.  Proposals
come from a symmetric pair affinity matrix P (catalog similarity, source-plane proximity, or both) through

* connected components, complete-link and average-link clustering of P over a grid of thresholds;
* anchor expansion: known families seeded first, then compatible images attached by average affinity;
* greedy pivot correlation clustering on logit(P) over several pivot orders;
* the same on noisy affinities drawn from Beta(20 P, 20 (1 - P));
* one-step split, merge and single-image moves of every proposal.

``repair_candidate_partitions`` grows C around its best members under any partition scorer.  The catalog pair model
(``catalog_pair_probability``) is a logistic model of pair features; it is only a proposal heuristic and never a
likelihood term, because pair terms count the evidence of one family many times.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LOGIT_EPS = 1.0e-6
REDSHIFT_COLUMNS = ("zspec_best", "catalog_z", "zphot_best", "image_zphot_family", "z_source")
DEFAULT_REDSHIFT_SIGMA = 0.5
PartitionScorer = Callable[[np.ndarray], np.ndarray]     # (partition, image) int -> (partition,) log score


@dataclass(frozen=True)
class LogisticFit:
    """MAP logistic pair model: coefficients in ``feature_names`` order."""
    coefficients: np.ndarray
    feature_names: list[str]
    success: bool
    objective: float


# ------------------------------------------------------------------ partitions
def canonicalize_partition(assignment) -> np.ndarray:
    """Relabel family ids by first occurrence, so equal partitions have equal label vectors."""
    values = np.asarray(assignment, dtype=int).reshape(-1)
    _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=int)
    rank[np.argsort(first)] = np.arange(first.size)
    return rank[inverse.reshape(-1)]


def same_family_matrix(assignment) -> np.ndarray:
    """(image, image) bool matrix, True where two images share a family."""
    canonical = canonicalize_partition(assignment)
    return canonical[:, None] == canonical[None, :]


def dedupe_partitions(partitions, *, n_images: int) -> np.ndarray:
    """Canonical, unique partitions of ``n_images`` images in first-seen order, shape (partition, image)."""
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for row in np.asarray(partitions, dtype=int).reshape(-1, n_images) if n_images else []:
        canonical = canonicalize_partition(row)
        unique.setdefault(tuple(canonical.tolist()), canonical)
    return np.asarray(list(unique.values()), dtype=int).reshape(-1, n_images)


def weighted_pair_probability(partitions, weights) -> np.ndarray:
    """P(z_i = z_j) = sum_m w_m 1[z_i^m = z_j^m] over weighted partitions, shape (image, image)."""
    partition_array = np.atleast_2d(np.asarray(partitions, dtype=int))
    weight_array = np.asarray(weights, dtype=float).reshape(-1)
    if partition_array.shape[0] != weight_array.size:
        raise ValueError("Number of partitions must match number of weights.")
    if weight_array.size == 0:
        return np.empty((0, 0), dtype=float)
    total = float(np.sum(weight_array))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Partition weights must have a positive finite sum.")
    same = partition_array[:, :, None] == partition_array[:, None, :]
    return np.einsum("m,mij->ij", weight_array / total, same.astype(float))


def local_partition_perturbations(assignment) -> list[np.ndarray]:
    """One-step neighbors of a partition: split one image out, merge two families, move one image to another family."""
    base = canonicalize_partition(assignment)
    if base.size <= 1:
        return []
    labels = np.unique(base)
    fresh = int(labels[-1]) + 1
    proposals: list[np.ndarray] = []
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            proposal = base.copy()
            proposal[proposal == right] = left
            proposals.append(proposal)
    for image in range(base.size):
        for target in labels:
            if target != base[image]:
                proposal = base.copy()
                proposal[image] = target
                proposals.append(proposal)
        if np.sum(base == base[image]) > 1:
            proposal = base.copy()
            proposal[image] = fresh
            proposals.append(proposal)
    return [canonicalize_partition(p) for p in proposals]


def repair_candidate_partitions(partitions, score: PartitionScorer, *, top_k: int = 200,
                                max_rounds: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Grow a candidate set with the one-step neighbors of its ``top_k`` best partitions under ``score``.

    Each round adds the neighbors, dedupes and rescores; it stops early when a round adds nothing.  Returns the
    candidates and their scores.
    """
    current = np.atleast_2d(np.asarray(partitions, dtype=int))
    n_images = current.shape[1]
    current = dedupe_partitions(current, n_images=n_images)
    scores = np.asarray(score(current), dtype=float).reshape(-1)
    for _ in range(int(max_rounds)):
        if current.shape[0] == 0 or int(top_k) <= 0:
            break
        order = np.argsort(-scores)[:int(top_k)]
        proposals = [p for index in order for p in local_partition_perturbations(current[index])]
        if not proposals:
            break
        grown = dedupe_partitions(np.vstack([current, proposals]), n_images=n_images)
        if grown.shape[0] == current.shape[0]:
            break
        current = grown
        scores = np.asarray(score(current), dtype=float).reshape(-1)
    return current, scores


# ------------------------------------------------------------------ catalog columns
def _numeric(images: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in images.columns:
        return np.full(len(images), default, dtype=float)
    return pd.to_numeric(images[column], errors="coerce").to_numpy(dtype=float)


def image_positions_arcsec(images: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Image positions in arcsec from ``x_obs``/``y_obs``, ``x_obs_arcsec``/``y_obs_arcsec``, ``x``/``y``, or
    ``ra``/``dec`` (tangent offsets from the median position, x positive east as in the PR benchmarks)."""
    for x_column, y_column in (("x_obs", "y_obs"), ("x_obs_arcsec", "y_obs_arcsec"), ("x", "y")):
        if {x_column, y_column}.issubset(images.columns):
            return _numeric(images, x_column), _numeric(images, y_column)
    if {"ra", "dec"}.issubset(images.columns):
        ra, dec = _numeric(images, "ra"), _numeric(images, "dec")
        ra0 = float(np.nanmedian(ra)) if np.isfinite(ra).any() else 0.0
        dec0 = float(np.nanmedian(dec)) if np.isfinite(dec).any() else 0.0
        return (ra - ra0) * math.cos(math.radians(dec0)) * 3600.0, (dec - dec0) * 3600.0
    return np.full(len(images), np.nan), np.full(len(images), np.nan)


def image_redshifts(images: pd.DataFrame, *, default_sigma: float = DEFAULT_REDSHIFT_SIGMA) -> tuple[np.ndarray, np.ndarray]:
    """Best redshift (first positive of ``REDSHIFT_COLUMNS``, NaN if none) and its 1-sigma uncertainty per image.

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


def image_reliability(images: pd.DataFrame) -> np.ndarray:
    """Catalog ``family_reliability`` per image, clipped to [0, 1]; 1 where missing."""
    values = _numeric(images, "family_reliability", 1.0)
    return np.clip(np.where(np.isfinite(values), values, 1.0), 0.0, 1.0)


def magnitude_columns(images: pd.DataFrame) -> list[str]:
    """Sorted ``mag_*`` columns."""
    return sorted(str(c) for c in images.columns if str(c).startswith("mag_"))


def centered_colors(images: pd.DataFrame) -> np.ndarray:
    """Magnitudes minus each image's mean over its finite bands (NaN rows with fewer than two), shape (image, band)."""
    columns = magnitude_columns(images)
    mags = np.column_stack([_numeric(images, c) for c in columns]) if columns else np.empty((len(images), 0))
    finite = np.isfinite(mags)
    enough = finite.sum(axis=1) >= 2
    mean = np.where(enough, np.nansum(np.where(finite, mags, 0.0), axis=1) / np.maximum(finite.sum(axis=1), 1), np.nan)
    return np.where(enough[:, None], mags - mean[:, None], np.nan)


# ------------------------------------------------------------------ catalog pair model
FEATURE_COLUMNS = ["intercept", "local_separation", "radius_similarity", "opposite_side_score", "z_similarity",
                   "color_similarity", "size_similarity", "ellipticity_similarity", "reliability_mean", "same_object_source"]
DEFAULT_COEFFICIENTS = {"intercept": -2.0, "local_separation": 0.2, "radius_similarity": 0.8, "opposite_side_score": 0.8,
                        "z_similarity": 2.0, "color_similarity": 1.5, "size_similarity": 0.4, "ellipticity_similarity": 0.3,
                        "reliability_mean": 0.6, "same_object_source": 0.1}


def _strings(images: pd.DataFrame, column: str) -> np.ndarray:
    if column not in images.columns:
        return np.full(len(images), "", dtype=object)
    return images[column].fillna("").astype(str).str.strip().to_numpy(dtype=object)


def build_pair_table(images: pd.DataFrame) -> pd.DataFrame:
    """Pair features for every unordered image pair (left < right), one row per pair."""
    work = images.reset_index(drop=True)
    n = len(work)
    left, right = np.triu_indices(n, k=1)
    x, y = image_positions_arcsec(work)
    z, z_sigma = image_redshifts(work)
    reliability = image_reliability(work)
    ids = next((work[c].fillna("").astype(str).to_numpy() for c in ("object_id", "image_label", "id") if c in work.columns),
               np.asarray([str(i) for i in range(n)]))

    radii, angles = np.hypot(x, y), np.arctan2(y, x)
    radius_scale = max(float(np.nanmedian(radii[np.isfinite(radii)])) if np.isfinite(radii).any() else 1.0, 1.0)
    radius_delta = np.abs(radii[left] - radii[right])
    angle_delta = np.abs(np.angle(np.exp(1j * (angles[left] - angles[right]))))
    z_delta = np.abs(z[left] - z[right])
    z_combined = np.hypot(z_sigma[left], z_sigma[right])

    colors = centered_colors(work)
    common = np.isfinite(colors[left]) & np.isfinite(colors[right])
    n_common = common.sum(axis=1)
    squared = np.where(common, np.square(colors[left] - colors[right]), 0.0)
    color_rms = np.where(n_common >= 2, np.sqrt(squared.sum(axis=1) / np.maximum(n_common, 1)), np.nan)

    def similarity(delta):
        return np.where(np.isfinite(delta), np.exp(-delta), 0.5)

    size, ellipticity = _numeric(work, "image_size_arcsec"), _numeric(work, "image_ellipticity")
    source, family = _strings(work, "object_source"), _strings(work, "family_id")
    return pd.DataFrame({
        "left_index": left, "right_index": right, "left_object_id": ids[left], "right_object_id": ids[right],
        "separation_arcsec": np.hypot(x[left] - x[right], y[left] - y[right]),
        "radius_similarity": np.where(np.isfinite(radius_delta), np.exp(-radius_delta / radius_scale), 0.5),
        "opposite_side_score": np.where(np.isfinite(angle_delta), 0.5 * (1.0 - np.cos(angle_delta)), 0.5),
        "z_delta": z_delta, "z_sigma_combined": z_combined,
        "z_similarity": np.where(np.isfinite(z_delta), np.exp(-0.5 * np.square(z_delta / np.maximum(z_combined, 1.0e-3))), 0.5),
        "color_rms": color_rms, "color_similarity": similarity(color_rms), "n_common_bands": n_common,
        "size_similarity": similarity(np.abs(size[left] - size[right])),
        "ellipticity_similarity": similarity(np.abs(ellipticity[left] - ellipticity[right])),
        "reliability_mean": 0.5 * (reliability[left] + reliability[right]),
        "same_object_source": ((source[left] == source[right]) & (source[left] != "")).astype(float),
        "same_input_family": (family[left] == family[right]) & (family[left] != ""),
    })


def feature_matrix(pair_table: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Logistic design matrix of the pair table, shape (pair, feature), and the feature names."""
    if pair_table.empty:
        return np.empty((0, len(FEATURE_COLUMNS))), list(FEATURE_COLUMNS)

    def column(name, fill):
        return pd.to_numeric(pair_table[name], errors="coerce").fillna(fill).to_numpy(dtype=float)

    separation = pd.to_numeric(pair_table["separation_arcsec"], errors="coerce").to_numpy(dtype=float)
    columns = {"intercept": np.ones(len(pair_table)),
               "local_separation": 1.0 / (1.0 + np.where(np.isfinite(separation), np.maximum(separation, 0.0), 100.0) / 10.0),
               "reliability_mean": column("reliability_mean", 1.0), "same_object_source": column("same_object_source", 0.0)}
    for name in ("radius_similarity", "opposite_side_score", "z_similarity", "color_similarity", "size_similarity",
                 "ellipticity_similarity"):
        columns[name] = column(name, 0.5)
    return np.column_stack([columns[name] for name in FEATURE_COLUMNS]), list(FEATURE_COLUMNS)


def create_anchor_labels(pair_table: pd.DataFrame, *, incompatible_z_delta: float = 0.8, incompatible_z_sigma: float = 3.0,
                         incompatible_color_rms: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """High-confidence pair labels (1 same family, 0 different, NaN unknown) and their weights.

    Pairs in one input family are labeled 1 (weight 8); pairs with incompatible redshifts (|dz| and dz/sigma both above
    the thresholds) 0 (weight 12); pairs with color rms above ``incompatible_color_rms`` over at least three common bands
    and no other label 0 (weight 5).
    """
    labels, weights = np.full(len(pair_table), np.nan), np.zeros(len(pair_table))
    if pair_table.empty:
        return labels, weights
    same = pair_table["same_input_family"].fillna(False).astype(bool).to_numpy() if "same_input_family" in pair_table else np.zeros(len(pair_table), bool)
    labels[same], weights[same] = 1.0, 8.0
    z_delta = pd.to_numeric(pair_table.get("z_delta"), errors="coerce").to_numpy(dtype=float)
    z_sigma = pd.to_numeric(pair_table.get("z_sigma_combined"), errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        z_bad = (z_delta >= float(incompatible_z_delta)) & (z_delta / np.maximum(z_sigma, 1.0e-3) >= float(incompatible_z_sigma))
    labels[z_bad], weights[z_bad] = 0.0, 12.0
    color_rms = pd.to_numeric(pair_table.get("color_rms"), errors="coerce").to_numpy(dtype=float)
    n_common = pd.to_numeric(pair_table.get("n_common_bands"), errors="coerce").fillna(0).to_numpy(dtype=float)
    color_bad = np.isfinite(color_rms) & (n_common >= 3) & (color_rms >= float(incompatible_color_rms)) & (weights <= 0.0)
    labels[color_bad], weights[color_bad] = 0.0, 5.0
    return labels, weights


def _sigmoid(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(values >= 0.0, 1.0 / (1.0 + np.exp(-np.abs(values))), np.exp(-np.abs(values)) / (1.0 + np.exp(-np.abs(values))))


def logit(probability) -> np.ndarray:
    """log(p / (1 - p)) with p clipped to [LOGIT_EPS, 1 - LOGIT_EPS]."""
    clipped = np.clip(np.asarray(probability, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(clipped) - np.log1p(-clipped)


def fit_logistic_map(x, y, *, sample_weight=None, gaussian_prior_sigma: float = 2.0, initial_coefficients=None,
                     feature_names: list[str] | None = None) -> LogisticFit:
    """MAP logistic regression with soft labels ``y`` in [0, 1] and an isotropic Gaussian prior on the coefficients."""
    x, y = np.asarray(x, dtype=float), np.clip(np.asarray(y, dtype=float).reshape(-1), 0.0, 1.0)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ValueError("x must be a (pair, feature) matrix with one row per label.")
    names = feature_names or [f"x{i}" for i in range(x.shape[1])]
    if x.shape[0] == 0:
        return LogisticFit(np.zeros(x.shape[1]), names, True, 0.0)
    weights = np.ones(y.size) if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    sigma2 = max(float(gaussian_prior_sigma), 1.0e-6) ** 2
    initial = np.zeros(x.shape[1]) if initial_coefficients is None else np.asarray(initial_coefficients, dtype=float)

    def objective(beta):
        probability = _sigmoid(x @ beta)
        clipped = np.clip(probability, LOGIT_EPS, 1.0 - LOGIT_EPS)
        loss = -np.sum(weights * (y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))) + 0.5 * np.sum(beta ** 2) / sigma2
        return float(loss), x.T @ (weights * (probability - y)) + beta / sigma2

    result = minimize(objective, initial if initial.size == x.shape[1] else np.zeros(x.shape[1]), jac=True, method="L-BFGS-B")
    return LogisticFit(np.asarray(result.x, dtype=float), names, bool(result.success), float(result.fun))


def pair_matrix(pair_table: pd.DataFrame, values, n_images: int) -> np.ndarray:
    """Symmetric (image, image) matrix from one value per pair-table row, ones on the diagonal."""
    matrix = np.eye(n_images)
    left, right = pair_table["left_index"].to_numpy(dtype=int), pair_table["right_index"].to_numpy(dtype=int)
    matrix[left, right] = matrix[right, left] = np.asarray(values, dtype=float)
    return matrix


def catalog_pair_probability(pair_table: pd.DataFrame, n_images: int, coefficients=None) -> np.ndarray:
    """Catalog pair affinity: the logistic pair model with anchored pairs forced to 0 or 1, shape (image, image)."""
    x, names = feature_matrix(pair_table)
    beta = np.asarray([DEFAULT_COEFFICIENTS[n] for n in names]) if coefficients is None else np.asarray(coefficients, dtype=float)
    values = _sigmoid(x @ beta) if len(pair_table) else np.empty(0)
    labels, weights = create_anchor_labels(pair_table)
    anchored = np.isfinite(labels) & (weights > 0.0)
    values[anchored] = np.where(labels[anchored] >= 0.5, 1.0 - LOGIT_EPS, LOGIT_EPS)
    return pair_matrix(pair_table, values, n_images)


# ------------------------------------------------------------------ proposals from an affinity matrix
def _union_find_labels(n: int, edges) -> np.ndarray:
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return canonicalize_partition([find(i) for i in range(n)])


def _labels_from_clusters(clusters: list[list[int]], n: int) -> np.ndarray:
    assignment = np.empty(n, dtype=int)
    for label, members in enumerate(clusters):
        assignment[members] = label
    return canonicalize_partition(assignment)


def _assignment_from_threshold(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the graph with edges where P >= threshold."""
    left, right = np.nonzero(np.triu(probability >= float(threshold), k=1))
    return _union_find_labels(probability.shape[0], zip(left.tolist(), right.tolist()))


def _assignment_from_complete_link_threshold(probability: np.ndarray, threshold: float) -> np.ndarray:
    """Sequential assignment: each image joins the family whose every member it links to at P >= threshold."""
    families: list[list[int]] = []
    for image in range(probability.shape[0]):
        best, best_score = -1, -np.inf
        for index, members in enumerate(families):
            values = probability[image, members]
            if values.min() >= float(threshold) and values.mean() > best_score:
                best, best_score = index, float(values.mean())
        if best >= 0:
            families[best].append(image)
        else:
            families.append([image])
    return _labels_from_clusters(families, probability.shape[0])


def _cluster_link_score(probability: np.ndarray, left: list[int], right: list[int], linkage: str) -> float:
    block = probability[np.ix_(left, right)]
    if linkage == "complete":
        return float(block.min())
    if linkage == "average":
        return float(block.mean())
    raise ValueError(f"Unknown linkage: {linkage}")


def _agglomerate(probability: np.ndarray, clusters: list[list[int]], threshold: float, linkage: str,
                 forbidden: Callable[[list[int], list[int]], bool] | None = None) -> np.ndarray:
    """Merge the best-linked pair of clusters while its link score is at least ``threshold``."""
    clusters = [list(c) for c in clusters]
    while len(clusters) > 1:
        best, best_score = None, -np.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if forbidden is not None and forbidden(clusters[i], clusters[j]):
                    continue
                score = _cluster_link_score(probability, clusters[i], clusters[j], linkage)
                if score > best_score:
                    best, best_score = (i, j), score
        if best is None or best_score < float(threshold):
            break
        i, j = best
        clusters[i] += clusters.pop(j)
    return _labels_from_clusters(clusters, probability.shape[0])


def _assignment_from_agglomerative_threshold(probability: np.ndarray, threshold: float, *, linkage: str) -> np.ndarray:
    return _agglomerate(probability, [[i] for i in range(probability.shape[0])], threshold, linkage)


def _anchor_structure(pair_table: pd.DataFrame, n_images: int) -> tuple[list[list[int]], set[tuple[int, int]]]:
    """Known families (unions of same-input-family pairs, size > 1) and the set of incompatible (left, right) pairs."""
    if pair_table.empty:
        return [], set()
    labels, weights = create_anchor_labels(pair_table)
    left, right = pair_table["left_index"].to_numpy(dtype=int), pair_table["right_index"].to_numpy(dtype=int)
    same = pair_table["same_input_family"].fillna(False).astype(bool).to_numpy() if "same_input_family" in pair_table else np.zeros(len(left), bool)
    assignment = _union_find_labels(n_images, zip(left[same].tolist(), right[same].tolist()))
    groups = [np.flatnonzero(assignment == label).tolist() for label in np.unique(assignment)]
    bad = np.isfinite(labels) & (labels < 0.5) & (weights > 0.0)
    return [g for g in groups if len(g) > 1], set(zip(left[bad].tolist(), right[bad].tolist()))


def _assignment_from_anchor_expansion(probability: np.ndarray, anchors: tuple[list[list[int]], set[tuple[int, int]]], *,
                                      threshold: float) -> np.ndarray:
    """Seed the known families, then average-link merge clusters with no incompatible cross pair."""
    groups, incompatible = anchors
    used = {i for g in groups for i in g}
    clusters = [list(g) for g in groups] + [[i] for i in range(probability.shape[0]) if i not in used]

    def forbidden(a, b):
        return any((min(i, j), max(i, j)) in incompatible for i in a for j in b)

    return _agglomerate(probability, clusters, threshold, "average", forbidden)


def _assignment_from_pivot_correlation(probability: np.ndarray, *, threshold: float = 0.0, order=None) -> np.ndarray:
    """Greedy pivot correlation clustering: each unassigned pivot takes every unassigned image with logit(P) >= threshold."""
    weights = logit(probability)
    n = probability.shape[0]
    unassigned = np.ones(n, dtype=bool)
    assignment = np.full(n, -1, dtype=int)
    label = 0
    for pivot in (np.arange(n) if order is None else np.asarray(order, dtype=int)):
        if 0 <= pivot < n and unassigned[pivot]:
            members = unassigned & (weights[pivot] >= float(threshold))
            members[pivot] = True
            assignment[members], unassigned[members] = label, False
            label += 1
    remaining = np.flatnonzero(unassigned)
    assignment[remaining] = label + np.arange(remaining.size)
    return canonicalize_partition(assignment)


def _pivot_orders(probability: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    n = probability.shape[0]
    if n == 0:
        return [np.empty(0, dtype=int)]
    offdiag = np.where(np.eye(n, dtype=bool), np.nan, probability)
    mean = np.nan_to_num(np.nanmean(offdiag, axis=1) if n > 1 else np.zeros(n))
    orders = [np.arange(n), np.argsort(-mean), np.argsort(mean)]
    orders += [rng.permutation(n) for _ in range(2 if n >= 15 else min(4, max(1, n)))]
    return orders


def generate_candidate_partitions(pair_table: pd.DataFrame, pair_probability, *, n_images: int, n_thresholds: int = 25,
                                  n_noisy_partitions: int = 100, rng: np.random.Generator | None = None,
                                  perturb: bool = True) -> np.ndarray:
    """Candidate partitions from one affinity matrix, canonical and unique, shape (partition, image).

    Always includes all-singletons, all-in-one, and the input families (from ``same_input_family``).  ``perturb`` adds
    the one-step neighbors of every proposal.
    """
    rng = np.random.default_rng() if rng is None else rng
    probability = np.asarray(pair_probability, dtype=float)
    if probability.shape != (n_images, n_images):
        raise ValueError("pair_probability must have shape (n_images, n_images).")
    anchors = _anchor_structure(pair_table, n_images)
    proposals = [np.arange(n_images), np.zeros(n_images, dtype=int)]
    groups, _ = anchors
    proposals.append(_labels_from_clusters(groups + [[i] for i in range(n_images) if all(i not in g for g in groups)], n_images))

    def from_matrix(matrix, thresholds, orders, logit_thresholds):
        for threshold in thresholds:
            proposals.append(_assignment_from_threshold(matrix, threshold))
            proposals.append(_assignment_from_agglomerative_threshold(matrix, threshold, linkage="complete"))
            proposals.append(_assignment_from_agglomerative_threshold(matrix, threshold, linkage="average"))
            proposals.append(_assignment_from_anchor_expansion(matrix, anchors, threshold=threshold))
        for threshold in logit_thresholds:
            for order in orders:
                proposals.append(_assignment_from_pivot_correlation(matrix, threshold=threshold, order=order))

    thresholds = np.linspace(0.05, 0.95, max(2, int(n_thresholds)))
    from_matrix(probability, thresholds, _pivot_orders(probability, rng),
                np.unique(np.concatenate([logit(thresholds), [-2.0, -1.0, 0.0, 1.0, 2.0]])))
    proposals += [_assignment_from_complete_link_threshold(probability, t) for t in thresholds]
    for _ in range(max(0, int(n_noisy_partitions))):
        noisy = rng.beta(np.maximum(probability * 20.0, 1.0e-3), np.maximum((1.0 - probability) * 20.0, 1.0e-3))
        noisy = np.triu(noisy, 1) + np.triu(noisy, 1).T + np.eye(n_images)
        threshold = float(rng.uniform(0.25, 0.75))
        from_matrix(noisy, [threshold], _pivot_orders(noisy, rng)[:1 if n_images >= 15 else 3], [float(logit(threshold))])
    base = dedupe_partitions(proposals, n_images=n_images)
    neighbors = [p for a in base for p in local_partition_perturbations(a)] if perturb else []
    return dedupe_partitions(np.vstack([base, np.asarray(neighbors, dtype=int).reshape(-1, n_images)]), n_images=n_images)
