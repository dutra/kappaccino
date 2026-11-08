from __future__ import annotations

import numpy as np
import pandas as pd

from lenscluster.partition.candidates import (build_pair_table, canonicalize_partition, catalog_pair_probability, create_anchor_labels,
                                              dedupe_partitions, feature_matrix, fit_logistic_map, generate_candidate_partitions,
                                              local_partition_perturbations, repair_candidate_partitions, same_family_matrix,
                                              weighted_pair_probability)
from lenscluster.partition.source_plane_proposals import (lens_pair_affinity, lens_source_cluster_partitions, precompute_source_plane_cache,
                                                          propose_partitions)

TRUTH = np.asarray([0, 0, 1, 1])


def _catalog() -> pd.DataFrame:
    """Two two-image families, A at z = 2 and B at z = 3.2, with secure redshifts and distinct colors."""
    rows = [("a1", "A", 10.00000, 2.00, (24.0, 23.7, 23.2), 0.25, 0.20, 0.95),
            ("a2", "A", 10.00025, 2.01, (25.0, 24.7, 24.2), 0.28, 0.22, 0.90),
            ("b1", "B", 10.01000, 3.20, (23.2, 23.1, 23.0), 0.40, 0.50, 0.85),
            ("b2", "B", 10.01025, 3.19, (24.2, 24.1, 24.0), 0.43, 0.48, 0.88)]
    return pd.DataFrame([{"object_id": oid, "family_id": fam, "ra": ra, "dec": 0.0, "zspec_best": z, "zspec_best_confidence": "secure",
                          "zspec_best_confidence_rank": 4, "mag_F606W": m[0], "mag_F814W": m[1], "mag_F160W": m[2],
                          "image_size_arcsec": size, "image_ellipticity": e, "family_reliability": rel, "object_source": "test"}
                         for oid, fam, ra, z, m, size, e, rel in rows])


def _two_source_images() -> pd.DataFrame:
    return pd.DataFrame({"object_id": ["a1", "a2", "b1", "b2"], "x_obs": [0.0, 0.02, 1.0, 1.02], "y_obs": [0.0] * 4,
                         "catalog_z": [2.0, 2.0, 3.0, 3.0], "catalog_z_sigma": [0.1] * 4})


def _ray_shooter(x, y, z_source):
    """Toy lens: the z = 3 images ray-shoot one arcsec closer to the origin, so both families map to beta ~ 0."""
    return np.asarray(x, dtype=float) - (0.0 if z_source < 2.5 else 1.0), np.asarray(y, dtype=float)


def _contains(partitions, assignment) -> bool:
    return any(np.array_equal(canonicalize_partition(row), assignment) for row in partitions)


def test_canonicalization_and_pair_probability_conservation() -> None:
    np.testing.assert_array_equal(canonicalize_partition([7, 7, 2, 9, 2]), [0, 0, 1, 2, 1])
    assert np.diag(same_family_matrix([3, 1, 3])).all()
    probability = weighted_pair_probability(np.asarray([[0, 0, 1], [0, 1, 1]]), [0.25, 0.75])
    np.testing.assert_allclose(np.diag(probability), 1.0)
    np.testing.assert_allclose([probability[0, 1], probability[1, 2], probability[0, 2]], [0.25, 0.75, 0.0])


def test_dedupe_and_perturbations_are_canonical_and_unique() -> None:
    np.testing.assert_array_equal(dedupe_partitions([[1, 1, 0], [0, 0, 1], [2, 0, 0]], n_images=3), [[0, 0, 1], [0, 1, 1]])
    neighbors = local_partition_perturbations([0, 0, 1])
    assert all(np.array_equal(p, canonicalize_partition(p)) for p in neighbors)
    assert _contains(neighbors, np.asarray([0, 0, 0])) and _contains(neighbors, np.asarray([0, 1, 2]))
    assert not _contains(neighbors, np.asarray([0, 0, 1]))


def test_logistic_fit_recovers_sensible_feature_signs() -> None:
    x = np.asarray([[1.0, 0.95, 0.90], [1.0, 0.90, 0.85], [1.0, 0.10, 0.20], [1.0, 0.20, 0.10]])
    fit = fit_logistic_map(x, [1.0, 1.0, 0.0, 0.0], gaussian_prior_sigma=5.0)
    assert fit.success and fit.coefficients[1] > 0.0 and fit.coefficients[2] > 0.0


def test_pair_table_features_and_anchors() -> None:
    pairs = build_pair_table(_catalog())
    assert len(pairs) == 6 and feature_matrix(pairs)[0].shape == (6, 10)
    by_pair = pairs.set_index(["left_index", "right_index"])
    assert by_pair.loc[(0, 1), "same_input_family"] and not by_pair.loc[(0, 2), "same_input_family"]
    np.testing.assert_allclose(by_pair.loc[(0, 1), "color_rms"], 0.0, atol=1e-12)    # same colors, 1 mag fainter
    labels, weights = create_anchor_labels(pairs, incompatible_z_delta=0.5)
    incompatible = pairs["z_delta"].to_numpy() > 0.5
    assert incompatible.any() and (labels[incompatible] == 0.0).all() and (weights[incompatible] >= 12.0).all()


def test_candidate_generation_from_catalog_contains_truth() -> None:
    pairs = build_pair_table(_catalog())
    partitions = generate_candidate_partitions(pairs, catalog_pair_probability(pairs, 4), n_images=4, n_thresholds=8,
                                               n_noisy_partitions=8, rng=np.random.default_rng(5))
    assert partitions.shape[1] == 4 and _contains(partitions, TRUTH)
    assert all(np.array_equal(row, canonicalize_partition(row)) for row in partitions)


def test_candidate_generation_includes_correlation_clustering_partition() -> None:
    matrix = np.full((4, 4), 0.05)
    np.fill_diagonal(matrix, 1.0)
    matrix[0, 1] = matrix[1, 0] = 0.9
    matrix[2, 3] = matrix[3, 2] = 0.85
    partitions = generate_candidate_partitions(build_pair_table(_catalog()), matrix, n_images=4, n_thresholds=4, n_noisy_partitions=0,
                                               rng=np.random.default_rng(11), perturb=False)
    assert _contains(partitions, TRUTH)


def test_repair_reattaches_singleton_under_any_scorer() -> None:
    def score(partitions):    # number of pairs that agree with the truth
        truth = same_family_matrix(TRUTH)
        return np.asarray([np.sum(same_family_matrix(p) == truth) for p in partitions], dtype=float)

    repaired, scores = repair_candidate_partitions(np.asarray([[0, 0, 1, 2]]), score, top_k=1, max_rounds=1)
    assert _contains(repaired, TRUTH)
    np.testing.assert_array_equal(repaired[np.argmax(scores)], TRUTH)


def test_lens_affinity_and_source_plane_clusters_find_families() -> None:
    cache = precompute_source_plane_cache(_two_source_images(), _ray_shooter, z_grid=np.asarray([2.0, 3.0]), position_sigma_arcsec=0.05)
    affinity = lens_pair_affinity(cache, source_scatter_arcsec=0.02)
    assert affinity[0, 1] > 0.9 and affinity[2, 3] > 0.9 and affinity[0, 2] < 0.1
    catalog = np.full((4, 4), 0.1)
    catalog[0, 1] = catalog[1, 0] = catalog[2, 3] = catalog[3, 2] = 0.9
    np.fill_diagonal(catalog, 1.0)
    clusters = lens_source_cluster_partitions(cache, catalog_probability=catalog, beta_radius_grid=(0.05,), min_redshift_weight=0.05,
                                              source_scatter_arcsec=0.02, beam_width=10)
    assert _contains(clusters, TRUTH)


def test_propose_partitions_with_lens_draws_contains_truth() -> None:
    partitions = propose_partitions(_two_source_images(), ray_shooters=(_ray_shooter,), z_grid=np.asarray([2.0, 3.0]),
                                    position_sigma_arcsec=0.05, source_scatter_arcsec=0.02, n_thresholds=5, n_noisy_partitions=3,
                                    rng=np.random.default_rng(0))
    assert _contains(partitions, TRUTH)
    assert len({tuple(row) for row in partitions.tolist()}) == len(partitions)


def test_missing_optional_columns_degrade_gracefully() -> None:
    images = pd.DataFrame({"object_id": ["x1", "x2", "x3"], "ra": [10.0, 10.001, 10.02], "dec": [0.0, 0.0, 0.0]})
    pairs = build_pair_table(images)
    probability = catalog_pair_probability(pairs, 3)
    assert np.isfinite(probability).all()
    partitions = propose_partitions(images, n_thresholds=5, n_noisy_partitions=5, rng=np.random.default_rng(7))
    assert partitions.shape[1] == 3 and _contains(partitions, np.asarray([0, 1, 2]))
