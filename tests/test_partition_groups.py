"""The uncertain-image likelihood summed over partitions (kappaccino.partition.groups) and its inputs."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from kappaccino import pipeline
from kappaccino.config import ImageDiagnosticsConfig, PartitionConfig, ReportingConfig, RuntimeConfig, SamplingConfig, SearchConfig
from kappaccino.likelihood.evaluator import Evaluator
from kappaccino.diagnostics.exact_images import ExactImageSolver
from kappaccino.partition import add_uncertain_images, write_partition_outputs
from kappaccino.partition import catalog_evidence as ce, groups, inputs, lens_evidence as le
from kappaccino.likelihood import smoothed as sl
from kappaccino.core.state import build_state
from kappaccino.io.hdf5 import load_arrays
from conftest import _brute_force_images as brute_force_images

SOURCES = [(2.0, (0.5, 0.3)), (3.0, (-1.0, 0.8)), (2.5, (2.0, -1.5))]       # (z, source position) of the new sources
COLORS = {0: (24.0, 23.5, 23.3), 1: (25.0, 24.2, 23.6), 2: (23.8, 23.7, 23.65), "trusted": (24.5, 24.0, 23.0)}
BANDS = ("F090W", "F150W", "F200W")


def _images_of(solver, evaluator, physical, z, source, *, min_magnification=0.3):
    packed = evaluator.packed_state_for_z(physical, z)
    x, y = brute_force_images(solver, packed, source, (0.0, 0.0), 40.0, pitch=0.1)
    mu = solver.magnification(packed, x, y)
    keep = np.abs(mu) > min_magnification
    return np.asarray(x)[keep], np.asarray(y)[keep], np.sign(mu[keep]).astype(int)


def _mock(config, root: Path, *, counterimage: bool = False):
    """Uncertain images of three new sources under the start lens (and, with ``counterimage``, one more image of
    trusted family 1 that its catalog omits), with photo-z and three-band photometry; returns the CSV path and truth."""
    state = build_state(config)
    evaluator = Evaluator(state)
    physical = evaluator.physical(jnp.asarray(state.initial_theta))
    solver = ExactImageSolver(evaluator, ImageDiagnosticsConfig())
    rng = np.random.default_rng(1)
    rows = []
    for family, (z, source) in enumerate(SOURCES):
        for x, y, parity in zip(*_images_of(solver, evaluator, physical, z, source)):
            rows.append(dict(truth=f"new{family}", colors=family, x=x, y=y, parity=parity, z=z))
    if counterimage:
        index = state.images.family_ids.index("1")
        source = np.asarray(evaluator.statistics(jnp.asarray(state.initial_theta))["source_by_family"])[index]
        x, y, parity = _images_of(solver, evaluator, physical, 2.0, tuple(source))
        observed = np.stack([state.images.x_obs, state.images.y_obs], -1)[np.asarray(state.images.family_index_per_image) == index]
        far = np.min(np.hypot(x[:, None] - observed[:, 0], y[:, None] - observed[:, 1]), axis=1) > 1.0
        k = int(np.flatnonzero(far)[0])
        rows.append(dict(truth="1", colors="trusted", x=x[k], y=y[k], parity=parity[k], z=2.0))
    table = pd.DataFrame(rows)
    _, ra0, dec0 = state.reference
    x = table["x"] + rng.normal(0.0, 0.03, len(table))
    y = table["y"] + rng.normal(0.0, 0.03, len(table))
    out = pd.DataFrame({"image_label": [f"u{k}" for k in range(len(table))], "ra": ra0 - x / (np.cos(np.radians(dec0)) * 3600.0),
                        "dec": dec0 + y / 3600.0, "parity": table["parity"],
                        "zphot_best": table["z"] + rng.normal(0.0, 0.15, len(table)), "catalog_z_sigma": 0.2})
    for b, band in enumerate(BANDS):
        out[f"mag_{band}"] = [COLORS[c][b] + rng.normal(0.0, 0.05) for c in table["colors"]]
        out[f"mag_err_{band}"] = 0.05
    path = root / "uncertain.csv"
    out.to_csv(path, index=False)
    return path, list(table["truth"])


def _with_partition(config, path, **kwargs):
    return replace(config, partition=PartitionConfig(catalog_path=str(path), source_prior_sigma_arcsec=20.0, **kwargs)).validate()


def _add(config, state=None, evaluator=None, **kwargs):
    state = build_state(config) if state is None else state
    evaluator = Evaluator(state, config.likelihood.likelihood_mode) if evaluator is None else evaluator
    return add_uncertain_images(config, state, evaluator, state.initial_theta, log=lambda _: None, **kwargs)


def _same(truth):
    return np.asarray(truth)[:, None] == np.asarray(truth)[None, :]


def test_redshift_scaling_of_the_reference_fields_is_exact(synthetic_config):
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    physical = evaluator.physical(jnp.asarray(state.initial_theta))
    xy = np.asarray([[3.2, 10.6], [-9.2, -1.2], [7.7, -10.4], [0.4, 0.2]])
    z_grid = np.asarray([1.5, 2.5, 4.0])
    images = le.lens_images(state, xy, np.zeros(4, dtype=int), z_grid)
    fields = le.grid_fields(le.reference_fields(evaluator, physical, images), images)
    for g, z in enumerate(z_grid):
        alpha, u = evaluator.total_fields_at(xy, evaluator.packed_state_for_z(physical, float(z)))
        np.testing.assert_allclose(np.asarray(fields["beta"][g]), xy - alpha, atol=1e-12)
        np.testing.assert_allclose(np.asarray(fields["jacobian"][g]), np.eye(2) - u, atol=1e-12)


def test_closed_form_source_integral_matches_quadrature(synthetic_config):
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    physical = evaluator.physical(jnp.asarray(state.initial_theta))
    xy = np.asarray([[3.27, 11.25], [4.56, -10.1], [10.65, -2.58]])
    images = le.lens_images(state, xy, np.zeros(3, dtype=int), np.asarray([3.0]), prior_sigma=20.0)
    scales = le.error_scales(evaluator, physical, images, images.ratio)
    terms = le.image_terms(le.grid_fields(le.reference_fields(evaluator, physical, images), images), scales, images)
    out = le.family_lens_evidence(terms, images, jnp.asarray([[0, 1, 2]]), jnp.ones((1, 3), dtype=bool))
    mean, covariance = np.asarray(out["source_mean"][0, 0]), np.asarray(out["source_covariance"][0, 0])
    beta, precision, logdet = (np.asarray(terms[k][0]) for k in ("beta", "precision", "logdet"))
    width = 12.0 * np.sqrt(np.linalg.eigvalsh(covariance).max())
    axis_x, axis_y = np.linspace(mean[0] - width, mean[0] + width, 601), np.linspace(mean[1] - width, mean[1] + width, 601)
    grid = np.stack(np.meshgrid(axis_x, axis_y), -1)
    log_density = -0.5 * np.sum(grid ** 2, -1) / 20.0 ** 2 - np.log(2.0 * np.pi * 20.0 ** 2)
    for b, p, ld in zip(beta, precision, logdet):
        r = grid - b
        log_density += -0.5 * (np.einsum("...i,ij,...j", r, p, r) + ld) - np.log(2.0 * np.pi)
    numeric = np.log(np.exp(log_density - log_density.max()).sum() * (axis_x[1] - axis_x[0]) * (axis_y[1] - axis_y[0])) + log_density.max()
    determinant_mean = np.asarray(terms["determinant"][0]) + np.einsum("ni,ni->n", np.asarray(terms["p"][0]), beta - mean)
    log_h = np.asarray(sl.log_expected_absolute_normal(jnp.asarray(determinant_mean), terms["determinant_variance"][0], 0))
    np.testing.assert_allclose(float(out["log_evidence"][0, 0]) - log_h.sum(), numeric, atol=1e-8)


def test_efficiency_grid_is_uniform_in_the_lensing_efficiency(synthetic_config):
    from kappaccino.lens.cosmology import dpie_sigma0_factor_from_config
    state = build_state(synthetic_config)
    grid = inputs.efficiency_spaced_grid(1.0, 5.0, 30, float(state.z_lens), state.cosmo_config)
    assert grid[0] == pytest.approx(1.0) and grid[-1] == pytest.approx(5.0) and np.all(np.diff(grid) > 0)
    efficiency = np.array([dpie_sigma0_factor_from_config(float(state.z_lens), float(z), state.cosmo_config) for z in grid])
    np.testing.assert_allclose(np.diff(efficiency), np.diff(efficiency).mean(), rtol=0.15)     # the trapezoid weights take the actual spacing
    assert np.diff(grid)[0] < np.diff(grid)[-1]                  # dense at low redshift


def test_photoz_outlier_term_is_a_uniform_mixture():
    table = pd.DataFrame({"catalog_z": [2.0, 4.0], "catalog_z_sigma": [0.1, 0.1]})
    data = ce.catalog_data(table, ())
    z_grid = np.linspace(1.0, 5.0, 41)
    index, mask = np.asarray([[0, 1]]), np.ones((1, 2), dtype=bool)
    plain = ce.catalog_log_evidence(data, index, mask, z_grid, None)
    mixed = ce.catalog_log_evidence(data, index, mask, z_grid, None, photoz_outlier_fraction=0.05)
    low, high = ce.PHOTOZ_OUTLIER_RANGE
    gaussian = lambda zhat: np.exp(-0.5 * ((z_grid - zhat) / 0.1) ** 2) / (0.1 * np.sqrt(2.0 * np.pi))
    expected = np.log(np.prod([0.95 * gaussian(z) + 0.05 / (high - low) for z in (2.0, 4.0)], axis=0))
    np.testing.assert_allclose((mixed - plain)[0], expected - np.log(gaussian(2.0) * gaussian(4.0)), rtol=1e-9, atol=1e-9)
    # Two images 20 sigma apart in photo-z can still be one source.
    assert np.max(mixed) > np.max(plain) + 50.0


def test_groups_recover_the_new_sources_without_new_parameters(synthetic_config, tmp_path):
    path, truth = _mock(synthetic_config, tmp_path)
    config = _with_partition(synthetic_config, path)
    state = build_state(config)
    extended, evaluator, start = _add(config, state)
    model = extended.uncertain_images
    assert extended.n_parameters == state.n_parameters and np.array_equal(start, state.initial_theta)
    for family in set(truth):                                    # every true family is in the library
        assert tuple(i for i, t in enumerate(truth) if t == family) in model.groups
    value, gradient = evaluator.value_and_grad(jnp.asarray(start))
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(gradient)))
    base, _ = Evaluator(state).value_and_grad(jnp.asarray(start))
    stats = evaluator.statistics(jnp.asarray(start))
    np.testing.assert_allclose(float(value) - float(base), float(model.log_likelihood(evaluator, jnp.asarray(start), stats)), rtol=1e-9)
    result = model.posterior(evaluator, start[None], np.random.default_rng(0), draws_per_sample=3)
    np.testing.assert_allclose(result["pair_probability"][_same(truth)], 1.0, atol=1e-3)
    np.testing.assert_allclose(result["pair_probability"][~_same(truth)], 0.0, atol=1e-3)
    np.testing.assert_allclose(result["probability"].sum(axis=1), 1.0, atol=1e-9)
    assert result["labels"].shape == (3, len(truth))
    best = [model.option_names()[k] for k in np.argmax(result["probability"], axis=1)]
    assert all(b.startswith("group") for b in best)


def _set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        for k in range(len(part)):
            yield part[:k] + [[first] + part[k]] + part[k + 1:]
        yield [[first]] + part


def test_the_library_sum_is_the_sum_over_every_partition(synthetic_config, tmp_path):
    """With margins wide enough for every group, the likelihood is the brute-force sum over all set partitions of
    w_G E_G per group and a_i per image alone, and the pair probabilities are its averages."""
    path, _ = _mock(synthetic_config, tmp_path)
    table = pd.read_csv(path).iloc[:6]
    table.to_csv(path, index=False)
    config = _with_partition(synthetic_config, path, library_pair_margin=1.0e6, library_group_margin=1.0e6)
    extended, evaluator, start = _add(config)
    model = extended.uncertain_images
    theta = jnp.asarray(start)
    stats = evaluator.statistics(theta)
    out = {k: np.asarray(v) for k, v in model.terms(evaluator, theta, stats).items()}
    assert len(model.groups) == 2 ** 6 - 6 - 1
    value = dict(zip(model.groups, model.log_group_weight + out["log_group"]))
    parts = list(_set_partitions(list(range(6))))
    scores = np.array([sum(value[tuple(sorted(b))] if len(b) > 1 else out["log_alone"][b[0]] for b in p) for p in parts])
    from scipy.special import logsumexp
    np.testing.assert_allclose(float(model.log_likelihood(evaluator, theta, stats)), logsumexp(scores), rtol=1e-10)
    weights = np.exp(scores - logsumexp(scores))
    pair = np.zeros((6, 6))
    share = np.exp(out["options"] - out["log_alone"][:, None])[:, 1:]
    for p, w in zip(parts, weights):
        for block in p:
            if len(block) > 1:
                pair[np.ix_(block, block)] += w
        alone = [b[0] for b in p if len(b) == 1]
        pair[np.ix_(alone, alone)] += w * (share[alone] @ share[alone].T)
    np.fill_diagonal(pair, 1.0)
    result = model.posterior(evaluator, np.asarray(start)[None], np.random.default_rng(0))
    np.testing.assert_allclose(result["pair_probability"], pair, atol=1e-10)


def test_group_weight_is_the_sum_over_new_sources():
    log_new = np.log(np.asarray([[0.1, 0.3], [0.2, 0.2], [0.5, 0.05]]))
    index, mask = np.asarray([[0, 1, 2], [0, 2, 0]]), np.asarray([[True, True, True], [True, True, False]])
    expected = [np.log(0.1 * 0.2 * 0.5 + 0.3 * 0.2 * 0.05), np.log(0.1 * 0.5 + 0.3 * 0.05)]
    np.testing.assert_allclose(groups.group_log_weight(log_new, index, mask), expected, rtol=1e-12)


def test_packings_and_pruning_of_a_large_component():
    library = [(0, 1), (0, 2), (1, 2), (0, 1, 2), (2, 3)]
    packings = groups._packings([0, 1, 2, 3], list(range(len(library))), library, 100)
    # {} {01} {02} {12} {012} {23} {01,23}
    assert sorted(packings) == sorted([(), (0,), (1,), (2,), (3,), (4,), (0, 4)])
    with pytest.raises(RuntimeError):
        groups._packings([0, 1, 2, 3], list(range(len(library))), library, 3)
    assert groups._components(library, 5) == [[0, 1, 2, 3], [4]]
    assert sorted(groups._cliques(np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool), 3)) == [(0, 1), (0, 1, 2), (0, 2), (1, 2)]


def test_an_empty_library_leaves_every_image_alone(synthetic_config, tmp_path):
    """With no compatible groups, the sum has one partition per image (each image alone)."""
    path, truth = _mock(synthetic_config, tmp_path)
    table = pd.read_csv(path)
    table["zphot_best"] = np.linspace(1.0, 5.0, len(table))          # every pair disagrees in redshift
    table["catalog_z_sigma"] = 0.01
    table.to_csv(path, index=False)
    config = _with_partition(synthetic_config, path, photoz_outlier_fraction=0.0, library_pair_margin=0.0, library_group_margin=0.0)
    extended, evaluator, start = _add(config)
    model = extended.uncertain_images
    assert model.groups == () and model.packing_groups.shape == (len(truth), 0) and model.n_components == len(truth)
    stats = evaluator.statistics(jnp.asarray(start))
    out = model.terms(evaluator, jnp.asarray(start), stats)
    np.testing.assert_allclose(float(model.log_likelihood(evaluator, jnp.asarray(start), stats)), float(jnp.sum(out["log_alone"])), rtol=1e-12)
    result = model.posterior(evaluator, start[None], np.random.default_rng(0))
    np.testing.assert_allclose(result["probability"].sum(axis=1), 1.0, atol=1e-9)


def test_an_uncatalogued_counterimage_joins_its_trusted_family(synthetic_config, tmp_path):
    path, truth = _mock(synthetic_config, tmp_path, counterimage=True)
    config = _with_partition(synthetic_config, path)
    extended, evaluator, start = _add(config)
    model = extended.uncertain_images
    result = model.posterior(evaluator, start[None], np.random.default_rng(0))
    k = truth.index("1")
    assert result["probability"][k, model.option_names().index("1")] > 0.99


def test_claimed_images_are_a_library_group_with_their_prior(synthetic_config, tmp_path):
    """Images the catalog claims for one untrusted family always form a group, weighted by the association prior."""
    path, truth = _mock(synthetic_config, tmp_path)
    table = pd.read_csv(path)
    claimed = [i for i, t in enumerate(truth) if t == "new0"][:2] + [truth.index("new1")]      # a wrong claim
    table["claimed_family"] = ""
    table.loc[claimed, "claimed_family"] = "X"
    table["association_prior"] = np.nan
    table.loc[claimed, "association_prior"] = 0.9
    table.to_csv(path, index=False)
    config = _with_partition(synthetic_config, path, library_pair_margin=0.0, library_group_margin=0.0)
    extended, evaluator, start = _add(config)
    model = extended.uncertain_images
    assert tuple(sorted(claimed)) in model.groups
    result = model.posterior(evaluator, start[None], np.random.default_rng(0))
    assert result["pair_probability"][claimed[0], claimed[2]] < 0.01          # the data overrule the wrong claim
    unclaimed = replace(config, partition=replace(config.partition, association_weight=False))
    assert tuple(sorted(claimed)) not in _add(unclaimed)[0].uncertain_images.groups


def test_refits_rebuild_the_library_at_the_new_lens(synthetic_config, tmp_path):
    path, _ = _mock(synthetic_config, tmp_path)
    config = _with_partition(synthetic_config, path, library_refits=2)
    calls = []

    def refit(state, evaluator, theta):
        assert state.uncertain_images is not None and evaluator.state is state
        calls.append(np.asarray(theta))
        return np.asarray(theta) + 1.0e-4
    extended, _, start = _add(config, refit=refit)
    assert len(calls) == 2
    np.testing.assert_allclose(start, calls[0] + 2.0e-4)


def test_outputs_from_posterior_draws(synthetic_config, tmp_path):
    """uncertain_images.h5 and the tables from a few draws about the start (the final stage writes them from the NUTS draws)."""
    from types import SimpleNamespace
    path, truth = _mock(synthetic_config, tmp_path)
    config = _with_partition(synthetic_config, path, posterior_draws=5)
    extended, evaluator, start = _add(config)
    samples = start + 0.01 * np.random.default_rng(4).normal(size=(6, start.size))
    stage = SimpleNamespace(artifacts=tmp_path / "artifacts", tables=tmp_path / "tables")
    stage.artifacts.mkdir(), stage.tables.mkdir()
    write_partition_outputs(extended, evaluator, samples, stage, draws=5, seed=3, draws_per_sample=2, log=lambda _: None)
    saved = load_arrays(stage.artifacts / groups.OUTPUT_FILE)
    model = extended.uncertain_images
    assert saved["probability"].shape == (len(truth), len(model.option_names()))
    assert saved["responsibilities"].shape[0] == 5 and saved["assignment"].shape == (10, len(truth))
    assert len(saved["groups"]) == len(model.groups)
    np.testing.assert_allclose(saved["probability"].sum(axis=1), 1.0, atol=1e-9)
    images = pd.read_csv(stage.tables / "uncertain_images.csv")
    assert list(images["image_label"]) == [f"u{k}" for k in range(len(truth))]
    assert (stage.tables / "uncertain_pairs.csv").is_file()


def test_population_mixture_and_color_scatter_are_recovered():
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal([1.0, 0.0], 0.1, (400, 2)), rng.normal([3.0, 1.0], 0.1, (400, 2))])
    mixture = ce.fit_gaussian_mixture(data, n_components=2, seed=1)
    np.testing.assert_allclose(np.sort(mixture.means[:, 0]), [1.0, 3.0], atol=0.03)
    # Families of four images sharing colors with a 0.1 mag same-source scatter and 0.02 mag errors.
    rows = []
    for family in range(40):
        z = rng.uniform(1.0, 3.0)
        intrinsic = rng.normal(0.0, 0.5, 3)
        for _ in range(4):
            mags = 24.0 + intrinsic + rng.normal(0.0, np.sqrt(0.1 ** 2 / 2), 3) + rng.normal(0.0, 0.02, 3)
            rows.append({"catalog_z": z, "catalog_z_sigma": 0.01, **{f"mag_{b}": m for b, m in zip(BANDS, mags)},
                         **{f"mag_err_{b}": 0.02 for b in BANDS}})
    table = pd.DataFrame(rows)
    population = ce.fit_population(table, BANDS, n_components=2)
    data = ce.catalog_data(table, BANDS)
    index = np.arange(len(table)).reshape(40, 4)
    scatter = ce.fit_color_scatter(data, index, np.ones(index.shape, dtype=bool), np.unique(data.z_mean), population, outlier_fraction=0.0)
    assert 0.07 < scatter < 0.14


def test_partition_config_is_validated(synthetic_config, tmp_path):
    path = tmp_path / "uncertain.csv"
    path.write_text("image_label,ra,dec\n")
    for bad in (dict(unrelated_fraction=1.0), dict(outlier_fraction=-0.1), dict(redshift_grid=(0.1, 3.0, 8)), dict(posterior_draws=0),
                dict(photoz_outlier_fraction=1.0), dict(library_pair_margin=-1.0), dict(library_group_margin=float("nan")),
                dict(library_max_group=1), dict(library_refits=-1)):
        with pytest.raises(ValueError):
            replace(synthetic_config, partition=PartitionConfig(catalog_path=str(path), **bad)).validate()
    with pytest.raises(ValueError):
        replace(synthetic_config, partition=PartitionConfig(catalog_path=str(tmp_path / "missing.csv"))).validate()


def _likelihood_mode(config, mode):
    if mode == "smoothed":
        return config
    kernel = "knee" if mode == "knee" else "image_plane"
    return replace(config, likelihood=replace(config.likelihood, likelihood_mode="gp_image_plane", gp_kernel=kernel)).validate()


@pytest.mark.parametrize("mode", ["image_plane", "knee"])
def test_groups_under_the_gp_likelihoods(synthetic_config, tmp_path, mode):
    path, truth = _mock(synthetic_config, tmp_path)
    config = _with_partition(_likelihood_mode(synthetic_config, mode), path)
    state = build_state(config)
    base = Evaluator(state, config.likelihood.likelihood_mode)
    extended, evaluator, start = _add(config, state, base)
    value, gradient = evaluator.value_and_grad(jnp.asarray(start))
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(gradient)))
    result = extended.uncertain_images.posterior(evaluator, start[None], np.random.default_rng(0))
    assert np.all(result["pair_probability"][_same(truth)] > 0.99) and np.all(result["pair_probability"][~_same(truth)] < 0.01)
    if mode == "knee":
        # An uncertain image at a trusted family's redshift gets that family's deflection floor.
        physical = base.physical(jnp.asarray(state.initial_theta))
        model = extended.uncertain_images
        scales = le.error_scales(base, physical, model.images, jnp.asarray(model.trusted_ratio))
        trusted = np.asarray(base.error_scales(physical)["sigma_def"])
        family = np.asarray(state.images.family_index_per_image)
        for k in range(len(model.trusted_ids)):
            np.testing.assert_allclose(np.asarray(scales["sigma_def"])[k], trusted[family == k][0], rtol=1e-10)


def test_groups_sit_inside_the_membership_sum(synthetic_config, tmp_path):
    """With enumerated uncertain members, every member configuration evaluates the uncertain images with its own lens."""
    from scipy.special import logsumexp
    from kappaccino.core.state import with_member_classes
    path, _ = _mock(synthetic_config, tmp_path)
    p = 0.6
    config = _with_partition(replace(synthetic_config, members=replace(synthetic_config.members, membership="marginalize",
                                                                        membership_default_probability=p)), path)
    state = with_member_classes(build_state(config), np.asarray([0, 1]))
    extended, evaluator, start = _add(config, state, Evaluator(state))
    model = extended.uncertain_images
    assert len(evaluator.membership_enumerated_positions) > 0
    theta = jnp.asarray(start)
    physical = evaluator.physical(theta)
    terms = []
    for z in np.asarray(evaluator.membership_configurations):
        stats = evaluator.statistics(theta, membership=jnp.asarray(z))
        uncertain = model.log_likelihood(evaluator, theta, stats, membership=jnp.asarray(z))
        terms.append(float(jnp.sum(stats["loglike"]) + stats["source_correction"] + uncertain) + float(np.sum(np.where(z, np.log(p), np.log1p(-p)))))
    rest = (evaluator.far_field_log_barrier(evaluator.fields(physical)["far_cut_radii"])
            + evaluator.ellipticity_log_prior(evaluator.packed_state(physical)[0]) + evaluator.kinematics_log_likelihood(physical))
    np.testing.assert_allclose(float(evaluator.loglike(theta)), logsumexp(terms) + float(rest), rtol=1e-10)
    # The option probabilities average each configuration's, weighted by the configuration's posterior.
    weight = np.exp(np.asarray(terms) - logsumexp(terms))
    expected = 0.0
    for w, z in zip(weight, np.asarray(evaluator.membership_configurations)):
        out = {k: np.asarray(v) for k, v in model.terms(evaluator, theta, evaluator.statistics(theta, membership=jnp.asarray(z)), jnp.asarray(z)).items()}
        expected = expected + w * model._at_lens(out["options"], out["log_alone"], out["score"])["responsibilities"]
    result = model.posterior(evaluator, np.asarray(theta)[None], np.random.default_rng(0))
    np.testing.assert_allclose(result["probability"], expected, atol=1e-10)
    np.testing.assert_allclose(result["probability"].sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.slow
def test_final_stage_samples_the_lens_with_the_uncertain_images(synthetic_config, tmp_path):
    """NUTS through both stages with uncertain images: the refits, the chains at the last refit, the outputs, resume."""
    path, truth = _mock(synthetic_config, tmp_path)
    config = replace(_with_partition(synthetic_config, path, posterior_draws=4, library_refits=1),
                     runtime=RuntimeConfig(seed=3, chains=1, nuts_chain_method="sequential", resume=True, skip_plots=True),
                     search=SearchConfig(n_starts=2, member_starts=1, lbfgs_max_iter=20, polish_max_iter=5),
                     sampling=SamplingConfig(warmup=5, samples=5, max_tree_depth=3),
                     reporting=ReportingConfig(draws=1, image_draws=1, extra_images=False)).validate()
    result = pipeline.run(config)
    assert result.state.uncertain_images is not None and result.posterior.samples.shape[1] == result.init.state.n_parameters
    saved = load_arrays(result.paths.final.artifacts / groups.OUTPUT_FILE)
    assert saved["probability"].shape == (len(truth), len(result.state.uncertain_images.option_names()))
    assert pd.read_csv(result.paths.final.tables / "uncertain_images.csv").shape[0] == len(truth)
    resumed = pipeline.run(config)
    assert resumed.resumed and resumed.state.uncertain_images.groups == result.state.uncertain_images.groups
