from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from kappaccino import pipeline
from kappaccino.config import ImageDiagnosticsConfig, PartitionConfig, RuntimeConfig, SamplingConfig, SearchConfig
from kappaccino.evaluator import Evaluator
from kappaccino.exact_images import ExactImageSolver, packed_to_kwargs_lens
from kappaccino.partition import add_slots, extend_theta
from kappaccino.partition import catalog_evidence as ce, lens_evidence as le
from kappaccino import smoothed_likelihood as sl
from kappaccino.state import build_state

SOURCES = [(2.0, (0.5, 0.3)), (3.0, (-1.0, 0.8)), (2.5, (2.0, -1.5))]       # (z, source position) of the new sources
COLORS = {0: (24.0, 23.5, 23.3), 1: (25.0, 24.2, 23.6), 2: (23.8, 23.7, 23.65), "trusted": (24.5, 24.0, 23.0)}
BANDS = ("F090W", "F150W", "F200W")


def _images_of(solver, evaluator, physical, z, source, *, min_magnification=0.3):
    packed = evaluator.packed_state_for_z(physical, z)
    _, equation = solver.lenstronomy_model(z)
    x, y = equation.image_position_from_source(*source, packed_to_kwargs_lens(packed), min_distance=0.1, search_window=80,
                                               num_iter_max=50, precision_limit=1e-10)
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
                        "dec": dec0 + y / 3600.0, "sigma_arcsec": 0.05, "parity": table["parity"],
                        "zphot_best": table["z"] + rng.normal(0.0, 0.15, len(table)), "catalog_z_sigma": 0.2})
    for b, band in enumerate(BANDS):
        out[f"mag_{band}"] = [COLORS[c][b] + rng.normal(0.0, 0.05) for c in table["colors"]]
        out[f"mag_err_{band}"] = 0.05
    path = root / "uncertain.csv"
    out.to_csv(path, index=False)
    return path, list(table["truth"])


def _with_partition(config, path, **kwargs):
    return replace(config, partition=PartitionConfig(catalog_path=str(path), source_prior_sigma_arcsec=20.0, **kwargs)).validate()


def test_redshift_scaling_of_the_reference_fields_is_exact(synthetic_config):
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    physical = evaluator.physical(jnp.asarray(state.initial_theta))
    xy = np.asarray([[3.2, 10.6], [-9.2, -1.2], [7.7, -10.4], [0.4, 0.2]])
    z_grid = np.asarray([1.5, 2.5, 4.0])
    images = le.lens_images(state, xy, np.full(4, 0.05), np.zeros(4, dtype=int), z_grid)
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
    images = le.lens_images(state, xy, np.full(3, 0.05), np.zeros(3, dtype=int), np.asarray([3.0]), prior_sigma=20.0)
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


def test_slots_seed_the_new_sources_and_assign_their_images(synthetic_config, tmp_path):
    path, truth = _mock(synthetic_config, tmp_path)
    config = _with_partition(synthetic_config, path)
    state = build_state(config)
    extended, evaluator = add_slots(config, state, Evaluator(state), state.initial_theta, log=lambda _: None)
    model = extended.slots
    assert model.n_slots == 3 and extended.n_parameters == state.n_parameters + model.n_parameters
    theta = extend_theta(extended, state.initial_theta)
    value, gradient = evaluator.value_and_grad(jnp.asarray(theta))
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(gradient)))
    result = model.posterior(evaluator, theta[None], np.random.default_rng(0))
    same = np.asarray(truth)[:, None] == np.asarray(truth)[None, :]
    np.testing.assert_allclose(result["pair_probability"][same], 1.0, atol=1e-3)
    np.testing.assert_allclose(result["pair_probability"][~same], 0.0, atol=1e-3)
    best = [model.option_names()[k] for k in np.argmax(result["probability"], axis=1)]
    assert all(b.startswith("slot") for b in best)


def test_an_uncatalogued_counterimage_joins_its_trusted_family(synthetic_config, tmp_path):
    path, truth = _mock(synthetic_config, tmp_path, counterimage=True)
    config = _with_partition(synthetic_config, path)
    state = build_state(config)
    extended, evaluator = add_slots(config, state, Evaluator(state), state.initial_theta, log=lambda _: None)
    model = extended.slots
    result = model.posterior(evaluator, extend_theta(extended, state.initial_theta)[None], np.random.default_rng(0))
    k = truth.index("1")
    assert result["probability"][k, model.option_names().index("1")] > 0.99
    assert model.n_slots == 3                                  # no new slot was seeded for the trusted family's image


def test_slot_prior_correction_turns_unit_offsets_into_the_physical_priors(synthetic_config, tmp_path):
    path, _ = _mock(synthetic_config, tmp_path)
    config = _with_partition(synthetic_config, path)
    state = build_state(config)
    extended, _ = add_slots(config, state, Evaluator(state), state.initial_theta, log=lambda _: None)
    model = extended.slots
    from scipy.stats import multivariate_normal
    from kappaccino.partition.slots import COLOR_SCALE, LOG_DZ_SCALE
    population = model.population
    rng = np.random.default_rng(3)
    for _ in range(3):
        latent = rng.normal(size=model.n_parameters)
        values = model.slot_values(jnp.asarray(np.concatenate([state.initial_theta, latent])))
        beta, z, colors = (np.asarray(values[k]) for k in ("beta", "z", "colors"))
        log_beta = np.sum(-0.5 * np.sum(beta ** 2, -1) / 20.0 ** 2 - np.log(2.0 * np.pi * 20.0 ** 2))
        x = np.column_stack([z, colors])
        log_pop = np.sum(np.log(sum(w * multivariate_normal(m, c).pdf(x) for w, m, c in
                                    zip(population.weights, population.means, population.covariances))))
        # |d beta / d u| = prod scale, |d z / d v| = b (z - z_lens), |d theta / d w| = s_c^d
        log_jacobian = (np.sum(np.log(model.scale)) + model.n_slots * np.log(LOG_DZ_SCALE) + np.sum(np.log(z - model.z_lens))
                        + model.n_slots * model.n_colors * np.log(COLOR_SCALE))
        unit = -0.5 * np.sum(latent ** 2) - 0.5 * latent.size * np.log(2.0 * np.pi)
        np.testing.assert_allclose(float(model.log_prior_correction(values)), log_beta + log_pop + log_jacobian - unit, rtol=1e-10)


def test_final_stage_with_slots_writes_assignments(synthetic_config, tmp_path):
    path, truth = _mock(synthetic_config, tmp_path)
    config = replace(_with_partition(synthetic_config, path, posterior_draws=5),
                     runtime=RuntimeConfig(seed=3, chains=1, resume=True, nuts_chain_method="sequential", skip_plots=True),
                     search=SearchConfig(n_starts=2, lbfgs_max_iter=20, polish_max_iter=5),
                     sampling=SamplingConfig(warmup=5, samples=5, max_tree_depth=3),
                     image_diagnostics=ImageDiagnosticsConfig(posterior_image_diagnostic_draws=0)).validate()
    result = pipeline.run(config)
    assert result.state.slots is not None and result.posterior.samples.shape[1] == result.state.n_parameters
    saved = np.load(result.paths.final.artifacts / "slots.npz")
    assert saved["probability"].shape == (len(truth), len(result.state.slots.option_names()))
    np.testing.assert_allclose(saved["probability"].sum(axis=1), 1.0, atol=1e-9)
    images = pd.read_csv(result.paths.final.tables / "uncertain_images.csv")
    assert list(images["image_label"]) == [f"u{k}" for k in range(len(truth))]
    resumed = pipeline.run(config)
    assert resumed.resumed and resumed.state.n_parameters == result.state.n_parameters


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
    for bad in (dict(unrelated_fraction=1.0), dict(outlier_fraction=-0.1), dict(redshift_grid=(0.1, 3.0, 8)), dict(posterior_draws=0)):
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
def test_slots_under_the_gp_likelihoods(synthetic_config, tmp_path, mode):
    path, truth = _mock(synthetic_config, tmp_path)
    config = _with_partition(_likelihood_mode(synthetic_config, mode), path)
    state = build_state(config)
    base = Evaluator(state, config.likelihood.likelihood_mode)
    extended, evaluator = add_slots(config, state, base, state.initial_theta, log=lambda _: None)
    theta = extend_theta(extended, state.initial_theta)
    value, gradient = evaluator.value_and_grad(jnp.asarray(theta))
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(gradient)))
    result = extended.slots.posterior(evaluator, theta[None], np.random.default_rng(0))
    same = np.asarray(truth)[:, None] == np.asarray(truth)[None, :]
    assert np.all(result["pair_probability"][same] > 0.99) and np.all(result["pair_probability"][~same] < 0.01)
    if mode == "knee":
        # An uncertain image at a trusted family's redshift gets that family's deflection floor.
        physical = base.physical(jnp.asarray(state.initial_theta))
        model = extended.slots
        scales = le.error_scales(base, physical, model.images, jnp.asarray(model.trusted_ratio))
        trusted = np.asarray(base.error_scales(physical)["sigma_def"])
        family = np.asarray(state.images.family_index_per_image)
        for k in range(len(model.trusted_ids)):
            np.testing.assert_allclose(np.asarray(scales["sigma_def"])[k], trusted[family == k][0], rtol=1e-10)


def test_slots_sit_inside_the_membership_sum(synthetic_config, tmp_path):
    """With enumerated uncertain members, every member configuration evaluates the uncertain images with its own lens."""
    from scipy.special import logsumexp
    from kappaccino.state import with_member_classes
    path, _ = _mock(synthetic_config, tmp_path)
    p = 0.6
    config = _with_partition(replace(synthetic_config, members=replace(synthetic_config.members, membership="marginalize",
                                                                        membership_default_probability=p)), path)
    state = with_member_classes(build_state(config), np.asarray([0, 1]))
    extended, evaluator = add_slots(config, state, Evaluator(state), state.initial_theta, log=lambda _: None)
    assert len(evaluator.membership_enumerated_positions) > 0
    theta = jnp.asarray(extend_theta(extended, state.initial_theta))
    physical = evaluator.physical(theta)
    terms = []
    for z in np.asarray(evaluator.membership_configurations):
        stats = evaluator.statistics(theta, membership=jnp.asarray(z))
        slots = extended.slots.log_likelihood(evaluator, theta, stats, membership=jnp.asarray(z))
        terms.append(float(jnp.sum(stats["loglike"]) + stats["source_correction"] + slots) + float(np.sum(np.where(z, np.log(p), np.log1p(-p)))))
    rest = (evaluator.far_field_log_barrier(evaluator.fields(physical)["far_cut_radii"])
            + evaluator.ellipticity_log_prior(evaluator.packed_state(physical)[0]) + evaluator.kinematics_log_likelihood(physical))
    np.testing.assert_allclose(float(evaluator.loglike(theta)), logsumexp(terms) + float(rest), rtol=1e-10)
    # The assignment probabilities average each configuration's, weighted by the configuration's posterior.
    weight = np.exp(np.asarray(terms) - logsumexp(terms))
    expected = 0.0
    for w, z in zip(weight, np.asarray(evaluator.membership_configurations)):
        log_terms = np.asarray(extended.slots.log_terms(evaluator, theta, evaluator.statistics(theta, membership=jnp.asarray(z)), jnp.asarray(z)))
        expected = expected + w * np.exp(log_terms - logsumexp(log_terms, axis=1, keepdims=True))
    result = extended.slots.posterior(evaluator, np.asarray(theta)[None], np.random.default_rng(0))
    np.testing.assert_allclose(result["probability"], expected, atol=1e-10)
    np.testing.assert_allclose(result["probability"].sum(axis=1), 1.0, atol=1e-9)
