"""Independent matrix references for the profiled source-plane likelihood."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from lenscluster.config import LikelihoodConfig, resolve_source_plane_covariance_floor


FAMILY = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
MASK = np.array([True, True, True, True, True, False])


def inputs(theta=None):
    if theta is None:
        theta = jnp.zeros(9)
    beta = jnp.array([[.2, -.1], [.1, .3], [-.1, .2], [.5, .1], [.3, -.2], [1., 2.]])
    beta = beta + theta[0] * jnp.arange(12).reshape(6, 2) / 20
    a = jnp.array([[[.7, .3], [-.1, 1.2]], [[1.1, -.4], [.2, .6]],
                   [[.5, .2], [.3, .8]], [[.8, -.2], [.1, .5]],
                   [[1.3, .4], [-.2, .7]], [[.3, .2], [.1, .8]]])
    a = a.at[:, 0, 0].add(theta[1]).at[:, 0, 1].add(theta[2])
    a = a.at[:, 1, 0].add(theta[3]).at[:, 1, 1].add(theta[4])
    return dict(beta_x=beta[:, 0], beta_y=beta[:, 1], family_idx=jnp.asarray(FAMILY), n_families=3,
                sigma_per_image=jnp.linspace(.1, .3, 6) * jnp.exp(theta[5]),
                reliability_per_image=jnp.array([.9, .95, .8, .99, .7, .9]),
                image_has_constraint=jnp.asarray(MASK), source_sigma_int=.08 * jnp.exp(theta[6]),
                scatter_var_x=jnp.linspace(.001, .008, 6) * jnp.exp(theta[7]),
                scatter_var_y=jnp.linspace(.009, .002, 6) * jnp.exp(theta[8]),
                jac_a00=a[:, 0, 0], jac_a01=a[:, 0, 1], jac_a10=a[:, 1, 0], jac_a11=a[:, 1, 1],
                covariance_floor=0., outlier_sigma_arcsec=10.)


def reference(p, xp=jnp, loss="gaussian", cap=0.):
    """Independent dense solves with explicit family slices; no production helpers."""
    a = xp.stack([xp.stack([p['jac_a00'], p['jac_a01']], -1),
                  xp.stack([p['jac_a10'], p['jac_a11']], -1)], -2)
    cov = p['sigma_per_image'][:, None, None] ** 2 * (a @ xp.swapaxes(a, -1, -2))
    cov = cov + xp.eye(2) * (p['source_sigma_int'] ** 2 + xp.stack([p['scatter_var_x'], p['scatter_var_y']], -1))[..., None, :]
    beta = xp.stack([p['beta_x'], p['beta_y']], -1)
    precision = xp.linalg.inv(cov)
    centers = xp.stack([xp.linalg.solve(precision[FAMILY == f].sum(0),
                      xp.einsum('nij,nj->i', precision[FAMILY == f], beta[FAMILY == f])) for f in range(3)])
    residual = beta - centers[FAMILY]
    if cap:
        # Match the established smooth cap (tested separately below).
        dx, dy = cs._smooth_residual_cap(residual[:, 0], residual[:, 1], cap)
        residual = xp.stack([dx, dy], -1)
    quad = xp.einsum('ni,ni->n', residual, xp.linalg.solve(cov, residual[..., None])[..., 0])
    logdet = xp.linalg.slogdet(cov)[1]
    ll = -.5 * (quad + logdet + 2 * xp.log(2 * xp.pi))
    if loss == 'student-t':
        nu = 4.
        ll = -xp.log(2 * xp.pi) - .5 * logdet - (nu + 2) / 2 * xp.log1p(quad / nu)
    out = -.5 * (xp.sum(residual ** 2, -1) / p['outlier_sigma_arcsec'] ** 2 + 2 * xp.log(2 * xp.pi * p['outlier_sigma_arcsec'] ** 2))
    r = xp.clip(p['reliability_per_image'], 1e-6, 1 - 1e-6)
    total = xp.where(p['image_has_constraint'], xp.logaddexp(xp.log(r) + ll, xp.log1p(-r) + out), 0).sum()
    return total, cov, centers, quad, logdet


def stats(p):
    return cs._source_plane_statistics(*(p[k] for k in (
        'beta_x', 'beta_y', 'family_idx', 'n_families', 'sigma_per_image', 'source_sigma_int',
        'scatter_var_x', 'scatter_var_y', 'jac_a00', 'jac_a01', 'jac_a10', 'jac_a11')))


@pytest.mark.parametrize('loss,cap', [('gaussian', 0.), ('student-t', 0.), ('gaussian', .3)])
def test_full_covariance_reference(loss, cap):
    p = inputs()
    expected = reference(p, loss=loss, cap=cap)
    actual = cs._full_covariance_source_plane_bin_loglike(**p, residual_loss=loss, max_residual_arcsec=cap)
    np.testing.assert_allclose(actual, expected[0], rtol=1e-12, atol=1e-12)
    if cap == 0 and loss == 'gaussian':
        numpy_p = {k: np.asarray(v) if hasattr(v, 'shape') else v for k, v in p.items()}
        dense = reference(numpy_p, np)
        covariance, chol, precision, _, centroid, logdet = stats(p)
        np.testing.assert_allclose(covariance, dense[1], atol=1e-14)
        np.testing.assert_allclose(centroid, dense[2], atol=1e-13)
        np.testing.assert_allclose(logdet, dense[4], atol=1e-13)
        residual = np.stack([p['beta_x'], p['beta_y']], -1) - np.asarray(centroid)[FAMILY]
        np.testing.assert_allclose(np.einsum('ni,nij,nj->n', residual, precision, residual), dense[3], atol=1e-12)
        np.testing.assert_allclose(chol @ jnp.swapaxes(chol, -1, -2), covariance, atol=1e-14)


def central_difference(fn, x, step=1e-5):
    return np.array([(float(fn(x + np.eye(len(x))[i] * step)) - float(fn(x - np.eye(len(x))[i] * step))) / (2 * step) for i in range(len(x))])


def test_general_gradients_against_new_reference_and_finite_difference():
    x = jnp.array([.02, -.03, .01, -.02, .04, .03, -.01, .04, -.02])
    fn = jax.jit(lambda t: cs._full_covariance_source_plane_bin_loglike(**inputs(t)))
    value, gradient = jax.jit(jax.value_and_grad(fn))(x)
    ref_value, ref_gradient = jax.jit(jax.value_and_grad(lambda t: reference(inputs(t))[0]))(x)
    np.testing.assert_allclose(value, ref_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gradient, ref_gradient, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(gradient, central_difference(fn, np.asarray(x)), rtol=1e-6, atol=1e-7)


def isotropic_inputs(t):
    p = inputs()
    a = jnp.linspace(.3, 1.1, 6) * jnp.exp(t[0])
    variance = jnp.linspace(.001, .008, 6) * jnp.exp(t[3])
    p.update(jac_a00=a, jac_a11=a, jac_a01=jnp.zeros(6), jac_a10=jnp.zeros(6),
             reliability_per_image=jnp.full(6, .95), scatter_var_x=variance, scatter_var_y=variance,
             source_sigma_int=.08 * jnp.exp(t[1]), beta_x=p['beta_x'] + t[2] * jnp.arange(6))
    return p


def legacy_isotropic(t):
    p = isotropic_inputs(t)
    variance = p['sigma_per_image'] ** 2 * jnp.abs(p['jac_a00'] * p['jac_a11']) + p['source_sigma_int'] ** 2 + p['scatter_var_x']
    weight = p['reliability_per_image'] / variance
    sw = jnp.zeros(3).at[p['family_idx']].add(weight)
    bx = jnp.zeros(3).at[p['family_idx']].add(weight * p['beta_x']) / sw
    by = jnp.zeros(3).at[p['family_idx']].add(weight * p['beta_y']) / sw
    r2 = (p['beta_x'] - bx[FAMILY]) ** 2 + (p['beta_y'] - by[FAMILY]) ** 2
    ll = -.5 * (r2 / variance + 2 * jnp.log(2 * jnp.pi * variance))
    out = -.5 * (r2 / 100 + 2 * jnp.log(2 * jnp.pi * 100))
    return jnp.where(p['image_has_constraint'], jnp.logaddexp(jnp.log(.95) + ll, jnp.log(.05) + out), 0).sum()


def test_legacy_value_and_only_isotropy_preserving_gradients():
    # All four variations preserve A=aI, isotropic variance, and equal family reliability.
    t = jnp.array([.1, -.2, .02, .1])
    new = jax.jit(jax.value_and_grad(lambda x: cs._full_covariance_source_plane_bin_loglike(**isotropic_inputs(x))))(t)
    old = jax.jit(jax.value_and_grad(legacy_isotropic))(t)
    np.testing.assert_allclose(new[0], old[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(new[1], old[1], rtol=1e-11, atol=1e-11)


def crossing_inputs(t):
    p = inputs()
    angle = .47
    r = jnp.array([[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]])
    a = r @ jnp.diag(jnp.array([t, 1.2])) @ r.T
    p.update(**{k: jnp.full(6, a[i, j]) for k, i, j in [('jac_a00', 0, 0), ('jac_a01', 0, 1), ('jac_a10', 1, 0), ('jac_a11', 1, 1)]})
    p.update(scatter_var_x=jnp.zeros(6), scatter_var_y=jnp.zeros(6), source_sigma_int=.001,
             beta_x=p['beta_x'] * .001, beta_y=p['beta_y'] * .001)
    return p


def test_critical_crossing_positive_scatter_values_and_gradients():
    fn = jax.jit(lambda t: cs._full_covariance_source_plane_bin_loglike(**crossing_inputs(t)))
    vg = jax.jit(jax.value_and_grad(fn))
    ref = jax.jit(jax.value_and_grad(lambda t: reference(crossing_inputs(t))[0]))
    for t in [-.1, -.001, -1e-5, 0., 1e-5, .001, .1]:
        value, gradient = vg(t)
        assert np.isfinite(value) and np.isfinite(gradient)
        rv, rg = ref(t)
        np.testing.assert_allclose(value, rv, rtol=1e-9, atol=1e-8)
        np.testing.assert_allclose(gradient, rg, rtol=1e-6, atol=1e-7)
        # The covariance condition number reaches ~1e5. Two step sizes avoid
        # interpreting float64 cancellation at the even function's origin as a gradient.
        for step in (1e-6, 2e-6):
            fd = (fn(t + step) - fn(t - step)) / (2 * step)
            np.testing.assert_allclose(gradient, fd, rtol=2e-4, atol=2e-5)


def test_source_configuration_defaults_and_rejection():
    assert LikelihoodConfig().source_plane_covariance_floor is None
    assert resolve_source_plane_covariance_floor(None, 'source', 'full-jacobian') == 0.
    assert resolve_source_plane_covariance_floor(None, 'source') == 1e-6
    assert resolve_source_plane_covariance_floor(None, 'local-jacobian') == 1e-6
    from test_validation import _minimal_stage4_surrogate_state
    state = _minimal_stage4_surrogate_state()
    with pytest.raises(ValueError, match='positive source-scatter'):
        cs.ClusterJAXEvaluator(state, .1, sample_likelihood_mode='source', source_plane_covariance_mode='full-jacobian')
    state.parameter_specs.append(cs._build_source_scatter_parameter_spec(len(state.parameter_specs)))
    for opts, message in [({'source_plane_covariance_floor': 1e-6}, 'covariance_floor=0'),
                           ({'likelihood_stabilizer_max_gain': 50.}, 'max_gain=0')]:
        with pytest.raises(ValueError, match=message):
            cs.ClusterJAXEvaluator(state, .1, sample_likelihood_mode='source', source_plane_covariance_mode='full-jacobian', **opts)
    state.parameter_specs[-1] = replace(state.parameter_specs[-1], physical_lower=0.)
    with pytest.raises(ValueError, match='positive source-scatter lower'):
        cs.ClusterJAXEvaluator(state, .1, sample_likelihood_mode='source', source_plane_covariance_mode='full-jacobian')


def test_nonpositive_scatter_rejected_by_kernel():
    for source_sigma in [0., -.1]:
        p = inputs(); p['source_sigma_int'] = source_sigma
        assert float(cs._full_covariance_source_plane_bin_loglike(**p)) == cs.BAD_LOG_LIKE


def make_evaluator(engine='full_flat', scatter=True, module=cs, vary_lens=False):
    from test_validation import (_minimal_two_bin_local_jacobian_state,
                                 _minimal_two_bin_local_jacobian_state_with_scaling_scatter)
    state = (_minimal_two_bin_local_jacobian_state_with_scaling_scatter() if scatter
             else _minimal_two_bin_local_jacobian_state())
    if vary_lens:
        # Retain the inactive-member parameter for the surrogate cache and add
        # a macro-halo position so timings include differentiation through lensing.
        center_index = len(state.parameter_specs)
        state.parameter_specs.append(replace(
            state.parameter_specs[0], name="halo.x", sample_name="halo_x", field="x_centre",
            component_family="large", lower=.2, upper=.6,
        ))
        centers = state.packed_lens_spec.x_center_param_index.copy()
        centers[0] = center_index
        state.packed_lens_spec = replace(state.packed_lens_spec, x_center_param_index=centers)
    state.parameter_specs.append(module._build_source_scatter_parameter_spec(len(state.parameter_specs)))
    evaluator = module.ClusterJAXEvaluator(state, .1, sampling_engine=engine,
                                          sample_likelihood_mode='source', source_plane_covariance_floor=0.,
                                          source_plane_covariance_mode=('full-jacobian' if module is cs else 'magnification'))
    theta = jnp.asarray(module._default_theta(state.parameter_specs))
    evaluator.refresh_source_metric_cache(theta, reason='covariance-test')
    evaluator.refresh_scaling_scatter_cache(theta, reason='covariance-test')
    if engine == 'refreshing_surrogate_flat':
        evaluator.refresh_surrogate(theta, reason='covariance-test')
    return evaluator, theta


@pytest.mark.parametrize('engine', ['full_flat', 'refreshing_surrogate_flat'])
@pytest.mark.parametrize('scatter', [False, True])
def test_complete_evaluator_and_reported_centroids(engine, scatter):
    evaluator, theta = make_evaluator(engine, scatter)
    value, gradient = jax.jit(jax.value_and_grad(evaluator._source_loglike_fn))(theta)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    data = evaluator.flat_critical_arc_data
    phys = evaluator._physical_parameter_vector(theta)
    vx, vy = evaluator._flat_scaling_scatter_extra_variance_from_physical(phys, data)
    if not scatter:
        np.testing.assert_array_equal(vx, 0.)
        np.testing.assert_array_equal(vy, 0.)
    summaries = evaluator._family_source_summary(theta)
    expected = 0.
    for f, fid in enumerate(data.family_ids):
        summary = summaries[fid]
        mask = np.asarray(data.global_family_index_per_image) == f
        beta = np.stack([summary['source_beta_x'], summary['source_beta_y']], -1)
        covariance = summary['source_covariance_arcsec2']
        precision = np.linalg.inv(covariance)
        center = np.linalg.solve(precision.sum(0), np.einsum('nij,nj->i', precision, beta))
        np.testing.assert_allclose([summary['source_x'], summary['source_y']], center, atol=1e-10)
        residual = beta - center
        quad = np.einsum('ni,nij,nj->n', residual, precision, residual)
        loglike = -.5 * (quad + np.linalg.slogdet(covariance)[1] + 2 * np.log(2 * np.pi))
        out_sigma = evaluator.source_plane_outlier_sigma_arcsec
        out = -.5 * (np.sum(residual**2, -1) / out_sigma**2 + 2 * np.log(2 * np.pi * out_sigma**2))
        rel = np.clip(np.asarray(data.reliability_per_image)[mask], 1e-6, 1-1e-6)
        expected += np.where(np.asarray(data.image_has_constraint)[mask],
                             np.logaddexp(np.log(rel) + loglike, np.log1p(-rel) + out), 0).sum()
    np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(gradient, central_difference(evaluator._source_loglike_fn, np.asarray(theta)), rtol=1e-4, atol=1e-5)


def test_source_delta_matches_fixed_covariance_quadratic_reference():
    p = inputs()
    dx = jnp.stack([jnp.linspace(-.01, .02, 6), jnp.linspace(.02, -.03, 6)])
    dy = jnp.flip(dx, axis=1)
    kwargs = {k: v for k, v in p.items() if k != 'outlier_sigma_arcsec'}
    actual = cs._linearized_full_covariance_source_plane_population_delta(delta_beta_x=dx, delta_beta_y=dy, **kwargs)
    base = reference(p)[3]
    expected = []
    for x, y in zip(dx, dy):
        shifted = dict(p, beta_x=p['beta_x'] + x, beta_y=p['beta_y'] + y)
        expected.append(-.5 * np.sum(np.where(MASK, reference(shifted)[3] - base, 0)))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_empty_family_slot_and_singleton_mask():
    p = inputs(); p['n_families'] = 4
    assert np.isfinite(cs._full_covariance_source_plane_bin_loglike(**p))
    np.testing.assert_allclose(stats(p)[4][3], 0.)
    p['beta_x'] = p['beta_x'].at[-1].set(1e5)
    np.testing.assert_allclose(cs._full_covariance_source_plane_bin_loglike(**p), cs._full_covariance_source_plane_bin_loglike(**inputs()), atol=1e-12)


def test_summary_plot_whitens_full_covariance():
    from lenscluster import plotting
    from lenscluster.model import EvaluationResult
    evaluator, theta = make_evaluator()
    summaries = evaluator._family_source_summary(theta)
    output = plotting._source_plane_residual_components(
        evaluator.state, EvaluationResult(loglike=0., family_predictions=summaries),
    )
    expected = []
    for family in evaluator.state.family_data:
        p = summaries[family.family_id]
        r = np.stack([p['source_beta_x'] - p['source_x'], p['source_beta_y'] - p['source_y']], -1)
        expected.append(np.linalg.solve(np.linalg.cholesky(p['source_covariance_arcsec2']), r[..., None])[..., 0])
    expected = np.concatenate(expected)
    np.testing.assert_allclose(output[2], expected[:, 0])
    np.testing.assert_allclose(output[3], expected[:, 1])
    np.testing.assert_allclose(output[4], np.linalg.norm(expected, axis=-1))


def test_stage0_zeros_member_variance_even_with_scatter_parameter():
    from test_validation import _minimal_two_bin_local_jacobian_state_with_scaling_scatter
    state = replace(_minimal_two_bin_local_jacobian_state_with_scaling_scatter(), perturbation_discovery_stage0=True)
    state.parameter_specs.append(cs._build_source_scatter_parameter_spec(len(state.parameter_specs)))
    evaluator = cs.ClusterJAXEvaluator(state, .1, sample_likelihood_mode='source', source_plane_covariance_mode='full-jacobian')
    theta = jnp.asarray(cs._default_theta(state.parameter_specs))
    evaluator.refresh_scaling_scatter_cache(theta)
    vx, vy = evaluator._flat_scaling_scatter_extra_variance_from_physical(
        evaluator._physical_parameter_vector(theta), evaluator.flat_critical_arc_data,
    )
    np.testing.assert_array_equal(vx, 0.)
    np.testing.assert_array_equal(vy, 0.)
    assert np.isfinite(jax.jit(jax.value_and_grad(evaluator._source_loglike_fn))(theta)[1]).all()


def test_active_population_diagnostic_uses_cached_jacobian(monkeypatch):
    # Exercise the existing diagnostic with isolated candidate-provider hooks.
    # This does not claim an end-to-end active-population inference run.
    evaluator, theta = make_evaluator(scatter=False)
    data = evaluator.flat_critical_arc_data
    bx, by = data.x_obs * .1, data.y_obs * .1
    shift = jnp.linspace(-.02, .01, len(bx))
    monkeypatch.setattr(evaluator, '_flat_active_population_candidate_indices', lambda: jnp.array([0]), raising=False)
    monkeypatch.setattr(evaluator, '_active_scaling_probability_from_physical', lambda _: jnp.array([.5]), raising=False)
    monkeypatch.setattr(evaluator, '_flat_active_population_gate_probabilities', lambda *_: jnp.array([.5]), raising=False)
    monkeypatch.setattr(evaluator, '_flat_active_population_exact_state',
                        lambda *a, **kw: (bx, by, None, jnp.array(False), None), raising=False)
    monkeypatch.setattr(evaluator, '_flat_active_population_alpha_replacement',
                        lambda *a: ((bx + shift)[None, :], by[None, :]), raising=False)
    metric = replace(evaluator.flat_source_metric_cache,
                     jac_a00=jnp.ones_like(bx), jac_a01=jnp.zeros_like(bx),
                     jac_a10=jnp.zeros_like(bx), jac_a11=jnp.ones_like(bx))
    evaluator.flat_source_metric_cache = metric
    result = evaluator._flat_active_population_diagnostics_impl(theta)
    evaluator.flat_source_metric_cache = replace(metric, inv_abs_mu=jnp.full_like(bx, 1e5))
    unchanged = evaluator._flat_active_population_diagnostics_impl(theta)
    np.testing.assert_allclose(result[2], unchanged[2], atol=1e-12)
    evaluator.flat_source_metric_cache = replace(metric, jac_a01=metric.jac_a01 + .5)
    changed = evaluator._flat_active_population_diagnostics_impl(theta)
    assert not np.allclose(result[2], changed[2])


@pytest.mark.parametrize('engine', ['full_flat', 'refreshing_surrogate_flat'])
def test_complete_evaluator_differentiates_lens_parameter(engine):
    evaluator, theta = make_evaluator(engine, scatter=False, vary_lens=True)
    fn = evaluator._source_loglike_fn
    gradient = jax.jit(jax.grad(fn))(theta)
    assert abs(float(gradient[-2])) > 1e-8
    # The installed NIE Hessian internally differences positions at 1e-10 arcsec.
    # A 0.01-arcsec external step avoids that backend's cancellation noise;
    # strict general covariance derivative checks above do not use this backend.
    fd = central_difference(fn, np.asarray(theta))
    shift = jnp.zeros_like(theta).at[-2].set(.01)
    fd[-2] = float((fn(theta + shift) - fn(theta - shift)) / .02)
    np.testing.assert_allclose(gradient, fd, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("mode", ["magnification", "unit"])
def test_legacy_modes_accept_original_configuration(mode):
    from test_validation import _minimal_stage4_surrogate_state
    evaluator = cs.ClusterJAXEvaluator(
        _minimal_stage4_surrogate_state(), .1, sample_likelihood_mode="source",
        source_plane_covariance_mode=mode, likelihood_stabilizer_max_gain=50.)
    assert evaluator.source_plane_covariance_floor == 1e-6
    assert not evaluator._full_source_covariance_enabled()
    p = inputs()
    entries = [p.pop(key) for key in ("jac_a00", "jac_a01", "jac_a10", "jac_a11")]
    p["source_sigma_int"] = 0.
    p["covariance_floor"] = 1e-6
    p["max_gain"] = 50.
    inv_mu = (jnp.ones_like(entries[0]) if mode == "unit"
              else jnp.clip(jnp.abs(entries[0]*entries[3]-entries[1]*entries[2]), 1e-6, 1e6))
    expected = cs._source_plane_bin_loglike(**p, inv_abs_mu=inv_mu)
    actual = evaluator._source_plane_loglike_for_covariance(
        **p, **dict(zip(("jac_a00", "jac_a01", "jac_a10", "jac_a11"), entries)))
    np.testing.assert_allclose(actual, expected, rtol=0., atol=0.)
