"""Independent high-precision references for the two-sided determinant moment."""
from dataclasses import replace
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import smoothed_likelihood as sl
from test_smoothed_integration import make_evaluator, small_config

mp = pytest.importorskip('mpmath')


def normal_reference(mean, variance):
    with mp.workdps(90):
        m, v = mp.mpf(float(mean)), mp.mpf(float(variance))
        sd = mp.sqrt(v)
        z = m/sd
        phi = mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
        e = mp.erf(z/mp.sqrt(2))
        h = 2*sd*phi + m*e
        gm, gv = e/h, phi/(sd*h)
        return np.array([float(mp.log(h)), float(gm), float(gv)])


def test_closed_form_values_and_derivatives_against_90_digit_reference(record_property):
    cases = [(s*z, s*s) for s in (1e-150, 1e-100, 1e-10, 1., 1e10, 1e100, 1e150)
             for z in (-1e8, -128.01, -128., -40., -38., -37.5, -30., -8.00001, -8.,
                       -7.99999, -1., 0., 1., 7.99999, 8., 8.00001, 30., 37.5, 38., 40., 128., 128.01, 1e8)]
    cases += [(1e200, 1e-300), (-1e200, 1e-300), (1e300, 1e300), (-1e300, 1e300)]
    mean, variance = jnp.asarray(cases).T
    value, grad = jax.jit(jax.vmap(jax.value_and_grad(
        sl.log_expected_absolute_normal, argnums=(0, 1))))(mean, variance)
    actual = np.stack((value, grad[0], grad[1]), -1)
    expected = np.stack([normal_reference(m,v) for m,v in cases])
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    record_property('value_max_absolute_error', float(np.max(np.abs(actual[:,0]-expected[:,0]))))
    record_property('gradient_max_scaled_error', float(np.max(
        np.abs(actual[:,1:]-expected[:,1:])/(1+np.abs(expected[:,1:])))))
    # Relative error with an absolute allowance near vanishing derivatives.
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=2e-13)


def test_tail_switch_and_log_variance_hessians():
    def transformed(q):
        variance = jnp.exp(q[1])
        return sl.log_expected_absolute_normal(q[0]*jnp.sqrt(variance), variance)
    cases = jnp.array([(z, np.log(v)) for v in (1e-300, 1e-20, 1., 1e20, 1e300)
                      for z in (-128.01, -40., -8.00001, -8., -7.99999, 0., 7.99999, 8., 8.00001, 40., 128.01)])
    value, grad = jax.jit(jax.vmap(jax.value_and_grad(transformed)))(cases)
    hessian = np.asarray(jax.jit(jax.vmap(jax.hessian(transformed)))(cases))
    expected_g, expected_h = [], []
    with mp.workdps(90):
        for z, logv in np.asarray(cases):
            z = mp.mpf(z)
            phi = mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
            e = mp.erf(z/mp.sqrt(2))
            g = 2*phi + z*e
            expected_g.append([float(e/g), .5])
            expected_h.append([[float(2*phi/g - (e/g)**2), 0.], [0., 0.]])
    assert np.isfinite(value).all() and np.isfinite(grad).all() and np.isfinite(hessian).all()
    np.testing.assert_allclose(grad, expected_g, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(hessian, expected_h, rtol=3e-11, atol=2e-13)


def test_tiny_log_moment_is_retained_when_absolute_moment_rounds_to_one():
    variance = np.array([1/z**2 for z in (8.0001, 9., 15., 25., 30., 35.)])
    actual = jax.jit(sl.log_expected_absolute_normal)(jnp.ones(6), jnp.asarray(variance))
    expected = []
    with mp.workdps(90):
        for v in variance:
            z = 1/mp.sqrt(mp.mpf(float(v)))
            phi = mp.exp(-z*z/2)/mp.sqrt(2*mp.pi)
            expected.append(float(mp.log1p(2*phi/z - mp.erfc(z/mp.sqrt(2)))))
    assert np.all(np.asarray(actual) > 0.)
    np.testing.assert_allclose(actual, expected, rtol=5e-10, atol=0.)


def test_unknown_parity_skips_one_sided_quadrature(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('unknown parity evaluated the one-sided moment')
    monkeypatch.setattr(sl, '_log_positive_normal_moment', forbidden)
    args, evaluator, theta = make_evaluator(tmp_path)
    assert evaluator._smoothed_likelihood_parity is None
    value, gradient = jax.jit(jax.value_and_grad(evaluator._source_loglike_impl))(theta)
    assert np.isfinite(value) and np.isfinite(gradient).all()


def test_mixed_parity_uses_closed_form_without_contaminating_known_rows():
    mean = jnp.array([1e200, -1e200, 0., -8., 8., 1., -1.])
    variance = jnp.array([1e-300, 1e-300, 0., 1., 1., 0., 0.])
    parity = jnp.array([0, 0, 0, 1, -1, 1, 1])
    fn = lambda m, v: sl.log_expected_absolute_normal(m, v, parity)
    actual = jax.jit(fn)(mean, variance)
    value, gradient = jax.jit(jax.value_and_grad(lambda m: jnp.sum(fn(m, variance))))(mean)
    # Degenerate and impossible deterministic branches retain -inf.
    assert np.isneginf(actual[2]) and np.isneginf(actual[-1])
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(actual[:2], np.log(np.abs(mean[:2])), rtol=1e-15)
    with mp.workdps(90):
        phi = mp.exp(-32)/mp.sqrt(2*mp.pi)
        probability = mp.erfc(8/mp.sqrt(2))/2
        branch_h = phi - 8*probability
        reference_log = float(mp.log(branch_h))
        reference_grad = float(probability/branch_h)
    np.testing.assert_allclose(actual[3:5], reference_log, rtol=2e-14, atol=4e-12)
    np.testing.assert_allclose(gradient[3:5], [reference_grad, -reference_grad], rtol=2e-12)


def test_zero_parity_array_matches_static_path_including_extreme_inputs():
    mean = jnp.array([[0., -8., 8.], [1e200, -1e200, -1.]])
    variance = jnp.array([[.09, 1., 1.], [1e-300, 1e-300, 0.]])
    known_zero = jnp.zeros((2, 3), dtype=int)
    outputs = []
    for parity in (None, 0, known_zero):
        fn = lambda m,v: jnp.sum(sl.log_expected_absolute_normal(m,v,parity))
        outputs.append(jax.jit(jax.value_and_grad(fn, argnums=(0,1)))(mean, variance))
    for value in outputs[1:]:
        for a,b in zip(jax.tree.leaves(value), jax.tree.leaves(outputs[0])):
            np.testing.assert_allclose(a,b,rtol=2e-14,atol=2e-14)


def test_supplied_parity_metadata_keeps_restricted_likelihood(tmp_path):
    config = small_config(tmp_path)
    metadata = tmp_path/'parity.csv'
    metadata.write_text('image_label,parity\n1.a,1\n2.b,-1\n')
    config = replace(config, model=replace(config.model, image_constraints=replace(
        config.model.image_constraints, metadata_path=str(metadata))))
    _, ev, theta = make_evaluator(tmp_path, config=config)
    assert ev._smoothed_likelihood_parity is not None
    np.testing.assert_array_equal(ev._smoothed_likelihood_parity, ev.smoothed_parity)
    value, gradient = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(theta)
    assert np.isfinite(value) and np.isfinite(gradient).all()
