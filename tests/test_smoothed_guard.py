"""Safe-reference rejection and posterior curvature regression checks."""
from dataclasses import replace
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from lenscluster.config import ShearHaloConfig, PriorConfig
from test_smoothed_integration import make_evaluator, small_config
from test_smoothed_batching import original_exact_rows, assert_tree_close


def conditional_reference(ev, theta):
    """The previous two rejection guards, independent of the select path."""
    physical = ev._physical_parameter_vector(theta)
    _, validity = ev._smoothed_packed_with_validity(physical)
    scales = physical[ev._smoothed_scale_indices]
    def inner(theta):
        safe = jnp.all(jnp.isfinite(scales**2) & (scales > 0.))
        return jax.lax.cond(safe, ev._smoothed_loglike_value, lambda _: jnp.asarray(-jnp.inf), theta)
    return jax.lax.cond(validity.is_valid & jnp.all(jnp.isfinite(theta)), inner,
                        lambda _: jnp.asarray(cs.BAD_LOG_LIKE), theta)


@pytest.mark.parametrize('parameterization', ['direct', 'likelihood-whitened'])
def test_guard_preserves_values_gradients_and_hessian(tmp_path, monkeypatch, parameterization):
    _, ev, theta = make_evaluator(tmp_path, parameterization=parameterization)
    reference = np.asarray(ev._smoothed_guard_reference).copy()
    points = jnp.asarray(theta + np.random.default_rng(519).normal(size=(3, len(theta)))*.01)
    actual = jax.jit(jax.vmap(jax.value_and_grad(ev._source_loglike_impl)))(points)
    actual_hessian = jax.jit(jax.hessian(ev._source_loglike_impl))(points[0])
    # Restore JAXtronomy + spatial autodiff, and both previous cond guards.
    with monkeypatch.context() as patch:
        patch.setattr(ev, '_smoothed_exact_rows', MethodType(original_exact_rows, ev))
        fn = lambda x: conditional_reference(ev, x)
        expected = jax.jit(jax.vmap(jax.value_and_grad(fn)))(points)
        expected_hessian = jax.jit(jax.hessian(fn))(points[0])
    assert_tree_close(actual, expected)
    np.testing.assert_allclose(actual_hessian, expected_hessian, rtol=2e-10, atol=2e-10)
    np.testing.assert_array_equal(ev._smoothed_guard_reference, reference)


def test_guard_rejected_proposals_have_same_values_and_zero_gradients(tmp_path):
    config = small_config(tmp_path)
    halo = config.model.large_halos[0]
    halo = replace(halo, priors={**halo.priors,
        'ellipticite': PriorConfig('uniform', lower=0., upper=.8),
        'angle_pos': PriorConfig('uniform', lower=-90., upper=90.)})
    config = replace(config, model=replace(config.model, large_halos=(halo,)))
    _, ev, theta = make_evaluator(tmp_path, config=config)
    theta = jnp.asarray(theta)
    points = [theta.at[i].set(value) for value in (jnp.nan, jnp.inf, -jnp.inf) for i in range(len(theta))]
    points += [theta.at[i].set(value) for i in np.asarray(ev._smoothed_scale_indices)
               for value in (-1000., 1000.)]
    shape_indices = [i for i, spec in enumerate(ev.state.parameter_specs) if spec.field in ('e1', 'e2')]
    assert shape_indices
    points += [theta.at[i].set(50.) for i in shape_indices]
    fast = jax.jit(jax.value_and_grad(ev._source_loglike_impl))
    slow = jax.jit(jax.value_and_grad(lambda x: conditional_reference(ev, x)))
    for point in points:
        av, ag = fast(point)
        bv, bg = slow(point)
        np.testing.assert_array_equal(av, bv)
        np.testing.assert_array_equal(ag, bg)
        np.testing.assert_array_equal(ag, np.zeros(len(theta)))


def test_guard_rejects_invalid_reference_instead_of_installing_it(tmp_path):
    _, ev, _ = make_evaluator(tmp_path)
    reference = np.asarray(ev._smoothed_guard_reference).copy()
    index = ev.state.smoothed_model['sigma_src_index']
    ev.state.svi_init_values = {ev.state.parameter_specs[index].sample_name: 1000.}
    with pytest.raises(ValueError, match='safe-reference guard'):
        ev._initialize_smoothed_guard_reference()
    np.testing.assert_array_equal(ev._smoothed_guard_reference, reference)
    # Finite likelihood but unusable derivative must also fail validation.
    ev.state.svi_init_values = {}
    ev._smoothed_loglike_value = lambda x: jnp.sqrt(x[0]-jax.lax.stop_gradient(x[0]))
    with pytest.raises(ValueError, match='safe-reference guard'):
        ev._initialize_smoothed_guard_reference()


def test_exact_shear_and_dpie_batches_match_spatial_autodiff(tmp_path):
    config = small_config(tmp_path)
    config = replace(config, model=replace(config.model, large_halos=config.model.large_halos + (
        ShearHaloConfig(id='shear', gamma=.03, angle_pos=27., z_lens=.3),)))
    _, ev, theta = make_evaluator(tmp_path, config=config)
    data = ev.flat_critical_arc_data
    xy = jnp.stack((data.x_obs, data.y_obs), -1)
    packed = ev._smoothed_packed(ev._physical_parameter_vector(jnp.asarray(theta)))
    indices = np.arange(len(ev.state.base_components))[::-1]
    assert_tree_close(ev._smoothed_exact_rows(xy, packed, indices, spatial_derivative=True),
                      original_exact_rows(ev, xy, packed, indices, spatial_derivative=True))
