from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from test_validation import _minimal_stage4_surrogate_state


def evaluator():
    state = _minimal_stage4_surrogate_state()
    # Connect the selected parameter to a real cached halo, including a circular
    # shape: its polar angle has no derivative but its mass derivative does.
    state.parameter_specs[0] = replace(state.parameter_specs[0], lower=80., upper=160.)
    indices = state.packed_lens_spec.sigma_ref_param_index.copy()
    indices[1] = 0
    state.packed_lens_spec = replace(state.packed_lens_spec, sigma_ref_param_index=indices)
    ev = cs.ClusterJAXEvaluator(state, .1, sampling_engine='refreshing_surrogate_flat',
                               sample_likelihood_mode='source', source_plane_covariance_mode='unit')
    return ev, cs._default_theta(state.parameter_specs)


@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('with_jacobian', [False, True])
def test_surrogate_autodiff_matches_finite_difference(flat, with_jacobian):
    ev, theta = evaluator()
    ref = jnp.asarray(theta)
    selected = ref[ev.surrogate_param_indices]
    if flat:
        result = ev._build_flat_inactive_surrogate_jacobian(theta, include_jacobian=with_jacobian)
        fn = lambda p: ev._flat_inactive_surrogate_concat(p, ref, include_jacobian=with_jacobian)
    else:
        data = ev.traced_bin_data[0]
        args = (data.effective_z_source, jnp.asarray(data.x_obs), jnp.asarray(data.y_obs))
        result = ev._build_inactive_surrogate_jacobian(theta, *args, include_jacobian=with_jacobian)
        fn = lambda p: ev._inactive_surrogate_concat(p, ref, *args, include_jacobian=with_jacobian)
    assert result is not None
    keys = ['inactive_alpha_dx_dparams', 'inactive_alpha_dy_dparams']
    if with_jacobian:
        keys += ['inactive_jacobian_delta_d'+a+'_dparams' for a in ('a00','a01','a10','a11')]
    derivative = np.concatenate([result[k].T for k in keys], axis=0)
    assert np.any(np.abs(derivative) > 0)
    # The installed NIE Hessian uses spatial finite differences, so use a
    # macroscopic parameter step to avoid amplification of its roundoff.
    step = 1.
    fd = np.stack([(np.asarray(fn(selected + step*jnp.eye(len(selected))[i]))
                    - np.asarray(fn(selected - step*jnp.eye(len(selected))[i]))) / (2*step)
                   for i in range(len(selected))], axis=-1)
    np.testing.assert_allclose(derivative, fd, rtol=2e-3, atol=2e-5)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(fn))(selected), derivative, rtol=1e-8, atol=1e-7)
    # The cached affine surrogate, including Jacobian blocks, must behave
    # identically under JIT, reverse-mode differentiation, and batching.
    base = fn(selected)
    cached = lambda p: base + jnp.asarray(derivative) @ (p-selected)
    np.testing.assert_allclose(jax.jit(jax.grad(lambda p: jnp.sum(cached(p))))(selected),
                               derivative.sum(0), rtol=1e-12, atol=1e-10)
    batch = jnp.stack([selected, selected+.01])
    np.testing.assert_allclose(jax.jit(jax.vmap(cached))(batch),
                               jnp.stack([cached(x) for x in batch]), rtol=1e-12, atol=1e-10)



@pytest.mark.parametrize('flat', [False, True])
@pytest.mark.parametrize('failure', ['invalid', 'nonfinite', 'nonfinite_derivative'])
def test_failed_refresh_preserves_previous_cache(flat, failure, monkeypatch):
    ev, theta = evaluator()
    ev.refresh_surrogate(theta)
    ev.refresh_source_metric_cache(theta)
    ev.refresh_scaling_scatter_cache(theta)
    value_grad = jax.jit(jax.value_and_grad(ev._source_loglike_fn))
    before = value_grad(jnp.asarray(theta))
    old_fn = ev._source_loglike_fn
    old = (ev.flat_surrogate_cache, ev.surrogate_cache_by_z, ev.surrogate_reference_params,
           ev.surrogate_reference_param_values, ev.full_refresh_count)
    if not flat:
        ev.sampling_engine = 'full_flat'  # Exercise the retained per-bin refresh path.
    if failure in {'nonfinite', 'nonfinite_derivative'}:
        name = '_flat_inactive_surrogate_concat' if flat else '_inactive_surrogate_concat'
        original = getattr(ev, name)
        proposed = theta + .1
        if failure == 'nonfinite':
            monkeypatch.setattr(ev, name, lambda *a, **kw: original(*a, **kw) * jnp.nan)
        else:
            # Finite primal, undefined tangent: must reject, not fill with zeros.
            monkeypatch.setattr(ev, name, lambda *a, **kw: original(*a, **kw)
                                + jnp.sqrt(jnp.sum((a[0]-jnp.asarray(proposed))**2)))
    else:
        proposed = np.full_like(theta, np.nan)
    with pytest.warns(RuntimeWarning, match='Retaining the previous'):
        ev.refresh_surrogate(proposed, reason='regression')
    assert ev.flat_surrogate_cache is old[0]
    assert ev.surrogate_cache_by_z is old[1]
    assert ev.surrogate_reference_params is old[2]
    assert ev.surrogate_reference_param_values is old[3]
    assert ev.full_refresh_count == old[4]
    assert ev._source_loglike_fn is old_fn
    after = value_grad(jnp.asarray(theta))
    for a,b in zip(before,after):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('flat', [False, True])
def test_initial_refresh_failure_is_explicit(flat):
    ev, theta = evaluator()
    if not flat:
        ev.sampling_engine = 'full_flat'
    with pytest.raises(ValueError, match='No valid previous cache'):
        ev.refresh_surrogate(np.full_like(theta, np.nan))
    assert ev.flat_surrogate_cache is None
    assert ev.surrogate_reference_params is None


def test_successful_refresh_retraces_and_moves_reference():
    ev, theta = evaluator()
    ev.refresh_surrogate(theta)
    old_fn, old_cache, count = ev._source_loglike_fn, ev.flat_surrogate_cache, ev.full_refresh_count
    new_theta = theta + 1.
    ev.refresh_surrogate(new_theta)
    assert ev._source_loglike_fn is not old_fn
    assert ev.flat_surrogate_cache is not old_cache
    assert ev.full_refresh_count == count + 1
    np.testing.assert_array_equal(ev.surrogate_reference_params, new_theta)
    assert not np.array_equal(ev.flat_surrogate_cache.inactive_alpha_x, old_cache.inactive_alpha_x)
