"""Failed refreshes preserve scatter contributions and report retention honestly."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from test_validation import _minimal_two_bin_local_jacobian_state_with_scaling_scatter


def evaluator(engine="refreshing_surrogate_flat"):
    state = _minimal_two_bin_local_jacobian_state_with_scaling_scatter()
    state.parameter_specs[0] = replace(state.parameter_specs[0], lower=80., upper=160.)
    indices = state.packed_lens_spec.sigma_ref_param_index.copy()
    indices[1] = 0
    state.packed_lens_spec = replace(state.packed_lens_spec, sigma_ref_param_index=indices)
    ev = cs.ClusterJAXEvaluator(state, .1, sampling_engine=engine,
                               sample_likelihood_mode='source', source_plane_covariance_mode='unit')
    theta = cs._default_theta(state.parameter_specs)
    if ev.surrogate_enabled:
        ev.refresh_surrogate(theta)
    ev.refresh_scaling_scatter_cache(theta)
    return ev, theta


@pytest.mark.parametrize('flat', [True, False])
@pytest.mark.parametrize('failure', ['invalid_reference', 'nonfinite_covariance'])
def test_scaling_cache_retention(flat, failure, monkeypatch):
    ev, theta = evaluator()
    value_grad = jax.jit(jax.value_and_grad(ev._source_loglike_fn))
    expected = value_grad(jnp.asarray(theta))
    old = (ev.scaling_scatter_cache_by_z, ev.scaling_scatter_reference_params,
           ev.flat_critical_arc_data, ev._source_loglike_fn)
    assert np.any(np.asarray(old[2].sigma_scatter_cov00) > 0)
    before = cs._svi_refresh_evaluator_snapshot(ev)
    if not flat:
        ev.sampling_engine = 'per_bin_test'
    candidate = np.full_like(theta, np.nan) if failure == 'invalid_reference' else theta
    with monkeypatch.context() as patch:
        if failure == 'nonfinite_covariance':
            original = cs._scatter_covariance_terms_from_member_derivatives
            patch.setattr(cs, '_scatter_covariance_terms_from_member_derivatives',
                          lambda *a: tuple(np.full_like(x, np.nan) for x in original(*a)))
        for _ in range(2):
            with pytest.warns(RuntimeWarning, match='Retaining the previous scaling-scatter cache'):
                ev.refresh_scaling_scatter_cache(candidate, reason='regression')
            current = (ev.scaling_scatter_cache_by_z, ev.scaling_scatter_reference_params,
                       ev.flat_critical_arc_data, ev._source_loglike_fn)
            assert all(a is b for a, b in zip(old, current))
            after = cs._svi_refresh_evaluator_snapshot(ev)
            rows = cs._svi_refresh_cache_status_rows(
                {k: cs._svi_refresh_cache_delta(before[k], after[k]) for k in before}, after)
            assert next(r for r in rows if r.name == 'scaling').status == 'retained'
            assert not any(r.name == 'scaling_structure' for r in rows)
            before = after
    ev.sampling_engine = 'refreshing_surrogate_flat'
    actual = jax.jit(jax.value_and_grad(ev._source_loglike_fn))(jnp.asarray(theta))
    for a, b in zip(actual, expected):
        assert np.isfinite(a).all()
        np.testing.assert_array_equal(a, b)
    ev.refresh_scaling_scatter_cache(theta)
    assert not cs._svi_refresh_evaluator_snapshot(ev)['scaling'].refresh_failed


@pytest.mark.parametrize('flat', [True, False])
def test_scaling_cache_retention_requires_valid_initial_cache(flat):
    ev, theta = evaluator()
    ev.scaling_scatter_cache_by_z = {}
    ev.scaling_scatter_reference_params = None
    if not flat:
        ev.sampling_engine = 'per_bin_test'
    with pytest.raises(ValueError, match='No valid previous cache.*initialization'):
        ev.refresh_scaling_scatter_cache(np.full_like(theta, np.nan))


def test_pre_nuts_retained_cache_message():
    ev, theta = evaluator()
    with pytest.warns(RuntimeWarning, match='Retaining the previous scaling-scatter cache'):
        message = cs._refresh_pre_nuts_caches(ev, np.full_like(theta, np.nan), 'pre_nuts')
    assert 'scaling=retained' in message
    assert 'freshly refreshed' not in message


def test_scaling_cache_retention_invalid_member_probe(monkeypatch):
    ev, theta = evaluator()
    old = (ev.scaling_scatter_cache_by_z, ev.scaling_scatter_reference_params,
           ev.flat_critical_arc_data, ev._source_loglike_fn)
    original = ev._build_flat_packed_lens_state_with_validity_from_physical

    def invalid_probe(*args, **kwargs):
        packed, validity = original(*args, **kwargs)
        if 'sigma_log_offset' in kwargs:
            validity = cs.PackedLensValidity(is_valid=jnp.asarray(False), reason_flags=validity.reason_flags)
        return packed, validity

    monkeypatch.setattr(ev, '_build_flat_packed_lens_state_with_validity_from_physical', invalid_probe)
    with pytest.warns(RuntimeWarning, match='Retaining the previous scaling-scatter cache'):
        ev.refresh_scaling_scatter_cache(theta)
    current = (ev.scaling_scatter_cache_by_z, ev.scaling_scatter_reference_params,
               ev.flat_critical_arc_data, ev._source_loglike_fn)
    assert all(a is b for a, b in zip(old, current))
