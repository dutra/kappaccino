"""Cached member responses preserve the local Gaussian and independent scatter."""
import copy
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from lenscluster import cluster_solver as cs
from test_latent_displacement import fixture, MODE

@pytest.fixture(scope='module')
def pair():
    exact,theta=fixture()
    cached=cs.ClusterJAXEvaluator(exact.state,.1,sampling_engine='refreshing_surrogate_flat',
        sample_likelihood_mode=MODE,latent_displacement_sigma_beta_arcsec=.03)
    cached.refresh_surrogate(theta)
    return exact,cached,jnp.asarray(theta)


def test_reference_values_and_gradients(pair):
    exact,cached,theta=pair
    assert set(cached.exact_scaling_component_indices)==set(cached.free_correction_scaling_component_indices)
    assert not set(cached.cached_scaling_component_indices)&set(cached.active_component_indices)
    a,b=exact._latent_displacement_statistics(theta),cached._latent_displacement_statistics(theta)
    for key in ['backprojected','jacobian','member_covariance','covariance','loglike']:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-9)
    for x,y in zip(jax.jit(jax.value_and_grad(exact._source_loglike_impl))(theta),
                   jax.jit(jax.value_and_grad(cached._source_loglike_impl))(theta)):
        np.testing.assert_allclose(x,y,rtol=1e-7,atol=1e-8)


def test_response_gram_off_reference_and_no_exact_calls(pair,monkeypatch):
    _,cached,theta=pair
    ev=copy.copy(cached)
    theta=theta.at[ev.surrogate_param_indices[0]].add(.2)
    cache=ev.flat_surrogate_cache
    response=cache.member_responses+jnp.tensordot(cache.member_response_derivatives,
        ev._surrogate_parameter_delta(theta),axes=([-1],[0]))
    cov=ev._latent_cached_member_covariance(theta)
    for i,name in enumerate(['sigma','mass']):
        expected=np.einsum('gix,giy->ixy',np.asarray(response[:,:,:,i]),np.asarray(response[:,:,:,i]))
        np.testing.assert_allclose(cov[name+'_cov00'],expected[:,0,0])
        np.testing.assert_allclose(cov[name+'_cov01'],expected[:,0,1])
        assert np.linalg.eigvalsh(expected).min()>=-1e-8
    def fail(*a,**kw): raise AssertionError('exact member scatter called in cached likelihood')
    monkeypatch.setattr(ev,'_direct_scaling_scatter_responses',fail)
    assert np.isfinite(ev._source_loglike_impl(theta))
    before=ev.export_latent_surrogate_cache()
    ev.refresh_scaling_scatter_cache(theta)
    ev.refresh_source_metric_cache(theta)
    after=ev.export_latent_surrogate_cache()
    for key in before['arrays']: np.testing.assert_array_equal(before['arrays'][key],after['arrays'][key])


def test_roundtrip_rejection_and_exact_validation(pair):
    exact,cached,theta=pair
    saved=cached.export_latent_surrogate_cache()
    state=copy.copy(cached.state)
    state.production_latent_surrogate_cache=saved
    ev=cs.ClusterJAXEvaluator(state,.1,sampling_engine='refreshing_surrogate_flat',
        sample_likelihood_mode=MODE,latent_displacement_sigma_beta_arcsec=.03)
    np.testing.assert_allclose(ev._source_loglike_impl(theta),cached._source_loglike_impl(theta))
    bad=copy.deepcopy(saved);bad['metadata']['version']=999
    with pytest.raises(ValueError,match='version'): ev.restore_latent_surrogate_cache(bad)
    validation=cs._output_evaluator_for_validation(None,state,ev,theta)
    assert validation is not ev and validation.sampling_engine=='full_flat'
    np.testing.assert_allclose(validation._source_loglike_impl(theta),exact._source_loglike_impl(theta))
    assert ev.flat_surrogate_cache is not None


def test_atomic_failure_retains_all_fields(pair,monkeypatch):
    _,cached,theta=pair
    ev=copy.copy(cached)
    ev._cache_refresh_failures={}
    old=ev.flat_surrogate_cache;ref=ev.surrogate_reference_params.copy()
    def bad(physical,data):
        return jnp.full(old.member_responses.shape,jnp.nan)
    monkeypatch.setattr(ev,'_direct_scaling_scatter_responses',bad)
    with pytest.warns(RuntimeWarning,match='Retaining'): ev.refresh_surrogate(theta)
    assert ev.flat_surrogate_cache is old
    np.testing.assert_array_equal(ev.surrogate_reference_params,ref)
    ev.flat_surrogate_cache=None;ev.surrogate_reference_params=None
    with pytest.raises(ValueError,match='No valid previous'): ev.refresh_surrogate(theta)


def test_nuts_freeze_guard_covers_warmup_and_restores_on_failure(pair):
    _,cached,theta=pair
    ev=copy.copy(cached)
    @cs._freeze_latent_surrogate_during_sampling
    def pretend_sampler(args,state,evaluator):
        assert evaluator._latent_surrogate_frozen
        evaluator.refresh_surrogate(theta,reason='inside_warmup')
    with pytest.raises(RuntimeError,match='NUTS run'):
        pretend_sampler(None,ev.state,ev)
    assert not ev._latent_surrogate_frozen


def test_empty_population_uses_exact_fallback(pair):
    exact,cached,theta=pair
    ev=copy.copy(cached)
    # Simulate a model for which the partition has no cached members.
    ev.surrogate_enabled=False
    ev.flat_surrogate_cache=None
    ev.cached_scaling_component_indices=np.empty(0,dtype=np.int32)
    ref=copy.copy(exact);ref.cached_scaling_component_indices=np.empty(0,dtype=np.int32)
    np.testing.assert_allclose(ev._source_loglike_impl(theta),ref._source_loglike_impl(theta))
    assert not ev.export_latent_surrogate_cache()['arrays']


def test_sampled_sigma_beta_remains_exact():
    base, _ = fixture()
    base.state.parameter_specs.append(cs._build_latent_displacement_sigma_beta_spec(.03, .7))
    ev = cs.ClusterJAXEvaluator(base.state, .1, sampling_engine='refreshing_surrogate_flat',
        sample_likelihood_mode=MODE, latent_displacement_sigma_beta_arcsec=.03)
    theta = jnp.asarray(cs._default_theta(base.state.parameter_specs))
    ev.refresh_surrogate(theta)
    index = ev.latent_displacement_sigma_beta_param_index
    assert index not in ev.surrogate_param_indices
    before = ev.flat_surrogate_cache
    exact = cs._output_evaluator_for_validation(None, ev.state, ev, theta)
    for shift in [0., .7]:
        point = theta.at[index].add(shift)
        np.testing.assert_allclose(ev._source_loglike_impl(point), exact._source_loglike_impl(point), atol=1e-9)
    gradient = jax.jit(jax.grad(ev._source_loglike_impl))(theta)
    step = 1e-4
    finite_difference = (ev._source_loglike_impl(theta.at[index].add(step))
                         - ev._source_loglike_impl(theta.at[index].add(-step))) / (2*step)
    np.testing.assert_allclose(gradient[index], finite_difference, rtol=1e-4, atol=1e-8)
    assert gradient[index] != 0 and ev.flat_surrogate_cache is before
