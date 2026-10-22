"""The smoothed engine is independent of the legacy evaluator and its state."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np

from lenscluster import cluster_solver as cs
from lenscluster.member_tiers import classify_member_tiers
from lenscluster.tiered_evaluator import TieredMembersEvaluator
from test_member_tiers import references
from test_smoothed_integration import make_evaluator, small_config


def test_factory_and_direct_constructor_skip_legacy_initialization(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('Legacy evaluator initialized or evaluated')
    for name in ('__init__', '_build_scaling_rank_diagnostics', '_source_loglike_impl',
                 '_flat_source_loglike_impl', '_flat_critical_arc_source_loglike_impl',
                 '_prepare_packed_spec_arrays'):
        monkeypatch.setattr(cs.ClusterJAXEvaluator, name, forbidden)
    args, ev, theta = make_evaluator(tmp_path)
    assert isinstance(ev,TieredMembersEvaluator)
    assert not isinstance(ev,cs.ClusterJAXEvaluator)
    assert not hasattr(ev,'_flat_source_loglike_impl')
    assert not hasattr(ev,'surrogate_cache_by_z')
    assert not hasattr(ev,'scaling_scatter_cache_by_z')
    assert not hasattr(ev.flat_critical_arc_data,'mass_scatter_cov00')
    assert not hasattr(ev.flat_critical_arc_data,'magnitude_per_image')
    direct=cs.ClusterJAXEvaluator(ev.state,.1,sampling_engine='full_flat',
        sample_likelihood_mode='smoothed-local-image-plane')
    assert isinstance(direct,TieredMembersEvaluator)
    for evaluator in (ev,direct):
        value,gradient=jax.jit(jax.value_and_grad(evaluator._source_loglike_impl))(theta)
        assert np.isfinite(value) and np.isfinite(gradient).all()


def test_all_sampled_all_exact_has_no_absorbed_or_far_field_calls(tmp_path, monkeypatch):
    from lenscluster import member_tiers
    config=small_config(tmp_path)
    config=replace(config,member_tiers=replace(config.member_tiers,top_sampled='all',top_exact='all'))
    args,initial,_=make_evaluator(tmp_path,config=config)
    meta=initial.state.smoothed_model
    assert not any(s.component_family=='member_scatter_offset' for s in initial.state.parameter_specs)
    record,diagnostics=classify_member_tiers(**references([.3,.01,.3,.01],[.003,.003,0.,0.]),
        member_ids=meta['member_ids'],component_indices=meta['member_components'],
        top_sampled='all',top_exact='all',reference_uncertainty_available=False)
    assert record.exact_sampled==(0,1,2,3)
    assert diagnostics['selection_settings']==dict(top_sampled='all',top_exact='all')
    args,ev,theta=make_evaluator(tmp_path,config=config,tiers=record.to_dict())
    assert len(ev.smoothed_far_positions)==len(ev.smoothed_absorbed_positions)==0
    assert sum(s.component_family=='member_scatter_offset' for s in ev.state.parameter_specs)==4
    def forbidden(*a, **kw):
        raise AssertionError('All-exact sampling evaluated far-field profiles')
    monkeypatch.setattr(member_tiers,'evaluate_farfield_sampling',forbidden)
    value,gradient=jax.jit(jax.value_and_grad(ev._source_loglike_impl))(theta)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    stats=ev._smoothed_statistics(jnp.asarray(theta))
    expected=stats['sigma_src']**2*jnp.eye(2)+ev.flat_critical_arc_data.sigma_per_image[:,None,None]**2* (
        stats['jacobian']@jnp.swapaxes(stats['jacobian'],-1,-2))
    np.testing.assert_allclose(stats['covariance'],expected,rtol=1e-14,atol=1e-14)
