"""Saved source coordinates, fixed classifications, and target-density identity."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from lenscluster.member_tiers import MemberTierRecord
from lenscluster.planning import compile_run_plan
from test_smoothed_integration import make_evaluator, small_config, MODE


def evaluator_at(path, **kwargs):
    path.mkdir(parents=True, exist_ok=True)
    return make_evaluator(path, **kwargs)


def fixed_evaluator(path, parameterization='likelihood-whitened'):
    _, preliminary, _ = evaluator_at(path)
    meta = preliminary.state.smoothed_model
    tiers = MemberTierRecord(tuple(meta['member_ids']), tuple(meta['member_components']),
        exact_sampled=(0,), exact_absorbed=(1,), approximate_sampled=(2,), approximate_absorbed=(3,))
    return evaluator_at(path, tiers=tiers.to_dict(), parameterization=parameterization)


def varied_draws(ev, theta):
    data = ev.flat_critical_arc_data
    draws = np.stack([theta.copy(), theta.copy()])
    draws[0, np.asarray(data.global_family_source_x_param_index)] = [.4, -.2]
    draws[0, np.asarray(data.global_family_source_y_param_index)] = [-.3, .5]
    draws[1] = draws[0]
    draws[1, 0] += 3.
    draws[1, ev.state.smoothed_model['sigma_src_index']] += .15
    draws[1, np.asarray(data.global_family_source_x_param_index)] += .1
    return draws


def posterior_from_draws(ev, draws, *, sampler='numpyro_nuts'):
    log_prob = np.asarray([float(ev._source_loglike_impl(jnp.asarray(t))
                                + cs._prior_log_prob(ev.state.parameter_specs, jnp.asarray(t))) for t in draws])
    return cs.PosteriorResults(samples=draws, log_prob=log_prob,
        accept_prob=np.ones(len(draws)), diverging=np.zeros(len(draws), dtype=bool),
        num_steps=np.ones(len(draws)), warmup_steps=0, sample_steps=len(draws), num_chains=1,
        grouped_samples=draws[None], grouped_log_prob=log_prob[None], sampler=sampler,
        median_fit=draws[0].copy(), init_diagnostics={'map_sample_index':0,'maximum_likelihood_sample_index':1})


def test_actual_evaluator_direct_whitened_density_change_of_variables(tmp_path):
    _, white, theta = evaluator_at(tmp_path/'whitened')
    _, direct, _ = evaluator_at(tmp_path/'direct', parameterization='direct')
    theta = varied_draws(white, theta)[0]
    physical = white.reported_physical_parameter_vector(theta)
    direct_theta = direct.reported_physical_to_latent_parameter_vector(physical)
    white_stats, direct_stats = white._smoothed_statistics(theta), direct._smoothed_statistics(direct_theta)
    np.testing.assert_allclose(white_stats['source_by_family'],direct_stats['source_by_family'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(white_stats['covariance'],direct_stats['covariance'],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(white_stats['determinant_mean'],direct_stats['determinant_mean'],rtol=1e-12,atol=1e-12)
    white_posterior = white._source_loglike_impl(theta)+cs._prior_log_prob(white.state.parameter_specs,theta)
    direct_posterior = direct._source_loglike_impl(direct_theta)+cs._prior_log_prob(direct.state.parameter_specs,direct_theta)
    log_jacobian = -.5*white_stats['whitening']['logdet_precision'].sum()
    np.testing.assert_allclose(white_posterior,direct_posterior+log_jacobian,rtol=1e-12,atol=2e-9)
    # The determinant term remains a likelihood at the sampled physical sources.
    np.testing.assert_allclose(white_stats['loglike'],direct_stats['loglike'],rtol=1e-12,atol=1e-10)
    source_x=white.flat_critical_arc_data.global_family_source_x_param_index
    source_y=white.flat_critical_arc_data.global_family_source_y_param_index
    def white_target(t):
        return white._source_loglike_impl(t)+cs._prior_log_prob(white.state.parameter_specs,t)
    def transformed_direct_target(t):
        stats=white._smoothed_statistics(t)
        source=stats['source_by_family']
        mapped=t.at[source_x].set(source[:,0]).at[source_y].set(source[:,1])
        return (direct._source_loglike_impl(mapped)+cs._prior_log_prob(direct.state.parameter_specs,mapped)
                -.5*stats['whitening']['logdet_precision'].sum())
    grad_white=jax.jit(jax.grad(white_target))(jnp.asarray(theta))
    grad_direct=jax.jit(jax.grad(transformed_direct_target))(jnp.asarray(theta))
    assert np.isfinite(grad_white).all() and np.isfinite(grad_direct).all()
    np.testing.assert_allclose(grad_white,grad_direct,rtol=3e-10,atol=2e-7)


@pytest.mark.parametrize('saved_engine', ['tiered_members', 'full_flat'])
def test_h5_roundtrip_preserves_physical_sources_tiers_and_density(tmp_path, saved_engine):
    args, ev, theta = fixed_evaluator(tmp_path/'model')
    draws = varied_draws(ev,theta)
    posterior = posterior_from_draws(ev,draws)
    expected = np.stack([ev.reported_physical_parameter_vector(t) for t in draws])
    converted = cs._posterior_results_to_reported_physical(posterior,ev.state.parameter_specs,ev)
    np.testing.assert_allclose(converted.samples,expected,rtol=1e-12,atol=1e-11)
    np.testing.assert_allclose(converted.grouped_samples,expected[None],rtol=1e-12,atol=1e-11)
    np.testing.assert_allclose(converted.median_fit,expected[0],rtol=1e-12,atol=1e-11)
    # Log densities keep their documented sampling-coordinate measure.
    np.testing.assert_array_equal(converted.log_prob,posterior.log_prob)
    path=tmp_path/'plot_bundle.h5'
    cs._save_plot_bundle_h5(path,ev.state,args,expected[0],converted)
    with h5py.File(path, 'r+') as handle:
        runtime = cs._read_h5_json(handle, 'runtime_args_json')
        assert runtime['sampling_engine'] == 'tiered_members'
        for key in ('sampling_engine', 'stage1_sampling_engine', 'stage2_sampling_engine'):
            runtime[key] = saved_engine
        del handle['runtime_args_json']
        cs._write_h5_json(handle, 'runtime_args_json', runtime)
    before = path.read_bytes()
    rebuilt,saved,arrays,diagnostics=cs._rebuild_state_from_h5(path)
    assert saved['sampling_engine'] == 'tiered_members'
    assert saved['stage1_sampling_engine'] == saved['stage2_sampling_engine'] == 'tiered_members'
    assert saved['sample_likelihood_mode']==MODE
    assert saved['member_tier_phase']=='fixed'
    assert rebuilt.smoothed_model==ev.state.smoothed_model
    assert rebuilt.smoothed_model['tiers']==ev.state.smoothed_model['tiers']
    assert rebuilt.source_position_parameterization=='likelihood-whitened'
    np.testing.assert_allclose(arrays['samples'],expected,rtol=1e-12,atol=1e-11)
    np.testing.assert_allclose(arrays['median_fit'],expected[0],rtol=1e-12,atol=1e-11)
    np.testing.assert_array_equal(arrays['log_prob'],posterior.log_prob)
    reloaded=cs._build_cluster_evaluator_from_args(argparse.Namespace(**saved),rebuilt)
    for physical,latent,log_prob in zip(arrays['samples'],draws,posterior.log_prob):
        restored=reloaded.reported_physical_to_latent_parameter_vector(physical)
        np.testing.assert_allclose(restored,latent,rtol=1e-10,atol=1e-10)
        restored_log_prob=reloaded._source_loglike_impl(restored)+cs._prior_log_prob(rebuilt.parameter_specs,restored)
        np.testing.assert_allclose(restored_log_prob,log_prob,rtol=1e-11,atol=1e-8)
    assert diagnostics['log_prob_coordinates']=='sampling_parameters_before_numpyro_unconstraining'
    # An independent interpreter rebuilds all lens and transport state from H5.
    script='''
import argparse, json, sys
from pathlib import Path
import numpy as np
from lenscluster import cluster_solver as cs
state,saved,arrays,_=cs._rebuild_state_from_h5(Path(sys.argv[1]))
evaluator=cs._build_cluster_evaluator_from_args(argparse.Namespace(**saved),state)
latent=evaluator.reported_physical_to_latent_parameter_vector(arrays['best_fit'])
log_prob=float(evaluator._source_loglike_impl(latent)+cs._prior_log_prob(state.parameter_specs,latent))
print('RESULT_JSON='+json.dumps({'latent':latent.tolist(),'log_prob':log_prob,'tiers':state.smoothed_model['tiers'],'engine':evaluator.sampling_engine}))
'''
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OMP_NUM_THREADS='1',JAX_NUM_CPU_DEVICES='1')
    child=subprocess.run(['/home/dutra/.conda/envs/lenstronomy/bin/python','-c',script,str(path)],
                         cwd=root,env=env,text=True,capture_output=True,timeout=120)
    assert child.returncode==0, child.stdout+'\n'+child.stderr
    record=json.loads(next(line.split('=',1)[1] for line in child.stdout.splitlines() if line.startswith('RESULT_JSON=')))
    np.testing.assert_allclose(record['latent'],draws[0],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(record['log_prob'],posterior.log_prob[0],rtol=1e-11,atol=1e-8)
    assert record['tiers']==ev.state.smoothed_model['tiers']
    assert record['engine'] == 'tiered_members'
    assert path.read_bytes() == before


def test_svi_checkpoint_sources_are_physical_and_reloadable(tmp_path):
    args,ev,theta=evaluator_at(tmp_path/'model')
    draws=varied_draws(ev,theta)
    posterior=posterior_from_draws(ev,draws,sampler='svi')
    run_dir=tmp_path/'stage0'
    cs._save_svi_checkpoint(args,ev.state,ev,run_dir,draws[0],posterior)
    rebuilt,saved,arrays,diagnostics=cs._load_artifacts(run_dir/'artifacts'/'svi')
    expected=np.stack([ev.reported_physical_parameter_vector(t) for t in draws])
    np.testing.assert_allclose(arrays['samples'],expected,rtol=1e-12,atol=1e-11)
    np.testing.assert_allclose(arrays['best_fit'],expected[0],rtol=1e-12,atol=1e-11)
    np.testing.assert_allclose(arrays['median_fit'],expected[0],rtol=1e-12,atol=1e-11)
    assert diagnostics['log_prob_coordinates']=='sampling_parameters_before_numpyro_unconstraining'
    assert diagnostics['posterior_role']=='svi_guide_before_sampling'
    assert rebuilt.smoothed_model==ev.state.smoothed_model
    np.testing.assert_array_equal(posterior.samples,draws)
    assert posterior.init_diagnostics=={'map_sample_index':0,'maximum_likelihood_sample_index':1}


def test_current_beta_bundle_for_legacy_likelihood_needs_no_smoothed_payload(tmp_path):
    from test_latent_displacement import configured
    from lenscluster.planning import compile_run_plan
    config=configured()
    args=cs._clone_args(compile_run_plan(config).runtime_args,sample_likelihood_mode='source',sampling_engine='full_flat')
    state=cs._build_state_from_inputs(args,fit_mode_override='joint')
    theta=cs._default_theta(state.parameter_specs)
    physical=cs._convert_theta_to_physical(theta,state.parameter_specs)
    posterior=cs.PosteriorResults(samples=physical[None],log_prob=np.zeros(1),accept_prob=np.ones(1),
        diverging=np.zeros(1,dtype=bool),num_steps=np.ones(1),warmup_steps=0,sample_steps=1,num_chains=1)
    path=tmp_path/'legacy.h5'
    cs._save_plot_bundle_h5(path,state,args,physical,posterior)
    # A current beta-parameterized legacy likelihood needs no smoothed payload.
    with h5py.File(path,'r+') as handle:
        node=handle['state/build_state_meta_json']
        payload=json.loads(node[()].decode())
        payload.pop('smoothed_model',None)
        del handle['state/build_state_meta_json']
        handle['state/build_state_meta_json']=json.dumps(payload)
    rebuilt,saved,arrays,_=cs._rebuild_state_from_h5(path)
    assert rebuilt.smoothed_model is None and saved['sample_likelihood_mode']=='source'
    assert [s.sample_name for s in rebuilt.parameter_specs]==[s.sample_name for s in state.parameter_specs]
    np.testing.assert_array_equal(arrays['best_fit'],physical)
    np.testing.assert_array_equal(arrays['samples'],physical[None])


def test_resume_rejects_likelihood_transport_prior_and_tier_changes(tmp_path):
    args=cs._clone_args(compile_run_plan(small_config(tmp_path)).runtime_args,
                       sample_likelihood_mode=MODE,member_tier_phase='initializer',member_tiers_payload=None)
    artifacts=tmp_path/'artifacts'
    artifacts.mkdir()
    with h5py.File(artifacts/'plot_bundle.h5','w') as handle:
        cs._write_h5_json(handle,'runtime_args_json',cs._args_payload(args))
    assert cs._stage_run_cosmology_compatible(args,tmp_path)[0]
    with h5py.File(artifacts/'plot_bundle.h5','r+') as handle:
        runtime = cs._read_h5_json(handle, 'runtime_args_json')
        runtime['sampling_engine'] = 'full_flat'
        del handle['runtime_args_json']
        cs._write_h5_json(handle, 'runtime_args_json', runtime)
    assert cs._stage_run_cosmology_compatible(args,tmp_path)[0]
    changes={
        'sample_likelihood_mode':'source',
        'source_position_parameterization':'direct',
        'smoothed_likelihood_revision':str(getattr(args,'smoothed_likelihood_revision',1))+'-incompatible',
        'source_position_prior_sigma_arcsec':getattr(args,'source_position_prior_sigma_arcsec',100.)*2,
        'source_scatter_prior_median_arcsec':getattr(args,'source_scatter_prior_median_arcsec',.02)*2,
        'member_tier_phase':'fixed',
        'member_tier_top_sampled':0,
        'member_tier_top_exact':2,
        'member_tiers_payload':{'different':'classification'},
        'sampling_engine':'refreshing_surrogate_flat',
    }
    for key,value in changes.items():
        compatible,reason=cs._stage_run_cosmology_compatible(cs._clone_args(args,**{key:value}),tmp_path)
        assert not compatible,(key,reason)
        assert 'smoothed likelihood' in reason
