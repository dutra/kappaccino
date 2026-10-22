"""Independent checks of the sampled radius exponent and its prior measure."""
from dataclasses import replace
import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import truncnorm

from lenscluster import cluster_solver as cs
from lenscluster.config import ScalingModelConfig
from lenscluster.lens_physics import LensPhysicsMixin
from lenscluster.mock_validation.runner import parameter_recovery_table
from lenscluster.output_paths import prepare_output
from lenscluster.planning import compile_run_plan
from test_smoothed_artifacts import fixed_evaluator, posterior_from_draws
from test_smoothed_integration import small_config, make_evaluator


def test_radius_prior_density_gradients_and_numpyro_sites(tmp_path):
    _, ev, theta = make_evaluator(tmp_path)
    specs = ev.state.parameter_specs
    assert all(s.field != 'gamma_ml' for s in specs)
    beta_index = next(i for i,s in enumerate(specs) if s.field == 'beta_radius')
    beta = specs[beta_index]
    assert (beta.mean,beta.std,beta.lower,beta.upper,beta.transform_kind)==(.5,.2,0.,np.inf,'identity')
    xs = jnp.array([0.,1e-12,.1,.5,2.])
    distribution = cs._distribution_for_spec(beta)
    actual = jax.jit(jax.vmap(distribution.log_prob))(xs)
    reference = truncnorm.logpdf(xs, -2.5, np.inf, loc=.5,scale=.2)
    np.testing.assert_allclose(actual,reference,rtol=2e-14,atol=2e-14)
    gradients=jax.jit(jax.vmap(jax.grad(distribution.log_prob)))(xs)
    np.testing.assert_allclose(gradients, (.5-xs)/.2**2,rtol=2e-14,atol=2e-14)
    with pytest.warns(UserWarning, match='Out-of-support'):
        assert np.isneginf(distribution.log_prob(-1e-10))
    # The complete prior contains beta's density exactly once, with no gamma factor.
    def log_prior(b):
        return cs._prior_log_prob(specs,jnp.asarray(theta).at[beta_index].set(b))
    for value in [.1,.8]:
        difference=log_prior(value)-log_prior(.5)
        expected=distribution.log_prob(value)-distribution.log_prob(.5)
        np.testing.assert_allclose(difference,expected,atol=2e-13)
        np.testing.assert_allclose(jax.grad(log_prior)(value),(.5-value)/.2**2,atol=2e-13)
    from numpyro import handlers
    trace=handlers.trace(handlers.seed(cs._sample_site_model(specs),jax.random.PRNGKey(0))).get_trace()
    assert beta.sample_name in trace
    assert not any('gamma_ml' in name for name in trace)


@pytest.mark.parametrize('alpha,beta',[(.25,0.),(.12,.03),(.25,.5),(.4,1.8)])
def test_exponent_equations_and_coordinate_gradients(alpha,beta):
    # A small independent power-law calculation checks the physical relation and chain rule.
    helper=LensPhysicsMixin()
    spec={'alpha_sigma_base':jnp.array([.25]),'alpha_sigma_param_index':jnp.array([0]),
          'beta_radius_base':jnp.array([.5]),'beta_radius_param_index':jnp.array([1])}
    a,b,g=helper._effective_scaling_exponents_from_physical(spec,jnp.array([alpha,beta]))
    np.testing.assert_array_equal(a,[alpha]);np.testing.assert_array_equal(b,[beta])
    np.testing.assert_allclose(g,[2*alpha+beta-1],atol=2e-16)
    lum=jnp.array([.001,.05,1.,5.,100.])
    def direct(x):
        a,b,_=helper._effective_scaling_exponents_from_physical(spec,x)
        return jnp.sum((120*lum**a)**2 * 40*lum**b)
    def mass_tilt(x):
        return jnp.sum(120**2 * 40*lum**(1+x[1]))
    old=jnp.array([alpha,2*alpha+beta-1])
    value,gradient=jax.jit(jax.value_and_grad(direct))(jnp.array([alpha,beta]))
    ref,rg=jax.value_and_grad(mass_tilt)(old)
    np.testing.assert_allclose(value,ref,rtol=3e-15)
    np.testing.assert_allclose(gradient,[rg[0]+2*rg[1],rg[1]],rtol=3e-15)


@pytest.mark.parametrize('changes',[
    {'potfile_beta_radius_prior_lower':-.01}, {'potfile_beta_radius_prior_mean':float('nan')},
    {'potfile_beta_radius_prior_std':float('inf')}, {'potfile_beta_radius_prior_upper':float('nan')},
    {'potfile_beta_radius_prior_upper':float('-inf')}, {'potfile_beta_radius_prior_std':0.},
])
def test_invalid_radius_prior_rejected(tmp_path,changes):
    config=small_config(tmp_path)
    with pytest.raises(ValueError,match='potfile_beta_radius_prior'):
        replace(config,scaling=replace(config.scaling,**changes)).validate()


def test_old_configuration_rejected(tmp_path):
    config=small_config(tmp_path)
    with pytest.raises(ValueError,match='fresh run.*beta_radius'):
        replace(config,scaling=replace(config.scaling,scaling_relation_mode='direct-exponents')).validate()
    with pytest.raises(TypeError,match='potfile_gamma_ml_prior_mean'):
        ScalingModelConfig(potfile_gamma_ml_prior_mean=.1)


@pytest.mark.parametrize('kind',['runtime','state','sample_site'])
def test_historical_artifact_rejected_without_rewriting(tmp_path,kind):
    args,ev,theta=fixed_evaluator(tmp_path)
    samples=np.stack([theta,theta])
    posterior=posterior_from_draws(ev,samples)
    physical=np.stack([ev.reported_physical_parameter_vector(t) for t in samples])
    converted=cs._posterior_results_to_reported_physical(posterior,ev.state.parameter_specs,ev)
    path=tmp_path/'bundle.h5'
    cs._save_plot_bundle_h5(path,ev.state,args,physical[0],converted)
    with h5py.File(path,'r+') as handle:
        group=handle if kind=='runtime' else handle['state']
        key='runtime_args_json' if kind=='runtime' else 'build_state_meta_json'
        data=cs._read_h5_json(group,key)
        if kind=='sample_site':
            next(s for s in data['parameter_specs'] if s['field']=='beta_radius')['field']='gamma_ml'
        else:data['scaling_relation_mode']='direct-exponents'
        del group[key];cs._write_h5_json(group,key,data)
    before=path.read_bytes()
    with pytest.raises(ValueError,match='fresh run.*beta_radius'):
        cs._rebuild_state_from_h5(path)
    assert path.read_bytes()==before


def test_manifest_rejects_old_scaling_and_prior_changes(tmp_path):
    config=small_config(tmp_path);plan=compile_run_plan(config)
    prepare_output(plan)
    path=plan.output.output_dir/plan.output.run_name/'config.json'
    saved=path.read_bytes()
    changed=replace(config,scaling=replace(config.scaling,potfile_beta_radius_prior_std=.3))
    with pytest.raises(ValueError,match='different --run-label'):
        prepare_output(compile_run_plan(changed))
    assert path.read_bytes()==saved
    payload=json.loads(saved);payload['config']['scaling']['scaling_relation_mode']='direct-exponents'
    path.write_text(json.dumps(payload))
    historical=path.read_bytes()
    with pytest.raises(ValueError,match='different --run-label'):
        prepare_output(plan)
    assert path.read_bytes()==historical


def test_joint_derived_gamma_recovery():
    names=['members.alpha_sigma','members.beta_radius']
    # Non-monotone correlated draws distinguish joint transforms from marginal summaries.
    samples=np.array([[.1,.9],[.2,.1],[.4,.3],[.3,.7]])
    best=samples[2]
    table=parameter_recovery_table(samples,names,{'members.gamma_ml':.1},best_fit=best).set_index('parameter')
    expected=2*samples[:,0]+samples[:,1]-1
    np.testing.assert_allclose(table.loc['members.gamma_ml',['q16','median','q84']].to_numpy(dtype=float),np.quantile(expected,[.16,.5,.84]))
    assert table.loc['members.gamma_ml','best_fit']==pytest.approx(2*best[0]+best[1]-1)
    assert table.loc['members.beta_radius','median']==pytest.approx(np.median(samples[:,1]))


def test_exact_radii_at_zero_and_away_from_initialization(tmp_path):
    _,ev,theta=fixed_evaluator(tmp_path)
    physical=ev._physical_parameter_vector(jnp.asarray(theta))
    spec=ev.packed_spec_jax
    members=np.asarray(ev.smoothed_member_components)
    beta_index=next(i for i,s in enumerate(ev.state.parameter_specs) if s.field=='beta_radius')
    core=ev._apply_param_updates(spec['core_ref_base'],spec['core_ref_param_index'],physical)
    cut=ev._apply_param_updates(spec['cut_ref_base'],spec['cut_ref_param_index'],physical)
    lum=np.asarray(spec['luminosity_ratio'])
    kpc=float(ev._kpc_per_arcsec_for_physical(physical))
    value_and_grad=jax.jit(jax.value_and_grad(ev._source_loglike_impl))
    for beta in [0.,1e-10,.3,1.8]:
        point=physical.at[beta_index].set(beta)
        packed,validity=ev._smoothed_packed_with_validity(point)
        expected_core=np.maximum(np.asarray(core)[members]*lum[members]**beta/kpc,cs.SAFE_RADIUS_MARGIN_ARCSEC)
        expected_cut=np.maximum(np.asarray(cut)[members]*lum[members]**beta/kpc,expected_core+cs.SAFE_RADIUS_MARGIN_ARCSEC)
        np.testing.assert_allclose(np.asarray(packed.Ra)[members],expected_core,rtol=3e-15)
        np.testing.assert_allclose(np.asarray(packed.Rs)[members],expected_cut,rtol=3e-15)
        current=jnp.asarray(theta).at[beta_index].set(beta)
        value,grad=value_and_grad(current)
        assert np.isfinite(value) and np.isfinite(grad).all()
        if beta == .3:
            plus,_=value_and_grad(current.at[beta_index].add(1e-5))
            minus,_=value_and_grad(current.at[beta_index].add(-1e-5))
            np.testing.assert_allclose(grad[beta_index],(plus-minus)/2e-5,rtol=2e-6,atol=1e-6)


def test_stage_transfer_keeps_sampled_beta_and_physical_sources(tmp_path):
    args,ev,theta=fixed_evaluator(tmp_path)
    beta_index=next(i for i,s in enumerate(ev.state.parameter_specs) if s.field=='beta_radius')
    point=jnp.asarray(theta).at[beta_index].set(.73)
    physical=np.asarray(ev.reported_physical_parameter_vector(point))
    initial={s.sample_name:float(v) for s,v in zip(ev.state.parameter_specs,physical)}
    state=cs._build_state_from_inputs(args,fit_mode_override='joint',svi_init_physical_values=initial)
    transferred=cs._build_cluster_evaluator_from_args(args,state)
    latent=cs._reference_theta_from_init_values(state.parameter_specs,state.svi_init_values,cs._default_theta(state.parameter_specs))
    reconstructed=transferred.reported_physical_parameter_vector(latent)
    np.testing.assert_allclose(reconstructed,physical,rtol=3e-12,atol=3e-11)
    assert reconstructed[beta_index]==pytest.approx(.73)


def test_resume_and_stage_summary_reject_old_parameterization(tmp_path):
    args,ev,theta=fixed_evaluator(tmp_path)
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    payload={'map_values':{'members_gamma_ml':.1},'means':{},'stds':{}}
    (artifacts/'stage1_prior_summary.json').write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='fresh run.*beta_radius'):
        cs._load_stage1_summary(artifacts)
    with h5py.File(artifacts/'plot_bundle.h5','w') as handle:
        cs._write_h5_json(handle,'runtime_args_json',{'scaling_relation_mode':'direct-exponents'})
    before=(artifacts/'plot_bundle.h5').read_bytes()
    with pytest.raises(ValueError,match='fresh run.*beta_radius'):
        cs._stage_run_cosmology_compatible(args,tmp_path)
    assert (artifacts/'plot_bundle.h5').read_bytes()==before


@pytest.mark.parametrize('plots_only',[False,True])
def test_joint_dispatch_cannot_skip_old_run_as_complete(tmp_path,monkeypatch,plots_only):
    config=small_config(tmp_path)
    args=cs._clone_args(compile_run_plan(config).runtime_args,fit_mode='joint',resume='all',skip_plots=True,plots_only=plots_only)
    root=tmp_path/'output'/'smoothed';artifacts=root/'artifacts';artifacts.mkdir(parents=True)
    path=artifacts/'plot_bundle.h5'
    with h5py.File(path,'w') as handle:
        cs._write_h5_json(handle,'runtime_args_json',{'scaling_relation_mode':'direct-exponents'})
    before=path.read_bytes()
    monkeypatch.setattr(cs,'_stage_run_complete',lambda path:True)
    with pytest.raises(ValueError,match='fresh run.*beta_radius'):
        cs._run_typed_plan_dispatch(args,{})
    assert path.read_bytes()==before


def test_mock_truth_records_beta_and_derived_gamma():
    from lenscluster.mock_validation import generation as mock
    config=mock.SingleBCGMockConfig(n_subhalos=4,subhalo_alpha_sigma=.31,subhalo_beta_radius=.62,
        subhalo_alpha_sigma_half_width=0.,subhalo_beta_radius_half_width=0.)
    realized=mock._realize_potfile_exponents(config,np.random.default_rng(1))
    truth=mock._truth_parameter_values(realized,4.4)
    assert truth['potfile.beta_radius']==pytest.approx(.62)
    assert truth['potfile.gamma_ml']==pytest.approx(.24)
    assert truth['potfile.sampled_beta_radius']==pytest.approx(.62)
    assert 'potfile.sampled_gamma_ml' not in truth
    assert mock._subhalo_mass_luminosity_exponent(realized)==pytest.approx(1.24)
