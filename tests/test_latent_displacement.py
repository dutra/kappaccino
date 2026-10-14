"""Independent references for the local Gaussian, and current-model integration."""
from dataclasses import replace
from pathlib import Path
import argparse
import json

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal
from scipy.integrate import quad

from lenscluster.latent_displacement import MODE, statistics, validate_settings
from lenscluster import cluster_solver as cs
from test_scaling_cache_retention import evaluator


def fixture():
    old, _ = evaluator('full_flat')
    state = old.state
    # Both redshift families retain independent, shared two-component sources.
    state.parameter_specs.extend(cs._build_source_position_parameter_specs(
        state.family_data, {str(f.family_id): (0., 0.) for f in state.family_data},
        start_index=len(state.parameter_specs), beta_prior_sigma_arcsec=2.,
        parameterization='prior-whitened'))
    state.source_position_parameterization = 'prior-whitened'
    ev = cs.ClusterJAXEvaluator(state, .1, sampling_engine='full_flat',
        sample_likelihood_mode=MODE, latent_displacement_sigma_beta_arcsec=.03)
    return ev, cs._default_theta(state.parameter_specs)


def example(theta):
    a = jnp.array([[theta[0], 2.], [theta[1], 1.]])
    x = jnp.array([[.1, .3], [-.3, .8], [.6, -.4]])
    b = x @ a.T + theta[2:4]
    sources = jnp.stack((theta[4:6], theta[6:8]))[jnp.array([0, 0, 1])]
    d = jnp.array([[1., -.3], [.4, .7]]) * theta[9]
    member = jnp.broadcast_to(d @ d.T, (3, 2, 2))
    return statistics(b, sources, jnp.broadcast_to(a, (3, 2, 2)),
                      jnp.array([.1, .2, .3]) * theta[8], theta[10], member, jnp.ones(3, bool))


def reference(theta):
    a = np.array([[theta[0], 2.], [theta[1], 1.]])
    b = np.array([[.1, .3], [-.3, .8], [.6, -.4]]) @ a.T + theta[2:4]
    source = np.array([theta[4:6], theta[4:6], theta[6:8]])
    d = np.array([[1., -.3], [.4, .7]]) * theta[9]
    cov = np.array([s*s*a@a.T + d@d.T + theta[10]**2*np.eye(2) for s in np.array([.1,.2,.3])*theta[8]])
    return sum(multivariate_normal.logpdf(r, cov=c) for r,c in zip(b-source,cov)),cov


def test_reference_derivatives_and_hessian():
    t=np.array([1., .2, .01, -.02, .05, .08, .1, -.1, 1., .04, .03])
    fn=jax.jit(lambda t: example(t)['loglike'].sum())
    actual=example(t); expected,cov=reference(t)
    np.testing.assert_allclose(actual['covariance'],cov,rtol=1e-14,atol=1e-16)
    np.testing.assert_allclose(fn(t),expected,rtol=1e-13)
    grad=jax.jit(jax.grad(fn))(t)
    for k in range(len(t)):
        h=1e-6;d=np.eye(len(t))[k]*h
        np.testing.assert_allclose(grad[k],(reference(t+d)[0]-reference(t-d)[0])/(2*h),rtol=2e-6,atol=1e-6)
    assert np.isfinite(jax.jit(jax.hessian(fn))(t)).all()
    assert np.isfinite(jax.jit(jax.vmap(fn))(jnp.stack([t,t+.001]))).all()


def test_rotation_mask_permutation_and_rank_deficiency():
    jac=np.array([[[1.,2.],[0.,1.]],[[0.,0.],[0.,2.]],[[np.nan]*2]*2])
    b=np.array([[.1,-.2],[.05,.02],[np.nan,np.nan]]); src=np.zeros((3,2))
    v=np.array([[[.01,.003],[.003,.02]],np.zeros((2,2)),np.full((2,2),np.nan)])
    mask=jnp.array([True,True,False]);sig=jnp.array([.1,.2,np.nan])
    def calc(b,src,jac,v,mask,sig):return statistics(b,src,jac,sig,.03,v,mask)
    out=jax.jit(calc)(b,src,jac,v,mask,sig)
    np.testing.assert_allclose(out['covariance'][1],np.diag([.0009,.1609]))
    assert np.isfinite(out['loglike']).all();assert out['loglike'][2]==0
    q=np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
    rot=calc(b@q.T,src@q.T,q@jac@q.T,q@v@q.T,mask,sig)
    np.testing.assert_allclose(out['loglike'],rot['loglike'],rtol=1e-12)
    ix=np.array([1,0,2]); np.testing.assert_allclose(out['loglike'].sum(),calc(b[ix],src[ix],jac[ix],v[ix],mask[ix],sig[ix])['loglike'].sum())


def test_marginalization_and_map():
    f=np.array([.03,-.02]); sigma=.2;sb=.03
    for a in [0.,.7]:
        c=sb**2+a*a*sigma**2
        integral=quad(lambda d: np.exp(-.5*((f[0]+a*d)/sb)**2-.5*(d/sigma)**2)/(2*np.pi*sb*sigma),-np.inf,np.inf,epsabs=1e-12)[0]
        np.testing.assert_allclose(integral,np.exp(-f[0]**2/(2*c))/np.sqrt(2*np.pi*c),rtol=1e-11)
    a=np.array([[1.,2.],[-.4,.1]]);c=sb**2*np.eye(2)+sigma**2*a@a.T
    delta=-np.linalg.solve(a.T@a+(sb/sigma)**2*np.eye(2),a.T@f)
    np.testing.assert_allclose(np.sum((f+a@delta)**2)/(2*sb**2)+np.sum(delta**2)/(2*sigma**2),.5*f@np.linalg.solve(c,f),rtol=1e-12)


def test_fold_crossing_and_source_contraction():
    def fold(c):
        x=.1;k=.7
        b=jnp.array([[.2,.5*k*(x-c)**2]])
        a=jnp.array([[[1.,0.],[0.,k*(x-c)]]])
        return statistics(b,jnp.array([[.2,.01]]),a,jnp.array([.2]),.03,jnp.zeros((1,2,2)),jnp.array([True]))['loglike'].sum()
    vg=jax.jit(jax.value_and_grad(fold))
    for c in np.linspace(.08,.12,9):
        val,grad=vg(c);assert np.isfinite([val,grad]).all()
        np.testing.assert_allclose(grad,(fold(c+1e-6)-fold(c-1e-6))/2e-6,rtol=1e-5,atol=1e-8)
    def scale(a):return statistics(jnp.zeros((1,2)),jnp.zeros((1,2)),a*jnp.eye(2)[None],jnp.array([.2]),.03,jnp.zeros((1,2,2)),jnp.array([True]))['loglike'].sum()
    assert scale(.1)>scale(1.) # expected contraction preference, not an implementation bug
    # A fold at its critical point misses a nonzero mean and variance of quadratic displacement.
    k=.7;sigma=.2
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    increments = .5*k*(np.sqrt(2)*sigma*nodes)**2
    mean = np.sum(weights*increments)/np.sqrt(np.pi)
    variance = np.sum(weights*(increments-mean)**2)/np.sqrt(np.pi)
    np.testing.assert_allclose([mean,variance],[k*sigma**2/2,k*k*sigma**4/2],rtol=1e-14)


@pytest.mark.parametrize('sigma',[None,0.,-.1,np.nan,np.inf])
def test_invalid_scale(sigma):
    with pytest.raises(ValueError,match='positive'): validate_settings(sigma,'full_flat','direct')


def test_evaluator_gradient_cache_independence_and_sources():
    ev,t=fixture();fn=jax.jit(ev._source_loglike_impl);vg=jax.jit(jax.value_and_grad(fn));v,g=vg(t)
    assert np.isfinite(v) and np.isfinite(g).all()
    for i in range(len(t)):
        h=1e-4 if i==0 else 1e-5;d=np.eye(len(t))[i]*h
        np.testing.assert_allclose(g[i],(fn(t+d)-fn(t-d))/(2*h),rtol=2e-4,atol=1e-7)
    assert np.isfinite(jax.jit(jax.hessian(fn))(t)).all()
    ev.scaling_scatter_cache_by_z={0.:{'sigma_cov00':np.array([np.nan])}}
    before=ev._source_loglike_fn
    ev.refresh_scaling_scatter_cache(t);ev.refresh_source_metric_cache(t)
    assert before is ev._source_loglike_fn
    assert cs._svi_refresh_evaluator_snapshot(ev)['source_metric'].direct
    assert 'source_metric=direct' in cs._refresh_pre_nuts_caches(ev,t,'test')
    np.testing.assert_array_equal(jax.jit(jax.value_and_grad(ev._source_loglike_impl))(t)[1],g)
    df=ev.latent_displacement_diagnostics(t)
    np.testing.assert_allclose(df.loglike.sum(),v)
    assert df.family_id.nunique()==2
    assert df.groupby('family_id').source_x_arcsec.nunique().eq(1).all()
    assert all(np.isnan(p['approx_image_rms_arcsec']) for p in ev.evaluate(t).family_predictions.values())


def test_resume_identity(tmp_path):
    path=tmp_path/'artifacts';path.mkdir()
    with h5py.File(path/'plot_bundle.h5','w') as h:
        h['runtime_args_json']=json.dumps(dict(sample_likelihood_mode=MODE,latent_displacement_sigma_beta_arcsec=.03))
    args=argparse.Namespace(sample_likelihood_mode=MODE,latent_displacement_sigma_beta_arcsec=.03)
    assert cs._stage_run_cosmology_compatible(args,tmp_path)[0]
    args.latent_displacement_sigma_beta_arcsec=.04
    assert not cs._stage_run_cosmology_compatible(args,tmp_path)[0]
    args.sample_likelihood_mode='source'
    assert not cs._stage_run_cosmology_compatible(args,tmp_path)[0]


def configured():
    from test_config import _minimal_sequential_config
    c=_minimal_sequential_config()
    return replace(c,workflow=replace(c.workflow,stage2_forward_mode=MODE,stage2_sampling_engine='full_flat'),
        likelihood=replace(c.likelihood,latent_displacement_sigma_beta_arcsec=.03),
        schedule=replace(c.schedule,svi_steps=(10,20,20),refresh_every=(None,100,100),
                         fit_method=('svi+nuts','svi+nuts'),warmup=(1,1),samples=(2,2),
                         sampling_refresh_runs=(1,1),max_tree_depth=(8,8)))


def test_configuration_handoff_pickle_and_invalid_settings():
    import pickle
    from lenscluster.planning import compile_run_plan
    c=configured();plan=compile_run_plan(c)
    assert plan.stages[-1].sample_likelihood_mode==MODE
    runtime=pickle.loads(pickle.dumps(plan.runtime_args))
    assert runtime.latent_displacement_sigma_beta_arcsec==.03
    assert cs._stage2_sample_likelihood_mode(runtime)==MODE
    assert cs._sample_likelihood_uses_explicit_beta(MODE)
    assert not cs._sample_likelihood_uses_image_scatter(MODE)
    for bad in [None,0.,np.nan]:
        with pytest.raises(ValueError,match='positive'):
            compile_run_plan(replace(c,likelihood=replace(c.likelihood,latent_displacement_sigma_beta_arcsec=bad)))
    with pytest.raises(ValueError,match='full_flat'):
        compile_run_plan(replace(c,workflow=replace(c.workflow,stage2_sampling_engine='refreshing_surrogate_flat')))
    with pytest.raises(ValueError,match='prior-whitened'):
        compile_run_plan(replace(c,workflow=replace(c.workflow,source_position_parameterization='conditional-whitened')))


def test_independent_differentiable_reference_and_family_source_gradient():
    t=jnp.array([1., .2, .01, -.02, .05, .08, .1, -.1, 1., .04, .03])
    def independent(t):
        a=jnp.array([[t[0],2.],[t[1],1.]])
        d=jnp.array([[1.,-.3],[.4,.7]])*t[9]
        total=0.
        for x,source,sig in zip([[.1,.3],[-.3,.8],[.6,-.4]],[t[4:6],t[4:6],t[6:8]],[.1,.2,.3]):
            residual=a@jnp.array(x)+t[2:4]-source
            c=(sig*t[8])**2*(a@a.T)+d@d.T+t[10]**2*jnp.eye(2)
            total-=.5*(residual@jnp.linalg.solve(c,residual)+jnp.linalg.slogdet(c)[1]+2*jnp.log(2*jnp.pi))
        return total
    np.testing.assert_allclose(jax.grad(lambda t:example(t)['loglike'].sum())(t),jax.grad(independent)(t),rtol=2e-12,atol=1e-10)


def test_zero_scatter_evaluator_vectorization_and_invalid_model():
    ev,t=fixture()
    ev.scaling_scatter_param_indices=np.array([],dtype=np.int32)
    stat=ev._latent_displacement_statistics(jnp.array(t))
    np.testing.assert_array_equal(stat['member_covariance'],0.)
    batch=jax.jit(jax.vmap(ev._source_loglike_impl))(jnp.stack([t,t]))
    np.testing.assert_allclose(batch[0],batch[1]);assert np.isfinite(batch).all()
    assert ev._source_loglike_impl(jnp.array(t).at[0].set(jnp.nan))==cs.BAD_LOG_LIKE
