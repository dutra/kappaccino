from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from test_scaling_cache_retention import evaluator


def reference(ev, theta, eps=1e-4):
    """Independent member-at-a-time central differences of full backprojection."""
    physical = ev._physical_parameter_vector(jnp.asarray(theta))
    data = ev.flat_critical_arc_data
    result = {}
    for field, direction in [('sigma', (1., -2., -2.)), ('mass', (0., 1., 1.))]:
        derivatives = []
        for member in ev._scaling_scatter_component_indices():
            row = np.zeros(len(ev.state.packed_lens_spec.profile_type))
            row[member] = eps
            values = []
            for sign in (1, -1):
                packed, valid = ev._build_flat_packed_lens_state_with_validity_from_physical(
                    physical, data, stop_gradient=False, sigma_log_offset=sign*direction[0]*row,
                    core_log_offset=sign*direction[1]*row, cut_log_offset=sign*direction[2]*row)
                assert valid.is_valid
                values.append(np.stack(ev._flat_ray_shooting_for_components(data.x_obs, data.y_obs, packed), -1))
            derivatives.append((values[0]-values[1])/(2*eps))
        d = np.asarray(derivatives)
        for ij, a, b in [('00', 0, 0), ('01', 0, 1), ('11', 1, 1)]:
            result[f'{field}_cov{ij}'] = (d[:,:,a]*d[:,:,b]).sum(0)
    return result


def test_direct_covariance_reference_jit_vmap_and_gradient():
    ev, theta = evaluator('full_flat')
    def covariance(t):
        return ev._direct_scaling_scatter_covariance(ev._physical_parameter_vector(t), ev.flat_critical_arc_data)
    compiled = jax.jit(covariance)
    actual = compiled(jnp.asarray(theta))
    for key, expected in reference(ev, theta).items():
        np.testing.assert_allclose(actual[key], expected, rtol=2e-5, atol=1e-9)
    batch = jnp.stack([jnp.asarray(theta), jnp.asarray(theta).at[0].add(.1)])
    batched = jax.jit(jax.vmap(covariance))(batch)
    for key in actual:
        np.testing.assert_allclose(batched[key][0], actual[key], rtol=1e-12)
    func = lambda t: sum(jnp.sum(x) for x in covariance(t).values())
    grad = jax.jit(jax.grad(func))(jnp.asarray(theta))
    assert np.isfinite(grad).all()
    h = .01
    plus, minus = theta.copy(), theta.copy()
    plus[0] += h; minus[0] -= h
    fd = (sum(x.sum() for x in reference(ev, plus).values())-sum(x.sum() for x in reference(ev, minus).values()))/(2*h)
    np.testing.assert_allclose(grad[0], fd, rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize('mode', ['source', 'local-jacobian', cs.SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE, cs.SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE])
def test_direct_likelihood_gradient_and_cache_independence(mode):
    ev, theta = evaluator('full_flat')
    ev.sample_likelihood_mode = mode
    ev.critical_arc_source_position_policy = cs.CRITICAL_ARC_SOURCE_POSITION_POLICY_CENTROID_FIXED
    fn = jax.jit(jax.value_and_grad(ev._source_loglike_impl))
    expected = fn(jnp.asarray(theta))
    assert all(np.isfinite(x).all() for x in expected)
    assert expected[0] > cs.BAD_LOG_LIKE
    for i in range(len(theta)):
        h = 1e-3 * max(1., abs(theta[i]))
        plus, minus = theta.copy(), theta.copy()
        plus[i] += h; minus[i] -= h
        fd = (float(fn(jnp.asarray(plus))[0])-float(fn(jnp.asarray(minus))[0]))/(2*h)
        np.testing.assert_allclose(expected[1][i], fd, rtol=3e-3, atol=1e-5)
    ev.scaling_scatter_cache_by_z = {2.: {key: np.array([np.nan]) for key in cs.SCALING_SCATTER_CACHE_KEYS}}
    ev.flat_critical_arc_data = replace(ev.flat_critical_arc_data, **{
        f'{field}_scatter_cov{ij}': jnp.full_like(ev.flat_critical_arc_data.x_obs, jnp.nan)
        for field in ('sigma','mass') for ij in ('00','01','11')})
    actual = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(jnp.asarray(theta))
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)
    for z in (2.,3.):
        assert all(np.isfinite(x).all() for x in ev._scaling_scatter_covariance_for_bin(ev._physical_parameter_vector(jnp.asarray(theta)),z).values())


def test_direct_refresh_noop_and_status():
    ev, theta = evaluator('full_flat')
    old = ev._source_loglike_fn, ev.flat_critical_arc_data
    ev.refresh_scaling_scatter_cache(theta * np.nan)
    assert ev._source_loglike_fn is old[0] and ev.flat_critical_arc_data is old[1]
    assert ev.scaling_scatter_cache_by_z == {} and ev.scaling_scatter_reference_params is None
    snap = cs._svi_refresh_evaluator_snapshot(ev)
    rows = cs._svi_refresh_cache_status_rows({k:cs._svi_refresh_cache_delta(v,v) for k,v in snap.items()},snap)
    assert next(r for r in rows if r.name=='scaling').status == 'direct'


@pytest.mark.parametrize('disabled', ['stage0','no_members','no_parameters'])
def test_direct_scatter_zero_cases(disabled):
    ev, theta = evaluator('full_flat')
    if disabled=='stage0': ev.state.perturbation_discovery_stage0=True
    elif disabled=='no_members': ev.cached_scaling_component_indices=np.array([],dtype=int)
    else: ev.scaling_scatter_param_indices=np.array([],dtype=int)
    result=jax.jit(lambda t: ev._direct_scaling_scatter_covariance(ev._physical_parameter_vector(t),ev.flat_critical_arc_data))(jnp.asarray(theta))
    for x in result.values(): np.testing.assert_array_equal(x,0.)


@pytest.mark.parametrize('variation', ['cut', 'softening', 'shape', 'center'])
def test_direct_covariance_outer_parameter_gradients(variation):
    ev, theta = evaluator('full_flat')
    ev.packed_spec_jax['sigma_ref_param_index'] = jnp.array([-1,-1])
    if variation == 'cut':
        ev.packed_spec_jax['cut_ref_param_index'] = jnp.array([-1,0])
        theta[0] = 1.001  # Valid but close to the radius-ordering boundary.
    elif variation == 'softening':
        ev.log_softening_length_param_index = 0
        theta[0] = np.log(.2)
    elif variation == 'shape':
        ev.packed_spec_jax['e1_param_index'] = jnp.array([-1,0])
        theta[0] = .15
    else:
        ev.packed_spec_jax['x_center_param_index'] = jnp.array([-1,0])
        theta[0] = 1.2
    def fn(t):
        result = ev._direct_scaling_scatter_covariance(ev._physical_parameter_vector(t),ev.flat_critical_arc_data)
        return sum(jnp.sum(x) for x in result.values())
    gradient=jax.jit(jax.grad(fn))(jnp.asarray(theta))
    assert np.isfinite(gradient).all()
    step=1e-5
    plus,minus=theta.copy(),theta.copy()
    plus[0]+=step; minus[0]-=step
    fd=(float(fn(jnp.asarray(plus)))-float(fn(jnp.asarray(minus))))/(2*step)
    np.testing.assert_allclose(gradient[0],fd,rtol=2e-3,atol=1e-6)
    actual=ev._direct_scaling_scatter_covariance(ev._physical_parameter_vector(jnp.asarray(theta)),ev.flat_critical_arc_data)
    for key,value in reference(ev,theta,eps=1e-5).items():
        np.testing.assert_allclose(actual[key],value,rtol=2e-3,atol=1e-6)


def test_multiple_independent_members_and_reporting_ignore_cache():
    ev, theta = evaluator('full_flat')
    ev.cached_scaling_component_indices=np.array([0,1],dtype=np.int32)
    for key,value in {'component_family':[1,1], 'sigma_ref_base':[900.,120.],
                      'core_ref_base':[5.,1.], 'cut_ref_base':[100.,20.]}.items():
        ev.packed_spec_jax[key]=jnp.asarray(value)
    actual=ev._direct_scaling_scatter_covariance(ev._physical_parameter_vector(jnp.asarray(theta)),ev.flat_critical_arc_data)
    for key,value in reference(ev,theta).items():
        np.testing.assert_allclose(actual[key],value,rtol=2e-5,atol=1e-8)
    before=ev._family_source_summary(theta)
    ev.scaling_scatter_cache_by_z={z:{k:np.full(2,np.nan) for k in cs.SCALING_SCATTER_CACHE_KEYS} for z in (2.,3.)}
    after=ev._family_source_summary(theta)
    for family in before:
        for key in ('source_x','source_y','source_plane_rms'):
            np.testing.assert_allclose(before[family][key],after[family][key])


def test_invalid_full_flat_model_has_rejection_value_and_finite_gradient():
    ev, theta = evaluator('full_flat')
    ev.packed_spec_jax['cut_ref_param_index'] = jnp.array([-1,0])
    theta[0] = .1
    value, gradient = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(jnp.asarray(theta))
    assert value == cs.BAD_LOG_LIKE
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient, 0.)


def test_arc_reporting_uses_direct_covariance(monkeypatch):
    ev, theta = evaluator('full_flat')
    ev.sample_likelihood_mode = cs.SAMPLE_LIKELIHOOD_CRITICAL_ARC_ANISOTROPIC_IMAGE_PLANE
    monkeypatch.setattr(cs, '_arc_aware_image_support_from_local_linearization',
                        lambda *a, **kw: {key: np.asarray(kw[key]) for key in ('scatter_cov00','scatter_cov01','scatter_cov11')})
    family=ev.state.family_data[0]
    before=ev._arc_aware_image_support_details(theta,family,0.,0.)
    assert 'scatter_cov00' in before and np.any(before['scatter_cov00'] > 0)
    ev.scaling_scatter_cache_by_z={z:{k:np.full(2,np.nan) for k in cs.SCALING_SCATTER_CACHE_KEYS} for z in (2.,3.)}
    ev.flat_critical_arc_data=replace(ev.flat_critical_arc_data, **{
        f'{field}_scatter_cov{ij}':jnp.full_like(ev.flat_critical_arc_data.x_obs,jnp.nan)
        for field in ('sigma','mass') for ij in ('00','01','11')})
    after=ev._arc_aware_image_support_details(theta,family,0.,0.)
    for key in before: np.testing.assert_array_equal(before[key],after[key])


def test_both_scatter_amplitudes_have_current_covariance_gradients():
    base, _ = evaluator('full_flat')
    state=base.state
    mass_index=len(state.parameter_specs)
    state.parameter_specs=[*state.parameter_specs,replace(state.parameter_specs[-1],
        name='potfile.mass_log_scatter',sample_name='potfile_mass_log_scatter',field='mass_log_scatter')]
    indices=state.packed_lens_spec.mass_log_scatter_param_index.copy()
    indices[1]=mass_index
    state.packed_lens_spec=replace(state.packed_lens_spec,mass_log_scatter_param_index=indices)
    ev=cs.ClusterJAXEvaluator(state,.1,sampling_engine='full_flat',sample_likelihood_mode='source',source_plane_covariance_mode='unit')
    theta=cs._default_theta(state.parameter_specs)
    fn=jax.jit(jax.value_and_grad(ev._source_loglike_impl))
    value,grad=fn(jnp.asarray(theta))
    assert value>cs.BAD_LOG_LIKE and np.isfinite(grad).all()
    for index in (mass_index-1,mass_index):
        plus,minus=theta.copy(),theta.copy()
        plus[index]+=.001; minus[index]-=.001
        fd=(float(fn(jnp.asarray(plus))[0])-float(fn(jnp.asarray(minus))[0]))/.002
        assert abs(grad[index])>0
        np.testing.assert_allclose(grad[index],fd,rtol=1e-4,atol=1e-6)
