"""Block-metric layout and unconstrained curvature against analytic models."""
from dataclasses import replace
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS
from numpyro.infer.hmc_util import _initialize_mass_matrix
import pytest

from lenscluster.model import ParameterSpec
from lenscluster.smoothed_sampling import prepare_block_mass


def spec(name, family='large', site=None, index=None):
    return ParameterSpec(name=name, sample_name=name, potential_id='test', profile_type=0,
                         field='test', prior_kind='normal', lower=-np.inf, upper=np.inf,
                         step=.1, mean=0., std=1., component_family=family,
                         sample_site_name=site, sample_site_index=index)


def block_fixture():
    # Deliberately put global parameter order opposite NumPyro's sorted order.
    specs = [spec('z_global'), spec('a_global'),
             spec('source_x', 'source_position', 'source_positions', 0),
             spec('source_y', 'source_position', 'source_positions', 1),
             spec('member_0', 'member_scatter_offset', 'member_velocity_offsets', 0)]
    precision = np.array([[4., 1.], [1., 3.]])
    def model():
        z = numpyro.sample('z_global', dist.Normal(0, 1))
        a = numpyro.sample('a_global', dist.Normal(0, 1))
        u = numpyro.sample('source_positions', dist.Normal(jnp.zeros(2), 1).to_event(1))
        offset = numpyro.sample('member_velocity_offsets', dist.Normal(jnp.zeros(1), 1).to_event(1))
        x = jnp.array([a, z])
        numpyro.factor('coupled', -.5*x@jnp.asarray(precision-np.eye(2))@x
                       + .6*a*u[0]+.2*z*offset[0])
    return specs, model, precision


def test_partition_sorting_and_fixed_latent_curvature():
    specs, model, precision = block_fixture()
    dense, inverse, diagnostic = prepare_block_mass(None, model, specs, np.array([.2, -.1, .4, .3, -.2]))
    assert dense == [('a_global', 'z_global')]
    diagonal = ('member_velocity_offsets', 'source_positions')
    assert set(inverse) == {dense[0], diagonal}
    np.testing.assert_allclose(inverse[dense[0]], np.linalg.inv(precision), rtol=2e-14)
    np.testing.assert_array_equal(inverse[diagonal], np.ones(3))
    assert diagnostic['dim'] == 2 and diagnostic['diagonal_dim'] == 3
    assert diagnostic['used'] and not diagnostic['finite_difference']
    # Installed NumPyro accepts this exact dictionary and block shape.
    initialized, sqrt, sqrt_inv = _initialize_mass_matrix(
        dict(a_global=jnp.array(-.1), z_global=jnp.array(.2),
             source_positions=jnp.array([.4,.3]), member_velocity_offsets=jnp.array([-.2])),
        inverse.copy(), dense)
    np.testing.assert_allclose(initialized[dense[0]], np.linalg.inv(precision))
    assert sqrt[diagonal].shape == sqrt_inv[diagonal].shape == (3,)


def test_actual_numpyro_kernel_initialization():
    specs, model, _ = block_fixture()
    dense, inverse, _ = prepare_block_mass(None, model, specs, np.zeros(5))
    kernel = NUTS(model, dense_mass=dense, inverse_mass_matrix=inverse,
                  adapt_mass_matrix=False, adapt_step_size=False)
    state = kernel.init(jax.random.PRNGKey(7), num_warmup=0,
                        init_params=dict(a_global=jnp.array(0.),z_global=jnp.array(0.),
                                         source_positions=jnp.zeros(2),member_velocity_offsets=jnp.zeros(1)))
    assert set(state.adapt_state.inverse_mass_matrix) == set(inverse)
    assert np.isfinite(state.potential_energy)


def test_global_hyperparameters_remain_dense():
    specs = [spec('sigma_src','source_scatter'), spec('s_sigma','member_velocity_scatter')]
    def model():
        numpyro.sample('sigma_src',dist.Normal(0,1))
        numpyro.sample('s_sigma',dist.Normal(0,1))
    dense, inverse, diagnostic = prepare_block_mass(None,model,specs,np.zeros(2))
    assert dense == [('s_sigma','sigma_src')]
    np.testing.assert_array_equal(inverse[dense[0]],np.eye(2))
    assert diagnostic['diagonal_dim'] == 0


def test_no_globals_and_no_parameters():
    specs = [spec('u','source_position'),spec('z','member_scatter_offset')]
    def model():
        numpyro.sample('u',dist.Normal(0,1))
        numpyro.sample('z',dist.Normal(0,1))
    dense,inverse,diagnostic=prepare_block_mass(None,model,specs,np.zeros(2))
    assert dense == [] and list(inverse) == [('u','z')]
    np.testing.assert_array_equal(inverse[('u','z')],np.ones(2))
    assert not diagnostic['used'] and diagnostic['reason']=='no_global_parameters'
    dense,inverse,diagnostic=prepare_block_mass(None,lambda:None,[],np.zeros(0))
    assert dense == [] and inverse == {} and diagnostic['total_dim']==0


def test_negative_curvature_is_floored_without_absolute_value():
    specs=[spec('a'),spec('b')]
    def model():
        a=numpyro.sample('a',dist.Normal(0,1))
        b=numpyro.sample('b',dist.Normal(0,1))
        numpyro.factor('curvature',1.5*a*a-1.5*b*b)
    dense,inverse,diagnostic=prepare_block_mass(None,model,specs,np.zeros(2))
    np.testing.assert_allclose(inverse[dense[0]],np.diag([1.,.25]))
    assert diagnostic['n_negative_eigenvalues']==1 and diagnostic['n_floored_eigenvalues']==1


def test_uniform_support_uses_unconstrained_potential():
    s=replace(spec('x'),prior_kind='uniform',lower=-1.,upper=1.)
    def model():
        x=numpyro.sample('x',dist.Uniform(-1,1))
        numpyro.factor('likelihood',-.5*(x/.2)**2)
    dense,inverse,_=prepare_block_mass(None,model,[s],np.array([0.]))
    # x=tanh(u/2). The likelihood curvature is .25/.2^2; -log(dx/du)
    # adds .5. Seeding the constrained Hessian would incorrectly give 1/25.
    np.testing.assert_allclose(inverse[dense[0]],np.array([[1/(.25/.2**2+.5)]]),rtol=2e-14)
    with pytest.raises(ValueError,match='boundary'):
        prepare_block_mass(None,model,[s],np.array([1.]))


def test_nonfinite_and_mixed_sites_rejected():
    def bad_model():
        x=numpyro.sample('x',dist.Normal(0,1))
        numpyro.factor('bad',jnp.sqrt(x))
    with pytest.raises(RuntimeError,match='non-finite'):
        prepare_block_mass(None,bad_model,[spec('x')],np.array([0.]))
    specs=[spec('a','source_position','mixed',0),spec('b','large','mixed',1)]
    with pytest.raises(ValueError,match='mixes global'):
        prepare_block_mass(None,lambda:None,specs,np.zeros(2))


def test_finite_difference_fallback(monkeypatch):
    specs,model,precision=block_fixture()
    def unsupported(*args,**kwargs):
        raise TypeError('forward derivative unavailable')
    monkeypatch.setattr(jax,'jacfwd',unsupported)
    dense,inverse,diagnostic=prepare_block_mass(None,model,specs,np.zeros(5))
    np.testing.assert_allclose(inverse[dense[0]],np.linalg.inv(precision),rtol=2e-12)
    assert diagnostic['finite_difference']
