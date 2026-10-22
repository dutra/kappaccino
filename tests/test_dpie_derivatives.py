"""Compare the analytic kernel to independent autodiff of JAXtronomy."""
import jax
import jax.numpy as jnp
import numpy as np

from jaxtronomy.LensModel.Profiles.dpie_nie_e import DPIE_NIE_E
from lenscluster.dpie_derivatives import dpie_fields_and_third_derivatives

jax.config.update('jax_enable_x64', True)


def reference_fields(p, xy):
    alpha = jnp.stack(DPIE_NIE_E.derivatives(xy[:, 0], xy[:, 1], *p), -1)
    def hessian(position):
        h = DPIE_NIE_E.hessian(position[:, 0], position[:, 1], *p)
        return jnp.stack((jnp.stack(h[:2], -1), jnp.stack(h[2:], -1)), -2)
    du = jnp.stack([jax.jvp(hessian, (xy,),
        (jnp.broadcast_to(jnp.asarray(direction), xy.shape),))[1]
        for direction in ((1., 0.), (0., 1.))], -1)
    return alpha, hessian(xy), du


def analytic_fields(p, xy):
    return dpie_fields_and_third_derivatives(xy[:, 0], xy[:, 1], *p)


def assert_scaled_close(actual, expected, tolerance=1e-12):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.max(np.abs(a-b)/(1+np.abs(b)), initial=0) < tolerance


def test_analytic_dpie_fields_and_parameter_gradients_400_profiles():
    rng = np.random.default_rng(91)
    parameters, positions = [], []
    for i in range(400):
        a = 10**rng.uniform(-3, 1)
        b = a * (1 + 10**rng.uniform(-10, 4))
        e = 0. if i % 5 == 0 else (1e-10 if i % 7 == 0 else rng.uniform(0, .7))
        angle = rng.uniform(-np.pi, np.pi)
        cx, cy = rng.normal(size=2)
        parameters.append([10**rng.uniform(-2, 1), a, b, e*np.cos(angle), e*np.sin(angle), cx, cy])
        positions.append([[cx, cy], [cx+a*1e-6, cy-a*1e-6], [cx+a*.1, cy+a*.2],
                          [cx+a, cy-a], [cx+b, cy+b], [cx+b*1e3, cy-b*1e3]])
    p, xy = jnp.asarray(parameters), jnp.asarray(positions)
    assert_scaled_close(jax.jit(jax.vmap(analytic_fields))(p, xy),
                        jax.jit(jax.vmap(reference_fields))(p, xy))
    assert_scaled_close(jax.jit(jax.vmap(jax.jacfwd(analytic_fields)))(p, xy),
                        jax.jit(jax.vmap(jax.jacfwd(reference_fields)))(p, xy))


def test_analytic_dpie_parameter_hessian_and_finite_differences():
    p = jnp.array([2., .3, 9., 0., 0., .2, -.1])
    xy = jnp.array([[.2, -.1], [.3, .4], [-3., 4.]])
    def objective(fn, p):
        return sum(jnp.sum(a**2) for a in fn(p, xy))
    for fn in (analytic_fields, reference_fields):
        result = jax.jit(jax.hessian(lambda p: objective(fn, p)))(p)
        if fn is analytic_fields:
            actual = result
        else:
            assert_scaled_close(actual, result, 2e-11)
    # Independent central differences of JAXtronomy's Hessian in image space.
    du = analytic_fields(p, xy)[2]
    for axis in range(2):
        step = jnp.eye(2)[axis] * 1e-5
        difference = (reference_fields(p, xy+step)[1] - reference_fields(p, xy-step)[1]) / 2e-5
        np.testing.assert_allclose(du[..., axis], difference, rtol=2e-7, atol=2e-7)
