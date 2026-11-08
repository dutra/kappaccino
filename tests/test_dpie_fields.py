"""Analytic dPIE alpha, U and dU against the jaxtronomy kernel differentiated by forward mode."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from kappaccino.lens import lensing


def _reference(x, y, sigma0, a, b, e1, e2, cx, cy):
    def alpha(p):
        ax, ay = lensing.DPIENIE.derivatives(p[0], p[1], sigma0, a, b, e1, e2, cx, cy)
        return jnp.stack((ax, ay))
    u = jax.jacfwd(alpha)
    du = jax.jacfwd(u)
    pts = jnp.stack((x, y), -1)
    return jax.vmap(alpha)(pts), jax.vmap(u)(pts), jax.vmap(du)(pts)


def test_analytic_fields_match_forward_mode():
    rng = np.random.default_rng(3)
    x, y = (jnp.asarray(rng.uniform(-30, 30, 40)) for _ in range(2))
    for sigma0, a, b, e1, e2, cx, cy in [(1.2, .3, 9., .1, -.2, 1.5, -2.), (.05, .01, 40., -.3, .25, -8., 4.), (3., 1., 2., 0., 0., 0., 0.)]:
        ref = _reference(x, y, sigma0, a, b, e1, e2, cx, cy)
        got = lensing.dpie_fields(x, y, sigma0, a, b, e1, e2, cx, cy)
        fixed = lensing.dpie_fields_from_geometry(lensing.dpie_geometry(x, y, e1, e2, cx, cy), sigma0, a, b)
        for r, g, f in zip(ref, got, fixed):
            np.testing.assert_allclose(g, r, rtol=1e-11, atol=1e-13)
            np.testing.assert_array_equal(g, f)
        alpha, u, du = lensing.dpie_fields(x, y, sigma0, a, b, e1, e2, cx, cy, derivative=False)
        assert du is None
        np.testing.assert_array_equal(alpha, got[0])
        np.testing.assert_array_equal(u, got[1])


def test_amplitude_and_radius_gradients_match():
    x, y = jnp.array([2., -5., 11.]), jnp.array([-3., 7., .5])
    geom = lensing.dpie_geometry(x, y, .15, -.1, .7, -.4)
    ref = lambda p: jnp.stack(lensing.DPIENIE.derivatives(x, y, p[0], p[1], p[2], .15, -.1, .7, -.4), -1).sum()
    new = lambda p: lensing.dpie_fields_from_geometry(geom, p[0], p[1], p[2])[0].sum()
    p = jnp.array([.8, .2, 15.])
    np.testing.assert_allclose(jax.grad(new)(p), jax.grad(ref)(p), rtol=1e-12)
