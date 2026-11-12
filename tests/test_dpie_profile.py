"""The dPIE kernel against an 80-digit principal-axis reference, its own forward-mode derivatives and its limits."""
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from kappaccino.lens.profiles import dpie


def reference_nie(x, y, theta, e1, e2, r):
    """Deflection of a non-singular isothermal ellipse from the principal-axis equations, in 80 digits; no q clipping."""
    x, y, theta, e1, e2, r = map(mp.mpf, (x, y, theta, e1, e2, r))
    e = mp.sqrt(e1 * e1 + e2 * e2)
    if not e:
        radius = mp.sqrt(r * r + x * x + y * y)
        return theta * x / (radius + r), theta * y / (radius + r)
    q = (1 - e) / (1 + e)
    phi = mp.atan2(e2, e1) / 2
    c, s = mp.cos(phi), mp.sin(phi)
    xr, yr = c * x + s * y, -s * x + c * y
    b, core, t = theta * mp.sqrt(q), r / mp.sqrt(q), mp.sqrt(1 - q * q)
    psi = mp.sqrt(q * q * (core * core + xr * xr) + yr * yr)
    ax, ay = b / t * mp.atan(t * xr / (psi + core)), b / t * mp.atanh(t * yr / (psi + q * q * core))
    return c * ax - s * ay, s * ax + c * ay


def reference_dpie(x, y, sigma0, a, b, e1, e2):
    """The dPIE as the difference of two isothermal ellipses of core ``a`` and ``b``."""
    a, b, sigma0 = map(mp.mpf, (a, b, sigma0))
    theta = 2 * sigma0 * a * b / (b - a)
    inner, outer = reference_nie(x, y, theta, e1, e2, a), reference_nie(x, y, theta, e1, e2, b)
    return tuple(i - o for i, o in zip(inner, outer))


def kernel(x, y, *parameters, derivative=True):
    """The kernel at one point, as arrays without the point axis."""
    alpha, u, du = dpie.fields(jnp.atleast_1d(jnp.asarray(x, dtype=jnp.float64)), jnp.atleast_1d(jnp.asarray(y, dtype=jnp.float64)),
                               *parameters, derivative=derivative)
    return alpha[0], u[0], (None if du is None else du[0])


def test_deflection_matches_the_high_precision_reference():
    rng = np.random.default_rng(385)
    with mp.workdps(80):
        for i in range(80):
            x, y = rng.normal(size=2) * 10
            e = 0.0 if i < 3 else rng.uniform(1e-12, 0.7)
            angle = rng.uniform(-np.pi, np.pi)
            e1, e2 = e * np.cos(angle), e * np.sin(angle)
            a = np.exp(rng.uniform(-4, 1))
            b = a + np.exp(rng.uniform(-5, 3))
            sigma0 = np.exp(rng.uniform(-3, 3))
            expected = np.array(reference_dpie(x, y, sigma0, a, b, e1, e2), float)
            np.testing.assert_allclose(kernel(x, y, sigma0, a, b, e1, e2, 0.0, 0.0)[0], expected, rtol=1e-13, atol=2e-15)


@pytest.mark.parametrize("parameters", [(2.0, 0.3, 8.0, 0.15, -0.2), (2.0, 0.3, 8.0, 0.3, 0.0), (2.0, 0.3, 0.3000000001, 0.0, 0.0)])
def test_jacobian_matches_the_high_precision_reference(parameters):
    with mp.workdps(80):
        for x, y in [(1.0, 0.0), (0.0, 1.0), (2.0, -3.0), (1e6, -2e6)]:
            expected = np.array([[mp.diff(lambda xx: reference_dpie(xx, y, *parameters)[i], mp.mpf(x)),
                                  mp.diff(lambda yy: reference_dpie(x, yy, *parameters)[i], mp.mpf(y))] for i in range(2)], float)
            u = np.asarray(kernel(x, y, *parameters, 0.0, 0.0)[1])
            np.testing.assert_allclose(u, expected, rtol=1e-11, atol=1e-14)
            np.testing.assert_allclose(u, u.T, rtol=0, atol=1e-15)


def test_analytic_derivatives_match_forward_mode_and_the_geometry_split():
    rng = np.random.default_rng(3)
    x, y = (jnp.asarray(rng.uniform(-30, 30, 40)) for _ in range(2))
    points = jnp.stack((x, y), -1)
    for parameters in [(1.2, 0.3, 9.0, 0.1, -0.2, 1.5, -2.0), (0.05, 0.01, 40.0, -0.3, 0.25, -8.0, 4.0), (3.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0)]:
        alpha_at = lambda p: kernel(p[0], p[1], *parameters, derivative=False)[0]
        alpha, u, du = dpie.fields(x, y, *parameters)
        np.testing.assert_allclose(u, jax.vmap(jax.jacfwd(alpha_at))(points), rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(du, jax.vmap(jax.jacfwd(jax.jacfwd(alpha_at)))(points), rtol=1e-10, atol=1e-12)
        sigma0, a, b, e1, e2, cx, cy = parameters
        fixed = dpie.fields_from_geometry(dpie.geometry(x, y, e1, e2, cx, cy), sigma0, a, b)
        for value, same in zip((alpha, u, du), fixed):
            np.testing.assert_array_equal(value, same)
        alpha_only, u_only, none = dpie.fields(x, y, *parameters, derivative=False)
        assert none is None
        np.testing.assert_array_equal(alpha_only, alpha)
        np.testing.assert_array_equal(u_only, u)


def test_batched_kernel_and_registry_interface():
    from kappaccino.lens import profiles
    x, y = jnp.linspace(-5.0, 5.0, 7), jnp.linspace(3.0, -4.0, 7)
    batch = [jnp.asarray(v) for v in ([1.0, 2.0], [0.2, 0.3], [5.0, 9.0], [0.1, 0.0], [-0.2, 0.0], [0.5, -1.0], [0.0, 2.0])]
    for flag in (True, False):
        rows = dpie.grouped_fields[flag](x, y, *batch)
        for k in range(2):
            single = dpie.fields(x, y, *(v[k] for v in batch), derivative=flag)
            for got, expected in zip(rows, single):
                assert (got is None and expected is None) or np.array_equal(got[k], expected)
    for code, entry in profiles.PROFILES.items():
        assert entry.code == code and entry.parameters == entry.kernel.PARAMETERS
        assert set(entry.kernel.grouped_fields) == {True, False} and callable(entry.kernel.fields)


def test_extreme_geometry_is_finite_and_accurate():
    x, y = jnp.array([1e-12, 1.0, 1e6]), jnp.array([-1e-12, 0.0, -2e6])
    parameters = (2.0, 1.0, 1.0 + 1e-10, 0.0, 0.0)                      # nearly equal radii, exactly round
    alpha, u, du = dpie.fields(x, y, *parameters, 0.0, 0.0)
    assert np.isfinite(alpha).all() and np.isfinite(u).all() and np.isfinite(du).all()
    with mp.workdps(80):
        for i in range(3):
            expected = np.array(reference_dpie(float(x[i]), float(y[i]), *parameters), float)
            np.testing.assert_allclose(alpha[i], expected, rtol=3e-14, atol=1e-25)


def test_gradients_in_every_parameter_are_finite_at_circularity_and_match_differences():
    p = jnp.array([2.0, 0.3, 8.0, 0.0, 0.0, 0.2, -0.4])
    alpha = lambda p: kernel(1.0, -2.0, *p, derivative=False)[0]
    jacobian = jax.jit(jax.jacfwd(alpha))
    second = jax.jit(jax.jacfwd(jax.jacfwd(alpha)))(p)
    assert np.isfinite(second).all()
    for k in range(len(p)):
        step = 1e-5 * max(1.0, abs(float(p[k])))
        delta = jnp.zeros_like(p).at[k].set(step)
        np.testing.assert_allclose(second[..., k], (jacobian(p + delta) - jacobian(p - delta)) / (2 * step), rtol=3e-7, atol=2e-8)
    # The centre is a shift of the position.
    np.testing.assert_allclose(alpha(p), kernel(0.8, -1.6, *p[:5], 0.0, 0.0, derivative=False)[0], rtol=2e-15, atol=2e-15)


def test_series_switch_derivatives():
    """atan(sqrt(u)) / sqrt(u) through the switch to its series at |u| = 0.01."""
    with mp.workdps(80):
        for value in [0.0, 0.009999, 0.010001, -0.009999, -0.010001]:
            f = lambda u: jnp.real(dpie._atan_sqrt_ratio(u + 0j))
            reference = lambda u: mp.hyper([mp.mpf("0.5"), 1], [mp.mpf("1.5")], -u)
            for order in range(4):
                np.testing.assert_allclose(f(jnp.asarray(value)), float(mp.diff(reference, mp.mpf(value), order)),
                                           rtol=(1e-9 if order == 3 else 3e-10), atol=2e-11)
                f = jax.grad(f)
