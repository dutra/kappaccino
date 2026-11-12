"""Elliptical NFW cluster halos: kernel, physical normalization and the end-to-end model."""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from kappaccino.lens.profiles import nfw
import numpyro.distributions as dist

from kappaccino import priors
from kappaccino.config import NFWHaloConfig
from kappaccino.likelihood.evaluator import Evaluator
from kappaccino.lens.cosmology import flat_wcdm_config, kpc_per_arcsec_from_config
from kappaccino.lens.profiles.nfw import nfw_angular_parameters
from kappaccino.core.state import build_state


def _circular_nfw_deflection(r, rs, alpha_rs):
    """|alpha|(r) of a circular NFW halo with deflection alpha_rs at the scale radius rs."""
    x = np.asarray(r, dtype=float) / rs
    h = np.log(x / 2)
    inner = x < 1
    h[inner] += 2 / np.sqrt(1 - x[inner] ** 2) * np.arctanh(np.sqrt((1 - x[inner]) / (1 + x[inner])))
    h[~inner] += 2 / np.sqrt(x[~inner] ** 2 - 1) * np.arctan(np.sqrt((x[~inner] - 1) / (x[~inner] + 1)))
    return alpha_rs * h / (x * (1 - np.log(2)))


def test_nfw_kernel_matches_analytic_circular_halo_and_finite_differences():
    # The 44-term CSE decomposition reproduces the circular NFW deflection to 5e-6 over five decades in radius.
    rs, cx, cy = 60.0, 3.0, -1.5
    r, phi = np.geomspace(1e-2, 1e3, 200), np.linspace(0, 2 * np.pi, 200)
    alpha_rs = np.linspace(5, 10, 200)
    alpha = nfw.fields(jnp.asarray(r * np.cos(phi) + cx), jnp.asarray(r * np.sin(phi) + cy), rs, jnp.asarray(alpha_rs),
                               0.0, 0.0, cx, cy, derivative=False)[0]
    magnitude = _circular_nfw_deflection(r, rs, alpha_rs)
    np.testing.assert_allclose(alpha, np.stack((magnitude * np.cos(phi), magnitude * np.sin(phi)), -1), rtol=1e-5, atol=1e-5 * magnitude.min())
    rng = np.random.default_rng(5)
    x, y = (jnp.asarray(rng.uniform(-40, 40, 30)) for _ in range(2))
    rs, e1, e2 = 60.0, 0.12, -0.2
    alpha_rs = jnp.asarray(rng.uniform(5, 10, 30))                      # per-point normalization, as per image
    alpha, u, du = nfw.fields(x, y, rs, alpha_rs, e1, e2, cx, cy)
    h = 1e-4
    for l, (dx, dy) in enumerate(((h, 0.0), (0.0, h))):
        plus = nfw.fields(x + dx, y + dy, rs, alpha_rs, e1, e2, cx, cy, derivative=False)[0]
        minus = nfw.fields(x - dx, y - dy, rs, alpha_rs, e1, e2, cx, cy, derivative=False)[0]
        np.testing.assert_allclose(u[..., l], (plus - minus) / (2 * h), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(u[:, 0, 1], u[:, 1, 0], rtol=1e-12)
    for l, (dx, dy) in enumerate(((h, 0.0), (0.0, h))):
        plus = nfw.fields(x + dx, y + dy, rs, alpha_rs, e1, e2, cx, cy, derivative=False)[1]
        minus = nfw.fields(x - dx, y - dy, rs, alpha_rs, e1, e2, cx, cy, derivative=False)[1]
        np.testing.assert_allclose(du[..., l], (plus - minus) / (2 * h), rtol=1e-5, atol=1e-9)
    alpha_only, u_only, none = nfw.fields(x, y, rs, alpha_rs, e1, e2, cx, cy, derivative=False)
    assert none is None
    np.testing.assert_array_equal(alpha_only, alpha)
    # Gradients with respect to the ellipticity are finite at e = 0 and match finite differences away from it.
    total = lambda e: jnp.sum(nfw.fields(x, y, rs, alpha_rs, e[0], e[1], cx, cy, derivative=False)[0] ** 2)
    assert np.all(np.isfinite(np.asarray(jax.grad(total)(jnp.zeros(2)))))
    e = jnp.array([0.12, -0.2])
    g = np.asarray(jax.grad(total)(e))
    for k in range(2):
        step = jnp.zeros(2).at[k].set(1e-6)
        np.testing.assert_allclose(g[k], (total(e + step) - total(e - step)) / 2e-6, rtol=1e-5)


def test_analytic_jacobian_derivative_matches_forward_mode():
    """dU against two forward-mode derivatives of the kernel's own Jacobian: elliptical, flat and round halos, points
    on the axes and at the centre, and a per-point normalization."""
    rng = np.random.default_rng(8)
    x = jnp.concatenate((jnp.asarray(rng.uniform(-60, 60, 40)), jnp.asarray([3.0, 3.0 + 1e-9, 20.0, 3.0])))
    y = jnp.concatenate((jnp.asarray(rng.uniform(-60, 60, 40)), jnp.asarray([-4.0, -4.0, -4.0, 10.0])))
    alpha_rs = jnp.asarray(rng.uniform(5, 15, x.size))
    for e1, e2 in ((0.21, -0.13), (0.55, 0.3), (0.2, 0.0), (0.0, 0.0)):
        parameters = (35.0, alpha_rs, e1, e2, 3.0, -4.0)

        def jacobian_at(point, scale):
            return nfw.fields(point[None, 0], point[None, 1], 35.0, scale, e1, e2, 3.0, -4.0, derivative=False)[1][0]

        alpha, u, du = nfw.fields(x, y, *parameters)
        expected = jax.vmap(jax.jacfwd(jacobian_at))(jnp.stack((x, y), -1), alpha_rs)
        assert np.isfinite(np.asarray(du)).all()
        np.testing.assert_allclose(du, expected, rtol=1e-11, atol=1e-13 * float(jnp.max(jnp.abs(expected))))
        np.testing.assert_allclose(du, jnp.swapaxes(du, -1, -2), rtol=0, atol=1e-13 * float(jnp.max(jnp.abs(du))))   # symmetric in every pair
        np.testing.assert_allclose(u, nfw.fields(x, y, *parameters, derivative=False)[1], rtol=1e-13, atol=1e-15)
    # The parameter gradients of dU match those of the forward-mode version.
    point = jnp.asarray([12.0, -7.0])
    analytic = lambda p: nfw.fields(point[None, 0], point[None, 1], p[0], 9.0, p[1], p[2], p[3], p[4])[2][0]
    forward = lambda p: jax.jacfwd(lambda xy: nfw.fields(xy[None, 0], xy[None, 1], p[0], 9.0, p[1], p[2], p[3], p[4], derivative=False)[1][0])(point)
    p = jnp.asarray([35.0, 0.21, -0.13, 3.0, -4.0])
    np.testing.assert_allclose(jax.jacfwd(analytic)(p), jax.jacfwd(forward)(p), rtol=1e-9, atol=1e-13)


def test_nfw_physical_normalization_matches_astropy():
    # alpha(r_s) = 4 kappa_s theta_s (1 - ln 2), with kappa_s = rho_s r_s / Sigma_crit, from astropy distances and constants.
    import astropy.units as u
    from astropy.constants import G, c as light
    from astropy.cosmology import FlatLambdaCDM
    z_lens, z_source, m200, conc = 0.5, 2.0, 1.0e15, 4.0
    astro = FlatLambdaCDM(H0=70.0, Om0=0.3)
    rho_c = astro.critical_density(z_lens).to(u.Msun / u.kpc ** 3).value
    r200 = (3 * m200 / (4 * np.pi * 200 * rho_c)) ** (1 / 3)
    rs_kpc = r200 / conc
    rho_s = 200 / 3 * rho_c * conc ** 3 / (np.log1p(conc) - conc / (1 + conc))
    d_l, d_s = astro.angular_diameter_distance(z_lens), astro.angular_diameter_distance(z_source)
    d_ls = astro.angular_diameter_distance(z_lens, z_source)
    sigma_crit = (light ** 2 / (4 * np.pi * G) * d_s / (d_l * d_ls)).to(u.Msun / u.kpc ** 2).value
    theta_s = (rs_kpc * u.kpc / d_l).to(u.dimensionless_unscaled).value * u.rad.to(u.arcsec)
    alpha_rs_ref = 4 * rho_s * rs_kpc / sigma_crit * theta_s * (1 - np.log(2))
    cosmo = flat_wcdm_config(h0=70.0, om0=0.3)
    rs_angle, alpha_rs = nfw_angular_parameters(np.log10(m200), rs_kpc, z_lens=z_lens, z_source=z_source,
                                                cosmo_config=cosmo, kpc_per_arcsec=kpc_per_arcsec_from_config(z_lens, cosmo))
    np.testing.assert_allclose(rs_angle, theta_s, rtol=3e-3)
    np.testing.assert_allclose(alpha_rs, alpha_rs_ref, rtol=3e-3)


def test_nfw_cluster_halo_end_to_end(synthetic_config):
    halo = NFWHaloConfig(id="halo", x_centre=0.0, y_centre=0.0, ellipticite=0.2, angle_pos=0.0, log10_m200=14.6, scale_radius_kpc=300.0, z_lens=0.3,
                         priors={"log10_m200": dist.Uniform(14.0, 15.2),
                                 "scale_radius_kpc": priors.lognormal_from_moments(300.0, 100.0, low=100.0, high=600.0),
                                 **priors.ellipticity_box(0.6)})
    config = replace(synthetic_config, model=replace(synthetic_config.model, large_halos=(halo,))).validate()
    state = build_state(config)
    names = [s.name for s in state.parameter_specs]
    assert {"halo.log10_m200", "halo.scale_radius_kpc", "halo.e1", "halo.e2"} <= set(names)
    assert int(np.asarray(state.packed_lens_spec.profile_type)[0]) == 12
    evaluator = Evaluator(state)
    theta = jnp.asarray(state.initial_theta)
    value, grad = evaluator.value_and_grad(theta)
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad)))
    for name in ("halo.log10_m200", "halo.scale_radius_kpc"):
        assert float(grad[names.index(name)]) != 0.0
