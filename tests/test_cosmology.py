"""The flat wCDM distances of kappaccino.lens.cosmology against astropy, and their derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, FlatwCDM

from kappaccino.lens import cosmology as c
from kappaccino.lens.profiles import nfw

Z = np.array([0.01, 0.05, 0.2, 0.348, 0.396, 0.507, 1.0, 2.0, 3.5, 6.0, 9.0, 12.0, 20.0])
# (H0, Om0, w0): the cosmologies of the clusters (M0416 and AS1063, Hera, Ares), and the corners of the cosmography priors.
LCDM = [(70.0, 0.3, -1.0), (72.0, 0.24, -1.0), (70.4, 0.272, -1.0), (67.7, 0.31, -1.0), (70.0, 0.05, -1.0), (70.0, 0.95, -1.0)]
WCDM = [(67.7, 0.31, -0.8), (70.0, 0.3, -2.0), (70.0, 0.3, -0.2), (70.0, 0.05, -2.0), (70.0, 0.95, -0.2), (70.0, 0.05, -0.2),
        (70.0, 0.95, -2.0), (73.0, 0.27, -1.2)]
# astropy integrates wCDM with scipy.quad (relative tolerance 1.5e-8) and has a closed form for flat LCDM.
RTOL = {"lcdm": 1.0e-11, "wcdm": 1.0e-8}


def astropy_cosmology(h0, om0, w0):
    return FlatLambdaCDM(H0=h0, Om0=om0) if w0 == -1.0 else FlatwCDM(H0=h0, Om0=om0, w0=w0)


CASES = [("lcdm", *p) for p in LCDM] + [("wcdm", *p) for p in WCDM]


@pytest.mark.parametrize("kind, h0, om0, w0", CASES)
def test_distances_match_astropy(kind, h0, om0, w0):
    reference, rtol = astropy_cosmology(h0, om0, w0), RTOL[kind]
    z = jnp.asarray(Z)
    np.testing.assert_allclose(c.flat_wcdm_efunc(z, om0, w0), reference.efunc(Z), rtol=1e-13)
    np.testing.assert_allclose(c.flat_wcdm_comoving_distance_mpc(z, h0, om0, w0), reference.comoving_distance(Z).value, rtol=rtol)
    np.testing.assert_allclose(c.flat_wcdm_angular_diameter_distance_mpc(z, h0, om0, w0),
                               reference.angular_diameter_distance(Z).value, rtol=rtol)
    np.testing.assert_allclose(c.flat_wcdm_kpc_per_arcsec(z, h0, om0, w0),
                               reference.kpc_proper_per_arcmin(Z).to_value(u.kpc / u.arcsec), rtol=rtol)


@pytest.mark.parametrize("kind, h0, om0, w0", CASES)
@pytest.mark.parametrize("z_lens", [0.348, 0.396, 0.507])
def test_lensing_distances_match_astropy(kind, h0, om0, w0, z_lens):
    reference, rtol = astropy_cosmology(h0, om0, w0), 10.0 * RTOL[kind]       # differences of distances lose a digit
    z_source = Z[Z > z_lens + 0.1]
    d_ls = reference.angular_diameter_distance(z_lens, z_source).value
    d_s, d_l = reference.angular_diameter_distance(z_source).value, reference.angular_diameter_distance(z_lens).value
    np.testing.assert_allclose(c.flat_wcdm_angular_diameter_distance_z1z2_mpc(z_lens, jnp.asarray(z_source), h0, om0, w0), d_ls, rtol=rtol)
    np.testing.assert_allclose(c.flat_wcdm_lensing_efficiency(z_lens, jnp.asarray(z_source), h0, om0, w0), d_ls / d_s, rtol=rtol)
    kpc_per_arcsec, efficiency, factor = c.flat_wcdm_lens_geometry_factors(z_lens, jnp.asarray(z_source), h0, om0, w0)
    np.testing.assert_allclose(kpc_per_arcsec, reference.kpc_proper_per_arcmin(z_lens).to_value(u.kpc / u.arcsec), rtol=rtol)
    np.testing.assert_allclose(efficiency, d_ls / d_s, rtol=rtol)
    np.testing.assert_allclose(factor, c.dpie_sigma0_factor_from_lensing_efficiency(d_ls / d_s), rtol=rtol)
    np.testing.assert_allclose(c.dpie_sigma0_factor(z_lens, jnp.asarray(z_source), h0, om0, w0), factor, rtol=1e-13)
    # Critical surface density per arcsec^2: exact against astropy's distances with this module's constants, and within
    # the differences of G and the solar mass (4e-4) against astropy's constants.
    ours = np.asarray(c.critical_surface_density_angle_msun_per_arcsec2(z_lens, jnp.asarray(z_source), h0, om0, w0))
    sigma_crit = c.C_LIGHT_M_S ** 2 / (4.0 * np.pi * c.G_SI) * d_s / (d_l * d_ls) * c.MPC_M / c.M_SUN_KG * (d_l * c.ARCSEC_TO_RAD) ** 2
    np.testing.assert_allclose(ours, sigma_crit, rtol=rtol)
    from astropy import constants as const
    astropy_sigma = (const.c ** 2 / (4.0 * np.pi * const.G) * (d_s * u.Mpc) / ((d_l * u.Mpc) * (d_ls * u.Mpc))).to(u.Msun / u.Mpc ** 2).value
    np.testing.assert_allclose(ours, astropy_sigma * (d_l * c.ARCSEC_TO_RAD) ** 2, rtol=1e-3)


@pytest.mark.parametrize("kind, h0, om0, w0", CASES)
def test_critical_density_matches_astropy(kind, h0, om0, w0):
    reference = astropy_cosmology(h0, om0, w0)
    expected = reference.critical_density(Z).to_value(u.Msun / u.kpc ** 3)
    np.testing.assert_allclose(nfw.flat_wcdm_critical_density_msun_kpc3(jnp.asarray(Z), h0, om0, w0), expected, rtol=1e-4)
    config = c.flat_wcdm_config(h0=h0, om0=om0, w0=w0)
    assert config["class"] == ("FlatLambdaCDM" if w0 == -1.0 else "FlatwCDM")
    np.testing.assert_allclose([nfw.critical_density_msun_kpc3(float(z), config) for z in Z], expected, rtol=1e-4)
    np.testing.assert_allclose(c.kpc_per_arcsec_from_config(0.396, config), reference.kpc_proper_per_arcmin(0.396).to_value(u.kpc / u.arcsec), rtol=1e-8)


def test_lower_quadrature_orders_converge():
    reference = FlatLambdaCDM(H0=70.0, Om0=0.3).comoving_distance(Z).value
    errors = [np.max(np.abs(np.asarray(c.flat_wcdm_comoving_distance_mpc(jnp.asarray(Z), 70.0, 0.3, -1.0, order=n)) / reference - 1.0))
              for n in (4, 8, 16, 32)]
    assert errors[0] > errors[1] > errors[2] and errors[2] < 1e-8 and errors[3] < 1e-12


def test_shapes_and_sources_in_front_of_the_lens():
    assert c.flat_wcdm_comoving_distance_mpc(1.0, 70.0, 0.3, -1.0).shape == ()
    assert c.flat_wcdm_comoving_distance_mpc(jnp.ones((3, 2)), 70.0, 0.3, -1.0).shape == (3, 2)
    assert float(c.flat_wcdm_comoving_distance_mpc(0.0, 70.0, 0.3, -1.0)) == 0.0
    efficiency = np.asarray(c.flat_wcdm_lensing_efficiency(0.4, jnp.asarray([0.2, 0.4, 0.8]), 70.0, 0.3, -1.0))
    assert np.isnan(efficiency[0]) and np.isnan(efficiency[1]) and efficiency[2] > 0
    assert np.isnan(np.asarray(c.flat_wcdm_angular_diameter_distance_z1z2_mpc(0.4, 0.2, 70.0, 0.3, -1.0)))


@pytest.mark.parametrize("om0, w0", [(0.3, -1.0), (0.15, -1.6), (0.7, -0.5)])
def test_derivatives_in_the_cosmology(om0, w0):
    """The gradients cosmography samples with: D_ls / D_s and kpc per arcsec in Om0 and w0, against astropy differences."""
    z_lens, z_source, step = 0.396, 2.0, 1.0e-4

    def astropy_values(om, w):
        cosmo = FlatwCDM(H0=70.0, Om0=om, w0=w)
        return np.array([(cosmo.angular_diameter_distance(z_lens, z_source) / cosmo.angular_diameter_distance(z_source)).value,
                         cosmo.kpc_proper_per_arcmin(z_lens).to_value(u.kpc / u.arcsec)])

    def ours(parameters):
        return jnp.stack([c.flat_wcdm_lensing_efficiency(z_lens, z_source, 70.0, parameters[0], parameters[1]),
                          c.flat_wcdm_kpc_per_arcsec(z_lens, 70.0, parameters[0], parameters[1])])

    jacobian = np.asarray(jax.jacfwd(ours)(jnp.asarray([om0, w0])))
    np.testing.assert_allclose(np.asarray(jax.jacrev(ours)(jnp.asarray([om0, w0]))), jacobian, rtol=1e-10)
    expected = np.stack([(astropy_values(om0 + step, w0) - astropy_values(om0 - step, w0)) / (2 * step),
                         (astropy_values(om0, w0 + step) - astropy_values(om0, w0 - step)) / (2 * step)], axis=1)
    np.testing.assert_allclose(jacobian, expected, rtol=2e-5)
    # The efficiency does not depend on H0; the kpc per arcsec scales as 1 / H0.
    d_h0 = np.asarray(jax.jacfwd(lambda h: jnp.stack([c.flat_wcdm_lensing_efficiency(z_lens, z_source, h, om0, w0),
                                                      c.flat_wcdm_kpc_per_arcsec(z_lens, h, om0, w0)]))(70.0))
    assert abs(d_h0[0]) < 1e-15
    np.testing.assert_allclose(d_h0[1], -float(c.flat_wcdm_kpc_per_arcsec(z_lens, 70.0, om0, w0)) / 70.0, rtol=1e-12)
