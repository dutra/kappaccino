from __future__ import annotations

from functools import lru_cache
from typing import Any

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# Gauss-Legendre order of the comoving-distance integral.  The integrand in ln(1 + z) is smooth, so 64 nodes give the
# distance to round-off out to z = 20 for 0.05 < Om0 < 0.95 and -2.5 < w0 < -0.2 (tests/test_cosmology.py).
DEFAULT_FLAT_WCDM_QUADRATURE_ORDER = 64
C_LIGHT_KM_S = 299792.458
C_LIGHT_M_S = 299792458.0
G_SI = 6.67384e-11
MPC_M = 3.08567758e22
M_SUN_KG = 1.9891e30
ARCSEC_TO_RAD = np.deg2rad(1.0 / 3600.0)


@lru_cache(maxsize=None)
def _gauss_legendre(order: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(int(order))


def flat_wcdm_config(
    *,
    h0: float = 70.0,
    om0: float = 0.3,
    ode0: float | None = None,
    w0: float = -1.0,
) -> dict[str, float | str]:
    ode0_value = 1.0 - float(om0) if ode0 is None else float(ode0)
    class_name = "FlatLambdaCDM" if abs(float(w0) + 1.0) < 1.0e-10 else "FlatwCDM"
    return {
        "class": class_name,
        "H0": float(h0),
        "Om0": float(om0),
        "Ode0": float(ode0_value),
        "w0": float(w0),
    }


def h0_from_config(cosmo_config: dict[str, Any] | None) -> float:
    if not isinstance(cosmo_config, dict):
        return 70.0
    return float(cosmo_config.get("H0", 70.0))


def om0_from_config(cosmo_config: dict[str, Any] | None) -> float:
    if not isinstance(cosmo_config, dict):
        return 0.3
    return float(cosmo_config.get("Om0", cosmo_config.get("omegaM", cosmo_config.get("omega", 0.3))))


def ode0_from_config(cosmo_config: dict[str, Any] | None) -> float:
    om0 = om0_from_config(cosmo_config)
    if not isinstance(cosmo_config, dict):
        return 1.0 - om0
    return float(cosmo_config.get("Ode0", cosmo_config.get("omegaX", cosmo_config.get("lambda", 1.0 - om0))))


def w0_from_config(cosmo_config: dict[str, Any] | None) -> float:
    if not isinstance(cosmo_config, dict):
        return -1.0
    return float(cosmo_config.get("w0", cosmo_config.get("wX", cosmo_config.get("w", -1.0))))


def flat_wcdm_efunc(z: float | jnp.ndarray, omega_m: float | jnp.ndarray, w0: float | jnp.ndarray) -> jnp.ndarray:
    r"""Dimensionless Hubble rate of flat wCDM with matter and dark energy only:
    :math:`E(z) = \sqrt{\Omega_m (1 + z)^3 + (1 - \Omega_m)(1 + z)^{3 (1 + w)}}`."""
    one_plus_z = 1.0 + jnp.asarray(z, dtype=jnp.float64)
    omega_m, w0 = jnp.asarray(omega_m, dtype=jnp.float64), jnp.asarray(w0, dtype=jnp.float64)
    return jnp.sqrt(omega_m * one_plus_z ** 3 + (1.0 - omega_m) * one_plus_z ** (3.0 * (1.0 + w0)))


def scale_factor_from_redshift(z: float | jnp.ndarray) -> jnp.ndarray:
    return 1.0 / (1.0 + jnp.asarray(z, dtype=jnp.float64))


def flat_wcdm_comoving_distance_mpc(
    z: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> jnp.ndarray:
    r"""Line-of-sight comoving distance in Mpc, :math:`(c / H_0) \int_0^z dz' / E(z')`, by Gauss-Legendre quadrature in
    :math:`u = \ln(1 + z')`.  Differentiable in ``z``, ``h0``, ``omega_m`` and ``w0``; ``z`` of any shape."""
    nodes, weights = _gauss_legendre(order)
    z = jnp.asarray(z, dtype=jnp.float64)
    u_max = jnp.log1p(z)
    u = 0.5 * u_max[..., None] * (jnp.asarray(nodes) + 1.0)
    one_plus_z = jnp.exp(u)
    integral = 0.5 * u_max * jnp.sum(jnp.asarray(weights) * one_plus_z / flat_wcdm_efunc(one_plus_z - 1.0, omega_m, w0), axis=-1)
    return jnp.asarray(C_LIGHT_KM_S, dtype=jnp.float64) / jnp.asarray(h0, dtype=jnp.float64) * integral


def flat_wcdm_angular_diameter_distance_mpc(
    z: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> jnp.ndarray:
    return flat_wcdm_comoving_distance_mpc(z, h0, omega_m, w0, order=order) * scale_factor_from_redshift(z)


def flat_wcdm_angular_diameter_distance_z1z2_mpc(
    z_lens: float | jnp.ndarray,
    z_source: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> jnp.ndarray:
    chi_lens = flat_wcdm_comoving_distance_mpc(z_lens, h0, omega_m, w0, order=order)
    chi_source = flat_wcdm_comoving_distance_mpc(z_source, h0, omega_m, w0, order=order)
    distance = (chi_source - chi_lens) / (1.0 + jnp.asarray(z_source, dtype=jnp.float64))
    return jnp.where((chi_source > chi_lens) & (distance > 0.0), distance, jnp.nan)


def flat_wcdm_lensing_efficiency(
    z_lens: float | jnp.ndarray,
    z_source: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> jnp.ndarray:
    chi_lens = flat_wcdm_comoving_distance_mpc(z_lens, h0, omega_m, w0, order=order)
    chi_source = flat_wcdm_comoving_distance_mpc(z_source, h0, omega_m, w0, order=order)
    efficiency = (chi_source - chi_lens) / chi_source
    return jnp.where((chi_source > 0.0) & (efficiency > 0.0), efficiency, jnp.nan)


def flat_wcdm_lens_geometry_factors(
    z_lens: float | jnp.ndarray,
    z_sources: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the lens-plane scale (kpc per arcsec), and per source the lensing efficiency and the dPIE sigma0 factor."""
    chi_lens = flat_wcdm_comoving_distance_mpc(z_lens, h0, omega_m, w0, order=order)
    chi_sources = flat_wcdm_comoving_distance_mpc(jnp.ravel(jnp.asarray(z_sources, dtype=jnp.float64)), h0, omega_m, w0, order=order)
    angular_diameter_lens_mpc = chi_lens * scale_factor_from_redshift(z_lens)
    kpc_per_arcsec = angular_diameter_lens_mpc * 1000.0 * jnp.asarray(ARCSEC_TO_RAD, dtype=jnp.float64)
    efficiency = (chi_sources - chi_lens) / chi_sources
    efficiency = jnp.where((chi_sources > 0.0) & (efficiency > 0.0), efficiency, jnp.nan)
    return kpc_per_arcsec, efficiency, dpie_sigma0_factor_from_lensing_efficiency(efficiency)


def flat_wcdm_kpc_per_arcsec(
    z: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> jnp.ndarray:
    angular_diameter_mpc = flat_wcdm_angular_diameter_distance_mpc(z, h0, omega_m, w0, order=order)
    return angular_diameter_mpc * 1000.0 * jnp.asarray(ARCSEC_TO_RAD, dtype=jnp.float64)


def kpc_per_arcsec_from_config(
    z: float,
    cosmo_config: dict[str, Any] | None,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> float:
    return float(
        np.asarray(
            flat_wcdm_kpc_per_arcsec(
                float(z),
                h0_from_config(cosmo_config),
                om0_from_config(cosmo_config),
                w0_from_config(cosmo_config),
                order=order,
            )
        )
    )


def dpie_sigma0_factor_from_lensing_efficiency(efficiency: float | jnp.ndarray) -> jnp.ndarray:
    return (
        (1.0 / jnp.asarray(C_LIGHT_KM_S, dtype=jnp.float64)) ** 2
        * 3.0
        * jnp.pi
        * jnp.asarray(efficiency, dtype=jnp.float64)
        / jnp.asarray(ARCSEC_TO_RAD, dtype=jnp.float64)
    )


def dpie_sigma0_factor(
    z_lens: float | jnp.ndarray,
    z_source: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> jnp.ndarray:
    return dpie_sigma0_factor_from_lensing_efficiency(
        flat_wcdm_lensing_efficiency(z_lens, z_source, h0, omega_m, w0, order=order)
    )


def dpie_sigma0_factor_from_config(
    z_lens: float,
    z_source: float,
    cosmo_config: dict[str, Any] | None,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> float:
    return float(
        np.asarray(
            dpie_sigma0_factor(
                float(z_lens),
                float(z_source),
                h0_from_config(cosmo_config),
                om0_from_config(cosmo_config),
                w0_from_config(cosmo_config),
                order=order,
            )
        )
    )


def dpie_sigma0_from_vel_disp(
    vel_disp: float,
    ra_arcsec: float,
    rs_arcsec: float,
    z_lens: float,
    z_source: float,
    cosmo_config: dict[str, Any] | None,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> float:
    if float(ra_arcsec) <= 0.0 or float(rs_arcsec) <= float(ra_arcsec):
        return float("nan")
    factor = dpie_sigma0_factor_from_config(z_lens, z_source, cosmo_config, order=order)
    sigma0 = float(vel_disp) ** 2 * factor / float(ra_arcsec)
    return float(sigma0)


def critical_surface_density_angle_msun_per_arcsec2(
    z_lens: float | jnp.ndarray,
    z_source: float | jnp.ndarray,
    h0: float | jnp.ndarray,
    omega_m: float | jnp.ndarray,
    w0: float | jnp.ndarray,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> jnp.ndarray:
    dd = flat_wcdm_angular_diameter_distance_mpc(z_lens, h0, omega_m, w0, order=order)
    ds = flat_wcdm_angular_diameter_distance_mpc(z_source, h0, omega_m, w0, order=order)
    dds = flat_wcdm_angular_diameter_distance_z1z2_mpc(z_lens, z_source, h0, omega_m, w0, order=order)
    factor = (
        jnp.asarray(C_LIGHT_M_S, dtype=jnp.float64) ** 2
        / (4.0 * jnp.pi * jnp.asarray(G_SI, dtype=jnp.float64))
        * jnp.asarray(MPC_M / M_SUN_KG, dtype=jnp.float64)
    )
    sigma_crit = ds / (dd * dds) * factor * (dd * jnp.asarray(ARCSEC_TO_RAD, dtype=jnp.float64)) ** 2
    return jnp.where((dd > 0.0) & (ds > 0.0) & (dds > 0.0), sigma_crit, jnp.nan)


def critical_surface_density_angle_from_config(
    z_lens: float,
    z_source: float,
    cosmo_config: dict[str, Any] | None,
    *,
    order: int = DEFAULT_FLAT_WCDM_QUADRATURE_ORDER,
) -> float:
    return float(
        np.asarray(
            critical_surface_density_angle_msun_per_arcsec2(
                float(z_lens),
                float(z_source),
                h0_from_config(cosmo_config),
                om0_from_config(cosmo_config),
                w0_from_config(cosmo_config),
                order=order,
            )
        )
    )

