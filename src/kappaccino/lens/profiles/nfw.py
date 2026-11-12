"""The elliptical NFW (Lenstool profile 12) as the cored-steep-ellipsoid (CSE) sum of Oguri (2021), with the
ellipticity in the convergence.  The 44-term set matches the NFW to about one part in a thousand.

Deflection, Jacobian and its derivative are linear in ``alpha_rs``, the deflection at the scale radius, so the kernel is
evaluated with unit normalization and scaled afterwards; the normalization may vary per point (per image).  The
derivative of the Jacobian is analytic, like the Jacobian.

The second half of the module converts a halo's :math:`M_{200}` and scale radius to the kernel's ``rs`` and
``alpha_rs``."""
from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import frame as _frame

PARAMETERS = ("nfw_rs", "alpha_rs", "e1", "e2", "center_x", "center_y")     # PackedLensState fields, in kernel order
G_KPC_KMS2_PER_MSUN = 4.30091e-6
DELTA_200 = 200.0

# Table 1 of Oguri (2021), kept as NumPy so importing this module does not initialize JAX.
_CSE_A = np.array([1.6489880e-18, 6.2744580e-16, 3.6466200e-17, 3.4592060e-15, 2.4573890e-14, 1.0593190e-13, 4.2115970e-13, 1.1428320e-12, 4.3912150e-12, 1.5565000e-11, 6.9512710e-11,
                   3.1474660e-10, 1.3791090e-09, 3.8297780e-09, 1.3848580e-08, 5.3709510e-08, 1.8043840e-07, 5.7886080e-07, 3.2052560e-06, 1.1024220e-05, 4.0939710e-05, 1.2822060e-04,
                   4.5755410e-04, 7.9952700e-04, 5.0137010e-03, 1.4035080e-02, 5.2307270e-02, 1.8989070e-01, 3.6434480e-01, 7.2037340e-01, 1.7176670e+00, 2.2175660e+00, 3.1874470e+00,
                   8.1948980e+00, 1.7652100e+01, 1.9743190e+01, 2.7836880e+01, 4.4823110e+01, 5.5988970e+01, 1.4264850e+02, 2.2798330e+02, 5.4013350e+02, 9.7436820e+02, 1.7751240e+03])
_CSE_S = np.array([1.0824110e-06, 8.7865660e-06, 3.2928680e-06, 1.8600190e-05, 3.2742310e-05, 6.2324850e-05, 9.2563330e-05, 1.5467620e-04, 2.0973210e-04, 3.3911400e-04, 5.1787900e-04,
                   8.6367360e-04, 1.4051520e-03, 2.1938550e-03, 3.1795720e-03, 4.9709870e-03, 7.6319700e-03, 1.1194130e-02, 1.8272670e-02, 2.9452510e-02, 4.5627230e-02, 6.7825090e-02,
                   1.5969870e-01, 1.1277510e-01, 2.1694690e-01, 3.4238350e-01, 5.1945270e-01, 8.6231850e-01, 1.3827370e+00, 2.0349290e+00, 3.4029790e+00, 5.5942760e+00, 8.0523450e+00,
                   1.3490450e+01, 2.6038250e+01, 4.7368230e+01, 6.5593200e+01, 1.0879320e+02, 1.4776730e+02, 2.4953410e+02, 4.3059990e+02, 7.7602060e+02, 2.1430570e+03, 1.9357490e+03])
_NFW_UNIT_NORMALIZATION = 1.0 / (1.0 - math.log(2.0))      # 4 rho0 Rs^3 / Rs for alpha_Rs = 1


def _cse_set_deflection(x, y, q):
    """Deflection of the unit-normalized product-averaged CSE set (Oguri 2021, eqs. 19-20) in the halo frame; the sum
    over the 44 terms is a broadcast rather than a loop so that reverse-mode differentiation stays cheap."""
    a, s = _CSE_A[:, None] / q, _CSE_S[:, None]
    xq, yq = x[None, :] * jnp.sqrt(q), y[None, :] * jnp.sqrt(q)
    psi = jnp.sqrt(q * q * (s * s + xq * xq) + yq * yq)
    phi = (psi + s) ** 2 + (1.0 - q * q) * xq * xq
    common = a * q * jnp.sqrt(q) / (s * psi * phi)
    return jnp.sum(common * xq * (psi + q * q * s), axis=0), jnp.sum(common * yq * (psi + s), axis=0)


def _cse_set_hessian(x, y, q):
    """Second derivatives (f_xx, f_xy, f_yy) of the same set (Oguri 2021, eqs. 21-23)."""
    a, s = _CSE_A[:, None] / q, _CSE_S[:, None]
    xq, yq = x[None, :] * jnp.sqrt(q), y[None, :] * jnp.sqrt(q)
    q2 = q * q
    psi = jnp.sqrt(q2 * (s * s + xq * xq) + yq * yq)
    phi = (psi + s) ** 2 + (1.0 - q2) * xq * xq
    pre = a * q2 / (s * phi)                                   # extra q from the two sqrt(q) coordinate factors
    psi3, psi2phi = psi ** 3, psi * psi * phi
    fxx = pre * (1.0 + q2 * s * (q2 * s * s + yq * yq) / psi3 - 2.0 * xq * xq * (psi + q2 * s) ** 2 / psi2phi)
    fyy = pre * (1.0 + q2 * s * (s * s + xq * xq) / psi3 - 2.0 * yq * yq * (psi + s) ** 2 / psi2phi)
    fxy = -pre * xq * yq * (q2 * s / psi3 + 2.0 * (psi + q2 * s) * (psi + s) / psi2phi)
    return jnp.sum(fxx, axis=0), jnp.sum(fxy, axis=0), jnp.sum(fyy, axis=0)


def _cse_set_hessian_and_third(x, y, q):
    """(f_xx, f_xy, f_yy) and the third derivatives (f_xxx, f_xxy, f_xyy, f_yyy) of the same set: the derivatives of
    Oguri (2021), eqs. 21-23, in x and y, written out term by term and summed.

    With :math:`\\psi^2 = q^2 (s^2 + x^2) + y^2`, :math:`B = \\psi + q^2 s`, :math:`C = \\psi + s` and
    :math:`\\phi = C^2 + (1 - q^2) x^2`, each term is :math:`f_{xx} = P (1 + T_1 - T_2)` and
    :math:`f_{yy} = P (1 + T_3 - T_4)` with :math:`P \\propto 1 / \\phi`, :math:`T_1 = q^2 s (q^2 s^2 + y^2) / \\psi^3`,
    :math:`T_2 = 2 x^2 B^2 / (\\psi^2 \\phi)`, :math:`T_3 = q^2 s (s^2 + x^2) / \\psi^3` and
    :math:`T_4 = 2 y^2 C^2 / (\\psi^2 \\phi)`; the third derivatives follow from the product rule with
    :math:`\\partial \\ln P = -\\partial \\ln \\phi`.  No term divides by x or y, so the axes and the centre are regular."""
    a, s = _CSE_A[:, None] / q, _CSE_S[:, None]
    root = jnp.sqrt(q)
    xq, yq = x[None, :] * root, y[None, :] * root
    q2 = q * q
    psi = jnp.sqrt(q2 * (s * s + xq * xq) + yq * yq)
    b, c = psi + q2 * s, psi + s
    phi = c * c + (1.0 - q2) * xq * xq
    inv_psi, inv_phi = 1.0 / psi, 1.0 / phi
    pre = a * q2 / s * inv_phi                                 # extra q from the two sqrt(q) coordinate factors
    psi_x, psi_y = q2 * xq * inv_psi, yq * inv_psi
    dln_phi_x, dln_phi_y = (2.0 * c * psi_x + 2.0 * (1.0 - q2) * xq) * inv_phi, 2.0 * c * psi_y * inv_phi
    dln_psi_x, dln_psi_y = psi_x * inv_psi, psi_y * inv_psi
    inv_psi3 = inv_psi ** 3
    w = inv_psi * inv_psi * inv_phi                            # 1 / (psi^2 phi)
    t1 = q2 * s * (q2 * s * s + yq * yq) * inv_psi3
    t2 = 2.0 * xq * xq * b * b * w
    t3 = q2 * s * (s * s + xq * xq) * inv_psi3
    t4 = 2.0 * yq * yq * c * c * w
    gxx, gyy = 1.0 + t1 - t2, 1.0 + t3 - t4
    fxy = -pre * xq * yq * (q2 * s * inv_psi3 + 2.0 * b * c * w)
    t1_x, t1_y = -3.0 * t1 * dln_psi_x, 2.0 * q2 * s * yq * inv_psi3 - 3.0 * t1 * dln_psi_y
    t2_x = 4.0 * xq * b * (b + xq * psi_x) * w - t2 * (2.0 * dln_psi_x + dln_phi_x)
    t2_y = 4.0 * xq * xq * b * psi_y * w - t2 * (2.0 * dln_psi_y + dln_phi_y)
    t3_x, t3_y = 2.0 * q2 * s * xq * inv_psi3 - 3.0 * t3 * dln_psi_x, -3.0 * t3 * dln_psi_y
    t4_x = 4.0 * yq * yq * c * psi_x * w - t4 * (2.0 * dln_psi_x + dln_phi_x)
    t4_y = 4.0 * yq * c * (c + yq * psi_y) * w - t4 * (2.0 * dln_psi_y + dln_phi_y)
    third = (pre * (t1_x - t2_x - dln_phi_x * gxx), pre * (t1_y - t2_y - dln_phi_y * gxx),
             pre * (t3_x - t4_x - dln_phi_x * gyy), pre * (t3_y - t4_y - dln_phi_y * gyy))
    total = lambda value: jnp.sum(value, axis=0)
    # One more sqrt(q) for the third coordinate derivative.
    return (total(pre * gxx), total(fxy), total(pre * gyy)), tuple(root * total(value) for value in third)


def fields(x, y, rs, alpha_rs, e1, e2, center_x, center_y, derivative=True):
    """alpha (N, 2), U = d alpha / d theta (N, 2, 2) and dU[..., i, j, l] = d U_ij / d theta_l (N, 2, 2, 2; None unless
    ``derivative``) of one elliptical NFW with scale radius ``rs`` (arcsec) and deflection ``alpha_rs`` at the scale radius."""
    scale = jnp.broadcast_to(jnp.asarray(alpha_rs, dtype=jnp.float64), jnp.shape(x))
    q, rotation = _frame.rotation(e1, e2)
    c, s = rotation[0, 0], rotation[0, 1]
    xr, yr = ((x - center_x) * c + (y - center_y) * s) / rs, (-(x - center_x) * s + (y - center_y) * c) / rs
    fx, fy = _cse_set_deflection(xr, yr, q)
    alpha = _frame.to_sky_vector(rotation, jnp.stack((fx, fy), -1)) * (_NFW_UNIT_NORMALIZATION * scale[:, None])
    if not derivative:
        u_frame = _frame.symmetric(*_cse_set_hessian(xr, yr, q))
        return alpha, _frame.to_sky_matrix(rotation, u_frame) * (_NFW_UNIT_NORMALIZATION / rs) * scale[:, None, None], None
    hessian, (fxxx, fxxy, fxyy, fyyy) = _cse_set_hessian_and_third(xr, yr, q)
    u = _frame.to_sky_matrix(rotation, _frame.symmetric(*hessian)) * (_NFW_UNIT_NORMALIZATION / rs) * scale[:, None, None]
    du_frame = jnp.stack((_frame.symmetric(fxxx, fxxy, fxyy), _frame.symmetric(fxxy, fxyy, fyyy)), -3)
    return alpha, u, _frame.to_sky_third(rotation, du_frame) * (_NFW_UNIT_NORMALIZATION / (rs * rs)) * scale[:, None, None, None]


# Batches over components; the ``derivative`` flag is static, so each variant is its own vmapped function.
grouped_fields = {
    True: jax.vmap(fields, in_axes=(None, None, 0, 0, 0, 0, 0, 0)),
    False: jax.vmap(functools.partial(fields, derivative=False), in_axes=(None, None, 0, 0, 0, 0, 0, 0)),
}


# ----------------------------------------------------------------------------- mass and scale radius to kernel parameters
def critical_density_msun_kpc3(z: float, cosmo_config: dict[str, Any] | None) -> float:
    """rho_crit(z) = 3 H(z)^2 / (8 pi G) for the flat (w)CDM cosmology of the run."""
    from ..cosmology import h0_from_config, ode0_from_config, om0_from_config, w0_from_config
    h0, om0, ode0, w0 = (f(cosmo_config) for f in (h0_from_config, om0_from_config, ode0_from_config, w0_from_config))
    e2 = om0 * (1.0 + z) ** 3 + ode0 * (1.0 + z) ** (3.0 * (1.0 + w0))
    hubble_kms_per_kpc = h0 / 1000.0 * math.sqrt(e2)
    return 3.0 * hubble_kms_per_kpc ** 2 / (8.0 * math.pi * G_KPC_KMS2_PER_MSUN)


def flat_wcdm_critical_density_msun_kpc3(z, h0, omega_m, w0):
    """rho_crit(z) = 3 H(z)^2 / (8 pi G) of flat wCDM in Msun kpc^-3, JAX-traceable in the cosmology (sampled Om0, w0)."""
    from ..cosmology import flat_wcdm_efunc
    hubble_kms_per_kpc = jnp.asarray(h0, dtype=jnp.float64) / 1000.0 * flat_wcdm_efunc(z, omega_m, w0)
    return 3.0 * hubble_kms_per_kpc ** 2 / (8.0 * jnp.pi * G_KPC_KMS2_PER_MSUN)


def nfw_scale_density(log10_m200, scale_radius_kpc, rho_crit_msun_kpc3):
    """rho_s [Msun/kpc^3] and the concentration of an NFW halo of mass M200 and scale radius r_s (JAX-traceable)."""
    m200 = jnp.power(10.0, log10_m200)
    r200 = jnp.cbrt(3.0 * m200 / (4.0 * jnp.pi * DELTA_200 * rho_crit_msun_kpc3))
    c = r200 / scale_radius_kpc
    rho_s = DELTA_200 / 3.0 * rho_crit_msun_kpc3 * c ** 3 / (jnp.log1p(c) - c / (1.0 + c))
    return rho_s, c


# alpha_Rs[arcsec] = 4 rho_s r_s^2 (1 - ln 2) x (4 pi G / c^2) (D_ls / D_s) / arcsec; the dPIE sigma0 factor of this
# code is 3 pi (D_ls / D_s) / (c^2 arcsec), so the NFW factor is that times 4 G / 3.
NFW_FACTOR_OVER_DPIE_FACTOR = 4.0 * G_KPC_KMS2_PER_MSUN / 3.0


def nfw_alpha_rs_arcsec(rho_s_msun_kpc3, scale_radius_kpc, dpie_sigma0_factor):
    """Deflection at the scale radius in arcsec from rho_s, r_s and the per-source dPIE sigma0 factor."""
    return 4.0 * rho_s_msun_kpc3 * scale_radius_kpc ** 2 * (1.0 - jnp.log(2.0)) * NFW_FACTOR_OVER_DPIE_FACTOR * dpie_sigma0_factor


def nfw_angular_parameters(log10_m200: float, scale_radius_kpc: float, *, z_lens: float, z_source: float,
                           cosmo_config: dict[str, Any], kpc_per_arcsec: float) -> tuple[float, float]:
    """(r_s in arcsec, deflection at r_s in arcsec) for one halo."""
    from ..cosmology import dpie_sigma0_factor_from_config
    rho_s, _ = nfw_scale_density(float(log10_m200), float(scale_radius_kpc), critical_density_msun_kpc3(z_lens, cosmo_config))
    factor = dpie_sigma0_factor_from_config(z_lens, z_source, cosmo_config)
    return float(scale_radius_kpc) / float(kpc_per_arcsec), float(nfw_alpha_rs_arcsec(rho_s, float(scale_radius_kpc), factor))

