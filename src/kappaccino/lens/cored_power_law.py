"""Lensing fields of the truncated cored power law (CPL), the member profile with a free inner slope.

The convergence in the elliptical radius :math:`\\xi^2 = q x'^2 + y'^2 / q` (the product-average convention of the
dPIE kernel, :math:`x'` along the major axis) is

.. math::

    \\kappa(\\xi) = A \\left[(\\xi^2 + a^2)^{-p} - (\\xi^2 + s^2)^{-p}\\right], \\qquad p = (\\gamma - 1)/2,

with core radius :math:`a`, cut radius :math:`s > a` and 3D logarithmic slope :math:`\\gamma` between the core and
the cut (:math:`\\rho \\propto r^{-\\gamma}`), steepened by 2 beyond the cut (:math:`r^{-\\gamma - 2}`).  At
:math:`\\gamma = 2` it is the dPIE.  The amplitude keeps the projected mass inside the cut radius equal to that of
the dPIE with the same normalization :math:`\\sigma_0` (the kernel's central convergence at :math:`\\gamma = 2`)
and radii.  With the enclosed-mass function

.. math::

    G_p(R) = \\frac{(R^2 + a^2)^{1-p} - a^{2-2p} - (R^2 + s^2)^{1-p} + s^{2-2p}}{1 - p}

(the circular deflection is :math:`\\alpha(R) = A\\, G_p(R) / R`), the amplitude is

.. math::

    A = \\frac{\\sigma_0 a s}{s - a} \\frac{G_{1/2}(s)}{G_p(s)}.

So :math:`\\gamma` moves mass between the center and the edge of the cut at fixed mass inside the cut, core and cut
radius.  The total mass is not held fixed: beyond the cut the convergence falls as :math:`R^{-\\gamma - 1}`, so the
mass outside the cut grows as the slope approaches 1.  The slope must lie in (1, 3): the mass inside the core
diverges at :math:`\\gamma \\ge 3` and the total mass at :math:`\\gamma \\le 1`.

The deflection and its Jacobian are the integrals of Keeton (2001, astro-ph/0102341) over
:math:`u \\in [0, 1]`, evaluated by Gauss-Legendre quadrature on two panels split where the elliptical radius of the
integrand reaches the core, :math:`u_c = \\min(1, a^2 / (q (x'^2 + y'^2)))`: linear in :math:`u` inside the core,
where the integrand is smooth, linear in :math:`\\ln u` outside it up to :math:`u_m = \\max(u_c, 1/2)`, where the
integrand is a power of :math:`u`, and linear on :math:`[u_m, 1]`, where the factor :math:`D^{-n-1/2}` of a flat
ellipse peaks.
The rule is accurate for any slope and core.  The third derivatives of the potential (the spatial derivative of the
Jacobian) follow from the same integrals.  Circular fields (far-field members, tests) have a closed form.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from .lensing import ellipticity_frame

SLOPE_LOWER = 1.0           # open bounds of the 3D slope gamma
SLOPE_UPPER = 3.0
PANEL_NODES = (12,16,12)   # Gauss-Legendre nodes on [0, u_c] (linear), [u_c, u_m] (in ln u) and [u_m, 1] (linear)
ELLIPSE_SPLIT = 0.5          # u_m = max(u_c, ELLIPSE_SPLIT): the factor D^(-n-1/2) peaks at u = 1 for flat ellipses
_SMALLEST_SPLIT = 1.0e-30


def _unit_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return 0.5 * (nodes + 1.0), 0.5 * weights       # Gauss-Legendre on [0, 1]


_RULES = tuple(_unit_rule(n) for n in PANEL_NODES)


def _nodes(split):
    """Nodes u and weights, shape (node, point), of the three-panel rule on [0, 1] for core splits ``split`` (point,)."""
    (t_core, w_core), (t_log, w_log), (t_end, w_end) = ((jnp.asarray(t)[:, None], jnp.asarray(w)[:, None]) for t, w in _RULES)
    split = split[None, :]
    middle = jnp.maximum(split, ELLIPSE_SPLIT)
    log_ratio = jnp.log(middle / split)
    u_log = split * jnp.exp(t_log * log_ratio)                      # uniform in ln u on [u_c, u_m]
    u = jnp.concatenate((split * t_core, u_log, middle + (1.0 - middle) * t_end), axis=0)
    weight = jnp.concatenate((split * w_core, log_ratio * u_log * w_log, (1.0 - middle) * w_end), axis=0)
    return u, weight


def enclosed(radius_sq, a, s, slope):
    """:math:`G_p(R)` at :math:`R^2` = ``radius_sq``: the projected mass inside :math:`R` over :math:`\\pi A`
    (arcsec^(2 - 2p)).  The difference :math:`(R^2 + a^2)^e - (R^2 + s^2)^e` (:math:`e = 1 - p`) is written with
    ``expm1`` so that it keeps its precision far outside the cut."""
    e = 1.0 - 0.5 * (slope - 1.0)
    log_a, log_s = jnp.log1p(a * a / radius_sq), jnp.log1p(s * s / radius_sq)
    outer = -jnp.exp(e * (jnp.log(radius_sq) + log_a)) * jnp.expm1(e * (log_s - log_a))
    return (outer + s ** (2 * e) - a ** (2 * e)) / e


def amplitude(sigma0, a, s, slope):
    """The CPL amplitude :math:`A` that keeps the dPIE's projected mass inside the cut radius (units of convergence
    times arcsec^(2p)); :math:`\\sigma_0 a s / (s - a)` at slope 2."""
    dpie = 2.0 * (jnp.sqrt(s * s + a * a) - a - (jnp.sqrt(2.0) - 1.0) * s) / (s - a)     # G_1/2(s) / (s - a)
    return sigma0 * a * s * dpie / enclosed(s * s, a, s, slope)


def convergence(xi_sq, sigma0, a, s, slope):
    """:math:`\\kappa` at elliptical radius squared ``xi_sq`` (arcsec^2)."""
    p = 0.5 * (slope - 1.0)
    return amplitude(sigma0, a, s, slope) * ((xi_sq + a * a) ** -p - (xi_sq + s * s) ** -p)


def circular_deflection(radius, sigma0, a, s, slope):
    """Deflection modulus :math:`\\alpha(R) = A\\, G_p(R) / R` of the circular CPL, in arcsec."""
    return amplitude(sigma0, a, s, slope) * enclosed(radius * radius, a, s, slope) / radius


def circular_fields(dx, dy, sigma0, a, s, slope):
    """alpha (..., 2), U = d alpha / d theta (..., 2, 2) and dU[..., i, j, l] = d U_ij / d theta_l (..., 2, 2, 2) of
    the circular CPL at offsets (dx, dy) from its center (any common shape; the parameters broadcast against it).

    With :math:`\\alpha = f(R) \\hat r`: :math:`U = (f/R) I + (f' - f/R) \\hat r \\hat r` and
    :math:`\\partial_l U_{ij} = A' \\hat r_l \\delta_{ij} + (B' - 2B/R) \\hat r_i \\hat r_j \\hat r_l
    + (B/R)(\\delta_{il} \\hat r_j + \\delta_{jl} \\hat r_i)` with :math:`A = f/R` and :math:`B = f' - f/R`."""
    radius = jnp.sqrt(dx * dx + dy * dy)
    f = lambda r: circular_deflection(r, sigma0, a, s, slope)
    ones = jnp.ones_like(radius)
    df = lambda r: jax.jvp(f, (r,), (ones,))[1]
    value, slope_f = jax.jvp(f, (radius,), (ones,))
    curvature = jax.jvp(df, (radius,), (ones,))[1]
    rhat = jnp.stack((dx / radius, dy / radius), -1)
    eye = jnp.eye(2)
    big_a = value / radius
    big_b = slope_f - big_a
    da = (slope_f - big_a) / radius                               # d(f/R)/dR
    db = curvature - da
    outer = rhat[..., :, None] * rhat[..., None, :]
    u = big_a[..., None, None] * eye + big_b[..., None, None] * outer
    triple = outer[..., None] * rhat[..., None, None, :]
    mixed = eye[:, None, :] * rhat[..., None, :, None] + eye[None, :, :] * rhat[..., :, None, None]
    du = (da[..., None, None, None] * eye[..., None] * rhat[..., None, None, :]
          + (db - 2 * big_b / radius)[..., None, None, None] * triple + (big_b / radius)[..., None, None, None] * mixed)
    return value[..., None] * rhat, u, du


def _frame_fields(xr, yr, q, a, s, slope, derivative):
    """Deflection, Hessian and (with ``derivative``) third derivatives of the potential in the principal frame for
    unit amplitude (:math:`A = 1`).

    With :math:`D = 1 - (1 - q^2) u`, :math:`\\zeta(u)^2 = u (x'^2 + y'^2 / D)` and
    :math:`\\kappa_K(\\zeta^2) = \\kappa(q \\zeta^2)`, the integrals :math:`J_n = \\int_0^1 \\kappa_K D^{-n-1/2} du`,
    :math:`K_n = \\int_0^1 u \\kappa_K' D^{-n-1/2} du` and :math:`L_n = \\int_0^1 u^2 \\kappa_K'' D^{-n-1/2} du`
    (primes: derivatives in :math:`\\zeta^2`) give (Keeton 2001) :math:`\\phi_x = q x J_0`, :math:`\\phi_y = q y J_1`,
    :math:`\\phi_{xx} = q (2 x^2 K_0 + J_0)`, :math:`\\phi_{xy} = 2 q x y K_1`, :math:`\\phi_{yy} = q (2 y^2 K_2 + J_1)`,
    and, from :math:`\\partial_x J_n = 2 x K_n`, :math:`\\partial_y J_n = 2 y K_{n+1}` (the same for :math:`K \\to L`),
    the third derivatives :math:`\\phi_{xxx} = q (6 x K_0 + 4 x^3 L_0)`, :math:`\\phi_{xxy} = q (2 y K_1 + 4 x^2 y L_1)`,
    :math:`\\phi_{xyy} = q (2 x K_1 + 4 x y^2 L_2)`, :math:`\\phi_{yyy} = q (6 y K_2 + 4 y^3 L_3)`."""
    p = 0.5 * (slope - 1.0)
    u, weight = _nodes(jnp.clip(a * a / (q * (xr * xr + yr * yr) + _SMALLEST_SPLIT), _SMALLEST_SPLIT, 1.0))
    inv_d = 1.0 / (1.0 - (1.0 - q * q) * u)
    w = q * u * (xr[None, :] ** 2 + yr[None, :] ** 2 * inv_d)       # the dPIE-convention xi^2 at each node
    inv_a, inv_s = 1.0 / (w + a * a), 1.0 / (w + s * s)
    pa, ps = jnp.exp(p * jnp.log(inv_a)), jnp.exp(p * jnp.log(inv_s))   # (w + r^2)^-p, one power per radius
    kappa = pa - ps
    dkappa = -p * q * (pa * inv_a - ps * inv_s)                      # d kappa_K / d zeta^2
    base = [weight * jnp.sqrt(inv_d)]                                # weight D^(-n-1/2), n = 0 ... 3
    for _ in range(3):
        base.append(base[-1] * inv_d)
    j = [jnp.sum(base[n] * kappa, axis=0) for n in range(2)]
    k = [jnp.sum(base[n] * u * dkappa, axis=0) for n in range(3)]
    x2, y2, xy = xr * xr, yr * yr, xr * yr
    alpha = (q * xr * j[0], q * yr * j[1])
    hessian = (q * (2 * x2 * k[0] + j[0]), 2 * q * xy * k[1], q * (2 * y2 * k[2] + j[1]))
    if not derivative:
        return alpha, hessian, None
    d2kappa = p * (p + 1.0) * q * q * (pa * inv_a * inv_a - ps * inv_s * inv_s)
    l = [jnp.sum(base[n] * u * u * d2kappa, axis=0) for n in range(4)]
    third = (q * (6 * xr * k[0] + 4 * x2 * xr * l[0]), q * (2 * yr * k[1] + 4 * x2 * yr * l[1]),
             q * (2 * xr * k[1] + 4 * xr * y2 * l[2]), q * (6 * yr * k[2] + 4 * y2 * yr * l[3]))
    return alpha, hessian, third


def cored_power_law_fields(x, y, sigma0, a, s, slope, e1, e2, center_x, center_y, derivative=True):
    """alpha (N, 2), U = d alpha / d theta (N, 2, 2) and dU[..., i, j, l] = d U_ij / d theta_l (N, 2, 2, 2; None
    unless ``derivative``) of one elliptical CPL at points (x, y) (arcsec).

    ``sigma0`` (the dPIE-equivalent normalization, scalar or one per point), core ``a`` and cut ``s`` (arcsec), 3D
    slope ``slope`` and the Cartesian ellipticity (e1, e2) of the dPIE kernel."""
    x, y = jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)
    scale = jnp.broadcast_to(amplitude(jnp.asarray(sigma0, dtype=jnp.float64), a, s, slope), jnp.shape(x))
    phi, q = ellipticity_frame(e1, e2)
    c, sn = jnp.cos(phi), jnp.sin(phi)
    rotation = jnp.array([[c, sn], [-sn, c]])                        # principal frame = rotation @ (theta - centre)
    dx, dy = x - center_x, y - center_y
    (ax, ay), (fxx, fxy, fyy), third = _frame_fields(dx * c + dy * sn, -dx * sn + dy * c, q, a, s, slope, derivative)
    alpha = jnp.einsum("ai,na->ni", rotation, jnp.stack((ax, ay), -1)) * scale[:, None]
    u_frame = jnp.stack((jnp.stack((fxx, fxy), -1), jnp.stack((fxy, fyy), -1)), -2)
    u = jnp.einsum("ai,bj,nab->nij", rotation, rotation, u_frame) * scale[:, None, None]
    if not derivative:
        return alpha, u, None
    fxxx, fxxy, fxyy, fyyy = third
    du_frame = jnp.stack((jnp.stack((jnp.stack((fxxx, fxxy), -1), jnp.stack((fxxy, fxyy), -1)), -2),
                          jnp.stack((jnp.stack((fxxy, fxyy), -1), jnp.stack((fxyy, fyyy), -1)), -2)), -3)
    du = jnp.einsum("ai,bj,ck,nabc->nijk", rotation, rotation, rotation, du_frame)
    return alpha, u, du * scale[:, None, None, None]


# Batches over components (sigma0 per component, scalar or per point); ``derivative`` is static.
grouped_cored_power_law_fields = {
    True: jax.vmap(cored_power_law_fields, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0)),
    False: jax.vmap(functools.partial(cored_power_law_fields, derivative=False), in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0, 0)),
}


# ----------------------------------------------------------------------------- stellar kinematics
JEANS_NODES = 512            # log-spaced radii of the Jeans integrals, from 1e-3 of the smaller of core and aperture
JEANS_SPAN = (1.0e-3, 1.0e3)  # to 1e3 cut radii


def aperture_dispersion_sq(slope, a, s, aperture, sigma0=1.0):
    """Luminosity-weighted line-of-sight velocity dispersion squared inside a circular aperture of radius
    ``aperture``, in units of :math:`4\\\\pi G \\\\Sigma_\\\\mathrm{cr} / 3` times arcsec (so only ratios are physical).

    Spherical, isotropic, light traces mass.  With the 3D density
    :math:`\\\\rho = \\\\rho_0 [(r^2 + a^2)^{-\\\\gamma/2} - (r^2 + s^2)^{-\\\\gamma/2}]`, whose projection is the CPL
    convergence (:math:`\\\\rho_0 = A\\\\, \\\\Gamma(\\\\gamma/2) / (\\\\sqrt{\\\\pi}\\\\, \\\\Gamma((\\\\gamma - 1)/2))`), the
    isotropic Jeans equation projected and integrated over the aperture reduces to one radial integral,

    .. math::

        \\\\sigma_\\\\mathrm{ap}^2 = \\\\frac{4 \\\\pi G}{3 M_\\\\mathrm{2D}(<R)} \\\\int_0^\\\\infty \\\\rho(r) \\\\frac{M(r)}{r^2}
        \\\\left[r^3 - (r^2 - m^2)^{3/2}\\\\right] dr, \\\\qquad m = \\\\min(r, R),

    with :math:`M(r)` the 3D enclosed mass and :math:`M_\\\\mathrm{2D}(<R) = \\\\pi A\\\\, G_p(R)` the projected mass
    inside the aperture.  Trapezoids in :math:`\\\\ln r` on ``JEANS_NODES`` points; the arguments broadcast."""
    from jax.scipy.special import gammaln
    slope, a, s, aperture = (jnp.asarray(v, dtype=jnp.float64) for v in (slope, a, s, aperture))
    slope, a, s, aperture = jnp.broadcast_arrays(slope, a, s, aperture)
    amp = amplitude(sigma0, a, s, slope)
    rho0 = amp * jnp.exp(gammaln(0.5 * slope) - gammaln(0.5 * (slope - 1.0))) / jnp.sqrt(jnp.pi)
    low = JEANS_SPAN[0] * jnp.minimum(a, aperture)
    high = JEANS_SPAN[1] * s
    t = jnp.linspace(0.0, 1.0, JEANS_NODES)
    log_r = jnp.log(low)[..., None] + t * (jnp.log(high) - jnp.log(low))[..., None]
    step = (jnp.log(high) - jnp.log(low))[..., None]  / (JEANS_NODES - 1)
    r = jnp.exp(log_r)
    r2 = r * r
    half = 0.5 * slope[..., None]
    rho = rho0[..., None] * ((r2 + a[..., None] ** 2) ** -half - (r2 + s[..., None] ** 2) ** -half)
    shell = 4.0 * jnp.pi * rho * r2 * r                               # dM / d ln r
    inner = (4.0 / 3.0) * jnp.pi * rho[..., :1] * r[..., :1] ** 3     # the mass inside the first node (flat core)
    mass = inner + jnp.concatenate((jnp.zeros_like(shell[..., :1]),
                                    jnp.cumsum(0.5 * (shell[..., 1:] + shell[..., :-1]), axis=-1) * step), axis=-1)
    # r^3 - (r^2 - m^2)^(3/2) with m = min(r, R): r^3 inside the aperture.  The branch keeps the power away from 0,
    # where its second derivative is infinite (a NaN Hessian through a zero tangent).
    outside = r > aperture[..., None]
    excess = jnp.where(outside, r2 - aperture[..., None] ** 2, 1.0)
    kernel = r2 * r - jnp.where(outside, excess ** 1.5, 0.0)
    integrand = rho * mass / r2 * kernel * r                          # d / d ln r
    numerator = jnp.sum(0.5 * (integrand[..., 1:] + integrand[..., :-1]), axis=-1) * step[..., 0]
    projected = jnp.pi * amp * enclosed(aperture * aperture, a, s, slope)
    return numerator / projected


def aperture_dispersion_ratio(slope, a, s, aperture):
    """:math:`\\\\sigma_\\\\mathrm{ap}(\\\\gamma) / \\\\sigma_\\\\mathrm{ap}(2)`: the aperture dispersion of the CPL over that
    of the dPIE with the same dispersion parameter and radii (so the same mass inside the cut); 1 at slope 2."""
    return jnp.sqrt(aperture_dispersion_sq(slope, a, s, aperture) / aperture_dispersion_sq(2.0, a, s, aperture))
