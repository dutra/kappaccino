"""The dPIE (dual pseudo-isothermal ellipse, Lenstool profile 81): the difference of two non-singular isothermal
ellipses with core radius ``a`` and cut radius ``b``,

.. math::

    \\kappa(\\xi) = \\frac{\\sigma_0 a b}{2 (b - a)} \\left[(\\xi^2 + a^2)^{-1/2} - (\\xi^2 + b^2)^{-1/2}\\right],

in the Cartesian ellipticity (e1, e2).  The fields are closed forms in the complex position, so nothing is rotated:
``geometry`` holds everything that depends on the position, the shape and the centre, and ``fields_from_geometry``
adds the amplitude and the radii.  A member with a fixed catalog shape and position reuses its geometry at the image
positions on every evaluation."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

PARAMETERS = ("sigma0", "Ra", "Rs", "e1", "e2", "center_x", "center_y")     # PackedLensState fields, in kernel order


def _atan_sqrt_ratio(u):
    """atan(sqrt(u))/sqrt(u), evaluated without a singular intermediate at zero."""
    small = jnp.abs(u) < 0.01
    root = jnp.sqrt(jnp.where(small, 1.0 + 0j, u))
    series = jnp.full_like(u, -1.0 / 19.0)
    for n in range(8, -1, -1):
        series = (-1.0) ** n / (2 * n + 1) + u * series
    return jnp.where(small, series, jnp.arctan(root) / root)


def _base_geometry(x, y, e1, e2, center_x, center_y):
    x, y = jnp.asarray(x) - center_x, jnp.asarray(y) - center_y
    e_sq = e1 * e1 + e2 * e2
    d = 1.0 - e_sq
    qxx = (1 + e_sq - 2 * e1) / d
    qxy = -2 * e2 / d
    qyy = (1 + e_sq + 2 * e1) / d
    rx, ry = qxx * x + qxy * y, qxy * x + qyy * y
    radius_sq = x * rx + y * ry
    e = e1 + 1j * e2
    v = (x + 1j * y) - e * (x - 1j * y)
    return x, y, d, e, v, radius_sq, rx, ry


def geometry(x, y, e1, e2, center_x, center_y):
    """Geometry-only quantities of the cartesian dPIE kernel at points (x, y): everything that depends on the
    position, the ellipticity and the centre but not on the amplitude or the radii.  For components with fixed
    catalog geometry this is a constant and can be precomputed once per set of evaluation points."""
    xs, ys, d, e, v, rsq, rx, ry = _base_geometry(x, y, e1, e2, center_x, center_y)
    e_sq = e1 * e1 + e2 * e2
    qxx, qxy, qyy = (1 + e_sq - 2 * e1) / d, -2 * e2 / d, (1 + e_sq + 2 * e1) / d
    c = jnp.conj(e) / d
    return dict(rsq=rsq, rx=rx, ry=ry, v=v, c=c, qxx=qxx * jnp.ones_like(rsq), qxy=qxy * jnp.ones_like(rsq), qyy=qyy * jnp.ones_like(rsq),
                vlx=(1 - e) * jnp.ones_like(v), vly=(1j * (1 + e)) * jnp.ones_like(v))


def fields_from_geometry(geom, sigma0, a, b, derivative=True):
    """alpha (N, 2), U = d alpha / d theta (N, 2, 2) and dU[..., i, j, l] = d U_ij / d theta_l (N, 2, 2, 2) of one
    cartesian dPIE component from its precomputed geometry; the analytic form of two jvps through the Hessian.
    With ``derivative`` False the second-derivative terms are skipped and dU is None."""
    rsq, c, v = geom["rsq"], geom["c"], geom["v"]
    r = (geom["rx"], geom["ry"])
    q = ((geom["qxx"], geom["qxy"]), (geom["qxy"], geom["qyy"]))
    vl = (geom["vlx"], geom["vly"])
    ua, ub = jnp.sqrt(a * a + rsq), jnp.sqrt(b * b + rsq)
    da, db = ua + a, ub + b
    T = ua + ub
    m = 1 + (a + b) / T
    D = da * db + c * v * v
    g = v * m / D
    k2 = c * (b - a) ** 2
    u = k2 * g * g
    ual = [rl / ua for rl in r]
    ubl = [rl / ub for rl in r]
    Tl = [ual[i] + ubl[i] for i in range(2)]
    ml = [-(a + b) * Tl[i] / T ** 2 for i in range(2)]
    Dl = [ual[i] * db + da * ubl[i] + 2 * c * v * vl[i] for i in range(2)]
    Nl = [vl[i] * m + v * ml[i] for i in range(2)]
    gl = [(Nl[i] - g * Dl[i]) / D for i in range(2)]
    gll = [[None, None], [None, None]]
    for i in range(2 if derivative else 0):
        for k in range(i, 2):
            ua_ik = q[i][k] / ua - r[i] * r[k] / ua ** 3
            ub_ik = q[i][k] / ub - r[i] * r[k] / ub ** 3
            T_ik = ua_ik + ub_ik
            m_ik = -(a + b) * (T_ik / T ** 2 - 2 * Tl[i] * Tl[k] / T ** 3)
            D_ik = ua_ik * db + ual[i] * ubl[k] + ual[k] * ubl[i] + da * ub_ik + 2 * c * vl[i] * vl[k]
            N_ik = vl[i] * ml[k] + vl[k] * ml[i] + v * m_ik
            gll[i][k] = gll[k][i] = (N_ik - gl[k] * Dl[i] - g * D_ik - gl[i] * Dl[k]) / D
    amp = 2 * sigma0 * a * b
    alpha = amp * g * _atan_sqrt_ratio(u)
    S = amp / (1 + u)
    ul = [2 * k2 * g * gl[i] for i in range(2)]
    Sl = [-S * ul[i] / (1 + u) for i in range(2)]
    ax, ay = S * gl[0], S * gl[1]
    hxy = 0.5 * (jnp.imag(ax) + jnp.real(ay))
    U = jnp.stack((jnp.stack((jnp.real(ax), hxy), -1), jnp.stack((hxy, jnp.imag(ay)), -1)), -2)
    if not derivative:
        return jnp.stack((jnp.real(alpha), jnp.imag(alpha)), -1), U, None
    dU_l = []
    for l in range(2):
        ax_l = Sl[l] * gl[0] + S * gll[0][l]
        ay_l = Sl[l] * gl[1] + S * gll[1][l]
        cross = 0.5 * (jnp.imag(ax_l) + jnp.real(ay_l))
        dU_l.append(jnp.stack((jnp.stack((jnp.real(ax_l), cross), -1), jnp.stack((cross, jnp.imag(ay_l)), -1)), -2))
    return jnp.stack((jnp.real(alpha), jnp.imag(alpha)), -1), U, jnp.stack(dU_l, -1)


def fields(x, y, sigma0, a, b, e1, e2, center_x, center_y, derivative=True):
    """alpha, U and dU (None unless ``derivative``) of one dPIE component at points (x, y) with variable geometry."""
    return fields_from_geometry(geometry(x, y, e1, e2, center_x, center_y), sigma0, a, b, derivative)


grouped_geometry = jax.vmap(geometry, in_axes=(None, None, 0, 0, 0, 0))
# Batches over components; the ``derivative`` flag is static, so each variant is its own vmapped function.
grouped_fields = {
    True: jax.vmap(fields, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0)),
    False: jax.vmap(functools.partial(fields, derivative=False), in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0)),
}
grouped_fields_from_geometry = {
    True: jax.vmap(fields_from_geometry, in_axes=(0, 0, 0, 0)),
    False: jax.vmap(functools.partial(fields_from_geometry, derivative=False), in_axes=(0, 0, 0, 0)),
}
