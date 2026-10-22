"""Exact DPIE_NIE_E fields and analytic spatial derivatives of its Hessian.

Uses the installed JAXtronomy Cartesian-ellipticity geometry and regular atan
function. The factored NIE difference and ``2 * sigma0 * Ra * Rs`` amplitude
match DPIE_NIE_E, including circular and nearly equal-radius limits. This is
an exact-kernel optimization, not the member far-field approximation.
"""
import jax
import jax.numpy as jnp
from jaxtronomy.LensModel.Profiles.nie_e import _geometry, _atan_sqrt_ratio


def dpie_fields_and_third_derivatives(x, y, sigma0, a, b, e1, e2, cx=0., cy=0.):
    """Return alpha, U=d(alpha)/d(theta), and dU[..., i, j, k]=dU_ij/dtheta_k.

    Spatial axes are trailing; x/y may be arrays. Parameters may be scalars or
    broadcast over images (e.g. source-redshift-dependent amplitudes). The
    physical domain is 0 < a < b and e1**2 + e2**2 < 1. Callers reject invalid
    proposals before evaluating this kernel. All parameter derivatives remain
    available through JAX, including second derivatives used for mass seeding.
    """
    x, y, d, e, v, rsq, rx, ry = _geometry(x, y, e1, e2, cx, cy)
    qxx = (1 + e1 * e1 + e2 * e2 - 2 * e1) / d
    qxy = -2 * e2 / d
    qyy = (1 + e1 * e1 + e2 * e2 + 2 * e1) / d
    ua, ub = jnp.sqrt(a * a + rsq), jnp.sqrt(b * b + rsq)
    da, db = ua + a, ub + b
    c = jnp.conj(e) / d
    T = ua + ub
    m = 1 + (a + b) / T
    D = da * db + c * v * v
    g = v * m / D
    k2 = c * (b - a) ** 2
    u = k2 * g * g
    r = (rx, ry)
    q = ((qxx, qxy), (qxy, qyy))
    vl = (1 - e, 1j * (1 + e))
    ual = [rl / ua for rl in r]
    ubl = [rl / ub for rl in r]
    Tl = [ual[i] + ubl[i] for i in range(2)]
    ml = [-(a + b) * Tl[i] / T ** 2 for i in range(2)]
    Dl = [ual[i] * db + da * ubl[i] + 2 * c * v * vl[i] for i in range(2)]
    Nl = [vl[i] * m + v * ml[i] for i in range(2)]
    gl = [(Nl[i] - g * Dl[i]) / D for i in range(2)]
    # For N=v*m and D*g=N, differentiating twice gives
    # D*g_ik = N_ik - D_ik*g - D_i*g_k - D_k*g_i.
    gll = [[None, None], [None, None]]
    for i in range(2):
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
    # d[g * atan(sqrt(k2*g**2))/sqrt(k2*g**2)]/dg = 1/(1+u).
    # Differentiate the same symmetrized Hessian as JAXtronomy.
    S = amp / (1 + u)
    ul = [2 * k2 * g * gl[i] for i in range(2)]
    Sl = [-S * ul[i] / (1 + u) for i in range(2)]
    ax, ay = S * gl[0], S * gl[1]
    hxy = 0.5 * (jnp.imag(ax) + jnp.real(ay))
    U = jnp.stack((jnp.stack((jnp.real(ax), hxy), -1), jnp.stack((hxy, jnp.imag(ay)), -1)), -2)
    dU_l = []
    for l in range(2):
        ax_l = Sl[l] * gl[0] + S * gll[0][l]
        ay_l = Sl[l] * gl[1] + S * gll[1][l]
        cross = 0.5 * (jnp.imag(ax_l) + jnp.real(ay_l))
        dU_l.append(jnp.stack((jnp.stack((jnp.real(ax_l), cross), -1), jnp.stack((cross, jnp.imag(ay_l)), -1)), -2))
    return jnp.stack((jnp.real(alpha), jnp.imag(alpha)), -1), U, jnp.stack(dU_l, -1)


grouped_dpie_fields_and_third_derivatives = jax.vmap(
    dpie_fields_and_third_derivatives, in_axes=(None, None, 0, 0, 0, 0, 0, 0, 0))
