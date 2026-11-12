"""Shared geometry of the elliptical kernels: the principal frame of an (e1, e2) ellipticity and the rotations of
the deflection, the Jacobian and its derivative from that frame back to the sky.

The ellipticity is the Cartesian pair :math:`(e_1, e_2) = e (\\cos 2\\phi, \\sin 2\\phi)` with
:math:`e = (1 - q)/(1 + q)`, axis ratio :math:`q` and position angle :math:`\\phi` of the major axis."""
from __future__ import annotations

import jax
import jax.numpy as jnp

_ELLIPTICITY_SINGULARITY = 1.0e-12


@jax.custom_jvp
def ellipticity_frame(e1, e2):
    """Position angle and axis ratio of the (e1, e2) ellipticity.  The lensing fields are even in the ellipticity, so
    their derivative vanishes at e = 0; the custom tangent below returns that limit instead of the 0/0 of arctan2."""
    c = jnp.sqrt(e1 * e1 + e2 * e2)
    return 0.5 * jnp.arctan2(e2, e1), (1.0 - c) / (1.0 + c)


@ellipticity_frame.defjvp
def _ellipticity_frame_jvp(primals, tangents):
    e1, e2 = primals
    t1, t2 = tangents
    c2 = e1 * e1 + e2 * e2
    regular = c2 > _ELLIPTICITY_SINGULARITY ** 2
    safe_c2 = jnp.where(regular, c2, 1.0)
    c = jnp.sqrt(safe_c2)
    dphi = jnp.where(regular, 0.5 * (e1 * t2 - e2 * t1) / safe_c2, 0.0)
    dq = jnp.where(regular, -2.0 / (1.0 + c) ** 2 * (e1 * t1 + e2 * t2) / c, 0.0)
    return ellipticity_frame(e1, e2), (dphi, dq)


def rotation(e1, e2):
    """Axis ratio q and the matrix R with principal frame = R (theta - centre), for the (e1, e2) ellipticity."""
    phi, q = ellipticity_frame(e1, e2)
    c, s = jnp.cos(phi), jnp.sin(phi)
    return q, jnp.array([[c, s], [-s, c]])


def symmetric(fxx, fxy, fyy):
    """The symmetric matrices [[fxx, fxy], [fxy, fyy]], shape (..., 2, 2)."""
    return jnp.stack((jnp.stack((fxx, fxy), -1), jnp.stack((fxy, fyy), -1)), -2)


def to_sky_vector(matrix, frame_vector):
    """A vector field (N, 2) given in the principal frame, on the sky."""
    return jnp.einsum("ai,na->ni", matrix, frame_vector)


def to_sky_matrix(matrix, frame_matrix):
    """A rank-2 field (N, 2, 2) given in the principal frame, on the sky."""
    return jnp.einsum("ai,bj,nab->nij", matrix, matrix, frame_matrix)


def to_sky_third(matrix, frame_third):
    """A rank-3 field (N, 2, 2, 2) given in the principal frame, on the sky."""
    return jnp.einsum("ai,bj,ck,nabc->nijk", matrix, matrix, matrix, frame_third)
