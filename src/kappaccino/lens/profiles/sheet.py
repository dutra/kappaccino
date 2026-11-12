"""Uniform mass sheet (profile code 101) about the reference origin: :math:`\\alpha = \\kappa \\theta`, Jacobian
:math:`\\kappa I`.  ``kappa`` is the convergence the source sees (the lensing efficiency already applied), one value
or one per point."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

PARAMETERS = ("kappa",)     # PackedLensState fields, in kernel order


def fields(x, y, kappa, derivative=True):
    """alpha (N, 2), U = d alpha / d theta (N, 2, 2) and dU (N, 2, 2, 2; zero, None unless ``derivative``)."""
    kappa = jnp.broadcast_to(jnp.asarray(kappa, dtype=jnp.float64), jnp.shape(x))
    alpha = kappa[..., None] * jnp.stack((x, y), -1)
    u = kappa[..., None, None] * jnp.eye(2)
    return alpha, u, (jnp.zeros(jnp.shape(x) + (2, 2, 2), dtype=u.dtype) if derivative else None)


grouped_fields = {
    True: jax.vmap(fields, in_axes=(None, None, 0)),
    False: jax.vmap(functools.partial(fields, derivative=False), in_axes=(None, None, 0)),
}
