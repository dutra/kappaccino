"""External shear (Lenstool profile 14): the deflection :math:`\\alpha = (\\gamma_1 x + \\gamma_2 y,\\; \\gamma_2 x -
\\gamma_1 y)` about the reference origin, with constant Jacobian [[gamma1, gamma2], [gamma2, -gamma1]]."""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

PARAMETERS = ("gamma1", "gamma2")     # PackedLensState fields, in kernel order


def fields(x, y, gamma1, gamma2, derivative=True):
    """alpha (N, 2), U = d alpha / d theta (N, 2, 2) and dU (N, 2, 2, 2; zero, None unless ``derivative``)."""
    ones = jnp.ones_like(x)
    alpha = jnp.stack((gamma1 * x + gamma2 * y, gamma2 * x - gamma1 * y), -1)
    u = jnp.stack((jnp.stack((gamma1 * ones, gamma2 * ones), -1), jnp.stack((gamma2 * ones, -gamma1 * ones), -1)), -2)
    return alpha, u, (jnp.zeros(jnp.shape(x) + (2, 2, 2), dtype=u.dtype) if derivative else None)


grouped_fields = {
    True: jax.vmap(fields, in_axes=(None, None, 0, 0)),
    False: jax.vmap(functools.partial(fields, derivative=False), in_axes=(None, None, 0, 0)),
}
