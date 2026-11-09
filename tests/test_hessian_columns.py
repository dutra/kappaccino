"""sampling.hessian_columns: the chunked forward-over-reverse Hessian equals the full Hessian, for every column subset
and a chunk that does not divide the dimension."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kappaccino.inference.sampling import hessian_columns


def _function(x):
    return jnp.sum(jnp.sin(x) * jnp.roll(x, 1)) + 0.5 * jnp.sum(x ** 2) * jnp.sum(jnp.exp(0.1 * x))


def test_hessian_columns_match_the_full_hessian():
    x = jnp.asarray(np.random.default_rng(1).normal(size=11))
    full = np.asarray(jax.hessian(_function)(x))
    np.testing.assert_allclose(hessian_columns(_function, x, chunk=4), full, rtol=1e-12, atol=1e-12)
    columns = np.asarray([7, 0, 3])
    np.testing.assert_allclose(hessian_columns(_function, x, columns, chunk=2), full[:, columns], rtol=1e-12, atol=1e-12)
    assert hessian_columns(_function, x, np.zeros(0, dtype=int)).shape == (11, 0)
