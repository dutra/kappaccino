"""Sampler diagnostics: E-BFMI, the adaptation summary and the dense member-offset mass block.  The diagnostics
of a real run are checked by the NUTS smoke test in test_pipeline.py."""
from __future__ import annotations

import numpy as np

from kappaccino.sampling import adaptation_diagnostics, e_bfmi


def test_e_bfmi_separates_independent_and_random_walk_energies():
    rng = np.random.default_rng(1)
    independent = rng.normal(size=4000)                       # energies redrawn each transition: E-BFMI near 2
    random_walk = np.cumsum(0.05 * rng.normal(size=4000))      # energies that barely move between transitions
    assert 1.8 < e_bfmi(independent) < 2.2 and e_bfmi(random_walk) < 0.05
    out = adaptation_diagnostics(np.array([0.1, 0.02]), np.stack([independent, random_walk]), np.full((2, 4000), 0.9),
                                 np.zeros((2, 4000), bool), np.stack([np.full(4000, 7.0), np.full(4000, 255.0)]), max_tree_depth=8)
    assert [c["step_size"] for c in out["per_chain"]] == [0.1, 0.02] and out["min_e_bfmi"] == out["per_chain"][1]["e_bfmi"]
    assert out["per_chain"][0]["tree_depth_saturation"] == 0.0 and out["per_chain"][1]["tree_depth_saturation"] == 1.0


def test_dense_member_offsets_join_the_hessian_seeded_block(synthetic_config):
    """SamplingConfig.dense_member_offsets: the member-offset sites move from the diagonal into the dense block, whose
    Hessian seed (chunked forward-over-reverse) is symmetric positive definite; the sources stay diagonal."""
    import numpy as np
    from kappaccino.evaluator import Evaluator
    from kappaccino.sampling import block_mass_seed, posterior_model
    from kappaccino.state import build_state
    state = build_state(synthetic_config)
    specs = state.parameter_specs
    model = posterior_model(specs, Evaluator(state).log_likelihood)
    theta = np.asarray(state.initial_theta, dtype=float)
    n_offsets = sum(s.component_family == "member_scatter_offset" for s in specs)
    assert n_offsets > 0
    plain, _, d0 = block_mass_seed(model, specs, theta)
    dense, inverse, d1 = block_mass_seed(model, specs, theta, dense_families=("member_scatter_offset",))
    assert d1["global_dim"] == d0["global_dim"] + n_offsets and d1["latent_dim"] == d0["latent_dim"] - n_offsets
    block = np.asarray(inverse[dense[0]])
    assert block.shape == (d1["global_dim"], d1["global_dim"])
    np.testing.assert_allclose(block, block.T, atol=1e-10)
    assert np.linalg.eigvalsh(block).min() > 0



def test_stiffest_direction_finds_the_largest_whitened_curvature():
    """Power iteration on W^T H W: for a quadratic with curvatures (1, 100, 4) the stiffest direction is the second axis
    (step limit 2/10); whitening that axis by 1/10 makes the third the stiffest (curvature 4)."""
    import jax.numpy as jnp
    from kappaccino.sampling import stiffest_direction
    curvature = jnp.asarray([1.0, 100.0, 4.0])
    potential = lambda x: 0.5 * jnp.sum(curvature * x * x)
    x0 = jnp.asarray([0.3, -0.2, 0.1])
    value, direction = stiffest_direction(potential, x0, lambda v, transpose=False: v, iterations=60)
    np.testing.assert_allclose(value, 100.0, rtol=1e-6)
    assert np.argmax(np.abs(direction)) == 1
    scale = jnp.asarray([1.0, 0.1, 1.0])
    value, direction = stiffest_direction(potential, x0, lambda v, transpose=False: scale * v, iterations=200)
    np.testing.assert_allclose(value, 4.0, rtol=1e-4)
    assert np.argmax(np.abs(direction)) == 2
