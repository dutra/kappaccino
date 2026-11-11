"""The batched global image search agrees with a brute-force grid search and recovers the observed images."""
from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from kappaccino.likelihood.evaluator import Evaluator
from kappaccino.diagnostics.exact_images import ExactImageSolver
from kappaccino.core.state import build_state


def test_global_search_matches_brute_force_and_recovers_observed_images(synthetic_config, brute_force_images):
    state = build_state(synthetic_config)
    evaluator = Evaluator(state)
    # The synthetic catalog does not sit on the model's images, so widen the window to contain them.
    config = replace(synthetic_config.image_diagnostics, exact_image_finder="global-jax", exact_image_min_distance_arcsec=0.1, exact_image_window_margin_arcsec=12.0)
    solver = ExactImageSolver(evaluator, config)
    theta = jnp.asarray(state.initial_theta)
    physical, sources = np.asarray(evaluator.physical(theta)), np.asarray(evaluator.source_positions(theta))
    family_by_id = {f.family_id: f for f in state.family_data}
    families = [family_by_id[fid] for fid in state.images.family_ids]
    found = solver.solve_all_global(families, physical, sources)
    assert len(found) == len(families)
    for k, family in enumerate(families):
        x, y = found[k]
        assert x.size >= 1
        # every solution solves the lens equation for this family's source
        packed = evaluator.packed_state_for_z(jnp.asarray(physical), family.z_source)
        bx, by, *_ = solver.ray_shoot_and_jacobian(packed, x, y)
        np.testing.assert_allclose(np.column_stack([bx, by]), np.broadcast_to(sources[k], (x.size, 2)), atol=5e-3)
        # the same image set as a brute-force search over the same window
        half = 0.5 * family.search_window + config.exact_image_window_margin_arcsec
        lx, ly = brute_force_images(solver, packed, sources[k], (family.x_center, family.y_center), half, pitch=0.1,
                                    min_separation=config.exact_image_min_separation_arcsec)
        d = np.hypot(x[:, None] - lx[None, :], y[:, None] - ly[None, :])
        assert x.size == lx.size and np.all(d.min(axis=1) < 0.01) and np.all(d.min(axis=0) < 0.01), (family.family_id, x.size, lx.size)
        # no duplicates
        if x.size > 1:
            pair = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) + np.eye(x.size) * 1e9
            assert pair.min() > 0.01
    details = solver.predict_all_global(state.images, families, physical, sources)
    assert len(details) == len(families) and all(np.isfinite(d["magnification"]).all() for d in details)
