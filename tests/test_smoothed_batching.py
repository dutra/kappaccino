"""Numerical contracts for fixed-geometry profiles and compiled reporting."""
from dataclasses import replace
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np

from lenscluster import cluster_solver as cs
from lenscluster.config import IndependentMemberHaloConfig
from lenscluster.member_tiers import MemberTierRecord
from test_smoothed_integration import make_evaluator, small_config


def original_exact_rows(self, xy, packed, components, *, spatial_derivative=False):
    """Pre-split grouped kernel, including its delta-A sign convention."""
    rows = self._flat_component_alpha_and_jacobian_delta_rows_for_components(
        xy[:, 0], xy[:, 1], packed, components)
    fields = (jnp.stack(rows[:2], -1),
              -jnp.stack((jnp.stack(rows[2:4], -1), jnp.stack(rows[4:6], -1)), -2))
    if spatial_derivative:
        hessian = lambda position: original_exact_rows(self, position, packed, components)[1]
        du = jnp.stack([jax.jvp(hessian, (xy,),
            (jnp.broadcast_to(jnp.asarray(direction), xy.shape),))[1]
            for direction in ((1., 0.), (0., 1.))], -1)
        return (*fields, du)
    return fields


def assert_tree_close(actual, expected):
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_fixed_geometry_rows_preserve_order_and_spatial_derivatives(tmp_path, monkeypatch):
    config = small_config(tmp_path)
    config = replace(config, model=replace(config.model, independent_member_halos=(
        IndependentMemberHaloConfig(population_id='members', catalog_id='g1'),)))
    _, ev, theta = make_evaluator(tmp_path, config=config)
    members = ev.smoothed_member_components
    others = np.setdiff1d(ev.smoothed_exact_components, members)
    data = ev.flat_critical_arc_data
    xy = jnp.stack((data.x_obs, data.y_obs), -1)
    packed = ev._smoothed_packed(ev._physical_parameter_vector(jnp.asarray(theta)))
    # Empty groups, all fixed, all general, and noncatalog ordering/duplicates.
    selections = [np.array([], dtype=int), members, others,
                  np.array([members[-1], others[-1], members[0], others[0], members[-1]])]
    for components in selections:
        def both(position):
            return (ev._smoothed_exact_rows(position, packed, components),
                    original_exact_rows(ev, position, packed, components))
        actual, expected = jax.jit(both)(xy)
        assert_tree_close(actual, expected)
        assert_tree_close(ev._smoothed_exact_rows(xy, packed, components, spatial_derivative=True),
                          original_exact_rows(ev, xy, packed, components, spatial_derivative=True))
        actual, expected = jax.jit(jax.jacfwd(lambda shift: both(xy + shift)))(jnp.zeros(2))
        assert_tree_close(actual, expected)
    # A population member with a free geometric parameter must fall back to the
    # general path and use its packed value, not silently freeze its catalog shape.
    spec = ev.state.packed_lens_spec
    indices = np.array(spec.e1_param_index, copy=True)
    indices[members[0]] = 0
    monkeypatch.setattr(spec, 'e1_param_index', indices)
    shifted = packed._replace(e1=packed.e1.at[members[0]].add(.07))
    assert_tree_close(ev._smoothed_exact_rows(xy, shifted, members),
                      original_exact_rows(ev, xy, shifted, members))


def test_complete_likelihood_and_gradients_match_unsplit_kernel(tmp_path, monkeypatch):
    _, initial, _ = make_evaluator(tmp_path)
    meta = initial.state.smoothed_model
    tiers = MemberTierRecord(tuple(meta['member_ids']), tuple(meta['member_components']),
        exact_sampled=(0,), exact_absorbed=(1,), approximate_sampled=(2,), approximate_absorbed=(3,))
    _, ev, theta = make_evaluator(tmp_path, tiers=tiers.to_dict())
    rng = np.random.default_rng(617)
    xs = jnp.asarray(np.stack([theta, theta + rng.normal(size=len(theta))*.02,
                              theta + rng.normal(size=len(theta))*.05]))
    fast = jax.jit(jax.vmap(jax.value_and_grad(ev._source_loglike_impl)))(xs)
    with monkeypatch.context() as patch:
        patch.setattr(ev, '_smoothed_exact_rows', MethodType(original_exact_rows, ev))
        old = jax.jit(jax.vmap(jax.value_and_grad(lambda x: ev._source_loglike_impl(x))))(xs)
    assert_tree_close(fast, old)
    assert np.isfinite(fast[0]).all() and np.isfinite(fast[1]).all()


def test_compiled_reporting_preserves_grouped_sources_and_physical_likelihood(tmp_path):
    _, ev, theta = make_evaluator(tmp_path)
    specs = ev.state.parameter_specs
    xs = jnp.asarray(theta + np.random.default_rng(21).normal(size=(6, len(theta)))*.015)
    expected = np.asarray(jax.vmap(ev._reported_physical_parameter_vector)(xs))
    expected = cs._add_dynamic_cut_reference_offsets_to_reported_array(expected, specs)
    grouped = np.asarray(xs).reshape((2, 3, -1))
    actual = cs._convert_sample_matrix_to_reported_physical(grouped, specs, ev)
    assert actual.shape == grouped.shape
    np.testing.assert_allclose(actual.reshape(expected.shape), expected, rtol=1e-12, atol=1e-12)
    expected_like = np.asarray(jax.vmap(lambda x: ev._smoothed_statistics(x)['loglike'].sum())(xs))
    for batch_size in (4, 6):  # Partial final chunk and a single complete batch.
        actual_like = cs._source_loglike_matrix(ev, np.asarray(xs), batch_size=batch_size)
        np.testing.assert_allclose(actual_like, expected_like, rtol=1e-12, atol=1e-12)
    assert cs._convert_sample_matrix_to_reported_physical(None, specs, ev) is None
    assert cs._convert_sample_matrix_to_reported_physical(grouped[:0], specs, ev).shape == grouped[:0].shape
    assert cs._source_loglike_matrix(ev, np.asarray(xs[:0])).shape == (0,)


def test_reporting_batches_rebuilt_when_geometry_is_reinitialized(tmp_path):
    _, ev, theta = make_evaluator(tmp_path)
    xs = jnp.asarray(theta)[None, :]
    before = cs._convert_sample_matrix_to_reported_physical(xs, ev.state.parameter_specs, ev)
    batch = ev._smoothed_reported_physical_batch
    ev.flat_critical_arc_data = replace(ev.flat_critical_arc_data,
        x_obs=ev.flat_critical_arc_data.x_obs + .2)
    ev._initialize_smoothed_model()
    assert ev._smoothed_reported_physical_batch is not batch
    after = cs._convert_sample_matrix_to_reported_physical(xs, ev.state.parameter_specs, ev)
    expected = cs._add_dynamic_cut_reference_offsets_to_reported_array(
        np.asarray(jax.vmap(ev._reported_physical_parameter_vector)(xs)), ev.state.parameter_specs)
    np.testing.assert_allclose(after, expected, rtol=1e-12, atol=1e-12)
    ix = ev.flat_critical_arc_data.global_family_source_x_param_index
    assert np.max(np.abs(after[:, ix] - before[:, ix])) > 1e-5
    like = cs._source_loglike_matrix(ev, np.asarray(xs))
    expected_like = np.asarray(jax.vmap(lambda x: ev._smoothed_statistics(x)['loglike'].sum())(xs))
    np.testing.assert_allclose(like, expected_like, rtol=1e-12, atol=1e-12)
