"""Canonical engine naming, legacy aliases, and unchanged scientific targets."""
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from lenscluster.member_tiers import MemberTierRecord
from lenscluster.output_paths import _identity, prepare_output
from lenscluster.planning import compile_run_plan
from lenscluster.sampling_engines import canonical_sampling_engine
from test_smoothed_integration import make_evaluator, small_config, MODE


def test_alias_plans_identity_and_manifest_compatibility(tmp_path):
    config = small_config(tmp_path)
    config = replace(config, paths=replace(config.paths, output_dir=tmp_path, run_name='run', layout='results-v1'))
    legacy = replace(config, workflow=replace(config.workflow, sampling_engine='full_flat',
                     stage1_sampling_engine='full_flat', stage2_sampling_engine='full_flat'))
    plan, old_plan = compile_run_plan(config), compile_run_plan(legacy)
    assert plan == old_plan
    assert plan.runtime_args.values == old_plan.runtime_args.values
    assert all(stage.sampling_engine == 'tiered_members' for stage in plan.stages)
    assert plan.config.workflow.sampling_engine == 'tiered_members'
    assert plan.run_metadata == old_plan.run_metadata
    current = config.to_json_dict()
    old = json.loads(json.dumps(current))
    for key in ('sampling_engine', 'stage1_sampling_engine', 'stage2_sampling_engine'):
        old['workflow'][key] = 'full_flat'
    assert _identity(current) == _identity(old)
    prepare_output(plan)
    manifest = tmp_path/'run/config.json'
    payload = json.loads(manifest.read_text())
    payload['config'] = old
    payload['identity']['workflow'].update({key:'full_flat' for key in (
        'sampling_engine', 'stage1_sampling_engine', 'stage2_sampling_engine')})
    manifest.write_text(json.dumps(payload))
    before = manifest.read_bytes()
    prepare_output(plan)
    assert manifest.read_bytes() == before
    changed = replace(config, member_tiers=replace(config.member_tiers, top_exact=0))
    with pytest.raises(ValueError, match='incompatible.*member_tiers'):
        prepare_output(compile_run_plan(changed))


@pytest.mark.parametrize('field', ['sampling_engine', 'stage1_sampling_engine', 'stage2_sampling_engine'])
def test_tiered_engine_restricted_to_smoothed_config(tmp_path, field):
    from test_latent_displacement import configured
    config = configured()
    config = replace(config, workflow=replace(config.workflow, **{field:'tiered_members'}))
    with pytest.raises(ValueError, match='tiered_members requires smoothed-local-image-plane'):
        compile_run_plan(config)


@pytest.mark.parametrize('mode', ['source', 'local-jacobian', 'critical-arc-anisotropic-image-plane',
                                  'isotropic-latent-displacement'])
def test_legacy_engine_names_unchanged(mode):
    assert canonical_sampling_engine('full_flat', mode) == 'full_flat'
    assert canonical_sampling_engine('refreshing_surrogate_flat', mode) == 'refreshing_surrogate_flat'
    with pytest.raises(ValueError, match='requires smoothed'):
        canonical_sampling_engine('tiered_members', mode)


@pytest.mark.parametrize('fixed', [False, True])
def test_direct_alias_density_gradients_counts_and_refresh(tmp_path, monkeypatch, fixed):
    args, initial, theta = make_evaluator(tmp_path)
    if fixed:
        meta = initial.state.smoothed_model
        tiers = MemberTierRecord(tuple(meta['member_ids']), tuple(meta['member_components']),
            exact_sampled=(0,), exact_absorbed=(1,), approximate_sampled=(2,), approximate_absorbed=(3,))
        args, initial, theta = make_evaluator(tmp_path, tiers=tiers.to_dict())
    alias = cs._build_cluster_evaluator_from_args(args, initial.state, sampling_engine='full_flat')
    assert alias.sampling_engine == initial.sampling_engine == 'tiered_members'
    np.testing.assert_array_equal(alias.smoothed_far_positions, initial.smoothed_far_positions)
    np.testing.assert_array_equal(alias.smoothed_absorbed_positions, initial.smoothed_absorbed_positions)
    for displacement in (0., .001):
        point = jnp.asarray(theta).at[0].add(displacement)
        value, gradient = jax.jit(jax.value_and_grad(initial._source_loglike_impl))(point)
        alias_value, alias_gradient = jax.jit(jax.value_and_grad(alias._source_loglike_impl))(point)
        assert np.isfinite(value) and np.isfinite(gradient).all()
        np.testing.assert_array_equal(value, alias_value)
        np.testing.assert_array_equal(gradient, alias_gradient)
    def forbidden(*a, **kw):
        raise AssertionError('Legacy refresh called')
    monkeypatch.setattr(initial, '_refresh_flat_scaling_scatter_cache_from_reference', forbidden, raising=False)
    monkeypatch.setattr(initial, '_refresh_flat_source_metric_cache_from_reference', forbidden, raising=False)
    initial.refresh_scaling_scatter_cache(theta)
    initial.refresh_source_metric_cache(theta)
    rows = dict((name, value) for name,value,_ in cs._stage_model_summary_rows(initial))
    assert rows['sampling_engine'] == 'tiered_members'
    expected = 2 if fixed else 4
    assert rows['exact_members'] == str(expected)
    assert rows['approximate_members'] == str(4-expected)
    assert rows['sampled_offsets'] == ('2' if fixed else '0')
    assert rows['absorbed_offsets'] == ('2' if fixed else '4')
    if not fixed:
        assert rows['initializer_members'] == 'exact_profiles_provisional_absorbed_scatter'
    messages = []
    monkeypatch.setattr(cs, '_log', lambda args, message, **kw: messages.append(message))
    cs._log_solver_active_approximation_warning(args, initial)
    text = '\n'.join(messages)
    assert 'sampling_engine=tiered_members' in text
    assert 'exact flattened critical-arc' not in text


def test_direct_constructor_rejects_tiered_legacy_mode(tmp_path):
    from test_source_covariance import make_evaluator as legacy_evaluator
    legacy, _ = legacy_evaluator()
    with pytest.raises(ValueError, match='tiered_members requires smoothed-local-image-plane'):
        cs.ClusterJAXEvaluator(legacy.state, .1, sampling_engine='tiered_members', sample_likelihood_mode='source')
