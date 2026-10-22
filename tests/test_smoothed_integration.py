"""Production evaluator contracts for the revised sampled-source likelihood."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from lenscluster.config import (
    CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig,
    LensModelConfig, MemberPopulationConfig, PriorConfig, RunPathsConfig,
    RuntimeConfig, StageScheduleConfig, WorkflowConfig, IndependentMemberHaloConfig,
)
from lenscluster.member_tiers import MemberTierRecord
from lenscluster.planning import compile_run_plan

MODE = 'smoothed-local-image-plane'


def small_config(tmp_path):
    images = tmp_path / 'images.cat'
    images.write_text('#REFERENCE 3\n'
        '1.a 3 5 1 1 0 2 24 1 .05\n1.b -4 -5 1 1 0 2 24 1 .05\n'
        '2.a 8 -6 1 1 0 3 24 1 .05\n2.b -7 9 1 1 0 3 24 1 .05\n')
    members = tmp_path / 'members.cat'
    members.write_text('#REFERENCE 3\n'
        'g1 2 3 1 .9 20 20 1 1\ng2 -6 7 1 .8 30 21 1 1\n'
        'g3 100 70 1 1 0 21 1 1\ng4 -100 -80 1 1 0 23 1 1\n')
    return LensClusterSolverConfig(
        model=LensModelConfig(cosmology=CosmologyConfig(H0=70., Om0=.3, Ode0=.7),
            large_halos=(DPIEHaloConfig(id='halo', x_centre=0., y_centre=0., angle_pos=0., v_disp=600., core_radius_kpc=5.,
                cut_radius_kpc=1000., z_lens=.3, ellipticite=.2,
                priors={'v_disp': PriorConfig('uniform', lower=500., upper=700.)}),),
            member_populations=(MemberPopulationConfig(id='members', catalog_path=str(members),
                mag0=20., corekpc=.2, sigma=100., cutkpc=10., z_lens=.3),),
            image_constraints=ImageConstraintsConfig(catalog_path=str(images), sigma_arcsec=.1)),
        paths=RunPathsConfig(output_dir=str(tmp_path/'output'), run_name='smoothed', layout='results-v1'),
        runtime=RuntimeConfig(chains=1, nuts_chain_method='sequential', dense_mass='lens-and-hyperparameters',
            skip_plots=True, skip_validation=True, quiet=True),
        workflow=WorkflowConfig(stage0_likelihood=MODE, stage1_likelihood=MODE, stage2_forward_mode=MODE,
            sampling_engine='tiered_members', stage1_sampling_engine='tiered_members', stage2_sampling_engine='tiered_members',
            source_position_parameterization='likelihood-whitened'),
        schedule=StageScheduleConfig(svi_steps=(2,2,2), refresh_every=(None,None,None),
            fit_method=('svi','svi+nuts'), warmup=(1,5), samples=(2,5),
            sampling_refresh_runs=(1,1), max_tree_depth=(3,3)))


def make_evaluator(tmp_path, *, tiers=None, parameterization='likelihood-whitened', config=None, engine='tiered_members'):
    config = small_config(tmp_path) if config is None else config
    config = replace(config, workflow=replace(config.workflow, source_position_parameterization=parameterization))
    args = cs._clone_args(compile_run_plan(config).runtime_args, sample_likelihood_mode=MODE,
        sampling_engine=engine, member_tier_phase='initializer' if tiers is None else 'fixed',
        member_tiers_payload=tiers, perturbation_discovery_stage0=False, refresh_every=None)
    state = cs._build_state_from_inputs(args, fit_mode_override='joint')
    evaluator = cs._build_cluster_evaluator_from_args(args, state)
    return args, evaluator, cs._default_theta(state.parameter_specs)


@pytest.mark.parametrize('fixed', [False, True])
def test_smoothed_path_never_evaluates_legacy_likelihoods_or_caches(tmp_path, monkeypatch, fixed):
    def forbidden(*args, **kwargs):
        raise AssertionError('Legacy likelihood/ranking/cache evaluated in smoothed mode')
    for name in (
        '_build_scaling_rank_diagnostics', '_flat_source_metric_arrays_from_cache',
        '_flat_source_loglike_impl', '_flat_local_jacobian_source_loglike_impl',
        '_flat_critical_arc_source_loglike_impl', '_flat_latent_displacement_loglike_impl',
        '_flat_refreshing_surrogate_source_loglike_impl',
        '_flat_refreshing_surrogate_local_jacobian_source_loglike_impl',
        '_flat_refreshing_surrogate_critical_arc_source_loglike_impl',
        '_refresh_flat_scaling_scatter_cache_from_reference',
        '_refresh_flat_source_metric_cache_from_reference', 'refresh_surrogate',
        '_full_covariance_source_summary', '_latent_displacement_source_summary',
    ):
        monkeypatch.setattr(cs.ClusterJAXEvaluator, name, forbidden)
    args, initial, theta = make_evaluator(tmp_path)
    if fixed:
        meta = initial.state.smoothed_model
        tiers = MemberTierRecord(tuple(meta['member_ids']), tuple(meta['member_components']),
            exact_sampled=(0,), exact_absorbed=(1,), approximate_sampled=(2,), approximate_absorbed=(3,))
        args, initial, theta = make_evaluator(tmp_path, tiers=tiers.to_dict())
    ev, theta = cs._prepare_direct_evaluator(args, initial.state)
    assert not ev.surrogate_enabled
    assert ev.flat_source_metric_cache is None
    assert ev.scaling_rank_df['importance'].isna().all()
    assert ev.scaling_rank_df['selection_mode'].str.startswith('smoothed_').all()
    actual = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(jnp.asarray(theta))
    expected = jax.jit(jax.value_and_grad(ev._flat_smoothed_loglike_impl))(jnp.asarray(theta))
    for x,y in zip(actual, expected):
        np.testing.assert_allclose(x,y,rtol=2e-12,atol=2e-10)
    assert np.isfinite(actual[0]) and np.isfinite(actual[1]).all()
    assert set(ev._family_source_summary(theta)) == {'1','2'}
    monkeypatch.setattr(cs, '_validation_source_centroid_weights', forbidden)
    monkeypatch.setattr(ev, '_solve_exact_images', lambda family, packed, sx, sy: (family.x_obs, family.y_obs))
    family = ev.state.family_data[0]
    prediction = ev._exact_family_prediction_details(theta, family)
    assert not prediction['failed']
    source = ev._source_position_for_family_numpy(theta, family.family_id)
    assert ev.validation_cache[family.family_id].last_source_x == pytest.approx(source[0])
    with pytest.raises(ValueError, match='legacy likelihood'):
        cs._build_cluster_evaluator_from_args(cs._clone_args(args, sample_likelihood_mode='source'), initial.state)
    with pytest.raises(ValueError, match='legacy perturbation-discovery'):
        cs._build_cluster_evaluator_from_args(args, replace(initial.state, perturbation_discovery_stage0=True))
    ev.refresh_scaling_scatter_cache(theta, reason='test_post_svi')
    ev.refresh_source_metric_cache(theta, reason='test_post_nuts')
    ev.sampling_engine = 'refreshing_surrogate_flat'
    with pytest.raises(ValueError, match='legacy engines are disabled'):
        ev._source_loglike_impl(jnp.asarray(theta))


def test_full_evaluator_gradient_and_physical_source_roundtrip(tmp_path):
    args, ev, theta = make_evaluator(tmp_path)
    value, gradient = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(theta)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    physical = ev.reported_physical_parameter_vector(theta)
    np.testing.assert_allclose(ev.reported_physical_to_latent_parameter_vector(physical), theta, atol=1e-10)
    for i in range(len(theta)):
        step = 1e-4 if i == 0 else 1e-5
        delta = np.eye(len(theta))[i] * step
        estimate = (ev._source_loglike_fn(theta+delta)-ev._source_loglike_fn(theta-delta))/(2*step)
        np.testing.assert_allclose(gradient[i], estimate, rtol=5e-4, atol=2e-5)
    assert len(ev.state.smoothed_model['member_ids']) == 4
    assert not any(s.field in {'sigma_log_scatter', 'mass_log_scatter'} for s in ev.state.parameter_specs)
    assert ev.state.source_position_parameterization == 'likelihood-whitened'
    summary = ev._smoothed_source_summary(theta)
    assert set(summary) == {'1', '2'}


def test_shared_redshift_bin_preserves_image_metadata_by_family(tmp_path):
    config = small_config(tmp_path)
    # Two families share a bin, with unequal image counts and interleaved input
    # rows. A bin-local image index cannot index an individual family's labels.
    Path(config.model.image_constraints.catalog_path).write_text('#REFERENCE 3\n'
        '2.b -7 9 1 1 0 2 24 1 .05\n1.c 2 -3 1 1 0 2 24 1 .05\n'
        '1.a 3 5 1 1 0 2 24 1 .05\n2.a 8 -6 1 1 0 2 24 1 .05\n'
        '1.b -4 -5 1 1 0 2 24 1 .05\n')
    metadata = tmp_path/'image_metadata.csv'
    expected = {'1.a':(.11,1), '1.b':(.22,-1), '1.c':(.33,0),
                '2.a':(.44,-1), '2.b':(.55,1)}
    metadata.write_text('image_label,sigma_obs_arcsec,parity\n'+''.join(
        f'{label},{sigma},{parity}\n' for label,(sigma,parity) in reversed(list(expected.items()))))
    config = replace(config, model=replace(config.model, image_constraints=replace(
        config.model.image_constraints, metadata_path=str(metadata))))
    _, ev, theta = make_evaluator(tmp_path, config=config)
    assert len(ev.state.bin_data) == 1 and len(ev.state.family_data) == 2
    data = ev.flat_critical_arc_data
    families = {str(f.family_id):f for f in ev.state.family_data}
    for f_index, family_id in enumerate(data.family_ids):
        rows = np.flatnonzero(np.asarray(data.global_family_index_per_image) == f_index)
        family = families[family_id]
        np.testing.assert_array_equal(np.asarray(data.x_obs)[rows], family.x_obs)
        np.testing.assert_allclose(np.asarray(data.sigma_per_image)[rows],
                                   [expected[label][0] for label in family.image_labels])
        np.testing.assert_array_equal(np.asarray(ev.smoothed_parity)[rows],
                                      [expected[label][1] for label in family.image_labels])
    value, grad = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(theta)
    assert np.isfinite(value) and np.isfinite(grad).all()


def test_sampled_offsets_fixed_radii_and_all_four_tiers(tmp_path):
    _, initial, _ = make_evaluator(tmp_path)
    metadata = initial.state.smoothed_model
    tiers = MemberTierRecord(tuple(metadata['member_ids']), tuple(metadata['member_components']),
        exact_sampled=(0,), exact_absorbed=(1,), approximate_sampled=(2,), approximate_absorbed=(3,))
    _, ev, theta = make_evaluator(tmp_path, tiers=tiers.to_dict())
    p = ev._physical_parameter_vector(jnp.asarray(theta))
    packed = ev._smoothed_packed(p)
    offset = metadata['member_components'][0]
    index = ev.state.smoothed_model['offset_indices'][offset]
    changed = p.at[index].set(1.)
    updated = ev._smoothed_packed(changed)
    np.testing.assert_array_equal(packed.Ra, updated.Ra)
    np.testing.assert_array_equal(packed.Rs, updated.Rs)
    scatter = p[ev.state.smoothed_model['scatter_indices'][offset]]
    np.testing.assert_allclose(updated.sigma0[offset]/packed.sigma0[offset], np.exp(2*scatter))
    value, gradient = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(theta)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    assert tuple(ev.smoothed_absorbed_positions) == (1,3)


def test_jitted_sampling_uses_current_approximate_profiles_and_matches_original(tmp_path, monkeypatch, record_property):
    from lenscluster import member_tiers as mt
    from jaxtronomy.LensModel.Profiles.dpie_nie_e import DPIE_NIE_E
    _, initial, _ = make_evaluator(tmp_path)
    meta = initial.state.smoothed_model
    tiers = MemberTierRecord(tuple(meta['member_ids']), tuple(meta['member_components']),
        exact_sampled=(0,), exact_absorbed=(1,), approximate_sampled=(2,), approximate_absorbed=(3,))
    _, ev, theta = make_evaluator(tmp_path, tiers=tiers.to_dict())
    offset = ev.state.smoothed_model['offset_indices'][meta['member_components'][2]]
    theta = jnp.asarray(theta)
    batch = jnp.stack((theta, theta.at[offset].set(theta[offset]+.7)))
    components_seen = []
    original_rows = ev._smoothed_exact_rows
    def rows(xy, packed, components, **kwargs):
        components_seen.append(tuple(components))
        return original_rows(xy, packed, components, **kwargs)
    monkeypatch.setattr(ev, '_smoothed_exact_rows', rows)
    fast = jax.jit(jax.vmap(jax.value_and_grad(ev._source_loglike_impl)))
    fast_values, fast_grads = jax.block_until_ready(fast(batch))
    assert components_seen
    assert all(set(indices) == set(ev.smoothed_exact_components) for indices in components_seen)
    assert not set(ev.smoothed_member_components[2:]) & set(ev.smoothed_exact_components)
    assert np.all(np.isfinite(fast_grads)) and np.all(np.abs(fast_grads[:,offset]) > 1e-10)
    assert abs(float(fast_values[1]-fast_values[0])) > 1e-10
    # Independently check exact diagnostic rows against direct JAXtronomy.
    physical = ev._physical_parameter_vector(batch[1])
    packed = ev._smoothed_packed(physical)
    exact = jax.jit(lambda p: ev._smoothed_fields(p, force_exact=True))(physical)
    data = ev.flat_critical_arc_data
    for i, component in enumerate(ev.smoothed_member_components):
        arguments = (data.x_obs, data.y_obs, packed.sigma0[component], packed.Ra[component], packed.Rs[component],
                     packed.e1[component], packed.e2[component], packed.center_x[component], packed.center_y[component])
        alpha = np.stack(DPIE_NIE_E.derivatives(*arguments), axis=-1)
        xx,xy,yx,yy = DPIE_NIE_E.hessian(*arguments)
        u = np.stack((np.stack((xx,xy),-1),np.stack((yx,yy),-1)),-2)
        np.testing.assert_allclose(exact['member_alpha'][i], alpha, rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(exact['member_jacobian'][i], u, rtol=2e-12, atol=2e-13)
    def unoptimized(geometry, sigma, core, cut):
        f = mt.evaluate_farfield(geometry, sigma, core, cut)
        return mt.SamplingMemberFields(f.alpha, f.jacobian, f.jacobian_derivative.sum(axis=0))
    monkeypatch.setattr(mt, 'evaluate_farfield_sampling', unoptimized)
    reference = jax.jit(jax.vmap(jax.value_and_grad(lambda p: ev._source_loglike_impl(p))))
    old_values, old_grads = jax.block_until_ready(reference(batch))
    record_property('likelihood_max_abs_error', float(jnp.max(jnp.abs(fast_values-old_values))))
    record_property('gradient_max_abs_error', float(jnp.max(jnp.abs(fast_grads-old_grads))))
    np.testing.assert_allclose(fast_values, old_values, rtol=2e-12, atol=2e-10)
    np.testing.assert_allclose(fast_grads, old_grads, rtol=2e-11, atol=2e-10)
    step = np.zeros(len(theta)); step[offset] = 1e-5
    fd = (reference(batch+step)[0]-reference(batch-step)[0])/(2e-5)
    np.testing.assert_allclose(fast_grads[:,offset], fd, rtol=2e-4, atol=2e-6)


def test_independent_halo_excluded_from_population_and_empty_population(tmp_path):
    config = small_config(tmp_path)
    config = replace(config, model=replace(config.model, independent_member_halos=(
        IndependentMemberHaloConfig(population_id='members', catalog_id='g1'),)))
    _, ev, theta = make_evaluator(tmp_path, config=config)
    assert ev.state.smoothed_model['member_ids'] == ['members:g2', 'members:g3', 'members:g4']
    assert len(ev.smoothed_exact_components) == 5
    assert np.isfinite(ev._source_loglike_fn(theta))
    config = replace(config, model=replace(config.model, member_populations=(), independent_member_halos=()))
    _, ev, theta = make_evaluator(tmp_path, config=config)
    assert ev.state.smoothed_model['member_ids'] == []
    value, grad = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(theta)
    assert np.isfinite(value) and np.isfinite(grad).all()


def test_masked_coordinates_and_unrepresentable_scatter(tmp_path):
    _, ev, theta = make_evaluator(tmp_path)
    data = ev.flat_critical_arc_data
    # Keep one active constraint in this family; discarded rows never enter kernels.
    ev.flat_critical_arc_data = replace(data, x_obs=data.x_obs.at[0].set(jnp.nan),
        y_obs=data.y_obs.at[0].set(jnp.nan), image_has_constraint=data.image_has_constraint.at[0].set(False))
    ev._initialize_smoothed_model()
    value, grad = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(theta)
    assert np.isfinite(value) and np.isfinite(grad).all()
    for index in (ev.state.smoothed_model['sigma_src_index'],
                  ev.state.smoothed_model['scatter_indices'][ev.smoothed_member_components[0]]):
        proposal = jnp.asarray(theta).at[index].set(1000.)
        value, grad = jax.jit(jax.value_and_grad(ev._source_loglike_impl))(proposal)
        assert value == -jnp.inf and np.isfinite(grad).all()


def test_stage_handoff_rewhitens_physical_sources_without_moving_priors(tmp_path):
    _, before, theta = make_evaluator(tmp_path)
    reported = before.reported_physical_parameter_vector(theta)
    handoff = dict(zip((s.sample_name for s in before.state.parameter_specs), reported))
    meta = before.state.smoothed_model
    tiers = MemberTierRecord(tuple(meta['member_ids']), tuple(meta['member_components']),
                             exact_sampled=tuple(range(4)))
    args, after, _ = make_evaluator(tmp_path, tiers=tiers.to_dict())
    state = cs._build_state_from_inputs(args, fit_mode_override='joint',
        svi_init_physical_values=handoff, previous_stage_best_values=handoff)
    ev, reference = cs._prepare_direct_evaluator(args, state)
    result = ev.reported_physical_parameter_vector(reference)
    for i, spec in enumerate(state.parameter_specs):
        if spec.component_family == 'source_position':
            assert result[i] == pytest.approx(handoff[spec.sample_name], abs=1e-10)
    assert state.smoothed_model['source_prior_mean'] == [0., 0.]
    assert state.smoothed_model['source_prior_sigma'] == 100.

