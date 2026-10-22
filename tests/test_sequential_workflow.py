"""Workflow boundaries after removing the optional fourth fitting stage."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from lenscluster import cluster_solver as solver, plotting
from lenscluster.config import WorkflowConfig
from lenscluster.output_paths import STAGES, output_paths, prepare_output
from lenscluster.planning import compile_run_plan
from lenscluster.runner import _stage_fit_controls_from_plan
from test_latent_displacement import configured


@pytest.mark.parametrize('field,value', [
    ('stage3_forward_mode', 'critical-arc-anisotropic'),
    ('exact_image_diagnostics_stage3', True),
])
def test_removed_workflow_fields_are_rejected(field, value):
    with pytest.raises(TypeError, match=field):
        WorkflowConfig(**{field: value})


@pytest.mark.parametrize('stage2,smoothed', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('resume', [False, 'all', 'fast'])
@pytest.mark.parametrize('fresh', [False, True])
def test_sequential_final_stage_and_handoff(tmp_path, monkeypatch, stage2, resume, fresh, smoothed):
    from test_smoothed_integration import small_config, MODE
    from lenscluster import smoothed_workflow
    config = small_config(tmp_path) if smoothed else configured()
    frozen = {'frozen_test_identity': 'unchanged'}
    comparisons = []
    monkeypatch.setattr(smoothed_workflow, 'classify_from_artifacts', lambda *a: frozen)
    def compare(args, path, payload):
        comparisons.append((path, payload))
        return {'changed_members': 2, 'classification_unchanged': True}
    monkeypatch.setattr(smoothed_workflow, 'report_stage1_membership', compare)
    audits = []
    def audit(args, path):
        audits.append(path)
        return {'approximation_validation': 'passed'}
    monkeypatch.setattr(smoothed_workflow, 'audit_from_artifacts', audit)
    config = replace(config,
        paths=replace(config.paths, output_dir=str(tmp_path), run_name='run', layout='results-v1'),
        runtime=replace(config.runtime, resume=resume),
        workflow=replace(config.workflow, stage2_fresh_process=fresh,
                         stage2_forward_mode=(MODE if smoothed else 'isotropic-latent-displacement') if stage2 else 'none'))
    if not stage2:
        config = replace(config, schedule=replace(config.schedule,
            svi_steps=config.schedule.svi_steps[:2], refresh_every=config.schedule.refresh_every[:2],
            **{key: getattr(config.schedule, key)[:1] for key in
               ('fit_method', 'warmup', 'samples', 'sampling_refresh_runs', 'max_tree_depth')}))
    plan = compile_run_plan(config)
    prepare_output(plan)
    root = tmp_path / 'run'
    final = 2 if stage2 else 1
    assert [stage.name for stage in plan.stages] == [STAGES[f'stage{i}'] for i in range(final + 1)]
    controls = _stage_fit_controls_from_plan(plan, solver)
    assert set(controls) == {f'stage{i}' for i in range(final + 1)}
    calls = []
    values = {'halo_v_disp': 1200.0}
    sources = {'1': (.1, -.2)}
    shapes = [{2: (.1, .2)}]
    for number in range(final):
        artifact = root / f'stages/stage{number}/artifacts/plot_bundle.h5'
        artifact.parent.mkdir(parents=True)
        artifact.touch()

    def run(process, args, mode, name, **kwargs):
        calls.append((process, args, name, kwargs))
        return tmp_path / name

    monkeypatch.setattr(solver, '_run_single_stage', lambda *a, **kw: run(False, *a, **kw))
    monkeypatch.setattr(solver, '_run_single_stage_in_fresh_process', lambda *a, **kw: run(True, *a, **kw))
    monkeypatch.setattr(solver, '_stage_run_complete', lambda *a: False)
    monkeypatch.setattr(solver, '_stage_run_checkpointed', lambda *a: False)
    monkeypatch.setattr(solver, '_require_fast_resume_cosmology_compatibility', lambda *a: None)
    monkeypatch.setattr(solver, '_physical_best_fit_values_from_artifacts', lambda *a: values)
    monkeypatch.setattr(solver, '_source_position_prior_values_from_artifacts', lambda *a: sources)
    monkeypatch.setattr(solver, '_selected_independent_by_potfile_from_artifacts', lambda *a: [{2, 5}])
    monkeypatch.setattr(solver, '_member_shape_values_from_artifacts', lambda *a: shapes)
    monkeypatch.setattr(solver, '_audit_frozen_independent_selection', lambda *a: None)
    monkeypatch.setattr(solver, '_write_sequential_run_summary_txt', lambda *a: (None, ''))
    monkeypatch.setattr(solver, '_log', lambda *a: None)
    monkeypatch.setattr(solver, '_log_stage_banner', lambda *a: None)
    solver._run_sequential_v2(plan.runtime_args, controls)
    expected = [final] if resume == 'fast' else list(range(final + 1))
    assert [Path(call[2]).name for call in calls] == [f'stage{i}' for i in expected]
    assert calls[-1][0] == (stage2 and fresh and resume != 'fast')
    assert calls[-1][3]['svi_init_physical_values'] == values
    if stage2:
        assert calls[-1][3]['source_position_prior_values'] == sources
        assert calls[-1][3]['member_shape_values_by_potfile'] == (None if smoothed else shapes)
        assert calls[-1][3]['sample_likelihood_mode'] == (MODE if smoothed else 'isotropic-latent-displacement')
    if smoothed:
        assert all(call[1].sampling_engine == 'tiered_members' for call in calls)
        assert comparisons == [(root / 'stages/stage1/artifacts', frozen)]
        assert audits == [root / 'stages/stage1/artifacts', root / 'stages/stage2/artifacts']
        assert all(call[1].member_tiers_payload == frozen for call in calls if Path(call[2]).name != 'stage0')
    summary = json.loads((root / 'tables/sequential_summary.json').read_text())
    assert not any('stage3' in key for key in summary)
    assert summary[f'stage{final}_run_dir'] == str(root / f'stages/stage{final}')
    assert output_paths(root / f'stages/stage{final}').products == root
    assert not (root / 'stages/stage3').exists()


def test_archived_stage3_manifest_is_rejected_without_changes(tmp_path):
    config = configured()
    config = replace(config, paths=replace(config.paths, output_dir=str(tmp_path), run_name='run', layout='results-v1'))
    plan = compile_run_plan(config)
    prepare_output(plan)
    root = tmp_path / 'run'
    manifest = root / 'config.json'
    saved = json.loads(manifest.read_text())
    saved['config']['workflow']['stage3_forward_mode'] = 'critical-arc-anisotropic'
    saved['stages']['stage3_free_source_forward_fit'] = 'stages/stage3'
    saved['final_stage'] = 'stage3_free_source_forward_fit'
    manifest.write_text(json.dumps(saved))
    artifact = root / 'stages/stage3/artifacts/plot_bundle.h5'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'archived result')
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='incompatible.*run-label'):
        prepare_output(plan)
    assert manifest.read_bytes() == before
    assert artifact.read_bytes() == b'archived result'


def test_stage2_exact_diagnostics_and_cutouts():
    args = compile_run_plan(configured()).runtime_args
    args = solver._clone_args(args, exact_image_diagnostics_stage2=True,
                              image_catalog_family_cutout_image_dir='images', image_catalog_family_cutouts=True)
    stage1 = Path(STAGES['stage1'])
    stage2 = Path(STAGES['stage2'])
    assert not solver._stage_exact_image_diagnostics_enabled(args, stage1)
    assert solver._stage_exact_image_diagnostics_enabled(args, stage2)
    assert not solver._force_quick_diagnostics_for_nonfinal_stage(args, str(stage2), str(stage1)).quick_diagnostics
    assert plotting._image_catalog_family_cutout_enabled(args, stage2)
    assert not plotting._image_catalog_family_cutout_enabled(args, stage1)


@pytest.mark.parametrize('stage', [1, 2])
def test_anisotropic_likelihood_remains_available(stage):
    config = configured()
    config = replace(config, workflow=replace(config.workflow, **{
        'stage1_likelihood' if stage == 1 else 'stage2_forward_mode': 'critical-arc-anisotropic'}))
    plan = compile_run_plan(config)
    assert plan.stages[stage].sample_likelihood_mode == 'critical-arc-anisotropic-image-plane'


@pytest.mark.parametrize('saved_stage', [1, 2])
def test_as1063_comparison_reads_selected_bundle_without_summary(tmp_path, saved_stage):
    import h5py
    import numpy as np
    import runpy
    script = runpy.run_path('scripts/make_as1063_potfile_comparison.py')
    bundle = tmp_path / 'stages/stage2/artifacts/plot_bundle.h5'
    bundle.parent.mkdir(parents=True)
    with h5py.File(bundle, 'w') as handle:
        handle['runtime_args_json'] = json.dumps({'run_name': f'fit/stages/stage{saved_stage}'})
        handle['posterior/samples'] = np.zeros((2, 1))
        handle['posterior/diverging'] = [False, True]
        handle['state/build_state_meta_json'] = '{}'
        handle['init_diagnostics_json'] = json.dumps({'nuts_quality_metrics': {'rhat_max': 1.01, 'ess_min': 100}})
    if saved_stage != 2:
        with pytest.raises(ValueError, match='selected stage'):
            script['_load_posterior'](bundle)
    else:
        samples, _, divergences, summary = script['_load_posterior'](bundle)
        assert samples.shape == (2, 1)
        assert divergences == 1
        assert summary == {'rhat_max': 1.01, 'ess_min': 100, 'stage_semantic_name': STAGES['stage2']}
    assert script['BUNDLE'].parent.parent.name == 'stage2'
    assert script['OUT'] == script['BUNDLE'].parent.parent / 'figures'
