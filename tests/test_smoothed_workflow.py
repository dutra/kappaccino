"""Artifact workflow contracts without launching inference."""
import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster import smoothed_workflow as workflow
from lenscluster.member_tiers import member_farfield_geometry, MemberTierRecord

jax.config.update("jax_enable_x64", True)


class SyntheticEvaluator:
    def __init__(self):
        self.state = SimpleNamespace(parameter_specs=[SimpleNamespace(sample_name=name) for name in
            ("source_x", "source_y", "sigma_src", "s_sigma")], smoothed_model=dict(
                phase="initializer", tiers=None, member_ids=["population:g"], member_components=[0],
                sigma_src_index=2, source_scatter_prior_median_arcsec=.02,
                member_velocity_scatter_prior_median=.02))
        self.flat_critical_arc_data = SimpleNamespace(
            sigma_per_image=jnp.array([.1, .1]), image_has_constraint=jnp.array([True, True]),
            global_family_source_x_param_index=jnp.array([0]),
            global_family_source_y_param_index=jnp.array([1]),
            global_family_index_per_image=jnp.array([0, 0]))
        self.smoothed_member_components = np.array([0])
        self.smoothed_absorbed_positions = np.array([0])
        self.smoothed_geometry = member_farfield_geometry(np.array([[-1., 0.], [1., 0.]]), np.array([[1000., 1000.]]))
        self.smoothed_parity = jnp.array([0, 1])

    def _smoothed_fields(self, physical, force_exact=False):
        assert force_exact
        return dict(backprojected=jnp.array([[.1, -.1], [.2, .1]]),
            jacobian=jnp.broadcast_to(jnp.diag(jnp.array([.8, .7])), (2, 2, 2)),
            determinant_gradient=jnp.tile(jnp.array([.02, .03]), (2, 1)),
            jacobian_spatial_derivative=jnp.ones((2, 2, 2, 2)) * .01,
            member_alpha=jnp.array([[[.01, 0.], [.01, 0.]]]),
            member_jacobian=jnp.tile(jnp.eye(2)[None, None] * .001, (1, 2, 1, 1)),
            member_sigma=physical[3:4])

    def _smoothed_packed(self, physical):
        return SimpleNamespace(sigma0=jnp.ones((1, 2)) * .01, Ra=jnp.array([.2]), Rs=jnp.array([1.]))

    def _smoothed_packed_with_validity(self, physical):
        # Simulate a model-domain violation whose packed clipping remains finite.
        return self._smoothed_packed(physical), SimpleNamespace(is_valid=physical[3] < .1)


@pytest.mark.parametrize("top", [False, True])
def test_stage1_diagnostic_preserves_frozen_artifacts_and_model(tmp_path, monkeypatch, top):
    from lenscluster import cluster_solver as cs
    ev = SyntheticEvaluator()
    frozen = MemberTierRecord(("population:g",), (0,), exact_sampled=(0,)).to_dict()
    ev.state.smoothed_model.update(phase="fixed", tiers=frozen)
    center = np.array([.3, -.4, .02, .02])
    samples = np.tile(center, (16, 1))
    samples[:, 0] += np.linspace(-.01, .01, 16)
    monkeypatch.setattr(workflow, "_load", lambda path: (
        ev.state, SimpleNamespace(), dict(best_fit=center, samples=samples), ev))
    messages = []
    monkeypatch.setattr(cs, "_log", lambda args, message, **kw: messages.append(message))
    (tmp_path / "member_tiers.json").write_text(json.dumps(frozen))
    before = (tmp_path / "member_tiers.json").read_bytes()
    args = SimpleNamespace(member_tier_top_sampled=0, member_tier_top_exact=1) if top else SimpleNamespace()
    report = workflow.report_stage1_membership(args, tmp_path, frozen)
    assert report["changed_members"] == 1
    assert report["changes"] == [dict(member_id="population:g", component_index=0,
                                      frozen="exact_sampled", proposed="exact_absorbed")]
    assert report["classification_unchanged"]
    assert ev.state.smoothed_model["tiers"] == frozen
    assert (tmp_path / "member_tiers.json").read_bytes() == before
    assert not (tmp_path / "member_tier_budgets.json").exists()
    assert (tmp_path / report["budgets_file"]).exists()
    assert any("inference membership unchanged" in message for message in messages)
    assert any("exact_sampled -> exact_absorbed: 1 members" in message for message in messages)
    assert workflow.report_stage1_membership(args, tmp_path, frozen) == report


@pytest.mark.parametrize("swap", [False, True])
def test_stage1_comparison_reports_swaps_with_unchanged_totals(tmp_path, monkeypatch, swap):
    from lenscluster import cluster_solver as cs
    names = ("exact_sampled", "exact_absorbed", "approximate_sampled", "approximate_absorbed")
    frozen = MemberTierRecord(tuple("abcd"), (0, 1, 2, 3),
                              **{name: (i,) for i, name in enumerate(names)}).to_dict()
    proposed = MemberTierRecord(tuple("abcd"), (0, 1, 2, 3),
        **{name: ((i + 1) % 4 if swap else i,) for i, name in enumerate(names)}).to_dict()
    monkeypatch.setattr(workflow, "classify_from_artifacts", lambda *a, **kw: proposed)
    monkeypatch.setattr(cs, "_log", lambda *a, **kw: None)
    report = workflow.report_stage1_membership(SimpleNamespace(), tmp_path, frozen)
    assert report["changed_members"] == (4 if swap else 0)
    for count in report["counts"].values():
        assert count == dict(frozen=1, proposed=1, entering=int(swap), leaving=int(swap))
    proposed["member_ids"][0] = "wrong"
    with pytest.raises(ValueError, match="identity"):
        workflow.report_stage1_membership(SimpleNamespace(), tmp_path, frozen)


def test_reference_statistics_preserve_physical_beta_and_apply_covariance_guard():
    ev = SyntheticEvaluator()
    vector = jnp.array([.3, -.4, .2, .02])
    actual = jax.jit(lambda p: workflow._reference_snapshot(ev, p, prior_guards=True))(vector)
    a = np.diag([.8, .7])
    covariance = .02**2 * np.eye(2) + .1**2 * a @ a.T
    cross = -.1**2 * a @ np.array([.02, .03])
    residual = np.array([[.1, -.1], [.2, .1]]) - np.array([.3, -.4])
    expected_mean = np.linalg.det(a) + residual @ np.linalg.solve(covariance, cross)
    np.testing.assert_allclose(actual["determinant_mean"], expected_mean, rtol=1e-13)
    assert actual["sigma_src"] == .02


@pytest.mark.parametrize('limit', [0, 'all'])
def test_top_ranking_artifact_resume_and_invalid_reference(tmp_path, monkeypatch, limit):
    ev = SyntheticEvaluator()
    arrays = dict(best_fit=np.array([.3,-.4,.02,.02]), samples=np.empty((0,4)))
    monkeypatch.setattr(workflow, '_load', lambda path: (ev.state, SimpleNamespace(), arrays, ev))
    args = SimpleNamespace(member_tier_top_sampled=limit, member_tier_top_exact=limit)
    payload = workflow.classify_from_artifacts(args, tmp_path)
    record=MemberTierRecord.from_dict(payload)
    assert (record.exact_sampled if limit=='all' else record.approximate_absorbed) == (0,)
    budgets = json.loads((tmp_path/'member_tier_budgets.json').read_text())
    assert budgets['selection_settings'] == dict(top_sampled=limit, top_exact=limit)
    assert budgets['rankings']['sampled']['member_ids'] == ['population:g']
    assert workflow.classify_from_artifacts(args, tmp_path) == payload
    with pytest.raises(ValueError, match='selection settings'):
        workflow.classify_from_artifacts(SimpleNamespace(), tmp_path)
    arrays['best_fit'][3] = .2
    with pytest.raises(ValueError, match='valid model references'):
        workflow.classify_from_artifacts(args, tmp_path/'invalid')


def test_saved_reference_ensemble_and_audit_every_posterior_draw(tmp_path, monkeypatch):
    ev = SyntheticEvaluator()
    center = np.array([.3, -.4, .02, .02])
    samples = np.tile(center, (23, 1))
    samples[:, 0] += np.linspace(-.01, .01, len(samples))
    arrays = dict(best_fit=center, samples=samples)
    runtime = SimpleNamespace()
    monkeypatch.setattr(workflow, "_load", lambda path: (ev.state, runtime, arrays, ev))
    payload = workflow.classify_from_artifacts(runtime, tmp_path)
    record = MemberTierRecord.from_dict(payload)
    assert record.reference_count == 17
    assert record.exact_absorbed == (0,)
    assert json.loads((tmp_path / "member_tiers.json").read_text()) == payload
    budgets = json.loads((tmp_path / "member_tier_budgets.json").read_text())
    assert budgets["fixed_physical_source_positions"]
    assert len(budgets["reference_sample_indices"]) == 16
    ev.state.smoothed_model.update(phase="fixed", tiers=payload)
    samples[-1, 2] = 1e-7
    report = workflow.audit_from_artifacts(runtime, tmp_path)
    assert report["reference_count"] == 24
    assert report['best_fit_template_errors']['scope'] == 'saved_best_fit'
    assert report['best_fit_template_errors']['physical_parameters'] == center.tolist()
    assert report['best_fit_template_errors']['approximation_validation'] == 'passed'
    needed = report['budget_required_members']
    assert needed['reference_count'] == 17
    assert needed['reference_indices'][0] == 0
    assert needed['reference_indices'][-1] == 23
    assert needed['required_counts'] == dict(sampled=1, exact=1)
    assert needed['classification_unchanged']
    assert report["approximation_validation"] == "failed"
    assert report["failed_reference_indices"] == [23]
    saved = np.load(tmp_path / report["exact_image_diagnostics_file"])
    assert saved["magnification"].shape == (24, 2)
    np.testing.assert_allclose(saved["magnification"], 1 / (.8 * .7))
    np.testing.assert_allclose(saved["source_by_family"][1:, 0], samples[:, :2])
    assert np.all(saved["absorbed_lnmu_variance"] > 0)
    assert not np.any(saved["critical_or_nonfinite_determinant"])
    assert not np.any(saved['determinant_degenerate'])
    assert np.all(saved['likelihood_valid'])
    assert saved['determinant_conditional_variance'].shape == (24, 2)
    np.testing.assert_allclose(saved["absorbed_magnification_variance"],
                               saved["absorbed_lnmu_variance"] / saved["determinant"]**2)
    assert json.loads((tmp_path / "member_tier_audit.json").read_text())["classification_unchanged"]


@pytest.mark.parametrize('error', [0., .001, .002, .003, float('nan')])
def test_best_fit_template_error_thresholds_and_red_rows(tmp_path, monkeypatch, error):
    from lenscluster import cluster_solver as cs
    from lenscluster.member_tiers import MemberTierThresholds
    from test_member_tiers import references
    data = references([.1], [error])
    snapshot = {key: value[0] for key, value in data.items()}
    snapshot['reference_valid'] = True
    record = MemberTierRecord(('g',), (0,), approximate_sampled=(0,))
    result = workflow._best_fit_template_errors(record, snapshot, MemberTierThresholds(), np.array([0]))
    failed = not np.isfinite(error) or error >= .002
    assert result['metrics'][0]['failed'] == failed
    assert result['metrics'][1]['failed'] == failed
    assert result['classification_unchanged']
    result['constrained_image_indices'] = [7]
    tables = []
    monkeypatch.setattr(cs, '_log', lambda *a, **kw: tables.append(kw['renderable']) if 'renderable' in kw else None)
    workflow._report_best_fit_template_errors(SimpleNamespace(), tmp_path/'stage2'/'artifacts', result)
    assert (tables[0].rows[0].style == 'bold red') == failed
    json.dumps(result, allow_nan=False)


def test_best_fit_error_report_keeps_conservative_budget_despite_cancellation():
    from lenscluster.member_tiers import MemberTierThresholds
    from test_member_tiers import references
    data = references([0., 0.], [.0015, -.0015])
    data['template_jacobian'][0, 0, 0] = np.eye(2)*.5
    snapshot = {key: value[0] for key, value in data.items()}
    snapshot['reference_valid'] = True
    record = MemberTierRecord(('a','b'), (0,1), approximate_sampled=(0,1))
    result = workflow._best_fit_template_errors(record, snapshot, MemberTierThresholds(), np.array([0]))
    assert not result['metrics'][0]['failed']
    assert result['metrics'][0]['maximum'] == 0.
    assert result['metrics'][1]['failed']
    assert result['metrics'][2]['failed']
    assert result['metrics'][3]['limit'] is None  # No invented Jacobian tolerance.


@pytest.mark.parametrize("n_draws", [0, 3, 16])
def test_no_uncertainty_draws_use_all_sampled_exact_fallback(tmp_path, monkeypatch, n_draws):
    ev = SyntheticEvaluator()
    center = np.array([.3, -.4, .02, .02])
    # Sixteen copies of a single draw do not establish a reference ensemble.
    arrays = dict(best_fit=center, samples=np.tile(center, (n_draws, 1)))
    monkeypatch.setattr(workflow, "_load", lambda path: (ev.state, SimpleNamespace(), arrays, ev))
    record = MemberTierRecord.from_dict(workflow.classify_from_artifacts(SimpleNamespace(), tmp_path))
    assert record.exact_sampled == (0,)
    assert record.selection_reason == "all_exact_sampled_fallback"


def test_finite_but_invalid_packed_models_cannot_pass_reference_or_posterior_audit(tmp_path, monkeypatch):
    ev = SyntheticEvaluator()
    center = np.array([.3, -.4, .02, .02])
    samples = np.tile(center, (17, 1))
    samples[:, 0] += np.linspace(-.01, .01, len(samples))
    samples[1, 3] = .2  # invalid despite the fake packed values being entirely finite
    arrays = dict(best_fit=center, samples=samples)
    monkeypatch.setattr(workflow, "_load", lambda path: (ev.state, SimpleNamespace(), arrays, ev))
    payload = workflow.classify_from_artifacts(SimpleNamespace(), tmp_path)
    assert MemberTierRecord.from_dict(payload).exact_sampled == (0,)
    budgets = json.loads((tmp_path / "member_tier_budgets.json").read_text())
    assert budgets["invalid_model_reference_indices"] == [2]
    assert budgets["approximation_validation"] == "failed"
    ev.state.smoothed_model.update(phase="fixed", tiers=payload)
    ev.smoothed_absorbed_positions = np.array([], dtype=int)
    report = workflow.audit_from_artifacts(SimpleNamespace(), tmp_path)
    assert report["approximation_validation"] == "failed"
    assert report["failed_reference_indices"] == [2]
    saved = np.load(tmp_path / report["exact_image_diagnostics_file"])
    assert np.flatnonzero(~saved["model_valid"]).tolist() == [2]


def test_resume_preserves_classes_without_recomputing_and_rejects_changed_references(tmp_path, monkeypatch):
    ev = SyntheticEvaluator()
    center = np.array([.3, -.4, .02, .02])
    samples = np.tile(center, (16, 1))
    samples[:, 0] += np.linspace(-.01, .01, len(samples))
    arrays = dict(best_fit=center, samples=samples)
    monkeypatch.setattr(workflow, "_load", lambda path: (ev.state, SimpleNamespace(), arrays, ev))
    payload = workflow.classify_from_artifacts(SimpleNamespace(), tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Resume must reuse its classification, not recalculate fields.")
    monkeypatch.setattr(workflow, "_reference_snapshot", forbidden)
    assert workflow.classify_from_artifacts(SimpleNamespace(), tmp_path) == payload
    samples[1, 0] += .001
    with pytest.raises(ValueError, match="ensemble"):
        workflow.classify_from_artifacts(SimpleNamespace(), tmp_path)


def test_resume_rejects_incomplete_or_incompatible_evidence(tmp_path, monkeypatch):
    ev = SyntheticEvaluator()
    arrays = dict(best_fit=np.array([.3, -.4, .02, .02]), samples=np.empty((0, 4)))
    monkeypatch.setattr(workflow, "_load", lambda path: (ev.state, SimpleNamespace(), arrays, ev))
    workflow.classify_from_artifacts(SimpleNamespace(), tmp_path)
    budgets_path = tmp_path / "member_tier_budgets.json"
    budgets = json.loads(budgets_path.read_text())
    budgets["workflow_revision"] = 999
    budgets_path.write_text(json.dumps(budgets))
    with pytest.raises(ValueError, match="revision"):
        workflow.classify_from_artifacts(SimpleNamespace(), tmp_path)
    budgets_path.unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        workflow.classify_from_artifacts(SimpleNamespace(), tmp_path)
