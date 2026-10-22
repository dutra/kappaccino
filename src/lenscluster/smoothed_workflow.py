"""Saved-artifact member classification and offline approximation audits.

Sources in the artifact arrays are physical beta. Reference determinant moments
are evaluated directly at those beta values, so removing absorbed covariance
never changes the source through its conditional whitening transformation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

from .member_tiers import (MemberTierRecord, MemberTierThresholds, evaluate_farfield,
                           classify_member_tiers, audit_member_tiers, required_top_counts)
from . import smoothed_likelihood as sl


REFERENCE_DRAWS = 16
AUDIT_BATCH_SIZE = 8
WORKFLOW_REVISION = 1
_REFERENCE_KEYS = ("exact_alpha", "exact_jacobian", "template_alpha", "template_jacobian",
                   "total_jacobian", "determinant_mean", "determinant_variance", "sigma_src", "s_sigma")


def _thresholds(args):
    return MemberTierThresholds(
        template_deflection=float(getattr(args, "member_tier_kernel_deflection_fraction", .1)),
        template_logh=float(getattr(args, "member_tier_kernel_log_h_tolerance", .1)),
        sampled_deflection=float(getattr(args, "member_tier_scatter_deflection_fraction", np.sqrt(.1))),
        sampled_logh=float(getattr(args, "member_tier_scatter_log_h_tolerance", np.sqrt(.1))),
        absorbed_variance=float(getattr(args, "member_tier_absorbed_variance_fraction", .25)))


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _load(artifacts_dir):
    from . import cluster_solver as cs
    from .planning import SolverRuntime
    state, saved_runtime, arrays, diagnostics = cs._load_artifacts(Path(artifacts_dir))
    if not isinstance(getattr(state, "smoothed_model", None), dict):
        raise ValueError("Member-tier workflow requires smoothed likelihood artifacts.")
    arrays, _ = cs._maybe_convert_loaded_posterior_arrays_to_physical(arrays, state.parameter_specs, diagnostics)
    runtime = SolverRuntime(saved_runtime)
    evaluator = cs._build_cluster_evaluator_from_args(runtime, state, sampling_engine="tiered_members", quick_diagnostics=True)
    return state, runtime, arrays, evaluator


def _posterior_samples(arrays, n_parameters):
    samples = np.asarray(arrays.get("samples", np.empty((0, n_parameters))), dtype=float)
    if samples.ndim != 2 or samples.shape[1] != n_parameters:
        raise ValueError("Saved physical posterior samples have incompatible parameter dimensions.")
    return samples


def _reference_snapshot(evaluator, physical, *, prior_guards):
    """JAX-compatible one-model calculation at fixed reported physical sources."""
    data, meta = evaluator.flat_critical_arc_data, evaluator.state.smoothed_model
    fields = evaluator._smoothed_fields(physical, force_exact=True)
    packed, packed_validity = evaluator._smoothed_packed_with_validity(physical)
    members = evaluator.smoothed_member_components
    template = evaluate_farfield(evaluator.smoothed_geometry, packed.sigma0[members], packed.Ra[members], packed.Rs[members])
    source = jnp.stack((physical[data.global_family_source_x_param_index],
                        physical[data.global_family_source_y_param_index]), axis=-1)
    source_images = source[data.global_family_index_per_image]
    sigma_src = physical[meta["sigma_src_index"]]
    if prior_guards:
        sigma_src = jnp.minimum(sigma_src, meta["source_scatter_prior_median_arcsec"])
    alpha, u, scatter = fields["member_alpha"], fields["member_jacobian"], fields["member_sigma"]
    def statistics(positions, source_scatter):
        return sl.statistics(fields["backprojected"], source_images, fields["jacobian"],
            fields["determinant_gradient"], data.sigma_per_image, source_scatter,
            alpha[positions], u[positions], scatter[positions], data.image_has_constraint,
            parity=evaluator.smoothed_parity, jacobian_spatial_derivative=fields["jacobian_spatial_derivative"])
    no_absorbed = statistics(np.asarray([], dtype=int), sigma_src)
    # Geometry may contain unconstrained singleton images. Only likelihood rows
    # enter the classification and its budgets.
    active = np.flatnonzero(np.asarray(data.image_has_constraint))
    result = dict(exact_alpha=alpha[:, active], exact_jacobian=u[:, active],
                  template_alpha=template.alpha[:, active], template_jacobian=template.jacobian[:, active],
                  total_jacobian=fields["jacobian"][active],
                  determinant_mean=no_absorbed["determinant_mean"][active],
                  determinant_variance=no_absorbed["determinant_variance"][active],
                  sigma_src=sigma_src, s_sigma=scatter,
                  reference_valid=packed_validity.is_valid & jnp.all(no_absorbed["valid"][active]))
    if not prior_guards:
        actual = statistics(evaluator.smoothed_absorbed_positions, sigma_src)
        result['reference_valid'] = result['reference_valid'] & jnp.all(actual['valid'][active])
        determinant = actual["determinant"][active]
        vd = actual["member_determinant_variance"][active]
        critical = (determinant == 0.) | ~jnp.isfinite(determinant)
        result["science"] = dict(determinant=determinant, magnification=1. / determinant,
            determinant_mean=actual['determinant_mean'][active],
            determinant_conditional_variance=actual['determinant_variance'][active],
            determinant_degenerate=actual['degenerate'][active], likelihood_valid=actual['valid'][active],
            absorbed_determinant_variance=vd, absorbed_lnmu_variance=vd / determinant**2,
            absorbed_magnification_variance=vd / determinant**4,
            critical_or_nonfinite_determinant=critical,
            nonfinite_magnification=~jnp.isfinite(1. / determinant),
            model_valid=packed_validity.is_valid,
            epsilon=actual["epsilon"][active], source_by_family=source,
            loglike=actual["loglike"][active])
    return result


def _stack_snapshots(snapshots):
    return {key: np.stack([snapshot[key] for snapshot in snapshots]) for key in _REFERENCE_KEYS}


def _active_parity(evaluator):
    active = np.asarray(evaluator.flat_critical_arc_data.image_has_constraint, dtype=bool)
    return np.asarray(evaluator.smoothed_parity)[active]


def classify_from_artifacts(args, stage0_artifacts_dir, *, diagnostic_only=False):
    """Persist classes/budgets, using separate files for diagnostic proposals."""
    path = Path(stage0_artifacts_dir)
    state, _runtime, arrays, evaluator = _load(path)
    meta = state.smoothed_model
    center = np.asarray(arrays["best_fit"], dtype=float)
    selection_settings = {name: getattr(args, f"member_tier_{name}", None) for name in ("top_sampled", "top_exact")}
    if center.shape != (len(state.parameter_specs),) or not np.isfinite(center).all():
        raise ValueError("Cannot classify from an invalid saved physical parameter vector.")
    samples = _posterior_samples(arrays, len(center))
    finite_indices = np.flatnonzero(np.isfinite(samples).all(axis=1))
    # Preserve the order of whole joint draws while excluding duplicated rows.
    unique_positions = (np.sort(np.unique(samples[finite_indices], axis=0, return_index=True)[1])
                        if len(finite_indices) else np.array([], dtype=int))
    usable_indices = finite_indices[unique_positions]
    uncertainty_available = len(usable_indices) >= REFERENCE_DRAWS
    indices = (usable_indices[np.linspace(0, len(usable_indices) - 1, REFERENCE_DRAWS).astype(int)]
               if uncertainty_available else np.array([], dtype=int))
    references = np.concatenate((center[None], samples[indices]), axis=0) if uncertainty_available else center[None]
    prefix = "member_tier_proposal" if diagnostic_only else "member_tiers"
    record_path = path / f"{prefix}.json"
    budgets_path = path / ("member_tier_proposal_budgets.json" if diagnostic_only else "member_tier_budgets.json")
    if record_path.exists() != budgets_path.exists():
        raise ValueError("Incomplete saved member classification: both tier membership and budget evidence are required.")
    if record_path.exists():
        payload = json.loads(record_path.read_text())
        saved_record = MemberTierRecord.from_dict(payload)
        saved_budgets = json.loads(budgets_path.read_text())
        saved_settings = saved_budgets.get("selection_settings", {"top_sampled": None, "top_exact": None})
        if saved_settings != selection_settings:
            raise ValueError("Saved member classification selection settings differ; use a new run label.")
        if any(value is not None for value in selection_settings.values()) and saved_budgets.get("ranking_revision") != 1:
            raise ValueError("Saved member ranking revision is missing or incompatible.")
        saved_record.validate_identity(meta["member_ids"], meta["member_components"])
        names = tuple(spec.sample_name for spec in state.parameter_specs)
        if saved_record.reference_parameter_names != names:
            raise ValueError("Saved member classification reference parameter ordering differs.")
        if saved_budgets.get("workflow_revision") != WORKFLOW_REVISION:
            raise ValueError("Saved member classification workflow revision is missing or incompatible.")
        if (saved_record.reference_count != len(references)
                or saved_budgets.get("reference_count") != saved_record.reference_count
                or np.asarray(saved_record.reference_parameters).shape != references.shape
                or not np.array_equal(np.asarray(saved_record.reference_parameters), references)):
            raise ValueError("Saved member classification reference ensemble differs from the stage artifacts.")
        if saved_budgets.get("thresholds") != vars(_thresholds(args)):
            raise ValueError("Saved member classification thresholds differ from the requested settings.")
        if saved_budgets.get("scope") != "evaluated_reference_models":
            raise ValueError("Saved member classification has missing or incompatible validation scope.")
        # Reuse the matching saved selection, including diagnostic proposals.
        _report_selection(args, saved_record, saved_budgets, diagnostic_only)
        return payload
    calculate = jax.jit(lambda vector: _reference_snapshot(evaluator, vector, prior_guards=True))
    snapshots = [jax.device_get(calculate(jnp.asarray(vector))) for vector in references]
    invalid_references = [index for index, snapshot in enumerate(snapshots) if not bool(snapshot["reference_valid"])]
    if invalid_references and any(value is not None for value in selection_settings.values()):
        raise ValueError(f"Explicit top-k member ranking requires valid model references; invalid indices: {invalid_references}.")
    record, diagnostics = classify_member_tiers(**_stack_snapshots(snapshots),
        member_ids=meta["member_ids"], component_indices=meta["member_components"],
        parity=_active_parity(evaluator), thresholds=_thresholds(args),
        source_prior_median=meta["source_scatter_prior_median_arcsec"],
        member_prior_median=meta["member_velocity_scatter_prior_median"],
        reference_uncertainty_available=uncertainty_available and not invalid_references,
        reference_parameter_names=tuple(spec.sample_name for spec in state.parameter_specs),
        reference_parameters=references, **selection_settings)
    diagnostics.update(workflow_revision=WORKFLOW_REVISION, diagnostic_only=diagnostic_only,
        selection_settings=selection_settings,
        reference_sample_indices=indices.tolist() if uncertainty_available else [],
        reference_parameter_space="reported_physical", first_reference="saved_best_fit",
        uncertainty_available=bool(uncertainty_available),
        unique_finite_draws_available=int(len(usable_indices)),
        nonfinite_saved_draw_indices=np.flatnonzero(~np.isfinite(samples).all(axis=1)).tolist(),
        invalid_model_reference_indices=invalid_references,
        fixed_physical_source_positions=True,
        source_guard_applied_to_determinant_covariance=True)
    if invalid_references:
        diagnostics["approximation_validation"] = "failed"
    payload = record.to_dict()
    _write_json(record_path, payload)
    _write_json(budgets_path, diagnostics)
    _report_selection(args, record, diagnostics, diagnostic_only)
    return payload


def _report_selection(args, record, budgets, diagnostic_only):
    from . import cluster_solver as cs
    settings = budgets.get("selection_settings", {"top_sampled": None, "top_exact": None})
    label = "diagnostic proposal" if diagnostic_only else "frozen selection"
    cs._log(args, f"[member tiers: {label}] top_sampled={settings['top_sampled']} "
            f"top_exact={settings['top_exact']}; effective sampled={len(record.sampled)} "
            f"exact={len(record.exact)} total={len(record.member_ids)}; policy={record.selection_reason}; "
            f"references={record.reference_count} uncertainty_available={budgets.get('uncertainty_available', False)}")
    if budgets.get("approximation_validation") != "passed":
        cs._log(args, f"[member tiers: {label}] WARNING: ERROR BUDGETS FAILED; membership remains fixed. "
                + ", ".join(f"{key}={value}" for key, value in budgets.items() if key.endswith("_max_ratio")))


def report_stage1_membership(args, artifacts_dir, frozen_payload):
    """Compare a diagnostic selection with the frozen inference classes."""
    from . import cluster_solver as cs
    frozen = MemberTierRecord.from_dict(frozen_payload)
    proposal = MemberTierRecord.from_dict(classify_from_artifacts(
        args, artifacts_dir, diagnostic_only=True))
    proposal.validate_identity(frozen.member_ids, frozen.component_indices)
    names = ("exact_sampled", "exact_absorbed", "approximate_sampled", "approximate_absorbed")
    old = {i: name for name in names for i in getattr(frozen, name)}
    new = {i: name for name in names for i in getattr(proposal, name)}
    changes = [dict(member_id=identity, component_index=frozen.component_indices[i],
                    frozen=old[i], proposed=new[i])
               for i, identity in enumerate(frozen.member_ids) if old[i] != new[i]]
    counts = {name: dict(frozen=len(getattr(frozen, name)), proposed=len(getattr(proposal, name)),
                        entering=sum(row["proposed"] == name for row in changes),
                        leaving=sum(row["frozen"] == name for row in changes)) for name in names}
    report = dict(diagnostic_only=True, classification_unchanged=True,
                  total_members=len(frozen.member_ids), changed_members=len(changes),
                  reference_count=proposal.reference_count, selection_reason=proposal.selection_reason,
                  counts=counts, changes=changes, frozen_classes=frozen.to_dict(),
                  proposal_file="member_tier_proposal.json",
                  budgets_file="member_tier_proposal_budgets.json")
    path = Path(artifacts_dir) / "member_tier_comparison.json"
    _write_json(path, report)
    title = (f"Stage 1 member diagnostic: {len(changes)}/{len(frozen.member_ids)} would change; "
             "inference membership unchanged")
    lines = [title, "Class | Frozen | Proposed | Entering | Leaving"]
    table = cs._RichTable(title=title, header_style="bold cyan", border_style="cyan") if cs._RichTable is not None else None
    if table is not None:
        for column in ("Class", "Frozen", "Proposed", "Entering", "Leaving"):
            table.add_column(column)
    for name, count in counts.items():
        row = (name.replace("_", "/"), *(str(count[key]) for key in ("frozen", "proposed", "entering", "leaving")))
        lines.append(" | ".join(row))
        if table is not None:
            table.add_row(*row)
    cs._log(args, "\n".join(lines), renderable=table)
    # Group transitions so swaps remain visible even when class totals agree.
    for before in names:
        for after in names:
            identities = [row["member_id"] for row in changes if row["frozen"] == before and row["proposed"] == after]
            if identities:
                cs._log(args, f"[member tiers: diagnostic] {before} -> {after}: {len(identities)} members")
    cs._log(args, f"[member tiers: diagnostic] references={proposal.reference_count} "
            f"selection={proposal.selection_reason}; identities and budgets: {path}")
    return report


def audit_from_artifacts(args, stage_artifacts_dir):
    """Audit the posterior median and every saved draw, without changing tiers.

    A failed audit is saved as a failed validation result, never an automatic
    reclassification or inference restart. Exact image-position fields and the
    first-order absorbed magnification-variance contribution are saved separately.
    """
    path = Path(stage_artifacts_dir)
    state, runtime, arrays, evaluator = _load(path)
    meta = state.smoothed_model
    if meta.get("phase") != "fixed" or meta.get("tiers") is None:
        raise ValueError("Posterior approximation audits require fixed saved member classes.")
    record = MemberTierRecord.from_dict(meta["tiers"])
    record.validate_identity(meta["member_ids"], meta["member_components"])
    samples = _posterior_samples(arrays, len(state.parameter_specs))
    median = np.median(samples, axis=0) if len(samples) else np.asarray(arrays["best_fit"], dtype=float)
    # Iterate fixed-size batches to avoid retaining all per-member fields.
    references = np.concatenate((median[None], samples), axis=0)
    calculate = jax.jit(lambda vector: _reference_snapshot(evaluator, vector, prior_guards=False))
    report = dict(approximation_validation="passed", scope="posterior_median_and_all_saved_draws",
                  reference_count=len(references), saved_draw_count=len(samples),
                  median_available=bool(len(samples)), failed_reference_indices=[],
                  reference_parameter_space="reported_physical", fixed_physical_source_positions=True,
                  classification_unchanged=True, thresholds=vars(_thresholds(runtime)))
    science = []
    count_indices = {0, *(1 + np.linspace(0, len(samples)-1, min(REFERENCE_DRAWS, len(samples))).astype(int))} if len(samples) else {0}
    count_snapshots = []
    for start in range(0, len(references), AUDIT_BATCH_SIZE):
        snapshots = [jax.device_get(calculate(jnp.asarray(vector))) for vector in references[start:start + AUDIT_BATCH_SIZE]]
        count_snapshots.extend(snapshot for offset, snapshot in enumerate(snapshots) if start + offset in count_indices)
        audit = audit_member_tiers(record, **_stack_snapshots(snapshots), parity=_active_parity(evaluator), thresholds=_thresholds(runtime))
        for key, value in audit.items():
            if key.endswith("_max_ratio"):
                previous = report.get(key, 0.)
                report[key] = None if previous is None or value is None else max(previous, value)
        invalid_references = [offset for offset, snapshot in enumerate(snapshots) if not bool(snapshot["reference_valid"])]
        if audit["approximation_validation"] != "passed" or invalid_references:
            report["approximation_validation"] = "failed"
            for offset, snapshot in enumerate(snapshots):
                individual = audit_member_tiers(record, **_stack_snapshots([snapshot]), parity=_active_parity(evaluator), thresholds=_thresholds(runtime))
                if individual["approximation_validation"] != "passed" or offset in invalid_references:
                    report["failed_reference_indices"].append(start + offset)
        science.extend(snapshot["science"] for snapshot in snapshots)
    report["nonfinite_saved_parameter_rows"] = np.flatnonzero(~np.isfinite(references).all(axis=1)).tolist()
    if report["nonfinite_saved_parameter_rows"]:
        report["approximation_validation"] = "failed"
    diagnostics_path = path / "smoothed_exact_image_diagnostics.npz"
    np.savez_compressed(diagnostics_path,
        **{key: np.stack([entry[key] for entry in science]) for key in science[0]},
        constrained_image_indices=np.flatnonzero(np.asarray(evaluator.flat_critical_arc_data.image_has_constraint)),
        parity=_active_parity(evaluator), reference_index=np.arange(len(references)))
    report["exact_image_diagnostics_file"] = diagnostics_path.name
    report["doubly_degenerate_reference_images"] = np.argwhere(
        np.stack([entry['determinant_degenerate'] for entry in science])).tolist()
    report["magnification_variance_convention"] = "First-order absorbed Var(ln mu)=V_D/D^2 and Var(mu)=V_D/D^4; divergent at D=0; no determinant floor."
    report["reference_index_convention"] = "0 is posterior median (best fit if no samples); j+1 is saved posterior row j."
    if all(bool(snapshot['reference_valid']) for snapshot in count_snapshots):
        needed = required_top_counts(record, **_stack_snapshots(count_snapshots),
                                     parity=_active_parity(evaluator), thresholds=_thresholds(runtime))
    else:
        needed = dict(status="unavailable", reason="invalid_model_references", classification_unchanged=True)
    needed.update(scope="posterior_median_and_up_to_16_deterministic_saved_draws",
                  reference_indices=sorted(int(i) for i in count_indices), reference_count=len(count_snapshots),
                  center="posterior_median" if len(samples) else "best_fit")
    report['budget_required_members'] = needed
    _report_required_counts(args, path, record, needed)
    best_snapshot = jax.device_get(calculate(jnp.asarray(arrays['best_fit'])))
    best_report = _best_fit_template_errors(record, best_snapshot, _thresholds(runtime), _active_parity(evaluator))
    best_report['constrained_image_indices'] = np.flatnonzero(np.asarray(evaluator.flat_critical_arc_data.image_has_constraint)).tolist()
    best_report['physical_parameters'] = [float(v) if np.isfinite(v) else None for v in np.asarray(arrays['best_fit'], dtype=float)]
    report['best_fit_template_errors'] = best_report
    _report_best_fit_template_errors(args, path, best_report)
    _write_json(path / "member_tier_audit.json", report)
    return report


def _best_fit_template_errors(record, snapshot, thresholds, parity):
    """Compare only members actually using templates, at the saved best fit."""
    audit = audit_member_tiers(record, **_stack_snapshots([snapshot]), parity=parity, thresholds=thresholds)
    positions = np.asarray(record.approximate, dtype=int)
    delta = np.asarray(snapshot['template_alpha'])[positions] - np.asarray(snapshot['exact_alpha'])[positions]
    delta_u = np.asarray(snapshot['template_jacobian'])[positions] - np.asarray(snapshot['exact_jacobian'])[positions]
    net = np.linalg.norm(delta.sum(axis=0), axis=-1)
    summed = np.linalg.norm(delta, axis=-1).sum(axis=0)
    jacobian = np.linalg.norm(delta_u.sum(axis=0), axis=(-2, -1))
    logh_ratios = audit['template_logh_ratios']
    logh = (np.asarray(logh_ratios)[0] * thresholds.template_logh if logh_ratios is not None
            else np.full(net.shape, np.nan))
    def clean(values):
        return [float(v) if np.isfinite(v) else None for v in values]
    metrics = []
    for label, values, limit in (
        ('Net deflection error [arcsec]', net, thresholds.template_deflection * float(snapshot['sigma_src'])),
        ('Summed member deflection errors [arcsec]', summed, thresholds.template_deflection * float(snapshot['sigma_src'])),
        ('Finite delta log h budget', logh, thresholds.template_logh),
        ('Net Jacobian error [Frobenius norm]', jacobian, None),
    ):
        invalid = ~np.isfinite(values)
        worst = int(np.flatnonzero(invalid)[0]) if invalid.any() else (int(np.argmax(values)) if values.size else None)
        value = float(values[worst]) if worst is not None and not invalid.any() else (0. if worst is None else None)
        failed = bool(invalid.any() or (limit is not None and (not np.isfinite(limit) or limit <= 0 or np.any(values >= limit))))
        metrics.append(dict(metric=label, maximum=value, limit=limit if limit is None or np.isfinite(limit) else None, failed=failed,
                            worst_constrained_image=worst, values=clean(values)))
    return dict(scope='saved_best_fit', prior_guards=False, classification_unchanged=True,
                approximate_member_ids=[record.member_ids[i] for i in positions],
                reference_valid=bool(snapshot['reference_valid']), metrics=metrics,
                approximation_validation='failed' if not snapshot['reference_valid'] or any(m['failed'] for m in metrics) else 'passed')


def _report_best_fit_template_errors(args, path, report):
    from . import cluster_solver as cs
    title = f"{path.parent.name}: best-fit far-field approximation versus exact dPIE"
    table = cs._RichTable(title=title, header_style='bold cyan', border_style='cyan')
    for name in ('Error metric', 'Maximum', 'Limit (<)', 'Image index', 'Status'):
        table.add_column(name)
    lines = [title]
    for metric in report['metrics']:
        index = metric['worst_constrained_image']
        image_index = str(report['constrained_image_indices'][index]) if index is not None else 'none'
        value = 'nonfinite' if metric['maximum'] is None else f"{metric['maximum']:.6g}"
        limit = 'not defined' if metric['limit'] is None else f"{metric['limit']:.6g}"
        status = 'FAIL' if metric['failed'] else ('INFO' if metric['limit'] is None else 'PASS')
        row = (metric['metric'], value, limit, image_index, status)
        table.add_row(*row, style='bold red' if metric['failed'] else None)
        lines.append(' | '.join(row))
    if not report['reference_valid']:
        table.add_row('Best-fit model validity', 'invalid', '', '', 'FAIL', style='bold red')
        lines.append('FAIL: invalid best-fit reference model')
    cs._log(args, '\n'.join(lines), renderable=table)
    cs._log(args, '[member errors] Actual best-fit scatter; frozen approximate members only; '
            'zero-based flattened image indices. Per-image errors saved in member_tier_audit.json.')


def _report_required_counts(args, path, record, needed):
    from . import cluster_solver as cs
    title = f"{path.parent.name}: members needed for error budgets (diagnostic only)"
    if needed['status'] != 'passed':
        cs._log(args, f"[{title}] unavailable: {needed['reason']}; membership unchanged")
        return
    table = cs._RichTable(title=title, header_style="bold cyan", border_style="cyan")
    for column in ('Treatment', 'Frozen', 'Required top-k', 'Difference'):
        table.add_column(column)
    lines = [title, 'Treatment | Frozen | Required top-k | Difference']
    for axis in ('sampled', 'exact'):
        current = len(getattr(record, axis))
        required = needed['required_counts'][axis]
        row = (axis, str(current), str(required), f'{required-current:+d}')
        table.add_row(*row)
        lines.append(' | '.join(row))
    cs._log(args, '\n'.join(lines), renderable=table)
    cs._log(args, f"[member budgets] scope={needed['scope']} references={needed['reference_count']}; "
            "actual posterior scatters; smallest passing ranked prefixes, not a global minimum; membership unchanged")
