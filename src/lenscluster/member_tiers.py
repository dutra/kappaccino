"""Fixed member classes and parameter-exact circular far-field geometry.

The approximation truncates the spatial series, never the sampled amplitudes or
radii. All Jacobians in this module are U = d alpha / d theta, not -U.
Classification certifies a finite saved reference ensemble, not a prior volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


TIER_SCHEMA_VERSION = 1
_TIER_NAMES = ("exact_sampled", "exact_absorbed", "approximate_sampled", "approximate_absorbed")


class FarFieldGeometry(NamedTuple):
    """Member, image, series-term, spatial-component arrays (arcsecond units)."""
    deflection: object
    jacobian: object
    jacobian_derivative: object
    valid_separation: object


class MemberFields(NamedTuple):
    alpha: object
    jacobian: object
    jacobian_derivative: object


def member_farfield_geometry(image_xy, member_xy) -> FarFieldGeometry:
    """Precompute rhat/R, rhat/R², rhat/R⁴ and their two spatial derivatives.

    Coincident member/image positions are flagged. Their template values are
    NaN, forcing those members into the exact class during classification.
    """
    images, members = np.asarray(image_xy, dtype=float), np.asarray(member_xy, dtype=float)
    if images.ndim != 2 or images.shape[1] != 2 or members.ndim != 2 or members.shape[1] != 2:
        raise ValueError("Image and member positions must have shapes (N, 2) and (M, 2).")
    if not np.isfinite(images).all() or not np.isfinite(members).all():
        raise ValueError("Geometry positions must be finite.")
    delta = images[None, :, :] - members[:, None, :]
    radius = np.linalg.norm(delta, axis=-1)
    valid = radius > 0
    safe_radius = np.where(valid, radius, 1.)
    direction = delta / safe_radius[..., None]
    eye = np.eye(2)
    outer = np.einsum("...i,...j->...ij", direction, direction)
    triple = np.einsum("...i,...j,...k->...ijk", direction, direction, direction)
    mixed = (np.einsum("ij,...k->...ijk", eye, direction)
             + np.einsum("ik,...j->...ijk", eye, direction)
             + np.einsum("jk,...i->...ijk", eye, direction))
    fields, jacobians, derivatives = [], [], []
    for n in (1, 2, 4):
        fields.append(direction / safe_radius[..., None] ** n)
        jacobians.append((eye - (n + 1) * outer) / safe_radius[..., None, None] ** (n + 1))
        derivatives.append((n + 1) * ((n + 3) * triple - mixed)
                           / safe_radius[..., None, None, None] ** (n + 2))
    def masked_stack(values, spatial_rank):
        array = np.stack(values, axis=2)
        return jnp.asarray(np.where(valid[(...,) + (None,) * (spatial_rank + 1)], array, np.nan))
    return FarFieldGeometry(masked_stack(fields, 1), masked_stack(jacobians, 2),
                            masked_stack(derivatives, 3), jnp.asarray(valid))


def _pair_parameter(value, shape):
    value = jnp.asarray(value)
    if value.ndim == 1:
        value = value[:, None]
    return jnp.broadcast_to(value, shape)


def _farfield_weights(geometry, sigma0, Ra, Rs, efficiency=None):
    shape = geometry.valid_separation.shape
    convergence, core, cut = (_pair_parameter(value, shape) for value in (sigma0, Ra, Rs))
    mass = 2 * convergence * core * cut
    if efficiency is not None:
        mass = mass * jnp.broadcast_to(jnp.asarray(efficiency), shape)
    c1 = core + cut
    return mass[..., None] * jnp.stack((jnp.ones_like(c1), -.5 * c1,
                                       .125 * c1 * (core**2 + cut**2)), axis=-1)


def evaluate_farfield(geometry: FarFieldGeometry, sigma0, Ra, Rs, efficiency=None) -> MemberFields:
    """Evaluate member rows using CURRENT packed central convergence and radii.

    sigma0/Ra/Rs accept (M,), (M,1), or (M,N); radii are angular arcseconds.
    The monopole is 2*sigma0*Ra*Rs in the active DPIE_NIE_E convention.
    If sigma0 already carries the per-image lensing efficiency, omit efficiency.
    Otherwise efficiency is (N,) and multiplies all fields exactly once.
    """
    weights = _farfield_weights(geometry, sigma0, Ra, Rs, efficiency)
    return MemberFields(jnp.einsum("mnk,mnki->mni", weights, geometry.deflection),
                        jnp.einsum("mnk,mnkij->mnij", weights, geometry.jacobian),
                        jnp.einsum("mnk,mnkijl->mnijl", weights, geometry.jacobian_derivative))


class SamplingMemberFields(NamedTuple):
    alpha: object
    jacobian: object
    total_jacobian_derivative: object


def evaluate_farfield_sampling(geometry: FarFieldGeometry, sigma0, Ra, Rs) -> SamplingMemberFields:
    """Batched current-parameter fields without per-member spatial-derivative output.

    Packed sigma0 already includes each image's redshift efficiency. Member
    deflections/Jacobians are needed for scatter covariance; only the summed
    spatial derivative is needed for determinant conditioning.
    """
    weights = _farfield_weights(geometry, sigma0, Ra, Rs)
    return SamplingMemberFields(
        jnp.einsum("mnk,mnki->mni", weights, geometry.deflection),
        jnp.einsum("mnk,mnkij->mnij", weights, geometry.jacobian),
        jnp.einsum("mnk,mnkijl->nijl", weights, geometry.jacobian_derivative))


@dataclass(frozen=True)
class MemberTierRecord:
    """Immutable catalog-order tier membership; indices are positions in member_ids."""
    member_ids: tuple[str, ...]
    component_indices: tuple[int, ...]
    exact_sampled: tuple[int, ...] = ()
    exact_absorbed: tuple[int, ...] = ()
    approximate_sampled: tuple[int, ...] = ()
    approximate_absorbed: tuple[int, ...] = ()
    reference_count: int = 0
    selection_reason: str = "reference_ensemble"
    reference_parameter_names: tuple[str, ...] = ()
    reference_parameters: tuple[tuple[float, ...], ...] = ()
    version: int = TIER_SCHEMA_VERSION

    def __post_init__(self):
        for name in ("member_ids", "component_indices", *_TIER_NAMES, "reference_parameter_names"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "reference_parameters", tuple(tuple(row) for row in self.reference_parameters))
        if self.version != TIER_SCHEMA_VERSION:
            raise ValueError(f"Unsupported member-tier schema version {self.version}.")
        n = len(self.member_ids)
        if len(self.component_indices) != n or len(set(self.member_ids)) != n or len(set(self.component_indices)) != n:
            raise ValueError("Member identities and component indices must be unique and aligned.")
        indices = [index for name in _TIER_NAMES for index in getattr(self, name)]
        if any(not isinstance(index, (int, np.integer)) for index in indices) or sorted(indices) != list(range(n)):
            raise ValueError("The four member tiers must partition catalog positions exactly once.")
        if self.reference_count < 0:
            raise ValueError("reference_count cannot be negative.")
        if self.reference_parameters:
            if len(self.reference_parameters) != self.reference_count:
                raise ValueError("Saved reference count does not match reference parameters.")
            if any(len(row) != len(self.reference_parameter_names) for row in self.reference_parameters):
                raise ValueError("Saved reference parameter ordering is inconsistent.")
            if not np.isfinite(np.asarray(self.reference_parameters, dtype=float)).all():
                raise ValueError("Saved reference parameters must be finite.")

    @property
    def sampled(self):
        return tuple(sorted(self.exact_sampled + self.approximate_sampled))

    @property
    def absorbed(self):
        return tuple(sorted(self.exact_absorbed + self.approximate_absorbed))

    @property
    def exact(self):
        return tuple(sorted(self.exact_sampled + self.exact_absorbed))

    @property
    def approximate(self):
        return tuple(sorted(self.approximate_sampled + self.approximate_absorbed))

    def to_dict(self):
        return {"version": self.version, "member_ids": list(self.member_ids),
                "component_indices": [int(index) for index in self.component_indices],
                **{name: [int(index) for index in getattr(self, name)] for name in _TIER_NAMES},
                "reference_count": self.reference_count, "selection_reason": self.selection_reason,
                "reference_parameter_names": list(self.reference_parameter_names),
                "reference_parameters": [list(row) for row in self.reference_parameters]}

    def validate_identity(self, member_ids, component_indices):
        """Reject a restored classification attached to reordered or changed rows."""
        if tuple(member_ids) != self.member_ids or tuple(component_indices) != self.component_indices:
            raise ValueError("Saved member-tier catalog identity/order does not match the current model.")

    @classmethod
    def from_dict(cls, payload):
        if "version" not in payload:
            raise ValueError("Saved member-tier record has no schema version.")
        return cls(**payload)


@dataclass(frozen=True)
class MemberTierThresholds:
    template_deflection: float = .1
    template_logh: float = .1
    sampled_deflection: float = float(np.sqrt(.1))
    sampled_logh: float = float(np.sqrt(.1))
    absorbed_variance: float = .25

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise ValueError("Member-tier tolerances must be finite and positive.")


def finite_logh_change(mean, variance, delta, parity=None):
    """max± |log h(mean ± delta, variance) - log h(mean, variance)|."""
    from .smoothed_likelihood import log_expected_absolute_normal
    mean, variance, delta = (jnp.asarray(value) for value in (mean, variance, delta))
    baseline = log_expected_absolute_normal(mean, variance, parity)
    plus = log_expected_absolute_normal(mean + delta, variance, parity)
    minus = log_expected_absolute_normal(mean - delta, variance, parity)
    return jnp.maximum(jnp.abs(plus - baseline), jnp.abs(minus - baseline))


def _determinant_response(total_jacobian, member_jacobian):
    a, u = total_jacobian[:, None], member_jacobian
    return a[..., 1, 1] * u[..., 0, 0] + a[..., 0, 0] * u[..., 1, 1] - a[..., 0, 1] * u[..., 1, 0] - a[..., 1, 0] * u[..., 0, 1]


def _reference_inputs(exact_alpha, exact_jacobian, template_alpha, template_jacobian,
                      total_jacobian, determinant_mean, determinant_variance, sigma_src, s_sigma,
                      parity, source_prior_median, member_prior_median, prior_guards):
    alpha, u, far, far_u, a, mean, variance = (np.asarray(value, dtype=float) for value in
        (exact_alpha, exact_jacobian, template_alpha, template_jacobian, total_jacobian,
         determinant_mean, determinant_variance))
    if alpha.ndim != 4 or alpha.shape[-1] != 2:
        raise ValueError("Exact alpha must have shape (references, members, images, 2).")
    r, m, n, _ = alpha.shape
    if far.shape != alpha.shape or u.shape != (r, m, n, 2, 2) or far_u.shape != u.shape:
        raise ValueError("Exact and approximate member field shapes are inconsistent.")
    if a.shape != (r, n, 2, 2) or mean.shape != (r, n) or variance.shape != mean.shape:
        raise ValueError("Reference Jacobian and determinant-statistic shapes are inconsistent.")
    source = np.broadcast_to(np.asarray(sigma_src, dtype=float), (r,)).copy()
    scatter = np.asarray(s_sigma, dtype=float)
    if scatter.ndim == 1:
        scatter = scatter[:, None]
    scatter = np.broadcast_to(scatter, (r, m)).copy()
    branch = np.zeros((r, n), dtype=int) if parity is None else np.broadcast_to(parity, (r, n))
    if not np.isin(branch, (-1, 0, 1)).all():
        raise ValueError("Parity must be -1, 0 (unknown), or +1.")
    finite = all(np.isfinite(value).all() for value in (alpha, u, a, mean, variance, source, scatter))
    valid = finite and np.all(source > 0) and np.all(scatter >= 0) and np.all(variance >= 0)
    if prior_guards:
        if (not np.isfinite(source_prior_median) or not np.isfinite(member_prior_median)
                or source_prior_median <= 0 or member_prior_median <= 0):
            raise ValueError("Selection prior medians must be finite and positive.")
        source = np.minimum(source, source_prior_median)
        scatter = np.maximum(scatter, member_prior_median)
    template_valid = np.isfinite(far).all(axis=(-1, 0, 2)) & np.isfinite(far_u).all(axis=(-1, -2, 0, 2))
    errors = np.linalg.norm(alpha - far, axis=-1)
    determinant_errors = np.abs(_determinant_response(a, u - far_u))
    # Invalid template members are promoted exactly; their unused error is zero.
    errors[:, ~template_valid] = 0.
    determinant_errors[:, ~template_valid] = 0.
    delta_alpha = 2 * scatter[..., None] * np.linalg.norm(alpha, axis=-1)
    delta_d = 2 * scatter[..., None] * np.abs(_determinant_response(a, u))
    return dict(r=r, m=m, n=n, valid=bool(valid), mean=mean, variance=variance, parity=branch,
                source=source, scatter=scatter, template_valid=template_valid,
                errors=errors, determinant_errors=determinant_errors,
                delta_alpha=delta_alpha, delta_d=delta_d)


def _budget_arrays(data, exact, sampled, thresholds):
    far, absorbed = ~exact, ~sampled
    position = data["errors"][:, far].sum(axis=1) / (thresholds.template_deflection * data["source"][:, None])
    determinant = np.asarray(finite_logh_change(data["mean"], data["variance"],
        data["determinant_errors"][:, far].sum(axis=1), data["parity"])) / thresholds.template_logh
    variance = np.square(data["delta_alpha"][:, absorbed]).sum(axis=1) / (
        thresholds.absorbed_variance * data["source"][:, None]**2)
    return position, determinant, variance


def _diagnostics(data, exact, sampled, thresholds):
    ratios = _budget_arrays(data, exact, sampled, thresholds)
    result = {"reference_count": data["r"], "image_count": data["n"],
              "thresholds": vars(thresholds).copy(), "scope": "evaluated_reference_models"}
    passed = data["valid"]
    for key, values in zip(("template_deflection", "template_logh", "absorbed_variance"), ratios):
        finite = np.isfinite(values).all()
        maximum = float(np.max(values)) if values.size and finite else (0. if not values.size else None)
        result[key + "_ratios"] = values.tolist() if finite else None
        result[key + "_max_ratio"] = maximum
        result[key + "_worst_reference_image"] = ([int(index) for index in np.unravel_index(np.argmax(values), values.shape)]
            if values.size and finite else None)
        passed = passed and finite and bool(np.all(values < 1.))
    absorbed = ~sampled
    individual_position = data["delta_alpha"][:, absorbed] / (
        thresholds.sampled_deflection * data["source"][:, None, None])
    individual_determinant = np.asarray(finite_logh_change(data["mean"][:, None], data["variance"][:, None],
        data["delta_d"][:, absorbed], data["parity"][:, None])) / thresholds.sampled_logh
    for key, values in (("absorbed_individual_deflection", individual_position),
                        ("absorbed_individual_logh", individual_determinant)):
        finite = np.isfinite(values).all()
        result[key + "_max_ratio"] = (float(np.max(values)) if values.size and finite
                                       else (0. if not values.size else None))
        # Individual promotion is strictly > its threshold, unlike collective budgets.
        passed = passed and finite and bool(np.all(values <= 1.))
    result["approximation_validation"] = "passed" if passed else "failed"
    return result


def classify_member_tiers(exact_alpha, exact_jacobian, template_alpha, template_jacobian,
                          total_jacobian, determinant_mean, determinant_variance, sigma_src, s_sigma,
                          *, member_ids, component_indices=None, parity=None, thresholds=None,
                          source_prior_median=.02, member_prior_median=.02,
                          reference_uncertainty_available=True, reference_parameter_names=(),
                          reference_parameters=(), top_sampled=None, top_exact=None):
    """Return (immutable tiers, diagnostics), enforcing all finite reference checks.

    Fields have shapes (R,M,N,2), (R,M,N,2,2), A=(R,N,2,2), nu/variance=(R,N).
    sigma_src is scalar or (R,), s_sigma scalar, (R,), or (R,M). Prior guards are
    used only in classification. Missing uncertainty triggers all-exact/sample.
    """
    thresholds = thresholds or MemberTierThresholds()
    for name, value in (("top_sampled", top_sampled), ("top_exact", top_exact)):
        if value != "all" and value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0):
            raise ValueError(f"{name} must be a nonnegative integer, 'all', or None.")
    data = _reference_inputs(exact_alpha, exact_jacobian, template_alpha, template_jacobian,
        total_jacobian, determinant_mean, determinant_variance, sigma_src, s_sigma, parity,
        source_prior_median, member_prior_median, True)
    n = data["m"]
    if len(member_ids) != n:
        raise ValueError("Member identities must match the member axis.")
    if len(set(member_ids)) != n:
        raise ValueError("Member identities must be unique.")
    identity_order = sorted(range(n), key=lambda index: str(member_ids[index]))
    def largest_contributor(values):
        # Retain catalog-order output indices, but make equal contributions
        # independent of a reordered input catalog.
        return int(identity_order[int(np.argmax(np.asarray(values)[identity_order]))])
    components = tuple(range(n)) if component_indices is None else tuple(component_indices)
    if top_sampled is not None or top_exact is not None:
        if not data["valid"] or data["r"] == 0:
            raise ValueError("Explicit top-k member ranking requires valid reference fields.")
        if top_sampled is None or top_exact is None:
            automatic, _ = classify_member_tiers(exact_alpha, exact_jacobian, template_alpha, template_jacobian,
                total_jacobian, determinant_mean, determinant_variance, sigma_src, s_sigma,
                member_ids=member_ids, component_indices=components, parity=parity, thresholds=thresholds,
                source_prior_median=source_prior_median, member_prior_median=member_prior_median,
                reference_uncertainty_available=reference_uncertainty_available,
                reference_parameter_names=reference_parameter_names, reference_parameters=reference_parameters)
            exact, sampled = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
            exact[list(automatic.exact)] = True
            sampled[list(automatic.sampled)] = True
        else:
            exact, sampled = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        ranking = {}
        for axis, count, mask, displacement, determinant, positional_limit, logh_limit in (
            ("sampled", top_sampled, sampled, data["delta_alpha"], data["delta_d"], thresholds.sampled_deflection, thresholds.sampled_logh),
            ("exact", top_exact, exact, data["errors"], data["determinant_errors"], thresholds.template_deflection, thresholds.template_logh),
        ):
            if count is None:
                continue
            count = n if count == "all" else count
            position_scores = displacement / (positional_limit * data["source"][:, None, None])
            determinant_scores = np.asarray(finite_logh_change(data["mean"][:, None], data["variance"][:, None],
                determinant, data["parity"][:, None])) / logh_limit
            scores = np.max(np.maximum(position_scores, determinant_scores), axis=(0, 2), initial=0.)
            mandatory = ~data["template_valid"] if axis == "exact" else np.zeros(n, dtype=bool)
            if int(mandatory.sum()) > count:
                raise ValueError(f"top_exact={count} cannot accommodate {int(mandatory.sum())} members with invalid far-field templates; increase top_exact or use None.")
            scores[mandatory] = np.inf
            if np.isnan(scores).any():
                raise ValueError(f"Explicit top_{axis} ranking has undefined error scores.")
            order = sorted(identity_order, key=lambda i: (not mandatory[i], -scores[i], str(member_ids[i])))
            mask[:] = False
            mask[order[:min(int(count), n)]] = True
            ranking[axis] = dict(member_ids=[str(member_ids[i]) for i in order],
                scores=[float(scores[i]) if np.isfinite(scores[i]) else None for i in order],
                infinite_score_member_ids=[str(member_ids[i]) for i in order if np.isinf(scores[i])],
                mandatory_exact_member_ids=[str(member_ids[i]) for i in order if mandatory[i]])
        record = MemberTierRecord(tuple(member_ids), components,
            tuple(np.flatnonzero(exact & sampled)), tuple(np.flatnonzero(exact & ~sampled)),
            tuple(np.flatnonzero(~exact & sampled)), tuple(np.flatnonzero(~exact & ~sampled)),
            reference_count=data["r"], selection_reason="impact_ranked_limits",
            reference_parameter_names=tuple(reference_parameter_names), reference_parameters=tuple(reference_parameters))
        diagnostics = _diagnostics(data, exact, sampled, thresholds)
        diagnostics.update(selection_reason=record.selection_reason, ranking_revision=1, rankings=ranking,
            selection_settings=dict(top_sampled=top_sampled, top_exact=top_exact),
            effective_counts=dict(sampled=int(sampled.sum()), exact=int(exact.sum())),
            reference_uncertainty_available=bool(reference_uncertainty_available),
            source_scatter_reference=data["source"].tolist(), member_scatter_reference=data["scatter"].tolist(),
            invalid_template_members=np.flatnonzero(~data["template_valid"]).tolist())
        return record, diagnostics
    fallback = not reference_uncertainty_available or not data["valid"] or data["r"] == 0
    exact = ~data["template_valid"].copy()
    sampled = np.zeros(n, dtype=bool)
    promotions = []
    if not fallback:
        scatter_logh = np.asarray(finite_logh_change(data["mean"][:, None], data["variance"][:, None],
                                                    data["delta_d"], data["parity"][:, None]))
        if not np.isfinite(scatter_logh).all():
            fallback = True
        else:
            sampled = np.any((data["delta_alpha"] > thresholds.sampled_deflection * data["source"][:, None, None])
                             | (scatter_logh > thresholds.sampled_logh), axis=(0, 2))
    if fallback:
        exact[:] = True
        sampled[:] = True
    else:
        # The worst reference/image is deterministic; contributor ties use ID.
        for _ in range(n + 1):
            position, determinant, variance = _budget_arrays(data, exact, sampled, thresholds)
            scores = np.stack((position, determinant))
            if not np.isfinite(scores).all():
                exact[:] = sampled[:] = True
                fallback = True
                break
            if scores.size == 0 or float(scores.max()) < 1.:
                break
            kind, ref, image = np.unravel_index(np.argmax(scores), scores.shape)
            contributions = data["errors" if kind == 0 else "determinant_errors"][ref, :, image]
            member = largest_contributor(np.where(~exact, contributions, -np.inf))
            exact[member] = True
            promotions.append({"member": member, "reason": "template_deflection" if kind == 0 else "template_logh",
                               "reference": int(ref), "image": int(image)})
        for _ in range(n + 1):
            variance = _budget_arrays(data, exact, sampled, thresholds)[2]
            if variance.size == 0 or float(variance.max()) < 1.:
                break
            ref, image = np.unravel_index(np.argmax(variance), variance.shape)
            contributions = np.square(data["delta_alpha"][ref, :, image])
            member = largest_contributor(np.where(~sampled, contributions, -np.inf))
            sampled[member] = True
            promotions.append({"member": member, "reason": "absorbed_variance", "reference": int(ref), "image": int(image)})
    record = MemberTierRecord(tuple(member_ids), components,
        tuple(np.flatnonzero(exact & sampled)), tuple(np.flatnonzero(exact & ~sampled)),
        tuple(np.flatnonzero(~exact & sampled)), tuple(np.flatnonzero(~exact & ~sampled)),
        reference_count=data["r"], selection_reason="all_exact_sampled_fallback" if fallback else "reference_ensemble",
        reference_parameter_names=tuple(reference_parameter_names), reference_parameters=tuple(reference_parameters))
    diagnostics = _diagnostics(data, exact, sampled, thresholds)
    diagnostics.update(promotions=promotions, selection_reason=record.selection_reason,
                       source_scatter_reference=data["source"].tolist(), member_scatter_reference=data["scatter"].tolist(),
                       invalid_template_members=np.flatnonzero(~data["template_valid"]).tolist())
    return record, diagnostics


def required_top_counts(record, *, thresholds=None, parity=None, **fields):
    """Smallest passing prefixes of impact rankings at actual posterior scatters.

    This is a diagnostic, not a globally minimal subset or a membership update.
    Scan prefixes explicitly because finite determinant changes need not be monotone.
    """
    thresholds = thresholds or MemberTierThresholds()
    data = _reference_inputs(**fields, parity=parity, source_prior_median=.02,
                             member_prior_median=.02, prior_guards=False)
    result = dict(status="unavailable", classification_unchanged=True,
                  method="smallest_passing_impact_ranked_prefix", prior_guards=False,
                  current_counts=dict(sampled=len(record.sampled), exact=len(record.exact)))
    if not data['valid'] or not data['r']:
        return dict(result, reason="invalid_reference_fields")
    masks, counts, identities = {}, {}, {}
    n = data['m']
    for axis, displacement, determinant, position_limit, logh_limit in (
        ('sampled', data['delta_alpha'], data['delta_d'], thresholds.sampled_deflection, thresholds.sampled_logh),
        ('exact', data['errors'], data['determinant_errors'], thresholds.template_deflection, thresholds.template_logh),
    ):
        score = np.max(np.maximum(displacement/(position_limit*data['source'][:, None, None]),
            np.asarray(finite_logh_change(data['mean'][:, None], data['variance'][:, None], determinant,
                                          data['parity'][:, None]))/logh_limit), axis=(0, 2), initial=0.)
        mandatory = ~data['template_valid'] if axis == 'exact' else np.zeros(n, dtype=bool)
        score[mandatory] = np.inf
        if np.isnan(score).any():
            return dict(result, reason="undefined_ranking_scores")
        order = sorted(range(n), key=lambda i: (not mandatory[i], -score[i], str(record.member_ids[i])))
        mask = np.zeros(n, dtype=bool)
        for k in range(n + 1):
            if k:
                mask[order[k-1]] = True
            if k < int(mandatory.sum()):
                continue
            if axis == 'exact':
                position, determinant_budget, _ = _budget_arrays(data, mask, np.ones(n, dtype=bool), thresholds)
                passed = all(np.isfinite(value).all() and np.all(value < 1.) for value in (position, determinant_budget))
            else:
                variance = np.square(data['delta_alpha'][:, ~mask]).sum(axis=1) / (
                    thresholds.absorbed_variance * data['source'][:, None]**2)
                passed = np.isfinite(variance).all() and np.all(variance < 1.) and np.all(score[~mask] <= 1.)
            if passed:
                counts[axis] = k
                identities[axis] = [record.member_ids[i] for i in order[:k]]
                masks[axis] = mask.copy()
                break
        else:
            return dict(result, reason="no_passing_prefix")
    return dict(result, status="passed", required_counts=counts, required_member_ids=identities,
                budgets=_diagnostics(data, masks['exact'], masks['sampled'], thresholds))


def audit_member_tiers(record, exact_alpha, exact_jacobian, template_alpha, template_jacobian,
                       total_jacobian, determinant_mean, determinant_variance, sigma_src, s_sigma,
                       *, parity=None, thresholds=None):
    """Audit fixed tiers with actual posterior scatters; never modifies membership."""
    thresholds = thresholds or MemberTierThresholds()
    data = _reference_inputs(exact_alpha, exact_jacobian, template_alpha, template_jacobian,
        total_jacobian, determinant_mean, determinant_variance, sigma_src, s_sigma, parity, .02, .02, False)
    if len(record.member_ids) != data["m"]:
        raise ValueError("Audit member axis does not match saved identities.")
    exact, sampled = np.zeros(data["m"], bool), np.zeros(data["m"], bool)
    exact[list(record.exact)] = sampled[list(record.sampled)] = True
    result = _diagnostics(data, exact, sampled, thresholds)
    if not np.all(data["template_valid"][list(record.approximate)]):
        result["approximation_validation"] = "failed"
    return result
