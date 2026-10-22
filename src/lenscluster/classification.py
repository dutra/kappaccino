"""Member classification: exact dPIE kernels near images, the circular far-field series elsewhere.

score_g = max over images i of max( (k r_cut,g / R_gi)^6 , ||C_i^-1/2 e_gi|| / tolerance )

The first term is the leading omitted term of the series normalized to the guard
distance k r_cut; members with any image inside k cut radii score above one and are
mandatory.  The second term is the whitened ellipticity error of the circular series,
with C_i = sigma_src^2 I + sigma_obs^2 A_i A_i^T, bounded and evaluated at one model.
After the per-member rule, the largest contributors are promoted until the summed
whitened ellipticity error of the remaining far-field members is below the tolerance
at every image.
"""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from .evaluator import Evaluator
from .farfield import evaluate_farfield_per_member
from .model import ModelState


def member_cut_radii_arcsec(state: ModelState, reference_cut_kpc: float, beta_radius: float) -> np.ndarray:
    """Cut radius of every population member for a reference cut radius and radius exponent."""
    spec = state.packed_lens_spec
    luminosity = np.asarray(spec.luminosity_ratio)[state.members.component_indices]
    return float(reference_cut_kpc) * luminosity ** float(beta_radius) / float(state.kpc_per_arcsec)


def scaling_values(state: ModelState, physical: np.ndarray) -> dict[str, float]:
    """Reference cut radius and radius exponent per population, read from a physical vector."""
    values: dict[str, float] = {}
    for spec in state.parameter_specs:
        if spec.component_family == "scaling" and spec.field in {"cutkpc", "beta_radius"}:
            values[spec.field] = float(physical[state.parameter_specs.index(spec)])
    return values


def classify_members(state: ModelState, evaluator: Evaluator, physical: np.ndarray, *,
                     cut_radius_arcsec: np.ndarray, guard_cut_radii: float, tolerance: float,
                     top_exact: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the exact member positions and the ranking diagnostics.

    ``physical`` supplies the Jacobians, source scatter and member deflections used by the
    ellipticity term; ``cut_radius_arcsec`` supplies the (conservative) cut radius per member.
    """
    members = state.members
    n_members = len(members.member_ids)
    if n_members == 0:
        return np.asarray([], dtype=np.int32), {"member_count": 0, "mandatory_count": 0, "exact_count": 0, "rankings": []}
    physical_jax = jnp.asarray(physical, dtype=jnp.float64)
    packed, _ = evaluator.packed_state(physical_jax)
    fields = evaluator.fields(physical_jax, packed, force_exact=True)
    exact_alpha = np.asarray(fields["member_alpha"])                     # (M, N, 2) elliptical dPIE
    components = jnp.asarray(members.component_indices)
    circular_alpha = np.asarray(evaluate_farfield_per_member(
        evaluator.geometry, jnp.take(packed.sigma0, components, axis=0), jnp.take(packed.Ra, components),
        jnp.take(packed.Rs, components))[0])                             # (M, N, 2) circular series
    # Whitened error: ||C^-1/2 e|| with C = sigma_src^2 I + sigma_obs^2 A A^T at this model.
    a = np.asarray(fields["jacobian"])
    sigma_obs = np.asarray(state.images.sigma_per_image)
    sigma_src = float(physical[members.sigma_src_index])
    covariance = sigma_src ** 2 * np.eye(2)[None] + sigma_obs[:, None, None] ** 2 * a @ np.swapaxes(a, 1, 2)
    precision = np.linalg.inv(covariance)
    error = exact_alpha - circular_alpha
    error = np.where(np.isfinite(error), error, np.inf)
    whitened = np.sqrt(np.einsum("mni,nij,mnj->mn", error, precision, error))
    separation = np.asarray(evaluator.geometry.separation)             # (M, N)
    ratio = (float(guard_cut_radii) * np.asarray(cut_radius_arcsec)[:, None] / np.where(separation > 0, separation, np.inf)) ** 6
    truncation_score = ratio.max(axis=1)
    ellipticity_score = (whitened / float(tolerance)).max(axis=1)
    score = np.maximum(truncation_score, ellipticity_score)
    score = np.where(np.isfinite(score), score, np.inf)
    exact = score >= 1.0
    promotions: list[dict[str, Any]] = []
    # Collective budget: summed whitened ellipticity error of far-field members per image.
    for _ in range(n_members):
        far_sum = np.where(~exact[:, None], np.nan_to_num(whitened, nan=np.inf, posinf=np.inf), 0.0).sum(axis=0)
        if far_sum.size == 0 or far_sum.max() < float(tolerance):
            break
        image = int(np.argmax(far_sum))
        candidates = np.where(~exact, whitened[:, image], -np.inf)
        member = int(np.argmax(candidates))
        exact[member] = True
        promotions.append({"member": members.member_ids[member], "image": state.images.image_labels[image],
                           "summed_whitened_error": float(far_sum[image])})
    mandatory = int(exact.sum())
    if top_exact is not None:
        if int(top_exact) < mandatory:
            raise ValueError(f"top_exact={top_exact} is below the {mandatory} members that must be exact.")
        order = np.argsort(-score, kind="stable")
        exact[order[:int(top_exact)]] = True
    order = sorted(range(n_members), key=lambda i: (-score[i], members.member_ids[i]))
    diagnostics = {
        "member_count": n_members, "mandatory_count": mandatory, "exact_count": int(exact.sum()),
        "guard_cut_radii": float(guard_cut_radii), "tolerance": float(tolerance), "top_exact": top_exact,
        "promotions": promotions,
        "rankings": [{"member": members.member_ids[i], "score": None if not np.isfinite(score[i]) else float(score[i]),
                      "truncation": None if not np.isfinite(truncation_score[i]) else float(truncation_score[i]),
                      "ellipticity": None if not np.isfinite(ellipticity_score[i]) else float(ellipticity_score[i]),
                      "cut_radius_arcsec": float(cut_radius_arcsec[i]),
                      "min_separation_arcsec": float(separation[i].min()), "exact": bool(exact[i])} for i in order],
    }
    return np.flatnonzero(exact).astype(np.int32), diagnostics
