"""Post-inference audit of the far-field approximation and the validity guard."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .evaluator import Evaluator
from .model import ModelState


def _whitened_error(evaluator: Evaluator, physical: np.ndarray, stats_far: dict[str, Any]) -> np.ndarray:
    """Per-image whitened difference between the far-field and all-exact backprojections."""
    exact = evaluator.fields(jnp.asarray(physical), force_exact=True)
    difference = np.asarray(stats_far["fields"]["backprojected"]) - np.asarray(exact["backprojected"])
    covariance = np.asarray(stats_far["covariance"])
    precision = np.linalg.inv(covariance)
    return np.sqrt(np.einsum("ni,nij,nj->n", difference, precision, difference))


def audit_posterior(state: ModelState, evaluator: Evaluator, samples: np.ndarray, *, tolerance: float,
                    guard_cut_radii: float, median_theta: np.ndarray | None = None) -> dict[str, Any]:
    """Minimum R/r_cut over draws, the whitened far-field error at the median, and guard margins."""
    samples = np.asarray(samples, dtype=float)
    far = evaluator.far_positions
    report: dict[str, Any] = {"far_member_count": int(len(far)), "exact_member_count": int(len(evaluator.exact_positions)),
                              "guard_cut_radii": float(guard_cut_radii), "tolerance": float(tolerance), "draws": int(len(samples))}
    if len(samples) == 0:
        report["approximation_validation"] = "no_draws"
        return report
    if len(far):
        cut_radii = jax.jit(jax.vmap(lambda theta: jnp.take(evaluator.packed_state(evaluator.physical(theta))[0].Rs, jnp.asarray(evaluator.far_components))))
        ratio = np.asarray(evaluator.far_min_separation)[None, :] / np.asarray(cut_radii(jnp.asarray(samples)))
        report["min_separation_over_cut_radius"] = float(ratio.min())
        report["draws_inside_guard"] = int((ratio.min(axis=1) < guard_cut_radii).sum())
        report["worst_member"] = str(state.members.member_ids[int(far[int(np.argmin(ratio.min(axis=0)))])])
    theta = np.asarray(median_theta if median_theta is not None else np.median(samples, axis=0), dtype=float)
    stats = evaluator.statistics(jnp.asarray(theta))
    whitened = _whitened_error(evaluator, np.asarray(stats["physical"]), stats) if len(far) else np.zeros(state.images.n_images)
    report["whitened_far_field_error_median_model"] = {"max": float(whitened.max()), "median": float(np.median(whitened)),
                                                       "worst_image": str(state.images.image_labels[int(np.argmax(whitened))])}
    passed = (report.get("draws_inside_guard", 0) == 0) and bool(whitened.max() < tolerance)
    report["approximation_validation"] = "passed" if passed else "failed"
    return report
