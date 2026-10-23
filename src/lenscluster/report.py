"""Plain-text reports of the member classification and the posterior audit."""
from __future__ import annotations

from typing import Any

import numpy as np

from .model import ModelState


def _fmt(value: Any, width: int = 9, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return f"{'inf' if value is not None else '-':>{width}}"
    return f"{float(value):{width}.{digits}g}"


def _rank_table(rankings: list[dict[str, Any]], title: str) -> list[str]:
    lines = [title, f"  {'member':<14}{'class':<10}{'score':>9}{'truncation':>11}{'ellipticity':>12}{'r_cut\"':>9}{'min R\"':>9}{'R/r_cut':>9}"]
    for r in rankings:
        ratio = r["min_separation_arcsec"] / r["cut_radius_arcsec"] if r["cut_radius_arcsec"] > 0 else float("inf")
        lines.append(f"  {r['member']:<14}{'exact' if r['exact'] else 'far-field':<10}{_fmt(r['score'])}{_fmt(r['truncation'], 11)}"
                     f"{_fmt(r['ellipticity'], 12)}{_fmt(r['cut_radius_arcsec'])}{_fmt(r['min_separation_arcsec'])}{_fmt(ratio)}")
    return lines


def classification_report(state: ModelState, *, provisional: dict[str, Any] | None = None, top: int = 15) -> str:
    """Counts, why each member is exact, the highest-ranked members, and changes since the provisional pass."""
    d = state.members.classification
    rankings = d.get("rankings", [])
    promoted = {p["member"] for p in d.get("promotions", [])}
    by_truncation = sum(1 for r in rankings if r["exact"] and (r["truncation"] or 0) >= 1)
    by_ellipticity = sum(1 for r in rankings if r["exact"] and (r["truncation"] or 0) < 1 and (r["ellipticity"] or 0) >= 1)
    by_rank = sum(1 for r in rankings if r["exact"] and (r["score"] or 0) < 1 and r["member"] not in promoted)
    far = [r for r in rankings if not r["exact"]]
    lines = ["=== Member classification ==="]
    for potfile in state.potfiles:
        pivot = potfile.get("pivot")
        if pivot:
            lines.append(f"  pivot {potfile['id']}: mag0 = {pivot['mag0']:.0f} (leverage-weighted member magnitude {pivot['weighted_mean_mag']:.2f}, "
                         f"median {pivot['member_median_mag']:.2f}); sigma and cutkpc priors moved from mag {pivot['prior_mag0']:.2f} "
                         f"by x{pivot['sigma_factor']:.3f} and x{pivot['cutkpc_factor']:.3f}: sigma* {potfile['sigma_nominal']:.1f} km/s, "
                         f"r_cut* {potfile['cutkpc_nominal']:.1f} kpc")
    lines += [
             f"  members {d['member_count']}: exact {d['exact_count']} (mandatory {d['mandatory_count']}), far-field {d['member_count'] - d['exact_count']}; "
             f"deviations: {state.packed_lens_spec.deviation_mode}, centered for {d.get('centered_count', 0)} members",
             f"  exact because an image lies within {d['guard_cut_radii']:g} r_cut: {by_truncation}; because of ellipticity error > {d['tolerance']:g}: {by_ellipticity}; "
             f"promoted by the collective budget: {len(promoted)}; added by top_exact: {by_rank}",
             f"  cut radii for classification: reference {d.get('reference_cut_kpc', float('nan')):.1f} kpc x {d.get('inflated_cut_kpc', 0) / max(d.get('reference_cut_kpc', 1), 1e-12):.1f} = "
             f"{d.get('inflated_cut_kpc', float('nan')):.1f} kpc, beta in [{d.get('beta_range', [d.get('beta_radius', float('nan'))] * 2)[0]:.2f}, {d.get('beta_range', [d.get('beta_radius', float('nan'))] * 2)[1]:.2f}]"
             + (f" (prior {d['prior_quantile']:.3f} quantiles or the current model, whichever is larger)" if d.get('prior_quantile') else "")]
    if far:
        worst = max(far, key=lambda r: r["score"] or 0)
        min_ratio = min(r["min_separation_arcsec"] / r["cut_radius_arcsec"] for r in far if r["cut_radius_arcsec"] > 0)
        lines.append(f"  far-field: highest score {worst['score']:.3g} ({worst['member']}), smallest R/r_cut {min_ratio:.2f} (guard {d['guard_cut_radii']:g}), "
                     f"summed ellipticity budget per image < {d['tolerance']:g}")
    lines += _rank_table(rankings[:top], f"  top {min(top, len(rankings))} members by score:")
    if far:
        lines += _rank_table(far[:5], "  far-field members closest to the threshold:")
    if provisional is not None:
        before = {r["member"] for r in provisional.get("rankings", []) if r["exact"]}
        after = {r["member"] for r in rankings if r["exact"]}
        lines.append(f"  change since the provisional pass: +{len(after - before)} exact, -{len(before - after)} exact "
                     f"(provisional {len(before)} -> final {len(after)})")
        if after - before:
            lines.append("    made exact: " + ", ".join(sorted(after - before)[:20]) + (" ..." if len(after - before) > 20 else ""))
        if before - after:
            lines.append("    made far-field: " + ", ".join(sorted(before - after)[:20]) + (" ..." if len(before - after) > 20 else ""))
    return "\n".join(lines)


def audit_report(state: ModelState, audit: dict[str, Any], *, posterior_summary: dict[str, Any] | None = None) -> str:
    """Guard margins over the draws, the far-field error at the median, and the median-model reclassification."""
    lines = ["=== Posterior audit of the far-field approximation ===",
             f"  draws {audit.get('draws')}, far-field members {audit.get('far_member_count')}, exact {audit.get('exact_member_count')}"]
    if "min_separation_over_cut_radius" in audit:
        lines.append(f"  guard: min R/r_cut over draws {audit['min_separation_over_cut_radius']:.2f} (limit {audit['guard_cut_radii']:g}), "
                     f"draws inside the guard {audit['draws_inside_guard']}, worst member {audit['worst_member']}")
    w = audit.get("whitened_far_field_error_median_model", {})
    if w:
        lines.append(f"  whitened far-field error at the median model: max {w['max']:.3g} (image {w['worst_image']}), median {w['median']:.3g}, "
                     f"tolerance {audit['tolerance']:g}")
    r = audit.get("reclassification_at_median")
    if r:
        lines.append(f"  reclassifying at the posterior median: exact {r['exact_count']} vs {r['current_exact_count']} now; "
                     f"would add {len(r['would_add'])}, remove {len(r['would_remove'])}")
        if r["would_add"]:
            lines.append("    would add: " + ", ".join(r["would_add"][:20]) + (" ..." if len(r["would_add"]) > 20 else ""))
        if r["would_remove"]:
            lines.append("    would remove: " + ", ".join(r["would_remove"][:20]) + (" ..." if len(r["would_remove"]) > 20 else ""))
    if posterior_summary:
        lines.append("  sampler: " + ", ".join(f"{k} {v}" for k, v in posterior_summary.items()))
    lines.append(f"  verdict: {audit.get('approximation_validation')}")
    return "\n".join(lines)
