"""The five paper figures of a mock-validation campaign, from each seed's recovery.json."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .aperture import COMPONENT_GROUPS
from .runner import RECOVERY_FILE

STYLE = Path(__file__).resolve().parent.parent / "style.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))

COLOR = "#2878A5"
TRUTH_COLOR = "0.2"
COMPONENT_COLORS = {"halo": "#2878A5", "bcg": "#8C6BB1", "members": "#2A9D8F"}
COMPONENT_LABELS = {"halo": "Main halo", "bcg": "BCG", "members": "Members"}
PAPER_PARAMETERS = (
    "halo.log10_mass_msun", "halo.x_centre", "halo.y_centre", "halo.v_disp", "halo.core_radius_kpc", "halo.cut_radius_kpc",
    "bcg.log10_mass_msun", "bcg.x_centre", "bcg.y_centre", "bcg.v_disp", "bcg.core_radius_kpc", "bcg.cut_radius_kpc",
    "potfile.log10_mass_msun", "potfile.sigma", "potfile.cutkpc", "potfile.alpha_sigma", "potfile.beta_radius", "potfile.gamma_ml",
)
PARAMETER_LABELS = {
    "log10_mass_msun": r"$\log_{10} M$", "x_centre": r"$x$", "y_centre": r"$y$", "v_disp": r"$\sigma$", "core_radius_kpc": r"$r_{\rm core}$",
    "cut_radius_kpc": r"$r_{\rm cut}$", "sigma": r"$\sigma^\star$", "cutkpc": r"$r_{\rm cut}^\star$", "alpha_sigma": r"$\alpha$",
    "beta_radius": r"$\beta$", "gamma_ml": r"$\gamma_{M/L}$",
}
PARAMETER_UNITS = {"x_centre": "arcsec", "y_centre": "arcsec", "v_disp": "km/s", "core_radius_kpc": "kpc", "cut_radius_kpc": "kpc",
                   "sigma": "km/s", "cutkpc": "kpc", "log10_mass_msun": r"$M_\odot$"}
COMPONENT_NAMES = {"halo": "Halo", "bcg": "BCG", "potfile": "Members"}


@dataclass
class Campaign:
    root: Path
    seeds: list[int]
    parameters: pd.DataFrame
    magnification: pd.DataFrame
    residual_draws: pd.DataFrame
    subhalos: list[dict[str, Any]]
    m2d: pd.DataFrame
    summary: pd.DataFrame


def load_campaign(root: str | Path) -> Campaign:
    root = Path(root)
    files = sorted(root.glob("seed_*/recovery/" + RECOVERY_FILE))
    if not files:
        raise FileNotFoundError(f"No seed_*/recovery/{RECOVERY_FILE} under {root}")
    seeds, parameters, magnification, draws, subhalos, m2d, summary = [], [], [], [], [], [], []
    for path in files:
        payload = json.loads(path.read_text())
        seed = int(payload["seed"])
        seeds.append(seed)
        for key, store in (("parameters", parameters), ("magnification", magnification), ("image_residual_draws", draws), ("m2d", m2d)):
            store.append(pd.DataFrame(payload[key]).assign(seed=seed))
        subhalos.append({"seed": seed, **payload["subhalos"]})
        summary.append({"seed": seed, "max_r_hat": payload.get("max_r_hat"), "min_n_eff": payload.get("min_n_eff"),
                        "divergences": payload.get("divergences"), "audit": payload.get("audit"), **payload.get("fit", {})})
    return Campaign(root, seeds, pd.concat(parameters, ignore_index=True), pd.concat(magnification, ignore_index=True),
                    pd.concat(draws, ignore_index=True), subhalos, pd.concat(m2d, ignore_index=True), pd.DataFrame(summary))


def representative_seed(campaign: Campaign) -> int:
    """The seed with the smallest maximum R-hat (the first seed when R-hat is unavailable)."""
    summary = campaign.summary.copy()
    summary["max_r_hat"] = pd.to_numeric(summary["max_r_hat"], errors="coerce")
    ordered = summary.sort_values(["max_r_hat", "seed"], na_position="last")
    return int(ordered["seed"].iloc[0])


# ----------------------------------------------------------------------------- parameter recovery
def _axis_label(parameter: str, role: str) -> str:
    component, field = parameter.split(".", 1)
    label = f"{COMPONENT_NAMES.get(component, component)} {PARAMETER_LABELS.get(field, field)}"
    unit = PARAMETER_UNITS.get(field)
    return f"{label} ({unit}, {role})" if unit else f"{label} ({role})"


def _draw_parameter_axis(ax, panel: pd.DataFrame, parameter: str) -> None:
    truth, median = panel["truth"].to_numpy(float), panel["median"].to_numpy(float)
    q16, q84 = panel["q16"].to_numpy(float), panel["q84"].to_numpy(float)
    ax.errorbar(truth, median, yerr=[np.maximum(0, median - q16), np.maximum(0, q84 - median)], fmt="o", color=COLOR, markersize=5,
                markeredgecolor="#1b4f6e", elinewidth=0.65, capsize=1.8, capthick=0.65, alpha=0.9, zorder=2)
    values = np.concatenate([truth, median, q16, q84])
    values = values[np.isfinite(values)]
    if values.size:
        pad = 0.05 * (values.max() - values.min() or 1.0)
        limits = (values.min() - pad, values.max() + pad)
        ax.plot(limits, limits, color="0.25", linewidth=1.0, alpha=0.7, zorder=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    ax.set_xlabel(_axis_label(parameter, "truth"), fontsize=11)
    ax.set_ylabel(_axis_label(parameter, "fit"), fontsize=11)
    ax.tick_params(labelsize=9)


def plot_parameter_recovery(campaign: Campaign, path: Path, *, ncols: int = 4) -> Path | None:
    df = campaign.parameters
    df = df[np.isfinite(df["truth"].to_numpy(float)) & np.isfinite(df["median"].to_numpy(float))]
    parameters = [p for p in PAPER_PARAMETERS if (df["parameter"] == p).any()]
    names = df["parameter"].astype(str)
    source = df[names.str.startswith("source.") & (names.str.endswith(".beta_x") | names.str.endswith(".beta_y"))]
    coordinates = [c for c in ("beta_x", "beta_y") if source["parameter"].str.endswith(c).any()]
    if not parameters and not coordinates:
        return None
    n_panels = len(parameters) + len(coordinates)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.45 * nrows), squeeze=False)
    flat = list(axes.ravel())
    for ax, parameter in zip(flat, parameters):
        _draw_parameter_axis(ax, df[df["parameter"] == parameter], parameter)
    for ax, coordinate in zip(flat[len(parameters):], coordinates):
        rows = source[source["parameter"].str.endswith(coordinate)]
        _draw_parameter_axis(ax, rows, f"source.{coordinate}")
        symbol = r"$\beta_x$" if coordinate == "beta_x" else r"$\beta_y$"
        ax.set_xlabel(f"Source {symbol} (arcsec, truth)", fontsize=11)
        ax.set_ylabel(f"Source {symbol} (arcsec, fit)", fontsize=11)
    for ax in flat[n_panels:]:
        ax.axis("off")
    fig.tight_layout(pad=0.35, w_pad=0.35, h_pad=0.35)
    fig.subplots_adjust(wspace=0.42, hspace=0.34)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- subhalo mass function
def _schechter_curve(selection: dict[str, Any], x: np.ndarray, normalization: float) -> np.ndarray | None:
    try:
        alpha, anchor, exponent = (float(selection[k]) for k in ("schechter_alpha", "dpie_mass_anchor_msun", "mass_luminosity_exponent"))
    except (KeyError, TypeError, ValueError):
        return None
    if not (anchor > 0 and exponent > 0 and normalization > 0):
        return None
    luminosity = np.power(np.power(10.0, x) / anchor, 1.0 / exponent)
    density = np.power(luminosity, alpha + 1.0) * np.exp(-luminosity)
    norm = float(np.trapezoid(density, x))
    return density / norm * normalization if norm > 0 else None


def plot_subhalo_mass_function(campaign: Campaign, path: Path) -> Path | None:
    truth_all = [np.asarray(s["truth_log10_mass"], float) for s in campaign.subhalos]
    recovered_all = [np.asarray(s["recovered_log10_mass"], float) for s in campaign.subhalos]
    values = np.concatenate([v for v in truth_all + recovered_all if v.size]) if any(v.size for v in truth_all + recovered_all) else np.asarray([])
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        lo, hi = lo - 0.25, hi + 0.25
    edges = np.linspace(lo, hi, int(np.clip(np.sqrt(values.size), 6, 18)) + 1)
    widths, centers = np.diff(edges), 0.5 * (edges[:-1] + edges[1:])
    truth_density = np.stack([np.histogram(v, bins=edges)[0] / widths for v in truth_all])
    recovered_density = np.stack([np.histogram(v, bins=edges)[0] / widths for v in recovered_all])
    residual = (recovered_density - truth_density) * widths
    truth_median, recovered_median = np.median(truth_density, axis=0), np.median(recovered_density, axis=0)
    r16, r50, r84 = np.percentile(residual, [16, 50, 84], axis=0)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(4.6, 5.5), sharex=True, gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08})
    top.stairs(recovered_median, edges, color=COLOR, linewidth=1.8, label="Recovered")
    top.stairs(truth_median, edges, color="0.25", linestyle="--", linewidth=1.6, label="Truth", zorder=4)
    curve = _schechter_curve(campaign.subhalos[0].get("selection", {}), centers, float(np.nansum(truth_median * widths)))
    if curve is not None:
        top.plot(centers, curve, color="0.45", linestyle=":", linewidth=1.4, label="Schechter")
    ceiling = max(1.0, float(np.nanmax(np.concatenate([truth_median, recovered_median]))) * 1.35)
    top.set_ylim(-0.02 * ceiling, ceiling)
    top.legend(loc="upper right", frameon=False, fontsize=10)
    bottom.axhline(0, color="0.35", linestyle="--", linewidth=1.0)
    bottom.errorbar(centers, r50, yerr=[r50 - r16, r84 - r50], fmt="o", color=COLOR, markersize=5, elinewidth=1.2, capsize=2.5, zorder=3)
    limit = max(1.0, float(np.abs(np.concatenate([r16, r84])).max()) * 1.15)
    bottom.set_ylim(-limit, limit)
    bottom.set_xlabel(r"$\log_{10}(M/M_\odot)$", fontsize=12)
    top.set_ylabel(r"$dN/d\log_{10} M$", fontsize=12)
    bottom.set_ylabel(r"$N_{\rm recovered}-N_{\rm true}$", fontsize=12)
    for ax in (top, bottom):
        ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.12, top=0.98)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- magnification
def plot_magnification_recovery(campaign: Campaign, path: Path) -> Path | None:
    df = campaign.magnification
    truth = np.abs(df["magnification_true_at_obs"].to_numpy(float))
    q16, q50, q84 = (np.abs(df[f"magnification_model_{q}"].to_numpy(float)) for q in ("q16", "q50", "q84"))
    lower, upper = np.minimum(np.minimum(q16, q50), q84), np.maximum(np.maximum(q16, q50), q84)
    valid = (truth > 0) & (q50 > 0) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return None
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    ax.errorbar(truth[valid], q50[valid], yerr=[np.maximum(0, q50 - lower)[valid], np.maximum(0, upper - q50)[valid]], fmt="o", color=COLOR,
                markersize=4, markeredgecolor="#1b4f6e", elinewidth=0.65, capsize=1.8, capthick=0.65, alpha=0.85, zorder=2)
    log_values = np.log10(np.concatenate([truth[valid], lower[valid][lower[valid] > 0], upper[valid]]))
    pad = max(0.08 * (log_values.max() - log_values.min()), 0.08)
    limits = (10 ** (log_values.min() - pad), 10 ** (log_values.max() + pad))
    ax.plot(limits, limits, color="0.25", linewidth=1.0, alpha=0.75, zorder=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel(r"Image $|\mu|$ (truth)")
    ax.set_ylabel(r"Image $|\mu|$ (fit)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- image residual distribution
def plot_image_residual_distribution(campaign: Campaign, path: Path, *, bins: int = 24) -> Path | None:
    df = campaign.residual_draws
    residual = pd.to_numeric(df["image_residual_arcsec"], errors="coerce").to_numpy(float)
    recovered = df["recovered"].astype(bool).to_numpy() & np.isfinite(residual) & (residual >= 0)
    if not recovered.any():
        return None
    x_max = float(np.ceil(np.quantile(residual[recovered], 0.99) * 1.10 / 0.05) * 0.05)
    edges = np.linspace(0.0, max(x_max, 1e-6), bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.stack([np.histogram(residual[recovered & (df["seed"] == seed).to_numpy() & (df["draw_index"] == draw).to_numpy()], bins=edges)[0]
                       for seed, draw in df.loc[recovered, ["seed", "draw_index"]].drop_duplicates().itertuples(index=False)])
    h16, h50, h84 = np.percentile(counts, [16, 50, 84], axis=0)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.fill_between(centers, h16, h84, color=COLOR, alpha=0.14, step="mid", linewidth=0.0)
    ax.step(centers, h50, where="mid", color=COLOR, linewidth=1.6)
    ax.set_ylabel("Number of images")
    ax.set_xlabel("Image-position residual (arcsec)")
    ax.set_xlim(0.0, edges[-1])
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- aperture mass components
def plot_m2d_components(campaign: Campaign, path: Path, *, seed: int | None = None) -> Path | None:
    seed = representative_seed(campaign) if seed is None else int(seed)
    rows = campaign.m2d[(campaign.m2d["seed"] == seed) & campaign.m2d["component"].isin(COMPONENT_GROUPS)]
    if rows.empty:
        return None
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(6.3, 6.0), sharex=True, gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.08})
    residuals = []
    for group in COMPONENT_GROUPS:
        sel = rows[rows["component"] == group].sort_values("radius_kpc")
        if sel.empty:
            continue
        radius, color = sel["radius_kpc"].to_numpy(float), COMPONENT_COLORS[group]
        top.plot(radius, sel["truth_m2d_msun"].to_numpy(float), color=color, linestyle=":", linewidth=1.2, alpha=0.9)
        top.plot(radius, sel["model_q50"].to_numpy(float), color=color, linewidth=1.6, label=COMPONENT_LABELS[group])
        q16, q50, q84 = (sel[f"ratio_{q}"].to_numpy(float) for q in ("q16", "q50", "q84"))
        bottom.fill_between(radius, q16, q84, color=color, alpha=0.10, linewidth=0.0)
        bottom.plot(radius, q50, color=color, linewidth=1.4)
        residuals.extend([q16, q84, q50])
    top.plot([], [], color="0.3", linestyle=":", linewidth=1.2, label="Truth")
    top.set_xscale("log")
    top.set_yscale("log")
    top.set_ylabel(r"$M_{\rm 2D}(<R)$ ($M_\odot$)")
    top.legend(loc="lower right", frameon=False, fontsize=11)
    bottom.axhline(1.0, color="0.2", linestyle="--", linewidth=1.0)
    finite = np.concatenate(residuals)
    finite = finite[np.isfinite(finite)] - 1.0
    limit = 0.1 if finite.size == 0 else float(min(0.5, max(0.1, 1.2 * np.max(np.abs(finite)))))
    bottom.set_ylim(1.0 - limit, 1.0 + limit)
    bottom.set_xlabel(r"$R$ (kpc)")
    bottom.set_ylabel(r"$M_{\rm model}/M_{\rm true}$")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- driver
def write_aggregate_plots(campaign_root: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    campaign = load_campaign(campaign_root)
    output = Path(output_dir) if output_dir is not None else campaign.root / "aggregate_plots"
    output.mkdir(parents=True, exist_ok=True)
    campaign.parameters.to_csv(output / "parameter_recovery_table.csv", index=False)
    campaign.summary.to_csv(output / "seed_summary.csv", index=False)
    plots = [plot_parameter_recovery(campaign, output / "mock_parameter_recovery_paper.pdf"),
             plot_subhalo_mass_function(campaign, output / "mock_aggregate_subhalo_recovery_shmf.pdf"),
             plot_magnification_recovery(campaign, output / "mock_magnification_recovery.pdf"),
             plot_m2d_components(campaign, output / "mock_m2d_aperture_recovery_components.pdf"),
             plot_image_residual_distribution(campaign, output / "mock_aggregate_image_residual_distribution.pdf")]
    return [p for p in plots if p is not None]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create the paper figures of a mock-validation campaign.")
    parser.add_argument("campaign_root", help="Directory holding the seed_* folders (validation_runs/<campaign>/<run_name>).")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    for path in write_aggregate_plots(args.campaign_root, args.output_dir):
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
