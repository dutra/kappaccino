"""Figures for one run: image recovery, residual histogram, corner, scaling relation, critical curves, truth recovery."""
from __future__ import annotations
from .style import CRITICAL_COLOR

import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from skimage.measure import find_contours

from .style import PAPER_STYLE
from ..likelihood.evaluator import Evaluator
from ..core.model import ModelState, ParameterSpec, PosteriorResults, member_log_deviations
from ..diagnostics.truth_recovery import PROFILE_INNER_ARCSEC, smoothed_digitized_curve
from ..core.utils import log10_dpie_mass_msun

STYLE = Path(__file__).with_name("style.mplstyle")
if STYLE.exists():
    plt.style.use(str(STYLE))

POINT_COLOR = "#3976a8"
MISSED_COLOR = "#c44e52"


def figure_path(figures_dir: Path, name: str, prefix: str | None) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / (f"{prefix}_{name}" if prefix else name)


def _finish(fig, path: Path, dpi: int = 220) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- image recovery
def posterior_residual_histogram_counts(draws: pd.DataFrame | None, bins: np.ndarray, *, min_recovered: int = 0) -> np.ndarray:
    """Raw counts for draws meeting the minimum recovered-image count."""
    if draws is None or draws.empty or "draw_index" not in draws:
        return np.empty((0, len(bins) - 1), dtype=int)
    counts = []
    for _, group in draws.groupby("draw_index", sort=True):
        values = pd.to_numeric(group["image_residual_arcsec"], errors="coerce").to_numpy(float)
        valid = group["recovered"].astype(bool).to_numpy() & np.isfinite(values) & (values >= 0)
        if valid.sum() < min_recovered:
            continue
        counts.append(np.histogram(values[valid], bins=bins)[0])
    return np.asarray(counts, dtype=int).reshape(-1, len(bins) - 1)


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_image_residual_histogram(image_df: pd.DataFrame, path: Path, *, draws: pd.DataFrame | None = None, log=None) -> Path:
    """Image-position residual histogram over the reporting draws, as the mock aggregate figure
    (``mock_validation.aggregate_plots.plot_image_residual_distribution``): the median count per bin over draws with
    the 16-84 per cent band, and the RMS as the median of the per-draw values (dashed) with their 16-84 per cent range
    (red shade)."""
    groups = []
    if draws is not None and not draws.empty:
        for _, group in draws.groupby("draw_index", sort=True):
            r = pd.to_numeric(group["image_residual_arcsec"], errors="coerce").to_numpy(float)
            r = r[group["recovered"].astype(bool).to_numpy() & np.isfinite(r) & (r >= 0)]
            if r.size:
                groups.append(r)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    if groups:
        pooled = np.concatenate(groups)
        x_max = float(np.ceil(np.quantile(pooled, 0.99) * 1.10 / 0.05) * 0.05)
        edges = np.linspace(0.0, max(x_max, 1e-6), 25)
        counts = np.stack([np.histogram(g, bins=edges)[0] for g in groups])
        h16, h50, h84 = np.percentile(counts, [16, 50, 84], axis=0)
        rms_low, rms, rms_high = (float(v) for v in np.percentile([np.sqrt(np.mean(g ** 2)) for g in groups], [16, 50, 84]))
        ax.stairs(h84, edges, baseline=h16, fill=True, color="tab:blue", alpha=0.08, linewidth=0.0, zorder=1)
        ax.stairs(h50, edges, color="tab:blue", linewidth=1.5, label="Residuals", zorder=3)
        ax.axvspan(rms_low, rms_high, color="#ba5a2a", alpha=0.15, linewidth=0, zorder=2)
        ax.axvline(rms, color="#ba5a2a", linestyle="--", linewidth=1.3, label="RMS", zorder=4)
        ax.set_xlim(0.0, edges[-1])
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_xlabel("Image-position residual (arcsec)")
    ax.set_ylabel("Number of images")
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    return path


def plot_image_recovery(image_df: pd.DataFrame, extra_df: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    x_obs, y_obs = image_df["x_obs_arcsec"].to_numpy(float), image_df["y_obs_arcsec"].to_numpy(float)
    x_model = image_df["x_model_q50"].fillna(image_df["x_model_arcsec"]).to_numpy(float)
    y_model = image_df["y_model_q50"].fillna(image_df["y_model_arcsec"]).to_numpy(float)
    recovered = (image_df["image_recovery_status"].astype(str) == "recovered").to_numpy()
    ax.scatter(x_obs[recovered], y_obs[recovered], marker="x", color=POINT_COLOR, s=18, linewidths=0.9, label="recovered")
    if (~recovered).any():
        ax.scatter(x_obs[~recovered], y_obs[~recovered], marker="x", color=MISSED_COLOR, s=18, linewidths=0.9, label="not recovered")
    finite = recovered & np.isfinite(x_model) & np.isfinite(y_model)
    if finite.any():
        xerr = [np.maximum(0, x_model - image_df["x_model_q16"].to_numpy(float))[finite], np.maximum(0, image_df["x_model_q84"].to_numpy(float) - x_model)[finite]]
        yerr = [np.maximum(0, y_model - image_df["y_model_q16"].to_numpy(float))[finite], np.maximum(0, image_df["y_model_q84"].to_numpy(float) - y_model)[finite]]
        ax.errorbar(x_model[finite], y_model[finite], xerr=np.nan_to_num(xerr), yerr=np.nan_to_num(yerr), fmt="o", color=POINT_COLOR,
                    alpha=0.75, markersize=3, elinewidth=0.8, capsize=1.5, label="model")
        for xo, yo, xm, ym in zip(x_obs[finite], y_obs[finite], x_model[finite], y_model[finite]):
            ax.plot([xo, xm], [yo, ym], color="0.6", lw=0.7)
    if extra_df is not None and not extra_df.empty:
        ax.scatter(extra_df["x_model_arcsec"], extra_df["y_model_arcsec"], marker="o", color="tab:blue", s=16, linewidths=0.0, label="extra")
    ax.invert_xaxis()
    ax.set_xlabel("x (arcsec)")
    ax.set_ylabel("y (arcsec)")
    ax.legend(loc="best", fontsize=8)
    residual = image_df["image_residual_q50"].fillna(image_df["image_residual_arcsec"]).to_numpy(float)
    index = np.arange(len(image_df))
    ok = np.isfinite(residual)
    if ok.any():
        yerr = [np.maximum(0, residual - image_df["image_residual_q16"].to_numpy(float))[ok], np.maximum(0, image_df["image_residual_q84"].to_numpy(float) - residual)[ok]]
        axes[1].errorbar(index[ok], residual[ok], yerr=np.nan_to_num(yerr), fmt="o", color="tab:blue", markersize=3)
    axes[1].set_xlabel("image index")
    axes[1].set_ylabel("image residual (arcsec)")
    if len(image_df) <= 40:
        axes[1].set_xticks(index)
        axes[1].set_xticklabels(image_df["image_label"].astype(str), rotation=90, fontsize=7)
    fig.tight_layout()
    return _finish(fig, path, dpi=180)


# ----------------------------------------------------------------------------- posterior
def plot_corner(samples: np.ndarray, specs: list[ParameterSpec], path: Path, *, grouped_physical=None) -> Path | None:
    from .corner import plot_parameter_corner
    keep = [(i, spec, spec.name) for i, spec in enumerate(specs)
            if spec.component_family not in {"source_position", "member_scatter_offset", "member_shape_offset"}]
    return plot_parameter_corner(samples, keep, path, grouped_physical=grouped_physical)


def member_scaling_table(state: ModelState, posterior: PosteriorResults) -> pd.DataFrame:
    """Posterior summary of every member's velocity dispersion, radii and mass, with its class."""
    physical = np.asarray(posterior.physical, dtype=float)
    spec_packed = state.packed_lens_spec
    exact = set(state.members.exact_positions.tolist())
    import jax
    dev_v, dev_r = (np.asarray(a) for a in jax.vmap(lambda row: member_log_deviations(spec_packed, row))(physical))   # (draws, components)
    rows = []
    for position, record in enumerate(state.scaling_component_records):
        c = int(record["component_index"])
        pop = record["potfile_id"]
        def value(index, base):                                    # a fixed relation parameter has index -1
            return physical[:, int(index[c])] if int(index[c]) >= 0 else np.full(len(physical), float(base[c]))
        sigma_ref = value(spec_packed.sigma_ref_param_index, spec_packed.sigma_ref_base)
        cut_ref = value(spec_packed.cut_ref_param_index, spec_packed.cut_ref_base)
        core_ref = value(spec_packed.core_ref_param_index, spec_packed.core_ref_base)
        alpha = value(spec_packed.alpha_sigma_param_index, spec_packed.alpha_sigma_base)
        beta = value(spec_packed.beta_radius_param_index, spec_packed.beta_radius_base)
        lum = float(spec_packed.luminosity_ratio[c])
        v_disp = sigma_ref * lum ** alpha * np.exp(dev_v[:, c])
        v_scaling = sigma_ref * lum ** alpha
        core, cut = core_ref * lum ** beta * np.exp(dev_r[:, c]), cut_ref * lum ** beta * np.exp(dev_r[:, c])
        # The dPIE total mass; a cored power-law member has the same mass inside its cut radius, not the same total.
        mass = log10_dpie_mass_msun(v_disp, cut)
        row = {"potfile_id": pop, "catalog_id": record["catalog_id"], "component_index": c, "catalog_mag": record["catalog_mag"],
               "luminosity_ratio": lum, "member_class": "exact" if position in exact else "far-field",
               "centered": bool(state.members.centered[position]) if len(state.members.centered) else False,
               "x_centre": record["x_centre"], "y_centre": record["y_centre"]}
        for name, values in (("v_disp", v_disp), ("v_disp_scaling", v_scaling), ("core_radius_kpc", core), ("cut_radius_kpc", cut),
                             ("log10_mass_msun", mass), ("velocity_offset", dev_v[:, c]), ("radius_offset", dev_r[:, c])):
            q16, q50, q84 = np.percentile(values, [16, 50, 84])
            row.update({f"{name}_p16": q16, f"{name}_median": q50, f"{name}_p84": q84})
        rows.append(row)
    return pd.DataFrame(rows)


def population_scaling_draws(table: pd.DataFrame, posterior_physical: np.ndarray,
                             specs: list[ParameterSpec], fixed: dict[str, float] | None = None) -> dict:
    """Joint population relations, without individual velocity/radius deviations.  ``fixed`` gives the value of a
    relation field that is not sampled (``"<population>.<field>"``: held fixed, e.g. a Lenstool ``vdslope 0``)."""
    fixed = fixed or {}
    names = {spec.name: i for i, spec in enumerate(specs)}
    result = {}
    for population, members in table.groupby("potfile_id", sort=True):
        # Recover the saved pivot, without depending on a possibly changed config.
        pivots = members["catalog_mag"].to_numpy() + 2.5 * np.log10(members["luminosity_ratio"].to_numpy())
        if not np.isfinite(pivots).all() or not np.allclose(pivots, pivots[0], rtol=0, atol=1e-8):
            raise ValueError(f"Inconsistent luminosity pivot for {population}")
        magnitude = np.linspace(members.catalog_mag.min(), members.catalog_mag.max(), 240)
        lum = 10. ** (-.4 * (magnitude - pivots[0]))
        def draws(field):
            key = f"{population}.{field}"
            if key not in names:
                if key not in fixed:
                    raise KeyError(f"{key} is neither sampled nor given as fixed")
                return np.full((posterior_physical.shape[0], 1), float(fixed[key]))
            return posterior_physical[:, names[key]][:, None]
        velocity = draws("sigma") * lum ** draws("alpha_sigma")
        radius = draws("cutkpc") * lum ** draws("beta_radius")
        result[population] = (magnitude, {"v_disp": velocity, "cut_radius_kpc": radius,
                                        "log10_mass_msun": log10_dpie_mass_msun(velocity, radius)})
    return result


@plt.rc_context({"font.family": "sans-serif", "font.size": 11, "axes.labelsize": 12,
                 "mathtext.fontset": "dejavusans", "mathtext.default": "regular",
                 "axes.linewidth": .9, "xtick.labelsize": 10, "ytick.labelsize": 10,
                 "xtick.major.width": .8, "ytick.major.width": .8,
                 "xtick.minor.width": .6, "ytick.minor.width": .6,
                 "xtick.major.size": 4, "ytick.major.size": 4,
                 "xtick.top": False, "ytick.right": False, "axes.grid": False,
                 "pdf.fonttype": 42})
def plot_scaling_relation(table: pd.DataFrame, posterior_physical: np.ndarray, specs: list[ParameterSpec], path: Path,
                          fixed: dict[str, float] | None = None) -> Path:
    from matplotlib.patches import Patch
    fields = [("v_disp", r"$\sigma$ (km s$^{-1}$)", True),
              ("cut_radius_kpc", r"$r_{\rm cut}$ (kpc)", True),
              ("log10_mass_msun", r"$\log_{10}(M/M_\odot)$", False)]
    relations = population_scaling_draws(table, np.asarray(posterior_physical), specs, fixed)
    names = {spec.name: i for i, spec in enumerate(specs)}
    pivots = {}
    for population, members in table.groupby("potfile_id"):
        member = members.iloc[0]
        pivot = float(member.catalog_mag + 2.5 * np.log10(member.luminosity_ratio))
        velocity = posterior_physical[:, names[f"{population}.sigma"]]
        radius = posterior_physical[:, names[f"{population}.cutkpc"]]
        pivots[population] = (pivot, {"v_disp": np.median(velocity), "cut_radius_kpc": np.median(radius),
                                     "log10_mass_msun": np.median(log10_dpie_mass_msun(velocity, radius))})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharex=True)
    handles = []
    relation_colors = ["black", "#77529B", "#237A66"]
    for ax, (field, label, logscale) in zip(axes, fields):
        for idx, (population, (mag, curves)) in enumerate(relations.items()):
            color = relation_colors[idx % len(relation_colors)]
            lo, med, hi = np.quantile(curves[field], [.16, .5, .84], axis=0)
            ax.fill_between(mag, lo, hi, color=color, alpha=.06, linewidth=0, zorder=1)
            line, = ax.plot(mag, med, color=color, lw=1.6, zorder=4,
                            label="Population relation" if len(relations) == 1 else str(population))
            if ax is axes[0]:
                handles.append(line)
            pivot, pivot_values = pivots[population]
            relative_luminosity = 10. ** (-.4 * (mag - pivot))
            if field == "v_disp":
                reference = pivot_values[field] * relative_luminosity ** .25
                reference_label = "Faber–Jackson"
            elif field == "cut_radius_kpc":
                reference = pivot_values[field] * relative_luminosity ** .5
                reference_label = "Faber–Jackson + constant M/L"
            else:
                reference = pivot_values[field] + np.log10(relative_luminosity)
                reference_label = "Constant M/L"
            reference_line, = ax.plot(mag, reference, color="tab:red", lw=1.3,
                                     ls=(0, (5, 3)), zorder=5, label=reference_label)
            if idx == 0:
                ax.legend(handles=[reference_line], loc="upper left", fontsize=8,
                          frameon=False, handlelength=2.3, handletextpad=.6)
            pivot_marker, = ax.plot(pivot, pivot_values[field], marker="o", ms=5.5, mfc="white",
                                   mec=color, mew=1.3, linestyle="none", zorder=10, label=r"$m_\star$")
            if ax is axes[0] and idx == 0:
                handles.append(pivot_marker)
        for kind, marker, color in (("far-field", "o", "tab:blue"), ("exact", "D", "tab:orange")):
            sub = table[table.member_class == kind]
            if sub.empty:
                continue
            yerr = np.array([sub[f"{field}_median"] - sub[f"{field}_p16"],
                             sub[f"{field}_p84"] - sub[f"{field}_median"]])
            container = ax.errorbar(sub.catalog_mag, sub[f"{field}_median"], yerr=yerr,
                fmt=marker, color=color, ecolor=matplotlib.colors.to_rgba(color, .23),
                markersize=3.1, markeredgewidth=0, elinewidth=.55, capsize=0,
                zorder=3 if kind == "exact" else 2,
                label=f"{kind.capitalize()} members")
            # Raise only the markers; retain the lighter error bars underneath the curves.
            container.lines[0].set_zorder(9 if kind == "exact" else 8)
            if ax is axes[0]:
                handles.append(container)
        if logscale:
            ax.set_yscale("log")
        ax.set_xlabel("Catalog magnitude")
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(False)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.margins(y=.10)
    lo, hi = table.catalog_mag.min(), table.catalog_mag.max()
    axes[0].set_xlim(hi + .03 * (hi-lo), lo - .03 * (hi-lo))
    handles.append(Patch(facecolor=relation_colors[0], alpha=.06, edgecolor="none", label="16–84% relation interval"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1.0), ncol=5,
               fontsize=9, frameon=False, handletextpad=.6, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, .91), w_pad=2.)
    return _finish(fig, path, dpi=250)


# ----------------------------------------------------------------------------- critical curves
def plot_critical_curves(state: ModelState, draws, z_source: float, path: Path, *, half_fov_arcsec: float | None = None,
                         grid_scale_arcsec: float = 0.5, log=print) -> Path:
    """Critical curves and caustics over the reporting draws (parametric plus each draw's GRF conditional mean): the zero
    contours of the per-pixel median tangential (solid) and radial (dashed) eigenvalues 1 - kappa -/+ |gamma|, shaded
    where their 16-84% range contains zero; the caustics are the per-point median over draws of the median curves
    mapped through each draw's lens; the sources are the medians over draws with their 16-84% range.  The grid spans
    ``half_fov_arcsec`` about the reference (default: the farthest image plus 20", rounded up to 10")."""
    from .jwst_rgb import draw_critical_interval
    if half_fov_arcsec is None:
        # The images and 20" around them, in steps of 10": the grid covers the constrained region, not empty sky.
        reach = float(np.max(np.abs(np.concatenate([state.images.x_obs, state.images.y_obs]))))
        half_fov_arcsec = 10.0 * np.ceil((reach + 20.0) / 10.0)
    n = max(2, int(round(2 * half_fov_arcsec / grid_scale_arcsec)) + 1)
    axis = np.linspace(-half_fov_arcsec, half_fov_arcsec, n)
    xx, yy = np.meshgrid(axis, axis)
    maps = draws.grid_maps("critical curves", xx, yy, z_source, log=log)
    shear = np.hypot(maps["gamma1"], maps["gamma2"])
    fig, (image_ax, source_ax) = plt.subplots(1, 2, figsize=(12, 5))
    pixel = np.arange(n, dtype=float)
    curves = []
    for eigenvalue, linestyle in ((1.0 - maps["kappa"] - shear, "-"), (1.0 - maps["kappa"] + shear, "--")):
        low, median, high = np.nanpercentile(eigenvalue, [16, 50, 84], axis=0)
        draw_critical_interval(image_ax, low, high, CRITICAL_COLOR, coordinates=(xx, yy))
        for vertices in find_contours(median, 0.0):
            cx, cy = np.interp(vertices[:, 1], pixel, axis), np.interp(vertices[:, 0], pixel, axis)
            image_ax.plot(cx, cy, color=CRITICAL_COLOR, linestyle=linestyle, linewidth=1.0)
            curves.append(np.stack((cx, cy), -1))
    if curves:
        xy = np.concatenate(curves)
        beta = np.median([xy - draws.deflections(k, xy, z_source) for k in range(len(draws))], axis=0)
        source_ax.scatter(beta[:, 0], beta[:, 1], color=CRITICAL_COLOR, s=2, alpha=0.75, linewidths=0.0)
    evaluator = draws.evaluator
    sources = np.stack([np.asarray(evaluator.source_positions(jnp.asarray(theta))) for theta in draws.samples])
    s16, s50, s84 = np.percentile(sources, [16, 50, 84], axis=0)
    cmap = plt.get_cmap("tab20", max(1, len(state.family_data)))
    for idx, family in enumerate(state.family_data):
        image_ax.scatter(family.x_obs, family.y_obs, color=cmap(idx), s=14, label=family.family_id)
    for idx, family_id in enumerate(state.images.family_ids):
        color_index = [f.family_id for f in state.family_data].index(family_id)
        source_ax.errorbar(s50[idx, 0], s50[idx, 1], xerr=[[s50[idx, 0] - s16[idx, 0]], [s84[idx, 0] - s50[idx, 0]]],
                           yerr=[[s50[idx, 1] - s16[idx, 1]], [s84[idx, 1] - s50[idx, 1]]], fmt="o", ms=3.5,
                           color=cmap(color_index), elinewidth=0.8, capsize=0)
    image_ax.invert_xaxis()
    image_ax.set_xlabel("x (arcsec)")
    image_ax.set_ylabel("y (arcsec)")
    source_ax.set_xlabel(r"$\beta_x$ (arcsec)")
    source_ax.set_ylabel(r"$\beta_y$ (arcsec)")
    fig.tight_layout()
    return _finish(fig, path, dpi=180)


# ----------------------------------------------------------------------------- truth recovery
def plot_truth_recovery_residuals(recovery: dict[str, Any], image_df: pd.DataFrame | None, path: Path) -> Path | None:
    truth, model = recovery["truth"], recovery["model"]
    if "abs_mu" not in truth:
        return None
    x, y = recovery["x"], recovery["y"]

    def fractional(m, t):
        valid = np.isfinite(m) & np.isfinite(t) & (t > 0)
        out = np.full(t.shape, np.nan)
        np.divide(m - t, t, out=out, where=valid)
        return out

    residuals = (fractional(model["kappa"]["median"], truth["kappa"]), fractional(model["abs_mu"]["median"], truth["abs_mu"]))
    norm = Normalize(vmin=-1.0, vmax=2.0, clip=True)
    # Crop the symmetric [-2, 2] palette, rather than stretching its blue half.
    cmap = LinearSegmentedColormap.from_list(
        "RdBu_r_cropped", plt.get_cmap("RdBu_r")(np.linspace(0.25, 1.0, 768)), N=768)
    cmap.set_bad("#e8e8e8")
    with plt.rc_context(fname=STYLE, rc=PAPER_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.17, top=0.79, wspace=0.07)
        meshes = [ax.pcolormesh(x, y, np.ma.masked_invalid(r), cmap=cmap, norm=norm, shading="nearest", rasterized=True) for ax, r in zip(axes[:2], residuals)]
        handles = []
        for det, color, width, label in ((truth["detA"], "#252525", 0.85, "Truth"), (model["detA"]["median"], "#d55e00", 0.7, "Model")):
            finite = det[np.isfinite(det)]
            if finite.size and finite.min() <= 0.0 <= finite.max():
                axes[2].contour(x, y, np.ma.masked_invalid(det), levels=[0.0], colors=[color], linewidths=[width])
                handles.append(Line2D([], [], color=color, linewidth=width, label=label))
        if image_df is not None and not image_df.empty:
            positions = image_df[["x_obs_arcsec", "y_obs_arcsec"]].to_numpy(float)
            for ax in axes:
                ax.scatter(*positions.T, s=4, facecolors="none", edgecolors="white", linewidths=0.9, zorder=5)
                ax.scatter(*positions.T, s=4, facecolors="none", edgecolors="#333333", linewidths=0.35, zorder=6)
            handles.append(Line2D([], [], marker="o", markersize=3, markerfacecolor="none", markeredgecolor="#333333", linestyle="none", label="Images"))
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (arcsec)")
            ax.tick_params(which="both", direction="in", top=True, right=True)
        axes[0].set_ylabel("y (arcsec)")
        if handles:
            axes[2].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3, frameon=False, fontsize=8, handlelength=1.3, columnspacing=0.7)
        fig.canvas.draw()
        for ax, mesh, label in zip(axes[:2], meshes, (r"$(\kappa_{\rm model}-\kappa_{\rm true})/\kappa_{\rm true}$", r"$(|\mu_{\rm model}|-|\mu_{\rm true}|)/|\mu_{\rm true}|$")):
            box = ax.get_position()
            cax = fig.add_axes([box.x0, box.y1 + 0.045, box.width, 0.035])
            cb = fig.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[-1, 0, 1, 2])
            cb.set_label(label, fontsize=8, labelpad=5)
            cb.ax.xaxis.set_label_position("top")
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.minorticks_off()
            cb.ax.tick_params(direction="in", length=3.5, width=0.8, labelsize=8, pad=2)
        return _finish(fig, path, dpi=600)


# Published FF-SIMS reconstructions drawn in the ratio panels: (name, approach in Table 1 of Meneghetti et al. 2017,
# colour).  Only the parametric models are drawn (the free-form ones stay in the digitized table); every model has its
# own colour, none of them blue (the colour of this work); a model is drawn only when the digitized table has it.
LITERATURE_MODELS = (("CATS", "Parametric", "#D55E00"), ("GLAFIC", "Parametric", "#009E73"),
                     ("Johnson-Sharon", "Parametric", "#CC79A7"), ("Zitrin-LTM-gauss", "Parametric", "#E69F00"),
                     ("Zitrin-NFW", "Parametric", "#882255"))
LITERATURE_GROUP_STYLE = {"Parametric": "-"}


def draw_literature_ratios(ax, literature: pd.DataFrame | None) -> dict:
    """Draw the digitized published ratios on ``ax`` (``smoothed_digitized_curve`` of the clean and partly hidden samples),
    returning the drawn lines by approach, in ``LITERATURE_MODELS`` order."""
    if literature is None:
        return {}
    groups = {}
    for name, group, color in LITERATURE_MODELS:
        curve = literature[(literature["model"] == name) & (literature["flag"] <= 1)]
        if len(curve) < 2:
            continue
        line, = ax.plot(*smoothed_digitized_curve(curve["r_arcsec"], curve["ratio"]), color=color,
                        linestyle=LITERATURE_GROUP_STYLE[group], linewidth=0.9, label=name, zorder=1.5)
        groups.setdefault(group, []).append(line)
    return groups


def literature_legends(fig, groups: dict, top: float) -> None:
    """One legend per approach above ``top`` (figure fraction), titled with the approach, stacked downward."""
    for group, lines in groups.items():
        legend = fig.legend(handles=lines, loc="lower center", bbox_to_anchor=(0.55, top), ncol=3,
                            title=f"{group} (Meneghetti et al. 2017)", title_fontsize=plt.rcParams["legend.fontsize"], handlelength=1.8, columnspacing=0.9,
                            labelspacing=0.25, borderaxespad=0.0)
        fig.canvas.draw()
        top = legend.get_window_extent().transformed(fig.transFigure.inverted()).y1 + 0.005


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_truth_ratio_panels(profile: pd.DataFrame, kappa_profile: pd.DataFrame, center: dict[str, Any], image_df: pd.DataFrame | None,
                            path: Path, literature_mass: pd.DataFrame | None = None, literature_kappa: pd.DataFrame | None = None) -> Path:
    """Model over truth for the enclosed projected mass (upper panel) and the azimuthally averaged convergence (lower panel)
    against radius, in the style of the enclosed-mass figure: posterior median and 16-84 per cent band, the published
    ratios (``draw_literature_ratios``, in one legend inside the mass panel) and ticks at the image radii."""
    fig, (ax_m, ax_k) = plt.subplots(2, 1, figsize=(3.5, 3.7), sharex=True, gridspec_kw={"hspace": 0.06})
    radii = None
    if image_df is not None and not image_df.empty and center is not None:
        radii = np.hypot(image_df["x_obs_arcsec"].to_numpy(float) - center["center_x_arcsec"], image_df["y_obs_arcsec"].to_numpy(float) - center["center_y_arcsec"])
    panels = ((ax_m, profile, "radius_arcsec", "m2d_ratio", literature_mass, r"$M_{\rm model}/M_{\rm true}$", (0.9, 1.1), [0.9, 0.95, 1.0, 1.05, 1.1]),
              (ax_k, kappa_profile, "radius_arcsec", "kappa_ratio", literature_kappa, r"$\kappa_{\rm model}/\kappa_{\rm true}$", (0.8, 1.2), [0.8, 0.9, 1.0, 1.1, 1.2]))
    groups = {}
    for ax, table, rcol, prefix, literature, label, ylim, yticks in panels:
        radius = table[rcol].to_numpy(float)
        lo, mid, hi = (table[f"{prefix}_{k}"].to_numpy(float) for k in ("q16", "median", "q84"))
        band = np.isfinite(lo) & np.isfinite(hi)
        if band.any():
            ax.fill_between(radius[band], lo[band], hi[band], color="#2878A5", alpha=0.22, linewidth=0.0)
        drawn = draw_literature_ratios(ax, literature)
        if sum(map(len, drawn.values())) > sum(map(len, groups.values())):
            groups = drawn
        ours, = ax.plot(radius, mid, color="#2878A5", linewidth=1.4, label="This work", zorder=2)
        ax.axhline(1.0, color="0.2", linestyle="--", linewidth=1.0)
        if radii is not None:
            ax.vlines(radii[radii > 0], 0.0, 0.10, transform=ax.get_xaxis_transform(), color="0.20", linewidth=0.85, alpha=0.9, zorder=5)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_ylabel(label)
    ax_m.set_xscale("log")
    shown = np.concatenate([profile["radius_arcsec"].to_numpy(float), kappa_profile["radius_arcsec"].to_numpy(float)])
    ax_k.set_xlim(PROFILE_INNER_ARCSEC, np.nanmax(shown) * 1.03)
    ax_k.set_xlabel(r"$R$ (arcsec)")
    # One legend in the upper left of the mass panel, on a mostly opaque box drawn above the curves.
    legend = ax_m.legend(handles=[ours, *[line for lines in groups.values() for line in lines]], loc="upper left", ncol=2,
                         frameon=True, framealpha=0.85, facecolor="white", edgecolor="none", handlelength=1.6,
                         columnspacing=0.8, labelspacing=0.2, borderpad=0.3, borderaxespad=0.3)
    legend.set_zorder(10)
    fig.align_ylabels((ax_m, ax_k))
    return _finish(fig, path, dpi=220)


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_m2d_aperture_ratio(profile: pd.DataFrame, center: dict[str, Any], image_df: pd.DataFrame | None, mass_scale: float, path: Path,
                            literature: pd.DataFrame | None = None) -> Path:
    """Enclosed projected mass of model and truth against aperture radius, and their ratio.  ``literature``
    (``truth_recovery.literature_m2d_ratios``) adds the digitized published ratios to the ratio panel
    (``draw_literature_ratios``), listed by approach in the legend."""
    fig, (ax, ax_resid) = plt.subplots(2, 1, figsize=(3.5, 3.7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.06})
    radius = profile["radius_arcsec"].to_numpy(float)
    truth = profile["kappa_true_sum"].to_numpy(float) * mass_scale
    model = profile["kappa_model_sum"].to_numpy(float) * mass_scale
    q16, q84 = profile["kappa_model_sum_q16"].to_numpy(float) * mass_scale, profile["kappa_model_sum_q84"].to_numpy(float) * mass_scale
    ratio = profile["m2d_ratio_median"].to_numpy(float)
    band = np.isfinite(q16) & np.isfinite(q84) & (q16 > 0)
    if band.any():
        ax.fill_between(radius[band], q16[band], q84[band], color="#2878A5", alpha=0.22, linewidth=0.0)
    ax.plot(radius, model, color="#2878A5", linewidth=1.8, label="Model")
    ax.plot(radius, truth, color="0.2", linestyle="--", linewidth=1.3, label="Truth")
    if image_df is not None and not image_df.empty and center is not None:
        radii = np.hypot(image_df["x_obs_arcsec"].to_numpy(float) - center["center_x_arcsec"], image_df["y_obs_arcsec"].to_numpy(float) - center["center_y_arcsec"])
        ax.vlines(radii[radii > 0], 0.0, 0.10, transform=ax.get_xaxis_transform(), color="0.20", linewidth=0.85, alpha=0.9, zorder=5)
    r16, r84 = profile["m2d_ratio_q16"].to_numpy(float), profile["m2d_ratio_q84"].to_numpy(float)
    rband = np.isfinite(r16) & np.isfinite(r84)
    if rband.any():
        ax_resid.fill_between(radius[rband], r16[rband], r84[rband], color="#2878A5", alpha=0.22, linewidth=0.0)
    literature_groups = draw_literature_ratios(ax_resid, literature)
    ax_resid.plot(radius, ratio, color="#2878A5", linewidth=1.3)
    ax_resid.axhline(1.0, color="0.2", linestyle="--", linewidth=1.0)
    ax_resid.set_ylim(0.9, 1.1)
    ax_resid.set_yticks([0.9, 0.95, 1.0, 1.05, 1.1])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax_resid.set_xlim(left=PROFILE_INNER_ARCSEC)
    ax_resid.set_xlabel(r"$R$ (arcsec)")
    ax_resid.set_ylabel(r"$M_{\rm model}/M_{\rm true}$")
    ax.set_ylabel(r"$M_{\rm 2D}(<R)$ ($M_\odot$)")
    shown = np.isfinite(truth) & np.isfinite(model) & (radius >= PROFILE_INNER_ARCSEC - 0.1) & (truth > 0) & (model > 0)
    if shown.any():
        values = np.concatenate([truth[shown], model[shown]])
        ax.set_ylim(values.min() / 1.35, values.max() * 1.35)
        ax.set_xlim(PROFILE_INNER_ARCSEC, radius[shown].max() * 1.03)
    ax.legend(loc="upper left", handlelength=2.3)
    if literature_groups:
        literature_legends(fig, literature_groups, ax.get_position().y1 + 0.008)
    return _finish(fig, path, dpi=220)
