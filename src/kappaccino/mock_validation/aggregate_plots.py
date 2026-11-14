"""Aggregate figures of a mock-validation campaign, from each seed's recovery.json."""
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
from scipy.integrate import quad
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter, NullLocator

from ..plotting.style import (APJ_COLUMN_WIDTH_IN, APJ_TEXT_WIDTH_IN, BLUE, DASHES, FIGURE_STYLE, GREEN, ORANGE, PAPER_STYLE, PINK,
                              YELLOW, save_page)
from .aperture import COMPONENT_GROUPS
from .runner import CONVERGENCE_MIN_NEFF, CONVERGENCE_RHAT, convergence_verdict, RECOVERY_FILE

STYLE = Path(__file__).resolve().parent.parent / "plotting" / "style.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))

COLOR = BLUE
TRUTH_COLOR = "0.2"
COMPONENT_COLORS = {"halo": BLUE, "bcg": ORANGE, "members": GREEN, "grf": PINK, "gp": "tab:red", "images": "tab:gray"}
COMPONENT_LABELS = {"halo": "Main halo", "bcg": "BCG", "members": "Members"}
PAPER_PARAMETERS = (
    "halo.log10_mass_msun", "halo.log10_aperture_mass_msun", "halo.x_centre", "halo.y_centre", "halo.v_disp", "halo.core_radius_kpc", "halo.cut_radius_kpc",
    "bcg.log10_mass_msun", "bcg.x_centre", "bcg.y_centre", "bcg.v_disp", "bcg.core_radius_kpc", "bcg.cut_radius_kpc",
    "images.sigma_grf", "images.lambda_b", "images.gp_knee_gamma",
    "images.sigma_grf2", "images.lambda_b2", "images.gp_knee_beta_low",
    "images.sigma_gp", "images.ell_gp", "images.sigma_meas",
    "potfile.log10_mass_msun", "potfile.sigma", "potfile.cutkpc", "potfile.alpha_sigma", "potfile.beta_radius", "potfile.gamma_ml",
)
PARAMETER_LABELS = {
    "sigma_grf": r"$\sigma_{\rm GRF}$", "lambda_b": r"$\lambda_{\rm b}$",
    "gp_knee_gamma": r"$\gamma_{\rm GRF}$", "sigma_grf2": r"$\sigma_{\alpha,2}$",
    "lambda_b2": r"$\lambda_{\rm b,2}$", "gp_knee_beta_low": r"$\beta_{\rm low}$",
    "sigma_gp": r"$\sigma_{\rm GP}$", "ell_gp": r"$\ell_{\rm GP}$",
    "sigma_meas": r"$\sigma_{\rm meas}$",
    "log10_aperture_mass_msun": r"$\log_{10} M_{\rm ap}$",
    "log10_mass_msun": r"$\log_{10} M_{\rm tot}$", "x_centre": r"$x$", "y_centre": r"$y$", "v_disp": r"$\sigma$", "core_radius_kpc": r"$r_{\rm core}$",
    "cut_radius_kpc": r"$r_{\rm cut}$", "sigma": r"$\sigma^\star$", "cutkpc": r"$r_{\rm cut}^\star$", "alpha_sigma": r"$\alpha$",
    "beta_radius": r"$\beta$", "gamma_ml": r"$\gamma_{M/L}$",
}
PARAMETER_UNITS = {"x_centre": "arcsec", "y_centre": "arcsec", "v_disp": "km/s", "core_radius_kpc": "kpc", "cut_radius_kpc": "kpc",
                   "sigma": "km/s", "cutkpc": "kpc", "log10_mass_msun": r"$M_\odot$"}
PARAMETER_UNITS.update({field: "arcsec" for field in (
    "sigma_grf", "lambda_b", "sigma_grf2", "lambda_b2", "sigma_gp", "ell_gp", "sigma_meas")})
COMPONENT_NAMES = {"halo": "Halo", "bcg": "BCG", "potfile": "Members",
                   "grf": "GRF", "gp": "Image-plane GP", "images": "Independent scatter"}


def _parameter_component(parameter):
    if parameter.startswith(("images.sigma_grf", "images.lambda_b", "images.gp_knee_")):     # the field: sigma_GRF, its second knee, lambda_b, slope
        return "grf"
    if parameter in ("images.sigma_gp", "images.ell_gp"):
        return "gp"
    return parameter.split(".")[0]




def _polish(ax):
    ax.grid(False)


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)
    return path


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


def flat_fit(fit: dict) -> dict:
    """The run's fit summary with each median-and-range entry ``{"median", "q16", "q84"}`` (over the reporting draws)
    as ``<key>`` (the median), ``<key>_q16`` and ``<key>_q84``."""
    out = {}
    for key, value in fit.items():
        if isinstance(value, dict) and "median" in value:
            out.update({key: value["median"], f"{key}_q16": value.get("q16"), f"{key}_q84": value.get("q84")})
        else:
            out[key] = value
    return out


def load_campaign(root: str | Path, *, converged_only: bool = False) -> Campaign:
    """All completed recovery tables by default, with convergence flags retained. With ``converged_only`` realizations whose sampler did not converge
    (max R-hat >= 1.1 or min n_eff < 50, recomputed from saved diagnostics) are left out of every table except the
    summary, which keeps all of them with their ``converged`` flag and reasons."""
    root = Path(root)
    files = sorted(root.glob("seed_*/recovery/" + RECOVERY_FILE))
    if not files:
        raise FileNotFoundError(f"No seed_*/recovery/{RECOVERY_FILE} under {root}")
    seeds, parameters, magnification, draws, subhalos, m2d, summary = [], [], [], [], [], [], []
    for path in files:
        payload = json.loads(path.read_text())
        seed = int(payload["seed"])
        converged, reasons = convergence_verdict({"max_r_hat": payload.get("max_r_hat"), "min_n_eff": payload.get("min_n_eff")})
        summary.append({"seed": seed, "mode": payload.get("mode"), "converged": converged, "convergence_reasons": "; ".join(reasons),
                        "max_r_hat": payload.get("max_r_hat"), "min_n_eff": payload.get("min_n_eff"),
                        "divergences": payload.get("divergences"), "audit": payload.get("audit"), "coverage_68": payload.get("coverage_68"),
                        "coverage_95": payload.get("coverage_95"), "scored_parameters": payload.get("scored_parameters"), **flat_fit(payload.get("fit", {}))})
        if converged_only and not converged:
            continue
        seeds.append(seed)
        for key, store in (("parameters", parameters), ("magnification", magnification), ("image_residual_draws", draws), ("m2d", m2d)):
            store.append(pd.DataFrame(payload[key]).assign(seed=seed))
        subhalos.append({"seed": seed, **payload["subhalos"]})
    if not seeds:
        raise ValueError(f"No converged realization under {root}")
    parameter_table = pd.concat(parameters, ignore_index=True)
    aperture_table = halo_aperture_parameters(pd.concat(m2d, ignore_index=True))
    if not aperture_table.empty:
        parameter_table = pd.concat([parameter_table, aperture_table], ignore_index=True)
    return Campaign(root, seeds, parameter_table, pd.concat(magnification, ignore_index=True),
                    pd.concat(draws, ignore_index=True), subhalos, pd.concat(m2d, ignore_index=True), pd.DataFrame(summary))



def halo_aperture_parameters(m2d: pd.DataFrame) -> pd.DataFrame:
    """Use the largest saved aperture inside each mock's outermost true image.

    Radii and centres are fixed by truth, not fitted. Transform saved mass
    quantiles monotonically to log mass; never invent missing 95% intervals.
    """
    if m2d.empty or "outer_image_radius_kpc" not in m2d:
        return pd.DataFrame()
    rows = []
    for seed, group in m2d[m2d.component == "halo"].groupby("seed"):
        inside = group[(group.radius_kpc > 0) & (group.radius_kpc <= group.outer_image_radius_kpc)]
        if inside.empty:
            continue
        aperture = inside.sort_values("radius_kpc").iloc[-1]
        values = [aperture.get(k, np.nan) for k in
                  ("truth_m2d_msun", "model_q16", "model_q50", "model_q84")]
        if not np.all(np.isfinite(values)) or np.min(values) <= 0:
            continue
        truth, lo, median, hi = np.log10(values)
        q025, q975 = [np.log10(aperture[k]) if k in aperture and aperture[k] > 0 else np.nan
                      for k in ("model_q025", "model_q975")]
        rows.append(dict(seed=seed, parameter="halo.log10_aperture_mass_msun",
                         truth=truth, median=median, q16=lo, q84=hi,
                         q025=q025, q975=q975, bias=median-truth,
                         covered_68=bool(lo <= truth <= hi),
                         covered_95=bool(q025 <= truth <= q975) if np.isfinite(q025+q975) else np.nan,
                         scored=True, aperture_radius_kpc=float(aperture.radius_kpc),
                         outer_image_radius_kpc=float(aperture.outer_image_radius_kpc),
                         aperture_definition="Largest saved radius inside outermost true image; true halo centre"))
    return pd.DataFrame(rows)


def representative_seed(campaign: Campaign) -> int:
    """The seed with the smallest maximum R-hat (the first seed when R-hat is unavailable)."""
    summary = campaign.summary[campaign.summary["seed"].isin(campaign.seeds)].copy()
    summary["max_r_hat"] = pd.to_numeric(summary["max_r_hat"], errors="coerce")
    ordered = summary.sort_values(["max_r_hat", "seed"], na_position="last")
    return int(ordered["seed"].iloc[0])


# ----------------------------------------------------------------------------- parameter recovery
def _axis_label(parameter: str, role: str) -> str:
    component, field = parameter.split(".", 1)
    label = PARAMETER_LABELS.get(field, field)
    if field == "log10_mass_msun":
        label = r"$\log_{10}(M_{\rm tot}/M_\odot)$"
    if field == "log10_aperture_mass_msun":
        label = r"$\log_{10}(M_{\rm ap}/M_\odot)$"
    unit = PARAMETER_UNITS.get(field)
    unit = None if field == "log10_mass_msun" else unit
    prefix = "True" if role == "truth" else "Fit"
    return f"{prefix} {label} ({unit})" if unit else f"{prefix} {label}"


def _draw_parameter_axis(ax, panel: pd.DataFrame, *, binned_color: str, binned_size: float = 3.4, point_color: str = COLOR) -> None:
    """Recovered against true value in one panel: every row with its 16-84% range, and means in bins of the truth."""
    truth, median = panel["truth"].to_numpy(float), panel["median"].to_numpy(float)
    q16, q84 = panel["q16"].to_numpy(float), panel["q84"].to_numpy(float)
    ax.errorbar(truth, median, yerr=[np.maximum(0, median - q16), np.maximum(0, q84 - median)], fmt="o", color=point_color, markersize=2.7,
                markeredgewidth=0, ecolor=matplotlib.colors.to_rgba(point_color, .2), elinewidth=.55, capsize=0, alpha=.5, zorder=2)
    binned = parameter_recovery_bins(panel)
    if not binned.empty:
        ax.errorbar(binned.truth_mean, binned.recovered_mean, yerr=binned.recovered_sem,
                    fmt="o", color=binned_color, markersize=binned_size, elinewidth=.8,
                    markeredgecolor="white", markeredgewidth=.4, capsize=1.5, zorder=4)
    values = np.concatenate([truth, median, q16, q84])
    values = values[np.isfinite(values)]
    if values.size:
        pad = 0.05 * (values.max() - values.min() or 1.0)
        limits = (values.min() - pad, values.max() + pad)
        ax.plot(limits, limits, color="0.25", linewidth=.7, linestyle="--", alpha=.7, zorder=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    _polish(ax)


def parameter_recovery_bins(panel: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """Mean recovery per truth bin, with complete mocks as the independent units."""
    data = panel[["seed", "truth", "median"]].copy()
    data = data[np.isfinite(data.truth) & np.isfinite(data["median"])]
    if data.empty:
        return pd.DataFrame()
    lo, hi = data.truth.min(), data.truth.max()
    if lo == hi:
        data["bin"] = 0
    else:
        edges = np.linspace(lo, hi, bins + 1)
        data["bin"] = np.clip(np.searchsorted(edges, data.truth, side="right") - 1, 0, bins - 1)
    rows = []
    for _, images in data.groupby("bin"):
        mocks = images.groupby("seed")[["truth", "median"]].mean()
        if len(mocks) < 2:
            continue
        rows.append({"truth_mean": mocks.truth.mean(), "recovered_mean": mocks["median"].mean(),
                     "recovered_sem": mocks["median"].std(ddof=1) / np.sqrt(len(mocks))})
    return pd.DataFrame(rows)


# The paper's recovery figure: one row of panels per model component, named beside its row.
RECOVERY_GROUPS = (("Halo", "halo.", BLUE), ("BCG", "bcg.", ORANGE), ("GRF + GP", "images.", PINK), ("Members", "potfile.", GREEN))
RECOVERY_SOURCES = ("Sources", YELLOW)
RECOVERY_POINT_COLOR = "0.5"          # one parameter of one mock


def _quantity_label(parameter: str) -> str:
    """A parameter's symbol and unit, as written inside its panel."""
    field = parameter.split(".", 1)[1]
    if field == "position_xy":
        return r"$\beta_x,\ \beta_y$ (arcsec)" if parameter.startswith("source.") else r"$x,\ y$ (arcsec)"
    if field == "log10_mass_msun":
        return r"$\log(M_{\rm tot}/M_\odot)$"
    if field == "log10_aperture_mass_msun":
        return r"$\log(M_{\rm ap}/M_\odot)$"
    unit = PARAMETER_UNITS.get(field)
    unit = {"km/s": r"km s$^{-1}$"}.get(unit, unit)
    label = PARAMETER_LABELS.get(field, field)
    return f"{label} ({unit})" if unit else label


def recovery_panels(parameters: pd.DataFrame) -> list[tuple[str, str, list[tuple[str, pd.DataFrame]]]]:
    """(component, colour, [(parameter, rows)]) for the recovery figure: the paper parameters of each component in
    order, with the x and y of a centre pooled into one panel, and the source positions as a last one-panel group."""
    df = parameters[np.isfinite(parameters["truth"].to_numpy(float)) & np.isfinite(parameters["median"].to_numpy(float))]
    names = df["parameter"].astype(str)
    groups = []
    for label, prefix, color in RECOVERY_GROUPS:
        panels, pooled = [], False
        for name in PAPER_PARAMETERS:
            if not name.startswith(prefix) or not (names == name).any():
                continue
            if name.endswith(("x_centre", "y_centre")):
                if not pooled:
                    panels.append((prefix + "position_xy", df[names.isin((prefix + "x_centre", prefix + "y_centre"))]))
                    pooled = True
                continue
            panels.append((name, df[names == name]))
        if panels:
            groups.append((label, color, panels))
    sources = df[names.str.startswith("source.") & names.str.endswith((".beta_x", ".beta_y"))]
    if not sources.empty:
        groups.append((*RECOVERY_SOURCES, [("source.position_xy", sources)]))
    return groups


def _recovery_rows(groups, ncols: int) -> list[list[tuple[str, str, str, pd.DataFrame, bool]]]:
    """Panels by row: every component starts a row and wraps at ``ncols``; a one-panel group (the sources) takes the
    first free slot instead of a row of its own.  Entries are (component, colour, parameter, rows, starts a component)."""
    rows: list[list] = []
    for label, color, panels in groups:
        if len(panels) == 1 and any(len(row) < ncols for row in rows):
            next(row for row in rows if len(row) < ncols).append((label, color, *panels[0], True))
            continue
        for start in range(0, len(panels), ncols):
            rows.append([(label, color, name, data, start == 0 and k == 0) for k, (name, data) in enumerate(panels[start:start + ncols])])
    return rows


@plt.rc_context(fname=STYLE, rc=FIGURE_STYLE)
def plot_parameter_recovery(campaign: Campaign, path: Path, *, ncols: int = 6) -> Path | None:
    """Recovered against true value of every paper parameter, one row of square panels per model component, sized for
    the full width of an ApJ page.  Gray points are one parameter of one mock with its 16-84% range; coloured points
    are means in bins of the true value with their standard error across mocks."""
    rows = _recovery_rows(recovery_panels(campaign.parameters), ncols)
    if not rows:
        return None
    pitch = APJ_TEXT_WIDTH_IN * (.997 - .078) / (ncols + .2 * (ncols - 1)) * 1.2     # one square panel and its gap, in inches
    fig, axes = plt.subplots(len(rows), ncols, figsize=(APJ_TEXT_WIDTH_IN, pitch * len(rows) + .5), squeeze=False)
    handles: dict[str, Any] = {}
    for axis_row, entries in zip(axes, rows):
        for ax in axis_row[len(entries):]:
            ax.axis("off")
        for ax, (component, color, parameter, data, first) in zip(axis_row, entries):
            _draw_parameter_axis(ax, data, binned_color=color, binned_size=3.6, point_color=RECOVERY_POINT_COLOR)
            ax.set_box_aspect(1)
            ax.tick_params(which="both", labelsize=7, pad=1.5)
            ax.tick_params(axis="y", labelrotation=90)          # upright labels would run into the next panel
            plt.setp(ax.get_yticklabels(), va="center")
            # The y axis repeats the x ticks, without those near the ends: their rotated labels would leave the panel.
            low, high = ax.get_ylim()
            ticks = [v for v in MaxNLocator(3).tick_values(low, high) if low + .13 * (high - low) <= v <= high - .13 * (high - low)]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.tick_params(which="major", length=2.8)
            ax.tick_params(which="minor", length=1.5)
            ax.text(.05, .95, _quantity_label(parameter), transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="0.1",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=.75), zorder=6)
            if first and ax is axis_row[0]:
                ax.annotate(component, (-.27, .5), xycoords="axes fraction", rotation=90, ha="center", va="center",
                            fontsize=9, color="0.1")
            elif first:                                       # a group that shares another component's row
                ax.text(.95, .05, component, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="0.1", zorder=6)
            containers = [c for c in ax.containers if c.has_yerr]
            handles.setdefault("One parameter in one mock", containers[0])
            if len(containers) > 1:
                handles.setdefault(component, containers[1])
    # Legend: the single mocks, then one binned-mean entry per component after a text-only heading.
    order = ["One parameter in one mock", *(label for label, _, _ in RECOVERY_GROUPS), RECOVERY_SOURCES[0]]
    names = [name for name in order if name in handles]
    entries = [handles[names[0]], Line2D([], [], linestyle="none"), *(handles[name] for name in names[1:])]
    legend = fig.legend(entries, [names[0], "Binned means:", *names[1:]], loc="upper center", ncol=len(entries), frameon=False,
                        fontsize=7.5, handletextpad=.25, columnspacing=1.0, borderaxespad=0, bbox_to_anchor=(.5, 1 - .03 / fig.get_figheight()))
    legend.legend_handles[1].set_visible(False)
    fig.supxlabel("True value", fontsize=9.5, y=.03 / fig.get_figheight())
    fig.supylabel("Recovered value", fontsize=9.5, x=.006)
    fig.subplots_adjust(left=.078, right=.997, bottom=.36 / fig.get_figheight(), top=1 - .23 / fig.get_figheight(),
                        wspace=.2, hspace=.2)
    return save_page(fig, path)


# ----------------------------------------------------------------------------- subhalo mass function
def _schechter_curve(selection: dict[str, Any], x: np.ndarray, normalization: float,
                     luminosity_range: tuple[float, float] | None = None) -> np.ndarray | None:
    """Input luminosity distribution mapped to dN/dlog10(M), without scatter.  L is in pivot units (L*); the Schechter
    cutoff L_c = 10^(-0.4 (schechter_magnitude - pivot_magnitude)) L*."""
    try:
        alpha, anchor, exponent = (float(selection[k]) for k in ("schechter_alpha", "dpie_mass_anchor_msun", "mass_luminosity_exponent"))
        cutoff = 10.0 ** (-0.4 * (float(selection["schechter_magnitude"]) - float(selection["pivot_magnitude"])))
        lower, upper = (float(v) for v in luminosity_range)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (np.all(np.isfinite([alpha, anchor, exponent, normalization, lower, upper]))
            and anchor > 0 and exponent > 0 and normalization > 0 and 0 < lower < upper):
        return None
    luminosity = np.power(np.power(10.0, x) / anchor, 1.0 / exponent)
    # p(L) is normalized over the generator's selection, not the plot range.
    norm = quad(lambda ell: ell ** alpha * np.exp(-ell / cutoff), lower, upper,
                epsabs=1e-12, epsrel=1e-12)[0]
    density = np.power(luminosity, alpha + 1.0) * np.exp(-luminosity / cutoff)
    density *= normalization * np.log(10.0) / (exponent * norm)
    return np.where((luminosity >= lower) & (luminosity <= upper), density, 0.0)


# Member-function axes: x = scale * log10(M/M*) / q (q = 1 for mass, p for luminosity); per-unit-L densities on log y.
MEMBER_FUNCTION_AXES = {
    "mass": dict(scale=1.0, luminosity=False, per_unit=False, x=r"$\log_{10}(M/M_\ast)$", y=r"$\langle dN/d\log_{10}M\rangle$"),
    "luminosity": dict(scale=1.0, luminosity=True, per_unit=True, x=r"$\log_{10}(L/L_\ast)$", y=r"$\langle dN/d(L/L_\ast)\rangle$"),
    "log_luminosity": dict(scale=1.0, luminosity=True, per_unit=False, x=r"$\log_{10}(L/L_\ast)$", y=r"$\langle dN/d\log_{10}L\rangle$"),
    "magnitude": dict(scale=-2.5, luminosity=True, per_unit=False, x=r"$m$", y=r"$\langle dN/dm\rangle$ (mag$^{-1}$)"),
}


def _draw_member_function(top, bottom, campaign: Campaign, axis: str):
    """Draw one member function (top) and its residual (bottom); returns the legend (handles, labels) or None.

    Truth and recovered masses of each mock are taken relative to its true M*, the mass of an L* member
    (``selection['dpie_mass_anchor_msun']``); on the luminosity axes they are converted with that mock's true relation,
    log10(L/L*) = log10(M/M*)/p (``selection['mass_luminosity_exponent']``), so every mock follows the same Schechter
    function; magnitudes are m = m* - 2.5 log10(L/L*), m* being the mock's ``selection['pivot_magnitude']``.
    ``"luminosity"`` is dN/d(L/L*) on log-log axes, where the faint-end slope is alpha; ``"log_luminosity"``
    is per dex (slope alpha + 1) and ``"magnitude"`` per magnitude against m - m* (slope -0.4(alpha + 1)), both on a
    linear axis like ``"mass"``.  The logarithmic slope of the Schechter curve is written at both ends: a straight-line
    fit over 0.3 dex in luminosity just inside the faintest luminosity every mock is complete to, and inside the
    brightest bin with true members.  Bottom: the recovered minus the true function of the same mock, mean and
    standard error over the mocks."""
    spec = MEMBER_FUNCTION_AXES[axis]
    scale, per_unit = spec["scale"], spec["per_unit"]
    anchors = [np.log10(float(s["selection"]["dpie_mass_anchor_msun"])) for s in campaign.subhalos]
    exponents = [float(s["selection"]["mass_luminosity_exponent"]) if spec["luminosity"] else 1.0 for s in campaign.subhalos]
    offsets = [float(s["selection"]["pivot_magnitude"]) if axis == "magnitude" else 0.0 for s in campaign.subhalos]
    truth_all = [o + scale * (np.asarray(s["truth_log10_mass"], float) - a) / q
                 for s, a, q, o in zip(campaign.subhalos, anchors, exponents, offsets)]
    recovered_all = [o + scale * (np.asarray(s["recovered_log10_mass"], float) - a) / q
                     for s, a, q, o in zip(campaign.subhalos, anchors, exponents, offsets)]
    values = np.concatenate([v for v in truth_all + recovered_all if v.size]) if any(v.size for v in truth_all + recovered_all) else np.asarray([])
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        lo, hi = lo - 0.25, hi + 0.25
    edges = np.linspace(lo, hi, int(np.clip(np.sqrt(values.size), 6, 18)) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Bin widths in the plotted unit: dex (or mag), or L/L* itself for dN/d(L/L*).
    widths = np.diff(10.0 ** edges) if per_unit else np.diff(edges)
    truth_density = np.stack([np.histogram(v, bins=edges)[0] / widths for v in truth_all])
    recovered_density = np.stack([np.histogram(v, bins=edges)[0] / widths for v in recovered_all])
    truth_mean, recovered_mean = truth_density.mean(axis=0), recovered_density.mean(axis=0)
    difference = recovered_density - truth_density              # paired: the same mock in both
    n_mocks = len(campaign.subhalos)
    # One independent unit is a complete mock, for both truth and recovery.
    zeros = np.zeros_like(truth_mean)
    sem = (lambda a: a.std(axis=0, ddof=1) / np.sqrt(n_mocks)) if n_mocks > 1 else (lambda a: zeros)
    truth_sem, recovered_sem, difference_sem = sem(truth_density), sem(recovered_density), sem(difference)
    # Limits from the histograms, never from the reference curve; headroom for the slope labels.
    high = float(np.max(np.concatenate([truth_mean + truth_sem, recovered_mean + recovered_sem])))
    if per_unit:
        top.set_xscale("linear")
        top.set_yscale("log")
        positive = np.concatenate([truth_mean[truth_mean > 0], recovered_mean[recovered_mean > 0]])
        floor, ceiling = .5 * float(positive.min()), 6. * high
    else:
        floor, ceiling = 0., 1.3 * high
    if n_mocks > 1:
        top.stairs(truth_mean + truth_sem, edges, baseline=np.maximum(truth_mean - truth_sem, floor), fill=True,
                   color="0.45", alpha=.22, linewidth=0, zorder=1)
    top.stairs(np.where(truth_mean > 0, truth_mean, np.nan) if per_unit else truth_mean, edges, color="0.1", linewidth=.9, zorder=3)
    truth_handle = (Patch(facecolor=matplotlib.colors.to_rgba("0.45", .22), edgecolor="none"), Line2D([], [], color="0.1", linewidth=.9))
    points = dict(fmt="o", color=COLOR, markersize=3.6, elinewidth=.8, capsize=1.6, capthick=.8, markeredgecolor="white", markeredgewidth=.5)
    shown = recovered_mean > 0 if per_unit else np.ones_like(recovered_mean, bool)
    lower = np.minimum(recovered_sem, recovered_mean - floor)[shown]   # on a log axis the bars stop at its floor
    recovered_bars = top.errorbar(centers[shown], recovered_mean[shown], yerr=[lower, recovered_sem[shown]], zorder=5, **points)
    handles, labels = [truth_handle, recovered_bars], ["Truth", "Recovered"]
    config_path = campaign.root / "campaign_config.json"
    luminosity_range = (json.loads(config_path.read_text()).get("mock", {}).get("luminosity_range")
                        if config_path.exists() else None)
    reference_x = np.linspace(edges[0], edges[-1], 2000)
    # The curve is per dex of M; per unit x it is q / |scale| times that, and per unit L/L* a further 1/(ln 10 L).
    curves = [_schechter_curve(subhalo.get("selection", {}), anchor + q * (reference_x - o) / scale, float(len(masses)), luminosity_range)
              for subhalo, anchor, q, o, masses in zip(campaign.subhalos, anchors, exponents, offsets, truth_all)]
    if all(curve is not None for curve in curves):
        mean_curve = np.mean([q / abs(scale) * curve for q, curve in zip(exponents, curves)], axis=0)
        if per_unit:
            mean_curve = mean_curve / (np.log(10.0) * 10.0 ** reference_x)
        # Not drawn outside the luminosity range the mocks are drawn from (no vertical drop at the magnitude limit).
        drawn = np.where((mean_curve > 0) & (mean_curve >= floor), mean_curve, np.nan)
        reference, = top.plot(reference_x, drawn,
                              color=ORANGE, linestyle=(0, (1, 1.6)), linewidth=1.5, dash_capstyle="round", zorder=4)
        handles.append(reference)
        labels.append("Schechter")
        # Faintest luminosity each mock is complete to, on this axis; the most central one bounds the faint window.
        limits = [o + scale * float(sub["selection"]["mass_luminosity_exponent"]) / q * np.log10(float(luminosity_range[0]))
                  for sub, q, o in zip(campaign.subhalos, exponents, offsets)]
        inward = np.sign(scale)                        # from the faint end toward the centre of the axis
        faint = max(limits) if inward > 0 else min(limits)
        filled = truth_mean > 0
        bright = float(edges[1:][filled].max()) if inward > 0 else float(edges[:-1][filled].min())
        width = .3 * abs(scale)
        for name, (start, end) in (("Faint", sorted((faint, faint + inward * width))),
                                   ("Bright", sorted((bright, bright - inward * width)))):
            window = (reference_x >= start) & (reference_x <= end) & (mean_curve > 0)
            if window.sum() < 2:  # the curve does not reach this end of the plotted range
                continue
            slope = np.polyfit(reference_x[window], np.log10(mean_curve[window]), 1)[0]
            # In the upper corner on the side of its end; on the log-log panel the faint end is high, so lower there.
            left = .5 * (start + end) < .5 * (edges[0] + edges[-1])
            value = f"{slope:+.2f}".replace("-", "\u2212")
            y = .3 if per_unit and left else .97
            top.text(.04 if left else .96, y, f"{name}-end\nslope {value}", transform=top.transAxes, color=ORANGE,
                     fontsize=7, ha="left" if left else "right", va="top", zorder=6)
    top.set_ylim(floor, ceiling)
    if not per_unit:
        top.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
    top.set_ylabel(spec["y"])
    bottom.axhline(0, color="0.1", linewidth=.7, linestyle=DASHES, zorder=1)
    bottom.errorbar(centers, difference.mean(axis=0), yerr=difference_sem, zorder=5, **points)
    reach = max(float(np.nanmax(np.abs(difference.mean(axis=0)) + difference_sem)) * 1.25, 1.0)
    bottom.set_ylim(-reach, reach)
    bottom.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))
    bottom.set_ylabel("Residual", fontsize=9)
    bottom.set_xlabel(spec["x"])
    step = edges[1] - edges[0]
    bottom.set_xlim(edges[0] - .35 * step, edges[-1] + .35 * step)
    return handles, labels


@plt.rc_context(fname=STYLE, rc=FIGURE_STYLE)
def plot_subhalo_mass_function(campaign: Campaign, path: Path) -> Path | None:
    """The member mass function against M/M* with its residual panel (see :func:`_draw_member_function`)."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(APJ_COLUMN_WIDTH_IN, 3.1), sharex=True,
                                      gridspec_kw={"height_ratios": [3.1, 1], "hspace": .07})
    legend = _draw_member_function(top, bottom, campaign, "mass")
    if legend is None:
        plt.close(fig)
        return None
    top.set_ylim(top.get_ylim()[0], top.get_ylim()[1] * 1.12)     # room for the legend row above the slope labels
    top.legend(*legend, loc="upper center", ncol=len(legend[0]), frameon=False, handlelength=1.5, columnspacing=.9,
               handletextpad=.4, borderaxespad=.35)
    for text in top.texts:
        text.set_y(.82)
    fig.align_ylabels((top, bottom))
    fig.subplots_adjust(left=.165, right=.985, bottom=.125, top=.985)
    return save_page(fig, path)


@plt.rc_context(fname=STYLE, rc=FIGURE_STYLE)
def plot_member_luminosity_function(campaign: Campaign, path: Path) -> Path | None:
    """The member mass function and the luminosity function in three forms, square panels each with its residual
    panel: dN/dlog M; dN/dL on log-log axes, where the faint-end slope is alpha; per dex of L (alpha + 1); and per
    magnitude (-0.4(alpha + 1)), as published cluster luminosity functions are drawn (see :func:`_draw_member_function`)."""
    forms = ("mass", "luminosity", "log_luminosity", "magnitude")
    # Panels as large as in the one-column mass-function figure (about 2.7 in), so panels and text keep its
    # proportions; the figure is scaled down as a whole on the page.
    fig, axes = plt.subplots(2, len(forms), figsize=(len(forms) * 3.05, 4.3), sharex="col",
                             gridspec_kw={"height_ratios": [1, .32], "hspace": .06, "wspace": .3})
    legend = None
    for column, axis in enumerate(forms):
        axes[0, column].set_box_aspect(1)
        axes[1, column].set_box_aspect(.32)
        legend = _draw_member_function(axes[0, column], axes[1, column], campaign, axis) or legend
    if legend is None:
        plt.close(fig)
        return None
    fig.legend(*legend, loc="upper center", ncol=len(legend[0]), frameon=False, handlelength=1.5, columnspacing=1.2,
               handletextpad=.4, bbox_to_anchor=(.5, 1.0))
    for column in range(len(forms)):
        fig.align_ylabels(axes[:, column])
    fig.subplots_adjust(left=.05, right=.995, bottom=.11, top=.93)
    return save_page(fig, path)


# ----------------------------------------------------------------------------- magnification
@plt.rc_context(fname=STYLE, rc=FIGURE_STYLE)
def plot_magnification_recovery(campaign: Campaign, path: Path) -> Path | None:
    """Top: recovered against true |mu| of every image (posterior median and 16-84% range), with means in bins of the
    true value.  Bottom: the ratio of the two; images whose ratio is off the panel are drawn as carets at its edge."""
    df = campaign.magnification
    truth = np.abs(df["magnification_true_at_obs"].to_numpy(float))
    q16, q50, q84 = (np.abs(df[f"magnification_model_{q}"].to_numpy(float)) for q in ("q16", "q50", "q84"))
    lower, upper = np.minimum(np.minimum(q16, q50), q84), np.maximum(np.maximum(q16, q50), q84)
    valid = (truth > 0) & (q50 > 0) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return None
    truth, q50, lower, upper = truth[valid], q50[valid], lower[valid], upper[valid]
    # A square main panel over a ratio panel, placed by hand so the main panel is exactly square.
    left, right, bottom_in, gap_in, ratio_in, top_in = .165, .985, .40, .07, .82, .06
    side_in = APJ_COLUMN_WIDTH_IN * (right - left)
    height = bottom_in + ratio_in + gap_in + side_in + top_in
    fig = plt.figure(figsize=(APJ_COLUMN_WIDTH_IN, height))
    ratio_ax = fig.add_axes([left, bottom_in / height, right - left, ratio_in / height])
    ax = fig.add_axes([left, (bottom_in + ratio_in + gap_in) / height, right - left, side_in / height], sharex=ratio_ax)
    faint = dict(fmt="o", color=COLOR, markersize=1.9, markeredgewidth=0, ecolor=matplotlib.colors.to_rgba(COLOR, .10),
                 elinewidth=.4, capsize=0, alpha=.38, zorder=2, rasterized=True)
    ax.errorbar(truth, q50, yerr=[np.maximum(0, q50 - lower), np.maximum(0, upper - q50)], **faint)
    ratio = q50 / truth
    reach = float(np.clip(np.quantile(np.abs(np.log10(ratio)), .995) * 1.15, np.log10(2.5), 1.0))     # decades shown each way
    inside = np.abs(np.log10(ratio)) <= reach
    ratio_ax.errorbar(truth[inside], ratio[inside], yerr=[np.maximum(0, q50 - lower)[inside] / truth[inside],
                                                          np.maximum(0, upper - q50)[inside] / truth[inside]], **faint)
    for sign, marker in ((1, "^"), (-1, "v")):
        off = np.sign(np.log10(ratio)) * ~inside == sign
        ratio_ax.plot(truth[off], np.full(off.sum(), 10 ** (sign * reach * .93)), marker, color=COLOR, markersize=2.6,
                      markeredgewidth=0, alpha=.7, linestyle="none", zorder=3)
    # The legend entry: marker and error bar together, at full opacity (the points themselves are faint).
    handles = [ax.errorbar([np.nan], [np.nan], yerr=[1.0], fmt="o", color=COLOR, markersize=2.8, markeredgewidth=0,
                           elinewidth=.6, capsize=0, label="Images")]
    stats = binned_magnification_summary(df)
    if not stats.empty:
        stats.to_csv(path.with_name(path.stem + "_bins.csv"), index=False)
        shown = stats[(stats.n_images > 7) & (stats.n_mocks >= 2)]
        if not shown.empty:
            binned = dict(fmt="o", color=ORANGE, markersize=4.4, elinewidth=1, markeredgecolor="white", markeredgewidth=.6,
                          capsize=2, capthick=1, zorder=5)
            handles.append(ax.errorbar(shown.truth_mean, shown.recovered_mean, yerr=shown.recovered_sem, label="Binned mean", **binned))
            ratio_ax.errorbar(shown.truth_mean, shown.recovered_mean / shown.truth_mean, yerr=shown.recovered_sem / shown.truth_mean, **binned)
    ax.legend(handles=handles, loc="upper left", frameon=False, borderaxespad=.5)
    log_values = np.log10(np.concatenate([truth, lower[lower > 0], upper]))
    pad = max(0.05 * np.ptp(log_values), 0.08)
    limits = (10 ** (log_values.min() - pad), 10 ** (log_values.max() + pad))
    exact = dict(color="0.1", linewidth=.7, linestyle=DASHES, zorder=3)
    ax.plot(limits, limits, **exact)
    ratio_ax.axhline(1.0, **exact)
    for axes in (ax, ratio_ax):
        axes.set_xscale("log")
        axes.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ratio_ax.set_ylim(10 ** -reach, 10 ** reach)
    for axis in (ax.xaxis, ax.yaxis):                      # a label every two decades, a tick at each decade
        axis.set_major_locator(LogLocator(base=100.0))
        axis.set_minor_locator(LogLocator(base=10.0, numticks=30))
        axis.set_minor_formatter(NullFormatter())
    marks = [v for v in (.1, .2, .5, 1, 2, 5, 10) if 10 ** (-reach * .9) <= v <= 10 ** (reach * .9)]
    ratio_ax.yaxis.set_major_locator(FixedLocator(marks[::2] if len(marks) > 5 else marks))
    ratio_ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ratio_ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"Recovered $|\mu|$")
    ratio_ax.set_ylabel("Recovered / true", fontsize=8.5)
    ratio_ax.set_xlabel(r"True $|\mu|$")
    fig.align_ylabels((ax, ratio_ax))
    return save_page(fig, path)


def binned_magnification_summary(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Equal-mock means and SEMs of saved median |mu|, binned by true |mu|.

    Summarize images within each mock first. Only occupied mocks contribute to
    a bin; these SEMs describe ensemble means, not individual posterior errors.
    """
    data = pd.DataFrame({"seed": df["seed"],
                         "truth": np.abs(df["magnification_true_at_obs"].to_numpy(float)),
                         "recovered": np.abs(df["magnification_model_q50"].to_numpy(float))})
    data = data[np.isfinite(data.truth) & np.isfinite(data.recovered) & (data.truth > 0) & (data.recovered > 0)].copy()
    if data.empty:
        return pd.DataFrame()
    lo, hi = data.truth.min(), data.truth.max()
    if hi == lo:
        lo, hi = lo / 1.1, hi * 1.1
    edges = np.geomspace(lo, hi, bins + 1)
    data["bin"] = np.clip(np.searchsorted(edges, data.truth, side="right") - 1, 0, bins - 1)
    rows = []
    for index, images in data.groupby("bin"):
        mocks = images.groupby("seed")[["truth", "recovered"]].mean()
        n = len(mocks)
        rows.append({"bin_lower": edges[index], "bin_upper": edges[index + 1],
                     "truth_mean": mocks.truth.mean(), "recovered_mean": mocks.recovered.mean(),
                     "recovered_sem": mocks.recovered.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                     "n_mocks": n, "n_images": len(images)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- image residual distribution
def plot_image_residual_distribution(campaign: Campaign, path: Path) -> Path | None:
    """The run figure (``plotting.plots.residual_histogram_figure``) over every posterior draw of every mock."""
    from ..plotting.plots import residual_histogram_figure
    df = campaign.residual_draws
    residual = pd.to_numeric(df["image_residual_arcsec"], errors="coerce").to_numpy(float)
    recovered = df["recovered"].astype(bool).to_numpy() & np.isfinite(residual) & (residual >= 0)
    if not recovered.any():
        return None
    groups = [residual[recovered & (df["seed"] == seed).to_numpy() & (df["draw_index"] == draw).to_numpy()]
              for seed, draw in df.loc[recovered, ["seed", "draw_index"]].drop_duplicates().itertuples(index=False)]
    return residual_histogram_figure(groups, path)


# ----------------------------------------------------------------------------- aperture mass components
def _label_curve(ax, x, y, at, text, color, *, above: bool = True, offset: float = 2.5) -> None:
    """Write ``text`` along a curve of log-log axes at x = ``at``, on one side of it.  The axes' limits and position
    must be final."""
    good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    value = lambda r: 10 ** np.interp(np.log10(r), np.log10(x[good]), np.log10(y[good]))
    (x0, y0), (x1, y1) = ax.transData.transform([(at / 1.25, value(at / 1.25)), (at * 1.25, value(at * 1.25))])
    angle = math.atan2(y1 - y0, x1 - x0)
    side = offset if above else -offset
    ax.annotate(text, (at, value(at)), xytext=(-math.sin(angle) * side, math.cos(angle) * side), textcoords="offset points",
                rotation=math.degrees(angle), rotation_mode="anchor", ha="center", va="bottom" if above else "top",
                fontsize=8, color=color, zorder=6, path_effects=[withStroke(linewidth=2.0, foreground="white")])


@plt.rc_context(fname=STYLE, rc=FIGURE_STYLE)
def plot_m2d_components(campaign: Campaign, path: Path, *, seed: int | None = None) -> Path | None:
    """Middle: enclosed mass per component for one representative realization, each curve named along itself.  Bottom:
    the campaign, median over realizations of each realization's posterior-median fractional mass error with the 16th
    to 84th percentile band across realizations, out to the median radius of the outermost image.  Top strip: where
    the images of all realizations sit."""
    seed = representative_seed(campaign) if seed is None else int(seed)
    groups = (*COMPONENT_GROUPS, "total")
    colors = {**COMPONENT_COLORS, "total": "0.05"}
    labels = {**COMPONENT_LABELS, "total": "Total"}
    m2d = campaign.m2d[campaign.m2d["component"].isin(groups)]
    rows = m2d[m2d["seed"] == seed]
    if rows.empty:
        return None
    outer = float(np.nanmedian(pd.to_numeric(m2d["outer_image_radius_kpc"], errors="coerce"))) if "outer_image_radius_kpc" in m2d else np.inf
    fig = plt.figure(figsize=(APJ_COLUMN_WIDTH_IN, 3.95))
    left, width = .185, .80
    bottom = fig.add_axes([left, .105, width, .215])
    top = fig.add_axes([left, .335, width, .555], sharex=bottom)
    strip = fig.add_axes([left, .890, width, .095], sharex=bottom)
    curve_radii = pd.to_numeric(rows["radius_kpc"], errors="coerce").to_numpy(float)
    curve_radii = curve_radii[np.isfinite(curve_radii) & (curve_radii > 0)]
    limits = (float(curve_radii.min()), float(curve_radii.max()))
    curves, reach = {}, .1
    for group in groups:
        sel = rows[rows["component"] == group].sort_values("radius_kpc")
        if sel.empty:
            continue
        radius, color = sel["radius_kpc"].to_numpy(float), colors[group]
        line_order = 2 if group == "total" else 3
        top.plot(radius, sel["truth_m2d_msun"].to_numpy(float), color=color, linestyle=(0, (3.5, 2)), linewidth=.9, zorder=line_order + 1)
        top.plot(radius, sel["model_q50"].to_numpy(float), color=color, linewidth=1.7, alpha=.85, zorder=line_order)
        curves[group] = (radius, sel["model_q50"].to_numpy(float))
        stats = m2d[m2d["component"] == group].groupby("radius_kpc")["ratio_q50"].agg(
            q16=lambda v: np.nanpercentile(v, 16), q50="median", q84=lambda v: np.nanpercentile(v, 84)).sort_index()
        inside = stats.index.to_numpy(float) <= outer
        r_c = stats.index.to_numpy(float)[inside]
        q16, q50, q84 = (stats[q].to_numpy(float)[inside] - 1.0 for q in ("q16", "q50", "q84"))
        bottom.fill_between(r_c, q16, q84, color=color, alpha=0.14, linewidth=0.0, zorder=0 if group == "total" else 1)
        bottom.plot(r_c, q50, color=color, linewidth=1.4, zorder=line_order)
        if q50.size:
            reach = max(reach, float(np.nanmax(np.abs(np.concatenate([q16, q50, q84])))))
    for ax in (top, bottom, strip):
        ax.set_xscale("log")
    top.set_yscale("log")
    top.set_xlim(limits)
    low, high = top.get_ylim()
    top.set_ylim(low, high * 1.6)
    top.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    # Each curve carries its own name; the legend only says which line style is which.
    place = lambda f: limits[0] * (limits[1] / limits[0]) ** f
    for group, fraction, above in (("total", .30, True), ("halo", .62, False), ("bcg", .30, False), ("members", .30, True)):
        if group in curves:
            _label_curve(top, *curves[group], place(fraction), labels[group], colors[group], above=above)
    top.legend([Line2D([], [], color="0.3", linewidth=1.7, alpha=.85), Line2D([], [], color="0.3", linewidth=.9, linestyle=(0, (3.5, 2)))],
               ["Recovered", "Truth"], loc="lower right", frameon=False, handlelength=2.0, borderaxespad=.6)
    # Pool every image, using each mock's own aperture centre and physical scale.
    image_radii = []
    for image_seed in campaign.seeds:
        truth_path = campaign.root / f"seed_{image_seed:08d}" / "mock" / "truth.json"
        if not truth_path.is_file():
            continue
        truth = json.loads(truth_path.read_text())
        images = truth.get("images", [])
        image_radii.extend([
            np.hypot(im["x_true_arcsec"] - truth["halo"]["x_centre"],
                     im["y_true_arcsec"] - truth["halo"]["y_centre"])
            * truth["kpc_per_arcsec"] for im in images])
    image_radii = np.asarray(image_radii)
    image_radii = image_radii[np.isfinite(image_radii) & (image_radii > 0)]
    strip.axis("off")
    if image_radii.size:
        edges = np.geomspace(*limits, 31)
        counts, _ = np.histogram(image_radii, bins=edges)
        strip.stairs(counts, edges, baseline=0, fill=True, color="0.80", linewidth=0, zorder=0)
        strip.stairs(counts, edges, baseline=0, color="0.35", linewidth=.7, zorder=1)
        strip.set_ylim(0, counts.max() * 1.08)
        strip.text(.015, .30, "Multiple images", transform=strip.transAxes, ha="left", va="bottom", fontsize=8, color="0.3")
    top.set_ylabel(r"$M_{\rm 2D}(<R)\ \ (M_\odot)$")
    top.tick_params(labelbottom=False)
    bottom.axhline(0.0, color="0.1", linestyle=DASHES, linewidth=.7, zorder=4)
    reach = min(math.ceil(reach * 1.05 / .1) * .1, .5)
    bottom.set_ylim(-reach, reach)
    bottom.set_xlabel(r"$R$ (kpc)")
    bottom.set_ylabel(r"$M_{\rm fit}/M_{\rm true}-1$")
    bottom.yaxis.set_major_locator(MaxNLocator(4, symmetric=True, prune="both"))
    bottom.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    fig.align_ylabels((top, bottom))
    return save_page(fig, path)


# ----------------------------------------------------------------------------- coverage
def coverage_table(campaign: Campaign) -> pd.DataFrame:
    """Per parameter across realizations: 68% and 95% coverage with binomial errors, median bias and median fractional
    bias.  Only scored rows count (finite truth inside the support); per-member offsets and source positions are pooled."""
    df = campaign.parameters.copy()
    if "scored" not in df:
        df["scored"] = np.isfinite(df["truth"].to_numpy(float))
    df = df[df["scored"].astype(bool)].copy()
    name = df["parameter"].astype(str)
    df["group"] = np.where(name.str.startswith("source.") & name.str.endswith("beta_x"), "source.beta_x",
                  np.where(name.str.startswith("source.") & name.str.endswith("beta_y"), "source.beta_y",
                  np.where(name.str.contains(r"\.member\d+\."), name.str.replace(r"\.member\d+\.", ".member.", regex=True), name)))
    rows = []
    for group, part in df.groupby("group", sort=False):
        n = len(part)
        for level in ("68", "95"):
            part[f"c{level}"] = pd.to_numeric(part[f"covered_{level}"], errors="coerce") if f"covered_{level}" in part else np.nan
        c68, c95 = float(part["c68"].mean()), float(part["c95"].mean()) if "covered_95" in part else np.nan
        rows.append({"parameter": group, "n": n, "coverage_68": c68, "coverage_68_err": math.sqrt(max(c68 * (1 - c68), 0.0) / n),
                     "coverage_95": c95, "coverage_95_err": math.sqrt(max(c95 * (1 - c95), 0.0) / part["c95"].count()) if np.isfinite(c95) else np.nan,
                     "median_bias": float(np.nanmedian(part["bias"])), "median_fractional_bias": float(np.nanmedian(part["fractional_bias"])) if "fractional_bias" in part else np.nan,
                     "median_truth_percentile": float(np.nanmedian(part["truth_percentile"]))})
    return pd.DataFrame(rows)


def _standardized_parameter_panels(campaign: Campaign, *, sources: bool = True):
    """Residuals in units of half the central 68% posterior width, not Gaussian z-scores."""
    df = campaign.parameters.copy()
    scale = (df["q84"] - df["q16"]) / 2
    valid = np.isfinite(df["truth"]) & np.isfinite(df["median"]) & np.isfinite(scale) & (scale > 0)
    df = df.loc[valid].copy()
    df["residual"] = (df["median"] - df["truth"]) / scale.loc[valid]
    df = df[np.isfinite(df["residual"])]
    panels = [(p, df[df["parameter"] == p].copy()) for p in PAPER_PARAMETERS if (df["parameter"] == p).any()]
    if sources:
        for coordinate in ("beta_x", "beta_y"):
            rows = df[df["parameter"].str.startswith("source.") & df["parameter"].str.endswith("." + coordinate)]
            if not rows.empty:
                panels.append(("source." + coordinate, rows.copy()))
    return panels


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_standardized_residuals(campaign: Campaign, path: Path, *, ncols: int = 4) -> Path | None:
    """Truth-dependent residuals; binned SEM uses mocks as the independent units, including for pooled sources."""
    panels = _standardized_parameter_panels(campaign)
    if not panels:
        return None
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.36 * nrows), squeeze=False)
    for ax, (parameter, data) in zip(axes.flat, panels):
        ax.axhline(0, color=".25", linestyle="--", linewidth=.8)
        ax.axhspan(-1, 1, color=".94", zorder=-3)
        raw = ax.scatter(data.truth, data.residual, s=5, color=COLOR, alpha=.22, linewidths=0,
                         rasterized=True, label="Individual estimates")
        data["bin"] = pd.cut(data.truth, bins=5, labels=False, include_lowest=True)
        mocks = data.groupby(["bin", "seed"])[["truth", "residual"]].mean()
        binned = mocks.groupby("bin").agg(x=("truth", "mean"), y=("residual", "mean"), sem=("residual", "sem")).dropna()
        trend = ax.errorbar(binned.x, binned.y, yerr=binned["sem"], fmt="o", markersize=4, color="tab:orange",
                            elinewidth=1, capsize=2, label="Mean residual and SEM across mocks")
        if parameter.startswith("source."):
            symbol = r"$\beta_x$" if parameter.endswith("beta_x") else r"$\beta_y$"
            label = f"True {symbol} (arcsec)"
        else:
            label = _axis_label(parameter, "truth")
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel(r"$(\mathrm{median}-\mathrm{truth})/s_{68}$", fontsize=9)
        ax.text(.04, .96, COMPONENT_NAMES.get(_parameter_component(parameter), "Sources"), transform=ax.transAxes,
                va="top", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        limit = max(3.5, data.residual.abs().max() * 1.1)
        ax.set_ylim(-limit, limit)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.legend([raw, trend], [raw.get_label(), trend.get_label()], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .965), h_pad=1.3, w_pad=1.1)
    return _save(fig, path)


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_bias_and_coverage(campaign: Campaign, path: Path) -> Path | None:
    """Global parameters only. Horizontal intervals describe residuals across mocks, not posterior intervals.

    Coverage uses the saved posterior-interval flags for scored rows with finite truth, independently of whether
    their central 68% width can be used to standardize the residual. Source coordinates are omitted.
    """
    from matplotlib.lines import Line2D

    panels = _standardized_parameter_panels(campaign, sources=False)
    if not panels:
        return None
    fig, (ax, coverage) = plt.subplots(1, 2, figsize=(9, max(3.5, .36 * len(panels) + 1)), sharey=True,
                                      gridspec_kw={"width_ratios": [2, 1]})
    labels, fractions = [], []
    for i, (parameter, data) in enumerate(panels):
        q = np.quantile(data.residual, [.025, .16, .5, .84, .975])
        component, field = parameter.split(".", 1)
        component = _parameter_component(parameter)
        color = COMPONENT_COLORS["members" if component == "potfile" else component]
        ax.plot(q[[0, 4]], [i, i], color=color, linewidth=1, alpha=.55)
        ax.plot(q[[1, 3]], [i, i], color=color, linewidth=4, solid_capstyle="butt")
        ax.plot(q[2], i, "o", color=color, markersize=5)
        rows = campaign.parameters
        rows = rows[(rows.parameter == parameter) & np.isfinite(rows.truth)]
        if "scored" in rows:
            rows = rows[rows.scored.astype(bool)]
        for column, marker, alpha in (("covered_68", "o", 1), ("covered_95", "s", .55)):
            if column in rows and rows[column].notna().any():
                fraction = rows[column].dropna().astype(float).mean()
                fractions.append(fraction)
                coverage.plot(fraction, i, marker, color=color, alpha=alpha, markersize=5)
        labels.append(COMPONENT_NAMES[component] + "  " + PARAMETER_LABELS.get(field, field))
    ax.axvspan(-1, 1, color=".94", zorder=-3)
    ax.axvline(0, color=".3", linestyle="--", linewidth=.8)
    coverage.axvline(.68, color=".3", linestyle="--", linewidth=.8)
    coverage.axvline(.95, color=".3", linestyle=":", linewidth=.8)
    ax.set_yticks(range(len(panels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$(\mathrm{median}-\mathrm{truth})/s_{68}$")
    coverage.set_xlabel("Empirical coverage")
    coverage.set_xlim(min(.35, min(fractions, default=.35) - .04), 1.03)
    coverage.tick_params(labelleft=False)
    fig.legend(handles=[Line2D([], [], color=".3", marker="o", linewidth=4, label="Median and central 68% across mocks"),
                        Line2D([], [], color=".3", linewidth=1, label="Central 95%"),
                        Line2D([], [], color=".3", marker="o", linestyle="", label="68% coverage"),
                        Line2D([], [], color=".3", marker="s", alpha=.55, linestyle="", label="95% coverage")],
               loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93))
    return _save(fig, path)


# ----------------------------------------------------------------------------- driver
def write_aggregate_plots(campaign_root: str | Path, output_dir: str | Path | None = None, *, converged_only: bool = False) -> list[Path]:
    """Write the campaign-level tables and figures from every completed realization's ``recovery.json``.

    Parameters
    ----------
    campaign_root : str or Path
        Directory holding the ``seed_*`` folders (``<output_dir>/<campaign_name>/<run_name>``).
    output_dir : str or Path or None, optional
        Directory for the outputs; None writes to ``<campaign_root>/aggregate_plots``.  Default ``None``.
    converged_only : bool, optional
        If True, leave out realizations whose sampler did not converge (max R-hat >= 1.01 or min n_eff < 50, from the
        saved diagnostics) from every table and figure except ``seed_summary.csv``.  If False, use every completed
        realization and only flag the unconverged ones.  Default ``False``.

    Returns
    -------
    list of Path
        The figures written (PDF); figures with nothing to show are not written and not listed.  The tables are not
        listed.

    Raises
    ------
    FileNotFoundError
        If there is no ``seed_*/recovery/recovery.json`` under ``campaign_root``.
    ValueError
        If ``converged_only`` is True and no realization converged.

    Notes
    -----
    Files written in the output directory (``aggregate_plots/`` relative to the campaign root by default):
    ``parameter_recovery_table.csv``, ``seed_summary.csv`` (one row per realization with its convergence verdict),
    ``campaign_selection.json`` (convergence rule, included and failed seeds), ``mock_coverage_table.csv``,
    ``mock_results.tex``, and the figures ``mock_parameter_recovery_paper.pdf``,
    ``mock_standardized_residuals.pdf``, ``mock_bias_and_coverage.pdf``,
    ``mock_aggregate_subhalo_recovery_shmf.pdf``, ``mock_aggregate_subhalo_recovery_luminosity.pdf``, ``mock_magnification_recovery.pdf``,
    ``mock_m2d_aperture_recovery_components.pdf``, ``mock_aggregate_image_residual_distribution.pdf``,
    ``mock_grf_gp_field_recovery.pdf``,
    ``mock_<block>_truth_deviation_corner.pdf`` (one per parameter block), ``mock_combined_truth_deviation_corner.pdf``
    and ``mock_aggregate_chain_traces.pdf``.  The field-recovery, deviation-corner and chain-trace figures also write
    companion data files next to them (``.csv``, ``.txt``, ``.h5`` and ``.json`` files of the same stem, and
    ``mock_truth_deviation_corners.json``).
    """
    campaign = load_campaign(campaign_root, converged_only=converged_only)
    output = Path(output_dir) if output_dir is not None else campaign.root / "aggregate_plots"
    output.mkdir(parents=True, exist_ok=True)
    campaign.parameters.to_csv(output / "parameter_recovery_table.csv", index=False)
    campaign.summary.to_csv(output / "seed_summary.csv", index=False)
    excluded = campaign.summary[~campaign.summary["converged"].astype(bool)]
    (output / "campaign_selection.json").write_text(json.dumps({
        "realizations": int(len(campaign.summary)), "converged": int(campaign.summary["converged"].astype(bool).sum()),
        "included": int(len(campaign.seeds)),
        "rule": f"max R-hat < {CONVERGENCE_RHAT} and min n_eff >= {CONVERGENCE_MIN_NEFF:.0f}",
        "selection_policy": "converged_only" if converged_only else "all_completed",
        "included_seeds": campaign.seeds,
        "all_realizations_used": not converged_only,
        "failed_convergence": [{"seed": int(r.seed), "reasons": r.convergence_reasons} for r in excluded.itertuples()],
        "excluded": [{"seed": int(r.seed), "reasons": r.convergence_reasons} for r in excluded.itertuples()] if converged_only else []}, indent=2) + "\n")
    coverage_table(campaign).to_csv(output / "mock_coverage_table.csv", index=False)
    from ..reporting.results_tex import write_mock_results
    write_mock_results(campaign, output, rhat_cut=CONVERGENCE_RHAT, min_ess=CONVERGENCE_MIN_NEFF)
    plots = [plot_parameter_recovery(campaign, output / "mock_parameter_recovery_paper.pdf"),
             plot_standardized_residuals(campaign, output / "mock_standardized_residuals.pdf"),
             plot_bias_and_coverage(campaign, output / "mock_bias_and_coverage.pdf"),
             plot_subhalo_mass_function(campaign, output / "mock_aggregate_subhalo_recovery_shmf.pdf"),
             plot_member_luminosity_function(campaign, output / "mock_aggregate_subhalo_recovery_luminosity.pdf"),
             plot_magnification_recovery(campaign, output / "mock_magnification_recovery.pdf"),
             plot_m2d_components(campaign, output / "mock_m2d_aperture_recovery_components.pdf"),
             plot_image_residual_distribution(campaign, output / "mock_aggregate_image_residual_distribution.pdf")]
    from .field_recovery_plots import plot_field_recovery
    plots.append(plot_field_recovery(campaign, output))
    from .deviation_corners import write_deviation_corners, write_combined_deviation_corner
    plots.extend(write_deviation_corners(campaign, output))
    plots.append(write_combined_deviation_corner(campaign, output))
    from .aggregate_chains import write_aggregate_chains
    plots.append(write_aggregate_chains(campaign, output))
    return [p for p in plots if p is not None]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create the paper figures of a mock-validation campaign.")
    parser.add_argument("campaign_root", help="Directory holding the seed_* folders (validation_runs/<campaign>/<run_name>).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--converged-only", action="store_true", help="Exclude failed convergence checks; default includes all completed realizations.")
    args = parser.parse_args(argv)
    for path in write_aggregate_plots(args.campaign_root, args.output_dir, converged_only=args.converged_only):
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
