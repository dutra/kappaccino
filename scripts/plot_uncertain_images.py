"""The paper figure of the uncertain-image test: what the uncertain images recover along the ladder of trusted families.

Top row, for one case (``--case``, default 3 trusted families, seed 1): maps of the fractional convergence error of the
fit with every family trusted, the fit that ignores the uncertain (photo-z) images, and the fit that sums the uncertain
images over their partitions (``kappaccino.partition.groups``).  Bottom row, against the number of trusted ("gold")
families (mean and standard deviation over the seeds, the seeds faint): the convergence bias and scatter near the
images, and the pair recall (filled) and precision (open) of the recovered partition.

Reads a scoring root of scripts/uncertain_images_ladder.py after scripts/score_uncertain_images.py, and writes
``<root>/scores/uncertain_images_paper.pdf``.

Example:

    python scripts/plot_uncertain_images.py results/uncertain_images/hera
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.io import fits  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from kappaccino.io.image_catalog import load_image_catalog  # noqa: E402
from kappaccino.io.sky import radec_to_offsets_arcsec  # noqa: E402
from score_uncertain_images import offsets, run_dir  # noqa: E402

plt.style.use(ROOT / "src" / "kappaccino" / "plotting" / "style.mplstyle")
PANEL = dict(fontsize=15, fontweight="bold", va="top")
FULL = ("all families gold", "0.55")
ARMS = {"ignore": ("photo-z images ignored", "#c0392b", "s"), "groups": ("photo-z images as groups", "#2e8b57", "D")}
TITLES = {"full": "all 19 families gold", "ignore": "photo-z images ignored", "groups": "photo-z images as groups"}


def _maps(fig, grid, root: Path, case: str, lens: pd.DataFrame, vmax: float, width: int) -> int:
    """The three convergence-error maps of ``case``; returns the case's number of trusted families."""
    settings = json.loads((root / "inputs" / case / "degrade.json").read_text())
    trusted = {str(t) for t in settings["trusted"]}
    catalog = load_image_catalog(ROOT / settings["source_catalog"])
    gold_all = catalog["family_id"].astype(str).isin(trusted).to_numpy()
    axes = []
    for k, fit in enumerate(("full", "ignore", "groups")):
        ax = fig.add_subplot(grid[0, width * k:width * (k + 1)], sharex=axes[0] if axes else None, sharey=axes[0] if axes else None)
        axes.append(ax)
        final = run_dir(root, case, fit)
        reference = json.loads((final / "artifacts" / "config.json").read_text())["model"]["reference"]
        fits_dir = final / "fits"
        truth = fits.getdata(fits_dir / "truth_recovery_kappa_true.fits").astype(float)
        model = fits.getdata(fits_dir / "truth_recovery_kappa_model_median.fits").astype(float)
        x, y = offsets(fits_dir / "truth_recovery_kappa_true.fits", reference)
        image = ax.pcolormesh(x, y, (model - truth) / truth, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True)
        ax.contour(x, y, truth, levels=[0.5, 1.0], colors="0.35", linewidths=0.8)
        ix, iy = radec_to_offsets_arcsec(catalog["ra"].to_numpy(float), catalog["dec"].to_numpy(float),
                                         float(reference["ra0_deg"]), float(reference["dec0_deg"]))
        gold = gold_all if fit != "full" else np.ones(len(ix), dtype=bool)
        ax.plot(ix[~gold], iy[~gold], "o", mfc="none", mec="k", ms=5, mew=0.9, label="photo-z image")
        ax.plot(ix[gold], iy[gold], "o", mfc="k", mec="white", ms=5.5, mew=0.7, label="gold image")
        row = lens[lens.fit == fit] if fit == "full" else lens[(lens.case == case) & (lens.fit == fit)]
        ax.text(0.04, 0.04, f"bias {100 * row.near_kappa_frac_bias.iloc[0]:+.1f}%\nscatter {100 * row.near_kappa_frac_scatter.iloc[0]:.1f}%",
                transform=ax.transAxes, fontsize=12, va="bottom", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
        ax.set_title(TITLES[fit], fontsize=14)
        ax.text(0.04, 0.96, "abc"[k], transform=ax.transAxes, **PANEL, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        span = 1.12 * np.abs(np.concatenate([ix, iy])).max()
        ax.set_xlim(span, -span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\Delta$RA (arcsec)")
        if k:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel(r"$\Delta$Dec (arcsec)")
    axes[1].legend(loc="lower right", fontsize=10, framealpha=0.85, handletextpad=0.2, borderpad=0.3)
    bar = fig.colorbar(image, ax=axes, shrink=0.9, pad=0.01)
    bar.set_label(r"$(\kappa_\mathrm{model} - \kappa_\mathrm{true}) / \kappa_\mathrm{true}$")
    return len(settings["trusted"])


def _trend(ax, table: pd.DataFrame, column: str, levels: list[int], color: str, marker: str, label: str | None, shift: float,
           scale: float = 100.0, fill: bool = True) -> None:
    """The seeds (faint) and their mean and standard deviation per number of trusted families."""
    rank = {n: i for i, n in enumerate(levels)}
    ax.plot(table.n_trusted.map(rank).to_numpy(float) + shift, table[column] * scale, marker, color=color, ms=4.5, alpha=0.35, mec="none")
    stat = table.groupby("n_trusted")[column].agg(["mean", "std"]).reindex(levels) * scale
    ax.errorbar(np.arange(len(levels)) + shift, stat["mean"], yerr=stat["std"], fmt="-" + marker, color=color, ms=8, lw=1.8, capsize=3,
                mfc=color if fill else "white", mew=1.5, label=label)


def paper_figure(root: Path, case: str, path: Path, vmax: float = 0.3) -> None:
    lens = pd.read_csv(root / "scores" / "lens_summary.csv")
    partition = pd.read_csv(root / "scores" / "partition_summary.csv")
    levels = sorted(lens.n_trusted.unique(), reverse=True)
    shifts = {"ignore": -0.08, "groups": 0.08}

    fig = plt.figure(figsize=(15, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 12, height_ratios=[1.15, 1])
    n_gold = _maps(fig, grid, root, case, lens, vmax, width=4)

    full = lens[lens.fit == "full"].iloc[0]
    bottom = [fig.add_subplot(grid[1, 4 * k:4 * (k + 1)]) for k in range(3)]
    for ax, column, ylabel in ((bottom[0], "near_kappa_frac_bias", r"$\kappa$ bias near images (%)"),
                               (bottom[1], "near_kappa_frac_scatter", r"$\kappa$ scatter near images (%)")):
        ax.axhline(100 * full[column], color=FULL[1], lw=1.5, label=FULL[0])
        for fit, (label, color, marker) in ARMS.items():
            _trend(ax, lens[lens.fit == fit], column, levels, color, marker, label, shifts[fit])
        ax.set_ylabel(ylabel)
    bottom[0].axhline(0, color="k", lw=0.8, ls=":")
    bottom[0].legend(loc="lower left", fontsize=10)

    _, color, marker = ARMS["groups"]
    _trend(bottom[2], partition, "pair_recall", levels, color, marker, "pair recall", 0.0, scale=1.0)
    _trend(bottom[2], partition, "pair_precision", levels, color, marker, "pair precision", 0.08, scale=1.0, fill=False)
    bottom[2].set_ylim(0, 1.05)
    bottom[2].set_ylabel("Partition of the photo-z images")
    bottom[2].legend(loc="lower left", fontsize=10)

    for k, ax in enumerate(bottom):
        ax.set_xticks(range(len(levels)), [str(n) for n in levels])
        ax.set_xlabel("Gold families (of 19)")
        ax.text(0.0, 1.02, "def"[k], transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
        if n_gold in levels:
            ax.axvspan(levels.index(n_gold) - 0.3, levels.index(n_gold) + 0.3, color="0.9", zorder=-1, lw=0)
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="results/<campaign>/<cluster>, scored")
    parser.add_argument("--case", default="n3_c0_s1", help="the case of the maps (default n3_c0_s1)")
    parser.add_argument("--output", type=Path, help="default <root>/scores/uncertain_images_paper.pdf")
    args = parser.parse_args(argv)
    path = args.output or args.root / "scores" / "uncertain_images_paper.pdf"
    paper_figure(args.root, args.case, path)
    print(path)


if __name__ == "__main__":
    main()
