"""Illustrate the local source-residual covariance near a critical curve.

Run with /home/dutra/.conda/envs/lenstronomy/bin/python.
Writes identical vector PDFs to plots/paper/ and paper/figures/.
All ellipse outlines are unit-Mahalanobis contours, not 68% contours.
The member covariance is set to zero only for this illustration.
"""

from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
NAME = "covariance_latent_displacement.pdf"
BLUE = "#28799B"
ORANGE = "#C26935"
INK = "#24272B"
MUTED = "#737980"
SIGMA_OBS = 0.3  # arcsec per image-plane coordinate
SIGMA_BETA = 0.02  # arcsec per source-plane coordinate
S_MAX, S_MIN = 0.2, 0.03
ANGLE = 28.0
PHI = np.deg2rad(ANGLE)
ROTATION = np.array([[np.cos(PHI), -np.sin(PHI)],
                     [np.sin(PHI), np.cos(PHI)]])
A = ROTATION @ np.diag([S_MAX, S_MIN]) @ ROTATION.T
PROPAGATED = SIGMA_OBS**2 * A @ A.T
TOTAL = PROPAGATED + SIGMA_BETA**2 * np.eye(2)
T_EXAMPLE = SIGMA_OBS * S_MIN / SIGMA_BETA


def verify_geometry():
    """Check the plotted geometry against the production residual model."""
    sys.path.insert(0, str(ROOT / "src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from lenscluster.latent_displacement import statistics

    # Include the singular Jacobian, not just a nearly critical example.
    small_values = np.array([S_MIN, 1e-5, 0.0])
    jacobians = np.array([ROTATION @ np.diag([S_MAX, s]) @ ROTATION.T
                          for s in small_values])
    expected = (SIGMA_OBS**2 * jacobians @ jacobians.swapaxes(-1, -2)
                + SIGMA_BETA**2 * np.eye(2))
    result = statistics(jnp.zeros((3, 2)), jnp.zeros((3, 2)),
                        jnp.asarray(jacobians), jnp.full(3, SIGMA_OBS),
                        SIGMA_BETA, jnp.zeros((3, 2, 2)), jnp.ones(3, bool))
    np.testing.assert_allclose(result["covariance"], expected, rtol=1e-13, atol=1e-16)
    np.testing.assert_allclose(expected[0], TOTAL, atol=1e-16)
    eigenvalues = np.linalg.eigvalsh(expected)
    np.testing.assert_allclose(eigenvalues[:, 0],
                               SIGMA_BETA**2 + (SIGMA_OBS * small_values)**2,
                               rtol=1e-13)
    np.testing.assert_allclose(eigenvalues[:, 1],
                               SIGMA_BETA**2 + (SIGMA_OBS * S_MAX)**2,
                               rtol=1e-13)
    assert np.isfinite(result["loglike"]).all()
    np.testing.assert_allclose(T_EXAMPLE, 0.45)
    np.testing.assert_allclose(np.sqrt(eigenvalues[0]) / SIGMA_BETA,
                               [np.hypot(1, T_EXAMPLE), np.sqrt(10)])
    np.testing.assert_allclose(np.sqrt(eigenvalues[-1, 0]), SIGMA_BETA)
    assert np.isclose(np.linalg.det(jacobians[-1]), 0, atol=1e-16)
    # Contours map to the stated physical covariance despite different display units.
    theta = np.linspace(0, 2*np.pi, 181)
    unit = np.array([np.cos(theta), np.sin(theta)])
    boundary = ROTATION @ np.diag(np.sqrt(eigenvalues[0, ::-1])) @ unit
    np.testing.assert_allclose(np.einsum("in,ij,jn->n", boundary,
                                        np.linalg.inv(TOTAL), boundary), 1, atol=1e-13)
    print("Covariance, contour geometry, and critical-limit checks passed.")


def arrow(ax, start, end, *, color=INK, lw=0.9, style="-|>", **kwargs):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=8,
                            linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                            zorder=6, **kwargs)
    ax.add_patch(patch)
    return patch


def make_figure():
    verify_geometry()
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.25, 3.2))
    left = fig.add_axes([0.035, 0.26, 0.215, 0.49])
    middle = fig.add_axes([0.315, 0.25, 0.30, 0.51])
    right = fig.add_axes([0.745, 0.29, 0.225, 0.44])
    left.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal")
    middle.set(xlim=(-3.65, 3.65), ylim=(-2.65, 2.65), aspect="equal")
    for ax in (left, middle):
        ax.set_axis_off()

    fig.text(0.5, 0.945,
             r"$\boldsymbol{C}_i=\sigma_{i,\mathrm{obs}}^2\boldsymbol{A}_i\boldsymbol{A}_i^{\mathsf{T}}"
             r"+\sigma_\beta^2\boldsymbol{I}+\boldsymbol{C}_{\mathrm{member},i}$",
             ha="center", va="center", fontsize=12)
    for x, label in [(0.035, "(a)"), (0.315, "(b)"), (0.705, "(c)")]:
        fig.text(x, 0.80, label, fontsize=11)
    fig.text(0.49, 0.80,
             r"$w_{\min}=\sqrt{\sigma_\beta^2+\sigma_{i,\mathrm{obs}}^2s_{\min}^2}$",
             ha="center", fontsize=10)

    major, minor = ROTATION[:, 0], ROTATION[:, 1]
    left.add_patch(Ellipse((0, 0), 2, 2, fill=False, edgecolor=BLUE,
                           lw=1.4, linestyle=(0, (4, 2.6))))
    left.plot(0, 0, "o", ms=3, color=INK, zorder=8)
    arrow(left, (0, 0), minor, color=INK)
    left.text(*(-0.08*major + 0.61*minor), r"$\boldsymbol{v}_{\min}$",
              ha="right", va="center", fontsize=10)
    arrow(left, (0, 0), major, color=BLUE, lw=0.8, style="-")
    left.text(*(0.6*major - 0.2*minor), "$1$", color=BLUE,
              ha="center", va="center", fontsize=10)
    # Mapping arrow uses figure coordinates; it does not imply equal display scales.
    mapping = FancyArrowPatch((0.252, 0.51), (0.307, 0.51),
                              transform=fig.transFigure, arrowstyle="-|>",
                              mutation_scale=10, lw=0.9, color=INK)
    fig.add_artist(mapping)
    fig.text(0.279, 0.56, r"$\boldsymbol{A}_i$", ha="center", fontsize=12)

    propagated_axes = SIGMA_OBS * np.array([S_MAX, S_MIN]) / SIGMA_BETA
    total_axes = np.sqrt(1 + propagated_axes**2)
    middle.add_patch(Ellipse((0, 0), 2, 2, fill=False, edgecolor=MUTED,
                             lw=0.9, linestyle=(0, (1, 2))))
    middle.add_patch(Ellipse((0, 0), *(2*propagated_axes), angle=ANGLE,
                             fill=False, edgecolor=BLUE, lw=1.3,
                             linestyle=(0, (4, 2.6))))
    middle.add_patch(Ellipse((0, 0), *(2*total_axes), angle=ANGLE,
                             fill=False, edgecolor=ORANGE, lw=1.6))
    middle.plot(0, 0, "o", ms=3, color=INK, zorder=8)
    arrow(middle, (0, 0), minor * total_axes[1], color=ORANGE)
    middle.text(*(-0.42*major + minor*1.5), r"$w_{\min}/\sigma_\beta$",
                color=ORANGE, ha="center", va="bottom", fontsize=10)
    middle.annotate(r"$\sigma_\beta$", xy=(0.7, -0.7), xytext=(1.8, -1.7),
                    color=MUTED, fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    t = np.linspace(0, 3, 401)
    right.plot(t, np.hypot(1, t), color=ORANGE, lw=1.6)
    right.plot(t, t, color=BLUE, lw=1.2, ls=(0, (4, 2.6)))
    right.axhline(1, color=MUTED, lw=0.7, ls=(0, (1, 2)))
    right.plot(0, 1, "o", ms=3.5, color=ORANGE, clip_on=False, zorder=6)
    right.plot(T_EXAMPLE, np.hypot(1, T_EXAMPLE), "o", ms=3.8,
               mfc="white", mec=ORANGE, mew=1.1, zorder=7)
    right.annotate("(b)", (T_EXAMPLE, np.hypot(1, T_EXAMPLE)),
                   xytext=(0.82, 0.45), fontsize=9, color=ORANGE,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=ORANGE))
    right.annotate(r"$w_{\min}\to\sigma_\beta$", xy=(0, 1), xytext=(0.08, 1.7),
                   fontsize=9, color=ORANGE,
                   arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
    right.text(1.45, 2.75, r"$\sqrt{1+t^2}$", fontsize=10, color=ORANGE)
    right.text(2.6, 2.05, "$t$", fontsize=10, color=BLUE)
    right.set(xlim=(0, 3.05), ylim=(0, 3.5),
              xlabel=r"$t=\sigma_{i,\mathrm{obs}}s_{\min}/\sigma_\beta$",
              ylabel=r"$w_{\min}/\sigma_\beta$")
    right.set_xticks([0, 1, 2, 3])
    right.set_yticks([0, 1, 2, 3])
    right.tick_params(direction="in", labelsize=9, length=3)
    right.xaxis.labelpad = 3
    right.yaxis.labelpad = 3

    fig.text(0.143, 0.235, r"$\delta\boldsymbol{x}_i/\sigma_{i,\mathrm{obs}}$",
             ha="center", fontsize=11)
    fig.text(0.143, 0.175, "Image plane", ha="center", fontsize=9, color=MUTED)
    fig.text(0.465, 0.235, r"$\boldsymbol{F}_i/\sigma_\beta$",
             ha="center", fontsize=11)
    fig.text(0.465, 0.175, "Source plane", ha="center", fontsize=9, color=MUTED)
    handles = [Line2D([], [], color=BLUE, lw=1.3, ls=(0, (4, 2.6)), label="Propagated positions"),
               Line2D([], [], color=ORANGE, lw=1.6, label=r"With $\sigma_\beta$"),
               Line2D([], [], color=MUTED, lw=0.9, ls=(0, (1, 2)), label=r"$\sigma_\beta$ reference")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
                ncol=3, fontsize=9, frameon=False, handlelength=2.4,
                columnspacing=2, handletextpad=0.6)

    output = ROOT / "plots" / "paper" / NAME
    paper = ROOT / "paper" / "figures" / NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    paper.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={"Title": "Local source-residual covariance near criticality",
                                 "Subject": "Illustrative covariance with zero member scatter"})
    plt.close(fig)
    shutil.copy2(output, paper)
    assert output.read_bytes() == paper.read_bytes()
    print(output)
    print(paper)


if __name__ == "__main__":
    make_figure()
