"""Figure 1: the local mapping of positional uncertainty into the source plane.

Run with /home/dutra/.conda/envs/lenstronomy/bin/python.
Writes only plots/paper/covariance.pdf, independent of the working directory.
The parameters are illustrative, not a fitted cluster. Every uncertainty contour
is a unit-Mahalanobis contour (39.3% enclosed probability in two dimensions).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "paper" / "covariance.pdf"
STYLE = ROOT / "src" / "lenscluster" / "style.mplstyle"
BLUE, ORANGE, INK = "#1674A4", "#C56B2B", "#252930"
PURPLE, MUTED = "#77529B", "#626971"
ANGLE = 25.0
# Drawing parameters only: chosen for legibility, never displayed as inferred values.
SIGMA_OBS, SIGMA_SRC = 0.1, 0.03  # arcsec, per coordinate
LAMBDA_STRETCH, LAMBDA_PERP = 0.12, 0.8
E = np.array([np.cos(np.deg2rad(ANGLE)), np.sin(np.deg2rad(ANGLE))])
N = np.array([-E[1], E[0]])
A = LAMBDA_STRETCH * np.outer(E, E) + LAMBDA_PERP * np.outer(N, N)
C_OBS = SIGMA_OBS**2 * A @ A.T
C_SRC = SIGMA_SRC**2 * np.eye(2)
C_TOTAL = C_OBS + C_SRC
SCALE = 6.0  # inches per arcsecond; identical in every covariance glyph


def verify_geometry():
    """Independent eigenvalue and mapped-contour checks of the plotted geometry."""
    np.testing.assert_allclose(A, A.T, rtol=0, atol=1e-16)
    np.testing.assert_allclose(A @ E, LAMBDA_STRETCH * E, atol=1e-16)
    np.testing.assert_allclose(A @ N, LAMBDA_PERP * N, atol=1e-16)
    values, vectors = np.linalg.eigh(C_TOTAL)
    expected = SIGMA_SRC**2 + SIGMA_OBS**2 * np.array([LAMBDA_STRETCH, LAMBDA_PERP])**2
    np.testing.assert_allclose(values, expected, rtol=2e-15)
    np.testing.assert_allclose(np.sqrt(np.linalg.eigvalsh(C_OBS)), SIGMA_OBS*np.abs([LAMBDA_STRETCH, LAMBDA_PERP]), rtol=2e-15)
    np.testing.assert_allclose(vectors[:, 1] @ E, 0.0, atol=2e-15)
    phase = np.linspace(0, 2*np.pi, 361)
    mapped = A @ (SIGMA_OBS * np.stack((np.cos(phase), np.sin(phase))))
    np.testing.assert_allclose(np.einsum('in,ij,jn->n', mapped, np.linalg.inv(C_OBS), mapped),
                               1.0, rtol=2e-14, atol=2e-14)
    # An isotropic source floor prevents the narrow axis from vanishing at a critical curve.
    critical_a = LAMBDA_PERP * np.outer(N, N)
    critical_c = SIGMA_SRC**2*np.eye(2) + SIGMA_OBS**2*critical_a@critical_a.T
    np.testing.assert_allclose(np.linalg.eigvalsh(critical_c)[0], SIGMA_SRC**2, rtol=2e-15)
    return np.sqrt(values)


def arrow(ax, start, end, *, color=INK, lw=0.9, style="-|>", size=8, zorder=6):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=size,
                                linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=zorder))


def contour(ax, center, covariance, *, color, lw=1.3, linestyle="solid", zorder=5):
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, 1]
    angle = np.rad2deg(np.arctan2(direction[1], direction[0]))
    ax.add_patch(Ellipse(center, 2*SCALE*np.sqrt(values[1]), 2*SCALE*np.sqrt(values[0]),
                         angle=angle, facecolor="none", edgecolor=color, linewidth=lw,
                         linestyle=linestyle, zorder=zorder))


def arc(ax, center):
    """Contextual arc, locally tangent to E; not a ray-traced cluster image."""
    t = np.linspace(-1.0, 1.0, 240)
    curve = center + t[:, None]*E + .10*t[:, None]**2*N
    taper = np.maximum(0, 1-t**2)**.7
    for width, shade in ((.115, "#E8E8E8"), (.082, "#D0D0D0"), (.046, "#B5B5B5")):
        edge = width*taper[:, None]*N
        ax.add_patch(Polygon(np.concatenate((curve+edge, (curve-edge)[::-1])),
                             closed=True, facecolor=shade, edgecolor="none", zorder=1))


def label(ax, xy, text, **kwargs):
    defaults = dict(ha="center", va="center", fontsize=plt.rcParams["font.size"], color=INK, zorder=12)
    defaults.update(kwargs)
    return ax.text(*xy, text, **defaults)


def make_figure():
    semiaxes = verify_geometry()
    plt.style.use(str(STYLE))
    # Scale the shared typography for this compact multi-diagram figure.
    plt.rcParams.update({
        "font.size": 0.8 * plt.rcParams["font.size"],
        "legend.fontsize": 0.8 * plt.rcParams["legend.fontsize"],
        "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.grid": False, "savefig.facecolor": "white",
    })
    annotation_size = plt.rcParams["legend.fontsize"]
    fig = plt.figure(figsize=(7.1, 3.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 7.1), ylim=(0, 3.2), aspect="equal")
    ax.set_axis_off()
    image = np.array([1.08, 1.40])
    observed = np.array([3.10, 1.40])
    intrinsic = np.array([4.20, 1.40])
    total = np.array([5.94, 1.40])


    arc(ax, image)
    contour(ax, image, SIGMA_OBS**2*np.eye(2), color=BLUE, lw=1.5)
    arrow(ax, image-.85*E, image+.85*E, color=PURPLE, style="<->", lw=1.1)
    arrow(ax, image, image+.73*N, color=MUTED, lw=.8)
    label(ax, image+.87*N, r'$\boldsymbol{e}_{\perp}$', color=MUTED, fontsize=annotation_size)
    label(ax, image+.94*E-.12*N, r'$\boldsymbol{e}_{\rm stretch}$', color=PURPLE,
          fontsize=annotation_size, ha="center", rotation=ANGLE)
    ax.plot(*image, "o", ms=3.2, color=INK, zorder=10)
    arrow(ax, image, image+np.array([0, -.60]), color=BLUE, lw=.9)
    label(ax, image+np.array([.25, -.28]), r'$\sigma_{\rm obs}$', color=BLUE, fontsize=annotation_size)
    label(ax, (1.08, .42), "Isotropic\nastrometry", color=BLUE)
    label(ax, (1.08, .04), r'$\boldsymbol{\Sigma}_{i,\rm obs}=\sigma_{\rm obs}^{2}\boldsymbol{I}$', color=BLUE)

    # Local mapping: this is a differential map, not an inverse-Jacobian image-error ellipse.
    arrow(ax, (2.02, 1.40), (2.63, 1.40), color=MUTED, lw=1.0)
    label(ax, (2.33, 2.12),
          r'$\delta\boldsymbol{\beta}=\boldsymbol{A}_i\,\delta\boldsymbol{\theta}$', fontsize=annotation_size)

    contour(ax, observed, C_OBS, color=BLUE, lw=1.5)
    contour(ax, intrinsic, C_SRC, color=ORANGE, lw=1.5, linestyle=(0, (3, 1.8)))
    for center in (observed, intrinsic):
        ax.plot(*center, "o", ms=2.5, color=INK, zorder=10)
    label(ax, (3.66, 1.40), "+", fontsize=20, color=MUTED)
    label(ax, (3.10, .42), "Mapped\nastrometry", color=BLUE)
    label(ax, (4.20, .42), "Source\nscatter", color=ORANGE)
    label(ax, (3.10, .04), r'$\sigma_{\rm obs}^{2}\boldsymbol{A}_i\boldsymbol{A}_i^{\mathsf{T}}$', color=BLUE, fontsize=annotation_size)
    label(ax, (4.20, .04), r'$\sigma_{\rm src}^{2}\boldsymbol{I}$', color=ORANGE)
    label(ax, (4.91, 1.40), "=", fontsize=20, color=MUTED)

    # Component outlines share the total's center and scale; covariance matrices add, not radii.
    contour(ax, total, C_OBS, color="#7FAEC6", lw=.95, zorder=4)
    contour(ax, total, C_SRC, color="#CF9771", lw=.95, linestyle=(0, (3, 1.8)), zorder=4)
    contour(ax, total, C_TOTAL, color=INK, lw=1.7)
    # Match arrow tips to the total covariance contour, not arbitrary direction lengths.
    minor, major = SCALE * semiaxes
    for direction, radius in ((E, minor), (N, major)):
        endpoint = radius * direction / SCALE
        np.testing.assert_allclose(endpoint @ np.linalg.solve(C_TOTAL, endpoint), 1.0, rtol=2e-15)
    arrow(ax, total-major*N, total+major*N, color=INK, style="<->", lw=.8)
    arrow(ax, total-minor*E, total+minor*E, color=PURPLE, style="<->", lw=.95)
    ax.plot(*total, "o", ms=2.8, color=INK, zorder=10)
    label(ax, total+.71*N, r'$\boldsymbol{e}_{+}$', fontsize=annotation_size)
    label(ax, total+.45*E+np.array([.02, .10]),
          r'$\boldsymbol{e}_{-}=\boldsymbol{e}_{\rm stretch}$', color=PURPLE,
          fontsize=annotation_size, ha="left")
    label(ax, (5.94, .42), r'$\sigma_{+}=\sqrt{c_{+}}\quad\sigma_{-}=\sqrt{c_{-}}$', fontsize=annotation_size)
    label(ax, (5.94, .04), r'$q_C=\sigma_{+}/\sigma_{-}$', fontsize=annotation_size)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Trim the outer margins on all sides, preserving the relative glyph scales.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = [*ax.texts, *ax.patches, *ax.lines]
    content = Bbox.union([artist.get_window_extent(renderer) for artist in artists])
    content = content.transformed(fig.dpi_scale_trans.inverted())
    crop = Bbox.from_extents(content.x0 - .04, content.y0 - .04,
                             content.x1 + .04, content.y1 + .04)
    fig.savefig(OUT, bbox_inches=crop, metadata={"Title": "Local propagation of positional covariance in strong lensing",
                              "Subject": "Illustrative unit-Mahalanobis contours; 39.3 percent enclosed probability in 2D",
                              "Creator": "scripts/make_covariance_figure.py"})
    plt.close(fig)
    print(f"Verified source semiaxes: {semiaxes[0]:.12f}, {semiaxes[1]:.12f} arcsec")
    print(f"Verified covariance axis ratio: {semiaxes[1]/semiaxes[0]:.12f}")
    print(OUT)


if __name__ == "__main__":
    make_figure()
