"""Build the paper's anisotropic covariance and activation-function figure.

Run with /home/dutra/.conda/envs/lenstronomy/bin/python.
Output: plots/paper/covariance.pdf, from any working directory.
Parameters are illustrative. The ellipse is a unit-Mahalanobis contour
(39.3% enclosed probability in two dimensions), not a 68% contour.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "plots" / "paper"
BLUE = "#28799B"
ORANGE = "#C26935"
INK = "#24272B"
MUTED = "#626971"
ANGLE = 28.0
SIGMA = 1.0
V = 8.0 * SIGMA**2  # Illustrative: parallel standard deviation = 3 sigma.
E = np.array([np.cos(np.deg2rad(ANGLE)), np.sin(np.deg2rad(ANGLE))])
N = np.array([-E[1], E[0]])
BASE = SIGMA**2 * np.eye(2)
ANISO = BASE + V * np.outer(E, E)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "savefig.facecolor": "white",
})


def arrow(ax, start, end, color=INK, lw=1.0, style="-|>", zorder=8):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0, zorder=zorder,
    ))


def arc(ax):
    """A faint, gently curved strip, locally tangent to the critical direction."""
    t = np.linspace(-4.15, 4.05, 260)
    center = t[:, None] * E + 0.039 * t[:, None] ** 2 * N
    taper = np.maximum(0, 1 - (t / 4.2) ** 2) ** 0.65
    # Nested vector polygons make a soft grey contextual arc.
    for width, color in [(0.29, "#F1F1F1"), (0.22, "#E7E7E7"),
                         (0.15, "#DCDCDC"), (0.075, "#CFCFCF")]:
        offset = (width * taper)[:, None] * N
        boundary = np.concatenate((center + offset, (center - offset)[::-1]))
        ax.add_patch(Polygon(boundary, closed=True, facecolor=color,
                             edgecolor="none", zorder=1))


def axes_compass(ax):
    origin = np.array([-3.75, 1.46])
    arrow(ax, origin, origin + [0.63, 0], color=MUTED, lw=0.65)
    arrow(ax, origin, origin + [0, 0.63], color=MUTED, lw=0.65)
    ax.text(*(origin + [0.76, -0.02]), r"$r_{i,x}$", fontsize=8,
            color=MUTED, va="center", ha="left")
    ax.text(*(origin + [-0.04, 0.74]), r"$r_{i,y}$", fontsize=8,
            color=MUTED, va="bottom", ha="center")


def make_figure():
    OUT.mkdir(parents=True, exist_ok=True)
    # Verify the illustrative covariance geometry.
    np.testing.assert_allclose(np.linalg.eigvalsh(ANISO), [SIGMA**2, 9*SIGMA**2])
    plt.rcParams.update({'axes.grid': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .7})
    fig = plt.figure(figsize=(7.25, 3.6))
    left = fig.add_axes([.025, .20, .46, .63])
    right = fig.add_axes([.60, .25, .37, .55])
    left.set(xlim=(-4.45, 4.45), ylim=(-2.7, 3.35), aspect='equal')
    left.set_axis_off()
    arc(left)
    axes_compass(left)
    e, n = E, N
    left.plot(*np.array([-3.6*e, 3.9*e]).T, color=MUTED, lw=.7, ls='--')
    arrow(left, 3.6*e, 3.95*e, color=MUTED, lw=.8)
    left.text(*(3.8*e+.3*n), r'$\boldsymbol{e}_{i,\mathrm{crit}}$', ha='center', fontsize=11)
    left.add_patch(Ellipse((0,0), 6, 2, angle=ANGLE, facecolor='none', edgecolor=ORANGE, lw=1.6))
    left.add_patch(Ellipse((0,0), 2, 2, facecolor='none', edgecolor=BLUE,
                          lw=1.1, ls=(0, (3, 2.5)), zorder=6))
    left.plot(0,0,'o',color=INK,ms=3.5,zorder=10)
    arrow(left, np.zeros(2), 3*e, color=ORANGE)
    arrow(left, np.zeros(2), n, color=ORANGE)
    left.text(*(-.43*e+.45*n),r'$\sigma_i$',color=ORANGE,
              ha='center',va='center',fontsize=12,zorder=12)
    left.text(*(1.7*e+.75*n),r'$\sqrt{\sigma_i^2+v_i}$',color=ORANGE,
              ha='center',rotation=ANGLE,fontsize=11)
    fig.text(.03,.91,'(a)',fontsize=11)
    fig.text(.57,.91,'(b)',fontsize=11)
    fig.text(.27,.91,r'$\boldsymbol{\Sigma}_i=\sigma_i^2\boldsymbol{I}+v_i\boldsymbol{e}_{i,\mathrm{crit}}\boldsymbol{e}_{i,\mathrm{crit}}^{\mathsf{T}}$',ha='center',fontsize=11)
    fig.text(.79,.91,r'$g_i=\left[1+\exp\!\left(\frac{s_{\mathrm{min},i}-s_{\mathrm{th}}}{s_{\mathrm{soft}}}\right)\right]^{-1}$',ha='center',fontsize=11)
    x = np.linspace(0,3,601)
    soft = .15  # s_soft / s_th
    activation = 1/(1+np.exp((x-1)/soft))
    assert np.all(np.diff(activation)<0)
    assert np.isclose(1/(1+np.exp(0)),.5)
    right.plot(x,activation,color=BLUE,lw=1.7)
    right.axvline(1,color='.65',lw=.7,ls=':')
    right.plot(1,.5,'o',color=BLUE,ms=3.5)
    right.text(.10,.79,'Arc-like',color=ORANGE,fontsize=10)
    right.text(1.76,.79,'Point-like',color=BLUE,fontsize=10)
    right.text(.10,.68,r'$g_i\simeq1$',color=MUTED,fontsize=10)
    right.text(1.76,.68,r'$g_i\simeq0$',color=MUTED,fontsize=10)
    right.annotate('',xy=(1+soft,.19),xytext=(1-soft,.19),
                   arrowprops={'arrowstyle':'<->','lw':.8,'color':INK,'shrinkA':0,'shrinkB':0})
    right.text(1,.07,r'$2s_{\mathrm{soft}}$',ha='center',fontsize=9)
    right.set(xlim=(0,3),ylim=(-.02,1.08),ylabel=r'$g_i$',xlabel=r'$s_{\mathrm{min},i}$')
    right.set_xticks([0,1,2,3],[r'$0$',r'$s_{\mathrm{th}}$',r'$2s_{\mathrm{th}}$',r'$3s_{\mathrm{th}}$'])
    right.set_yticks([0,.5,1],['0','0.5','1'])
    right.tick_params(direction='in',labelsize=9)
    fig.text(.265,.15,r'$v_i=g_i\left[(\sigma_{i,\mathrm{obs}}/s_{\mathrm{min},i})^2+\sigma_{\mathrm{arc}}^2\right]$',ha='center',fontsize=11)
    fig.text(.5,.04,r'Illustrative: $v_i=8\sigma_i^2$ in (a); $s_{\mathrm{soft}}=0.15s_{\mathrm{th}}$ in (b).',ha='center',fontsize=9,color=MUTED)
    fig.savefig(OUT/'covariance.pdf')
    plt.close(fig)
    print(OUT/'covariance.pdf')


if __name__ == "__main__":
    make_figure()
