"""Shared publication-sized overrides for style.mplstyle."""

# Critical curves and their uncertainty bands across all production figures.
CRITICAL_COLOR = '#559DFF'

# Scale the shared stylesheet for single-column figures and a 7.1-inch panel array.
PAPER_STYLE = {"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
               "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": .9,
               "xtick.major.size": 3.5, "ytick.major.size": 3.5,
               "xtick.major.width": .8, "ytick.major.width": .8,
               "xtick.minor.size": 2, "ytick.minor.size": 2,
               "xtick.minor.width": .5, "ytick.minor.width": .5,
               "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42}

# Paper palette (Okabe-Ito): separable under the common colour-vision deficiencies.  Blue is this work's model, orange
# the summary or reference drawn over it; in the mock figures the components keep one colour each.
BLUE, ORANGE, GREEN, PINK, YELLOW = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"
APJ_TEXT_WIDTH_IN = 7.1               # full width of a two-column ApJ page
APJ_COLUMN_WIDTH_IN = 3.35            # one column
# Line figures of the paper: fine frames and ticks, text sizes for a figure printed at its own size.
FIGURE_STYLE = {**PAPER_STYLE, "font.size": 8.5, "axes.labelsize": 9.5, "legend.fontsize": 8,
                "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": .6, "axes.labelpad": 3,
                "xtick.major.size": 3.2, "ytick.major.size": 3.2, "xtick.major.width": .6, "ytick.major.width": .6,
                "xtick.minor.size": 1.7, "ytick.minor.size": 1.7, "xtick.minor.width": .45, "ytick.minor.width": .45,
                "lines.solid_capstyle": "round", "legend.handletextpad": .5, "legend.labelspacing": .35}
DASHES = (0, (4, 2.5))                # the exact-recovery reference line


def save_page(fig, path):
    """Save at the figure's own size: paper figures are drawn at the journal's column or page width, and a tight box
    would change it."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox
    fig.savefig(path, dpi=300, bbox_inches=Bbox.from_bounds(0, 0, *fig.get_size_inches()))
    plt.close(fig)
    return path
