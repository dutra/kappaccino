"""Where does the mass field of an init best fit ask for more mass, and does a refit with halos there remove it?

Maps the shared conditional-mean GRF convergence at the init best fit of RUN over the image footprint, smooths it and
lists its strongest positive peaks inside the constrained region (written as JSON, the candidate positions of extra
halos).  With --after, a second panel shows the same field for a refit, on the same colour scale.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

from plot_grf_third_halo import init_best_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--after', type=Path, help='Refit whose field is drawn beside the first')
    parser.add_argument('--n-peaks', type=int, default=5)
    parser.add_argument('--mark', action='append', default=[], metavar='LABEL:X,Y', help='Reference position to draw (repeatable)')
    parser.add_argument('--smooth-arcsec', type=float, default=2.)
    parser.add_argument('--pixel-arcsec', type=float, default=0.4)
    parser.add_argument('--margin-arcsec', type=float, default=15.)
    parser.add_argument('--full-arcsec', type=float, default=5.)
    parser.add_argument('--zero-arcsec', type=float, default=10.)
    parser.add_argument('--peak-separation-arcsec', type=float, default=10.)
    args = parser.parse_args()
    from kappaccino.io import artifacts
    from kappaccino.io.image_catalog import load_image_catalog
    config = artifacts.load_config(args.run/'init/artifacts')
    z_source = float(config.truth.caustic_source_redshift)

    def fields(run, x=None, y=None):
        if x is None:
            # The grid needs the images in solver coordinates: build the state once through init_best_fit on a dummy point.
            _, _, _, cfg, state = init_best_fit(run, np.zeros((1, 1)), np.zeros((1, 1)), z_source, return_state=True)
            images = np.column_stack((np.asarray(state.images.x_obs), np.asarray(state.images.y_obs)))
            lo, hi = images.min(axis=0) - args.margin_arcsec, images.max(axis=0) + args.margin_arcsec
            x, y = np.meshgrid(np.arange(lo[0], hi[0], args.pixel_arcsec), np.arange(lo[1], hi[1], args.pixel_arcsec))
            return x, y, images
        _, grf, physical, cfg = init_best_fit(run, x, y, z_source)
        halos = {h.id: (physical.get(f'{h.id}.x_centre', getattr(h, 'x_centre', np.nan)), physical.get(f'{h.id}.y_centre', getattr(h, 'y_centre', np.nan)))
                 for h in cfg.model.large_halos if hasattr(h, 'x_centre')}
        scales = {k.split('.', 1)[1]: float(v) for k, v in physical.items() if k.startswith('images.')}
        return gaussian_filter(grf, args.smooth_arcsec / args.pixel_arcsec, mode='nearest'), halos, scales

    x, y, images = fields(args.run)
    distance = cKDTree(images).query(np.column_stack((x.ravel(), y.ravel())))[0].reshape(x.shape)
    weight = np.clip((args.zero_arcsec - distance) / (args.zero_arcsec - args.full_arcsec), 0., 1.)
    runs = [args.run] + ([args.after] if args.after else [])
    maps = [fields(run, x, y) for run in runs]
    before = maps[0][0]
    w = weight / weight.sum()
    rms = lambda values: float(np.sqrt(np.sum(w * values**2)))
    size = int(round(args.peak_separation_arcsec / args.pixel_arcsec))
    is_peak = (before == maximum_filter(before, size=size)) & (weight > .5) & (before > 0)
    order = np.argsort(before[is_peak])[::-1][:args.n_peaks]
    peaks = [dict(x=float(x[is_peak][k]), y=float(y[is_peak][k]), kappa=float(before[is_peak][k]), over_rms=float(before[is_peak][k] / rms(before)),
                  nearest_image_arcsec=float(distance[is_peak][k])) for k in order]
    marks = [(item.split(':')[0], *map(float, item.split(':')[1].split(','))) for item in args.mark]

    from cmcrameri import cm
    from kappaccino.plotting.style import PAPER_STYLE
    plt.style.use(Path(__file__).resolve().parents[1]/'src/kappaccino/plotting/style.mplstyle')
    plt.rcParams.update(PAPER_STYLE)
    limit = np.ceil(np.percentile(np.abs(before[weight > 0]), 99.5) * 500) / 500
    norm = TwoSlopeNorm(0., -limit, limit)
    extent = [x[0, 0] - args.pixel_arcsec / 2, x[0, -1] + args.pixel_arcsec / 2, y[0, 0] - args.pixel_arcsec / 2, y[-1, 0] + args.pixel_arcsec / 2]
    aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
    width = 3.3 * len(maps) + .9
    fig, axes = plt.subplots(1, len(maps), figsize=(width, 3.3 * aspect + .9), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    fig.subplots_adjust(left=.75 / width, right=1 - .75 / width, bottom=.5 / (3.3 * aspect + .9), top=1 - .42 / (3.3 * aspect + .9), wspace=.05)
    labels = ['Before', 'After'] if args.after else ['']
    for ax, (values, halos, _), label in zip(axes, maps, labels):
        ax.set_facecolor('#d9d9d9')
        ax.imshow(values, extent=extent, origin='lower', cmap=cm.vik, norm=norm, alpha=weight, interpolation='none')
        ax.contour(x, y, distance, levels=[args.zero_arcsec], colors='#292929', linewidths=.5)
        ax.scatter(images[:, 0], images[:, 1], s=3, facecolors='none', edgecolors='#292929', linewidths=.35, zorder=6)
        for name, (hx, hy) in halos.items():
            if np.isfinite(hx) and extent[0] < hx < extent[1] and extent[2] < hy < extent[3]:
                ax.scatter([hx], [hy], marker='D', s=22, c='white', edgecolors='black', linewidths=.7, zorder=8)
                ax.annotate(name, (hx, hy), xytext=(3, 3), textcoords='offset points', fontsize=6, zorder=9)
        for rank, peak in enumerate(peaks, 1):
            ax.scatter([peak['x']], [peak['y']], marker='x', s=36, c='black', linewidths=1.1, zorder=10)
            ax.annotate(str(rank), (peak['x'], peak['y']), xytext=(-9, -9), textcoords='offset points', fontsize=7, fontweight='bold', zorder=10)
        for name, mx, my in marks:
            ax.scatter([mx], [my], marker='o', s=70, facecolors='none', edgecolors='#00A651', linewidths=1.2, zorder=9)
        if label:
            ax.text(.04, .97, label, transform=ax.transAxes, va='top', fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel=r'$x$ (arcsec)')
        ax.set_aspect('equal')
    axes[0].set_ylabel(r'$y$ (arcsec)')
    handles = [Line2D([], [], marker='x', ls='none', ms=6, mec='black', label='Positive GRF peaks, ranked'),
               Line2D([], [], marker='D', ls='none', ms=5, mfc='white', mec='black', label='Halos of the fit')]
    if marks:
        handles.append(Line2D([], [], marker='o', ls='none', ms=8, mfc='none', mec='#00A651', label=', '.join(name for name, _, _ in marks)))
    handles.append(Line2D([], [], marker='o', ls='none', ms=3, mfc='none', mec='#292929', label='Images'))
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), frameon=False, fontsize=7, handletextpad=.4, columnspacing=1.)
    fig.canvas.draw()
    right = axes[-1].get_position()
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm.vik), cax=fig.add_axes([right.x1 + .012, right.y0, .014, right.height]), extend='both')
    bar.set_label(rf'$\delta\kappa_{{\rm GRF}}$ ($z_s={z_source:g}$)')
    bar.ax.tick_params(labelsize=7, length=2.5)
    target = (args.after or args.run)/'init/figures'
    target.mkdir(parents=True, exist_ok=True)
    output = target/(f'{config.paths.figure_prefix}_grf_peaks' + ('_before_after' if args.after else '') + '.pdf')
    fig.savefig(output, dpi=300)
    plt.close(fig)
    summary = dict(runs=[str(run) for run in runs], z_source=z_source, statistic='shared conditional mean at the init best fit',
                   smoothing_sigma_arcsec=args.smooth_arcsec, constrained_region=f'weight 1 within {args.full_arcsec:g}", 0 beyond {args.zero_arcsec:g}" of an image',
                   peaks=peaks, marks=[dict(label=name, x=mx, y=my, nearest_peak_arcsec=float(min(np.hypot(p['x'] - mx, p['y'] - my) for p in peaks)) if peaks else None,
                                            field_at_mark=[float(values[np.unravel_index(np.argmin(np.hypot(x - mx, y - my)), x.shape)]) for values, _, _ in maps])
                                       for name, mx, my in marks],
                   fields=[dict(run=str(run), rms=rms(values), minimum=float(values[weight > 0].min()), maximum=float(values[weight > 0].max()), error_scales=scales,
                                at_peaks=[float(values[np.unravel_index(np.argmin(np.hypot(x - p['x'], y - p['y'])), x.shape)]) for p in peaks])
                           for run, (values, _, scales) in zip(runs, maps)])
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(output)


if __name__ == '__main__':
    main()
