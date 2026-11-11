"""Which mass mismatch does the GRF correct: the member galaxies' peaks or the smooth cluster halo?

Top row, member scale: the true convergence residual of the parametric model and the GRF convergence, stacked in
annuli about the member galaxies of the constrained region, and their maps on scales below the split.
Bottom row, halo scale: the same two quantities above the split, along the axis through the two main halos, and
their maps.  Where the field corrects the model the GRF curve follows the residual curve.
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
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates

from plot_grf_compensation import load_maps, weighted_statistics

TARGET_COLOR, FIELD_COLOR = '#292929', '#D55E00'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--maps-suffix', default='', help='Read image_catalog_mass_kappa_*<suffix>.fits and add the suffix to the figure name')
    parser.add_argument('--split-arcsec', type=float, default=5., help='Gaussian sigma separating the member scale from the halo scale')
    parser.add_argument('--smooth-arcsec', type=float, default=1., help='Gaussian sigma of the member-scale maps')
    parser.add_argument('--member-magnitude', type=float, default=20.5, help='Faintest catalog magnitude of the stacked members')
    parser.add_argument('--stack-radius-arcsec', type=float, default=8.)
    args = parser.parse_args()
    m = load_maps(args.run, args.maps_suffix)
    x, y, pixel, weight = m.x, m.y, m.pixel, m.weight
    residual = m.truth - m.parametric
    residual = residual - np.sum(weight * residual) / np.sum(weight)  # the field has no uniform convergence
    large = {name: gaussian_filter(values, args.split_arcsec / pixel, mode='nearest') for name, values in (('target', residual), ('field', m.grf))}
    small = {name: gaussian_filter(values, args.smooth_arcsec / pixel, mode='nearest') - large[name]
             for name, values in (('target', residual), ('field', m.grf))}
    statistics = dict(member_scale=weighted_statistics(small['target'], small['field'], weight),
                      halo_scale=weighted_statistics(large['target'], large['field'], weight))

    def pixels(px, py):
        return [(np.asarray(py) - y[0, 0]) / (y[1, 0] - y[0, 0]), (np.asarray(px) - x[0, 0]) / (x[0, 1] - x[0, 0])]

    # The two main halos at their posterior medians; the members at their centres are part of the halo scale.
    state = json.loads((args.run/'init/artifacts/state.json').read_text())
    config = json.loads((m.final/'artifacts/config.json').read_text())
    with np.load(m.final/'artifacts/posterior.npz') as posterior:
        fitted = dict(zip((p['name'] for p in state['parameter_specs']), np.median(posterior['physical'], axis=0)))
    halos = np.array([[fitted.get(f"{h['id']}.{axis}", h[axis]) for axis in ('x_centre', 'y_centre')]
                      for h in config['model']['large_halos'] if h.get('x_centre') is not None][:2], dtype=float)
    assert len(halos) == 2, 'The halo-scale cut needs two cluster halos'

    members = pd.read_csv(m.final/'tables/member_diagnostics.csv')
    xy = members[['x_centre', 'y_centre']].to_numpy(float)
    member_weight = map_coordinates(weight, pixels(xy[:, 0], xy[:, 1]), order=1, mode='constant', cval=0.)
    at_halo = np.min(np.hypot(*(xy[:, None] - halos[None]).transpose(2, 0, 1)), axis=1) < 2.
    stacked = xy[(member_weight > .5) & (pd.to_numeric(members.catalog_mag, errors='coerce') < args.member_magnitude) & ~at_halo]
    edges = np.arange(0., args.stack_radius_arcsec + .5, .5)
    radius = (edges[:-1] + edges[1:]) / 2
    profiles = {name: np.full((len(stacked), len(radius)), np.nan) for name in ('target', 'field')}
    for index, (cx, cy) in enumerate(stacked):
        r = np.hypot(x - cx, y - cy)
        which = np.digitize(r, edges) - 1
        for name, values in (('target', residual), ('field', m.grf)):
            for k in range(len(radius)):
                profiles[name][index, k] = values[which == k].mean()

    # The cut through both halos, from 25 arcsec before the first to 25 arcsec beyond the second.
    separation = np.hypot(*(halos[1] - halos[0]))
    direction = (halos[1] - halos[0]) / separation
    s = np.arange(-25., separation + 25., .25)
    line = halos[0] + s[:, None] * direction
    cut = {name: map_coordinates(large[name], pixels(line[:, 0], line[:, 1]), order=1, mode='nearest') for name in large}
    cut_weight = map_coordinates(weight, pixels(line[:, 0], line[:, 1]), order=1, mode='constant', cval=0.)
    inside = cut_weight > 0

    from cmcrameri import cm
    from kappaccino.plotting.style import PAPER_STYLE
    plt.style.use(Path(__file__).resolve().parents[1]/'src/kappaccino/plotting/style.mplstyle')
    plt.rcParams.update(PAPER_STYLE)
    fig = plt.figure(figsize=(7.2, 4.7))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, .045], left=.095, right=.925, bottom=.095, top=.985, wspace=.07, hspace=.22)

    ax = fig.add_subplot(grid[0, 0])
    for name, color, label in (('target', TARGET_COLOR, r'$\kappa_{\rm true}-\kappa_{\rm parametric}$'), ('field', FIELD_COLOR, r'$\delta\kappa_{\rm GRF}$')):
        mean = profiles[name].mean(axis=0)
        error = profiles[name].std(axis=0, ddof=1) / np.sqrt(len(stacked))
        ax.fill_between(radius, mean - error, mean + error, color=color, alpha=.25, linewidth=0)
        ax.plot(radius, mean, color=color, lw=1.3, label=label)
    ax.axhline(0., color='#888888', lw=.6, zorder=0)
    ax.set(xlabel='Distance from member (arcsec)', ylabel=r'$\delta\kappa$', xlim=(0, edges[-1]))
    ax.legend(loc='lower right', frameon=False)
    ax.text(.95, .55, f'{len(stacked)} members stacked', transform=ax.transAxes, ha='right', fontsize=8)
    ax.set_box_aspect(1)

    ax = fig.add_subplot(grid[1, 0])
    for name, color in (('target', TARGET_COLOR), ('field', FIELD_COLOR)):
        ax.plot(s[inside], cut[name][inside], color=color, lw=1.3)
    ax.axhline(0., color='#888888', lw=.6, zorder=0)
    for position, halo in zip((0., separation), config['model']['large_halos']):
        ax.axvline(position, color='#888888', lw=.6, ls='--', zorder=0)
        ax.annotate(f"H{halo['id']}", (position, 1.), xycoords=('data', 'axes fraction'), xytext=(3, -3), textcoords='offset points', va='top', fontsize=8)
    ax.set(xlabel='Position along the halo axis (arcsec)', ylabel=r'$\delta\kappa$')
    ax.set_box_aspect(1)

    rows = [(small, stacked, None, f'Scales below {args.split_arcsec:g}"'), (large, None, line, f'Scales above {args.split_arcsec:g}"')]
    for row, (band, marks, path, label) in enumerate(rows):
        limit = np.ceil(np.percentile(np.abs(band['target'][weight > 0]), 97) * 200) / 200
        norm = TwoSlopeNorm(0., -limit, limit)
        for column, (name, alpha, text) in enumerate((('target', None, r'$\kappa_{\rm true}-\kappa_{\rm parametric}$'), ('field', weight, r'$\delta\kappa_{\rm GRF}$'))):
            ax = fig.add_subplot(grid[row, column + 1])
            ax.set_facecolor('#d9d9d9')
            ax.imshow(band[name], extent=m.extent, origin='lower', cmap=cm.vik, norm=norm, alpha=alpha, interpolation='none')
            ax.contour(x, y, m.distance, levels=[10.], colors='#292929', linewidths=.6)
            ax.scatter(m.images[:, 0], m.images[:, 1], s=5, facecolors='none', edgecolors='white', linewidths=1., zorder=5)
            ax.scatter(m.images[:, 0], m.images[:, 1], s=5, facecolors='none', edgecolors='#292929', linewidths=.45, zorder=6)
            if marks is not None:
                ax.scatter(marks[:, 0], marks[:, 1], s=16, marker='+', color='#00A651', linewidths=.9, zorder=7)
            if path is not None:
                ax.plot(path[inside, 0], path[inside, 1], color='#00A651', lw=1., zorder=7)
            ax.text(.04, .96, text, transform=ax.transAxes, va='top', fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
            if column == 0:
                ax.text(.04, .04, label, transform=ax.transAxes, va='bottom', fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
            ax.set(xlim=(m.bounds['xmin'], m.bounds['xmax']), ylim=(m.bounds['ymin'], m.bounds['ymax']), xlabel=r'$x$ (arcsec)' if row else '')
            ax.set_aspect('equal')
            ax.tick_params(labelleft=False, labelbottom=bool(row))
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm.vik), cax=fig.add_subplot(grid[row, 3]), extend='both')
        bar.set_label(r'$\delta\kappa$')
        bar.ax.tick_params(labelsize=7, length=2.5)
    output = m.final/'figures'/f"{m.provenance['cluster']}_grf_scale_split{args.maps_suffix}.pdf"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    peak = {name: float(profiles[name][:, :2].mean()) for name in profiles}
    summary = dict(run=str(args.run.resolve()), maps_suffix=args.maps_suffix, map_statistic=m.header.get('MAPSTAT', m.provenance['map_statistic']),
                   split_sigma_arcsec=args.split_arcsec, member_map_sigma_arcsec=args.smooth_arcsec,
                   stack=dict(members=len(stacked), faintest_magnitude=args.member_magnitude, annulus_arcsec=.5,
                              selection='constrained-region weight > 0.5; not within 2 arcsec of a cluster halo centre',
                              mean_within_1_arcsec=peak, band='standard error of the mean over members'),
                   halo_axis=dict(halos=halos.tolist(), cut='maps above the split, sampled every 0.25 arcsec inside the constrained region'),
                   target='true minus parametric convergence, weighted mean over the constrained region removed', weighted_statistics=statistics)
    output.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(weighted_statistics=statistics, stack_mean_within_1_arcsec=peak, members=len(stacked)), indent=2))
    print(output)


if __name__ == '__main__':
    main()
