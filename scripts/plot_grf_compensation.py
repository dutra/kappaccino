"""Does the GRF compensate for the real mass mismatch?  Three maps on one field: the true convergence residual of the
parametric model (what the field should equal), the GRF convergence, and the reduction of the absolute error.

Reads the saved image_catalog_mass_kappa_{model,grf}<suffix>.fits of a run with a truth map.  Target and field are
smoothed with the same Gaussian, the target's mean over the constrained region is removed (the field's uniform
convergence is projected out), and the field and the error reduction fade out away from the images.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree


def load_maps(run, maps_suffix='', full_arcsec=5., zero_arcsec=10.):
    """The saved parametric and GRF convergence maps of ``run``, the truth on the same pixels, the solver coordinates,
    and the weight of the constrained region: 1 within ``full_arcsec`` of the nearest image, linear to 0 at ``zero_arcsec``."""
    final = Path(run).resolve()/'final'
    parametric, header = fits.getdata(final/'fits'/f'image_catalog_mass_kappa_model{maps_suffix}.fits', header=True)
    grf, grf_header = fits.getdata(final/'fits'/f'image_catalog_mass_kappa_grf{maps_suffix}.fits', header=True)
    for field in ['CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2', 'ZSOURCE']:
        assert header[field] == grf_header[field]
    provenance = json.loads((final/'tables/image_catalog_mass_provenance.json').read_text())
    truth_native, truth_header = fits.getdata(provenance['truth_path'], header=True)
    assert np.isclose(truth_header['REDSHIFT'], header['ZSOURCE'])
    ny, nx = grf.shape
    yy, xx = np.indices(grf.shape)
    tx, ty = WCS(truth_header).world_to_pixel_values(*WCS(header).pixel_to_world_values(xx, yy))
    truth = map_coordinates(truth_native, [ty, tx], order=1, mode='constant', cval=np.nan)
    assert np.isfinite(truth).all(), 'Truth map does not cover the displayed field'
    # Solver coordinates of the pixel centres: x is positive west, so it runs against right ascension (mass_figure.overview_geometry).
    bounds = ({key: header[key.upper()] for key in ('xmin', 'xmax', 'ymin', 'ymax')} if 'XMIN' in header
              else provenance['bounds_arcsec'])
    xe = np.linspace(bounds['xmin'], bounds['xmax'], nx + 1)[::-1 if header['CDELT1'] > 0 else 1]
    ye = np.linspace(bounds['ymin'], bounds['ymax'], ny + 1)[::1 if header['CDELT2'] > 0 else -1]
    x, y = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2)
    images = pd.read_csv(final/'tables/image_fit_quality.csv')[['x_obs_arcsec', 'y_obs_arcsec']].to_numpy(float)
    distance = cKDTree(images).query(np.column_stack((x.ravel(), y.ravel())))[0].reshape(x.shape)
    weight = np.clip((zero_arcsec - distance) / (zero_arcsec - full_arcsec), 0., 1.)
    return SimpleNamespace(final=final, header=header, provenance=provenance, parametric=parametric, grf=grf, truth=truth,
                           bounds=bounds, x=x, y=y, extent=[xe[0], xe[-1], ye[0], ye[-1]], pixel=abs(xe[1] - xe[0]),
                           images=images, distance=distance, weight=weight)


def weighted_statistics(target, field, weight):
    """Agreement of ``field`` with ``target`` over the constrained region."""
    w = weight / weight.sum()
    mt, mf = np.sum(w * target), np.sum(w * field)
    covariance = np.sum(w * (target - mt) * (field - mf))
    variance_t, variance_f = np.sum(w * (target - mt)**2), np.sum(w * (field - mf)**2)
    gain = np.abs(target) - np.abs(target - field)
    return dict(correlation=float(covariance / np.sqrt(variance_t * variance_f)), slope=float(covariance / variance_f),
                rms_target=float(np.sqrt(np.sum(w * target**2))), rms_field=float(np.sqrt(np.sum(w * field**2))),
                rms_after=float(np.sqrt(np.sum(w * (target - field)**2))), improved_fraction=float(np.sum(w * (gain > 0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--maps-suffix', default='', help='Read image_catalog_mass_kappa_*<suffix>.fits and add the suffix to the figure name')
    parser.add_argument('--smooth-arcsec', type=float, default=2., help='Gaussian sigma applied to the target and the field')
    parser.add_argument('--full-arcsec', type=float, default=5., help='Distance to the nearest image within which the field is shown at full opacity')
    parser.add_argument('--zero-arcsec', type=float, default=10., help='Distance to the nearest image beyond which the field is not shown')
    args = parser.parse_args()
    m = load_maps(args.run, args.maps_suffix, args.full_arcsec, args.zero_arcsec)
    final, header, provenance, bounds, x, y, extent = m.final, m.header, m.provenance, m.bounds, m.x, m.y, m.extent
    images, distance, weight, pixel = m.images, m.distance, m.weight, m.pixel
    truth, parametric, grf = m.truth, m.parametric, m.grf

    sigma = args.smooth_arcsec / pixel
    target = gaussian_filter(truth - parametric, sigma, mode='nearest')
    field = gaussian_filter(grf, sigma, mode='nearest')
    offset = float(np.sum(weight * target) / np.sum(weight))
    target = target - offset
    gain = np.abs(target) - np.abs(target - field)

    statistics = dict(weighted_statistics(target, field, weight), removed_target_mean=offset)

    from kappaccino.plotting.style import PAPER_STYLE
    plt.style.use(Path(__file__).resolve().parents[1]/'src/kappaccino/plotting/style.mplstyle')
    plt.rcParams.update(PAPER_STYLE)
    from cmcrameri import cm
    limit = np.ceil(np.percentile(np.abs(target[weight > 0]), 95) * 100) / 100
    gain_limit = np.ceil(np.percentile(np.abs(gain[weight > 0]), 99) * 200) / 200
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.13, top=.83, wspace=.05)
    kappa_norm, gain_norm = TwoSlopeNorm(0., -limit, limit), TwoSlopeNorm(0., -gain_limit, gain_limit)
    panels = [(target, None, cm.vik, kappa_norm, r'$\kappa_{\rm true}-\kappa_{\rm parametric}$'),
              (field, weight, cm.vik, kappa_norm, r'$\delta\kappa_{\rm GRF}$'),
              (gain, weight, cm.bam, gain_norm, 'Error reduction')]
    meshes = []
    for ax, (values, alpha, cmap, norm, label) in zip(axes, panels):
        ax.set_facecolor('#d9d9d9')
        meshes.append(ax.imshow(values, extent=extent, origin='lower', cmap=cmap, norm=norm, alpha=alpha, interpolation='none'))
        ax.contour(x, y, distance, levels=[args.zero_arcsec], colors='#292929', linewidths=.6)
        # Image positions: a white ring under a dark one, legible on both ends of the colour scales.
        ax.scatter(images[:, 0], images[:, 1], s=7, facecolors='none', edgecolors='white', linewidths=1.1, zorder=5)
        ax.scatter(images[:, 0], images[:, 1], s=7, facecolors='none', edgecolors='#292929', linewidths=.5, zorder=6)
        ax.text(.04, .96, label, transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=2))
        ax.set(xlim=(bounds['xmin'], bounds['xmax']), ylim=(bounds['ymin'], bounds['ymax']), xlabel=r'$x$ (arcsec)')
        ax.set_aspect('equal')
    axes[0].set_ylabel(r'$y$ (arcsec)')
    fig.canvas.draw()
    left, right, last = axes[0].get_position(), axes[1].get_position(), axes[2].get_position()
    cax = fig.add_axes([left.x0, left.y1 + .03, right.x1 - left.x0, .025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=kappa_norm, cmap=cm.vik), cax=cax, orientation='horizontal', extend='both')
    cb.set_label(r'$\delta\kappa$ (left and middle)', labelpad=3)
    gax = fig.add_axes([last.x0, last.y1 + .03, last.width, .025])
    gb = fig.colorbar(plt.cm.ScalarMappable(norm=gain_norm, cmap=cm.bam), cax=gax, orientation='horizontal', extend='both')
    gb.set_label(r'Reduction of $|\kappa_{\rm model}-\kappa_{\rm true}|$', labelpad=3)
    for bar in (cb, gb):
        bar.ax.xaxis.set_label_position('top')
        bar.ax.xaxis.set_ticks_position('top')
        bar.ax.tick_params(labelsize=7, length=2.5)
    output = final/'figures'/f"{provenance['cluster']}_grf_compensation{args.maps_suffix}.pdf"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(
        run=str(args.run.resolve()), maps_suffix=args.maps_suffix, map_statistic=header.get('MAPSTAT', provenance['map_statistic']),
        z_source=float(header['ZSOURCE']), smoothing_sigma_arcsec=args.smooth_arcsec,
        weight=f'1 within {args.full_arcsec:g} arcsec of the nearest image, linear to 0 at {args.zero_arcsec:g} arcsec; the contour marks the outer limit',
        target='smoothed true minus parametric convergence, weighted mean removed (the field has no uniform convergence)',
        error_reduction='|target| - |target - smoothed GRF|; positive where the field moves the model toward the truth',
        color_limits=dict(kappa=float(limit), error_reduction=float(gain_limit)), weighted_statistics=statistics), indent=2) + '\n')
    print(json.dumps(statistics, indent=2))
    print(output)


if __name__ == '__main__':
    main()
