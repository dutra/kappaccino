"""Plot a saved signed GRF convergence map with matching parametric contours."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--relative', action='store_true', help='Show the correction as a fraction of total convergence')
    parser.add_argument('--cmap', default='vik')
    parser.add_argument('--maps-suffix', default='', help='Read image_catalog_mass_kappa_*<suffix>.fits and add the suffix to the figure name')
    args = parser.parse_args()
    final = args.run.resolve() / 'final'
    paths = {key: final/'fits'/f'image_catalog_mass_kappa_{suffix}{args.maps_suffix}.fits'
             for key, suffix in [('parametric', 'model'), ('grf', 'grf'), ('total', 'model_grf')]}
    maps, headers = {}, {}
    for key, path in paths.items():
        maps[key], headers[key] = fits.getdata(path, header=True)
    header = headers['grf']
    for key in maps:
        assert maps[key].shape == maps['grf'].shape
        for field in ['CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CRVAL1', 'CRVAL2', 'ZSOURCE']:
            assert headers[key][field] == header[field]
    np.testing.assert_allclose(maps['total'], maps['parametric'] + maps['grf'], atol=1e-12)
    provenance = json.loads((final/'tables/image_catalog_mass_provenance.json').read_text())
    prefix = provenance['cluster']
    truth_path = Path(provenance['truth_path'])
    truth_native, truth_header = fits.getdata(truth_path, header=True)
    assert np.isclose(truth_header['REDSHIFT'], header['ZSOURCE'])
    yy, xx = np.indices(maps['grf'].shape)
    sky = WCS(header).pixel_to_world_values(xx, yy)
    tx, ty = WCS(truth_header).world_to_pixel_values(*sky)
    truth = map_coordinates(truth_native, [ty, tx], order=1, mode='constant', cval=np.nan)
    assert np.isfinite(truth).all(), 'Truth map does not cover the displayed field'
    ny, nx = maps['grf'].shape
    # Solver coordinates of the pixel centres: x is positive west, so it runs against right ascension (mass_figure.overview_geometry).
    bounds = ({key: header[key.upper()] for key in ('xmin', 'xmax', 'ymin', 'ymax')} if 'XMIN' in header
              else provenance['bounds_arcsec'])  # maps on another field than the mass figure carry their own bounds
    xe = np.linspace(bounds['xmin'], bounds['xmax'], nx + 1)[::-1 if header['CDELT1'] > 0 else 1]
    ye = np.linspace(bounds['ymin'], bounds['ymax'], ny + 1)[::1 if header['CDELT2'] > 0 else -1]
    x, y = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    levels = [.25, .5, 1., 2., 4.]
    values = maps['grf']
    if args.relative:
        if not np.all(maps['total'] > 0):
            raise ValueError('Relative correction requires positive total convergence')
        values = values / maps['total']
    limit = float(np.max(np.abs(values)))
    limit = np.ceil(limit * 20) / 20
    style = Path(__file__).resolve().parents[1]/'src/kappaccino/plotting/style.mplstyle'
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(6.8, 5.7), layout='constrained')
    cmap = args.cmap
    if cmap in {'berlin', 'managua', 'vik'}:
        from cmcrameri import cm
        cmap = getattr(cm, cmap)
    elif cmap == 'cmocean.balance':
        from cmocean import cm
        cmap = cm.balance
    im = ax.imshow(values, extent=extent, origin='lower', cmap=cmap,
                   norm=TwoSlopeNorm(vmin=-limit, vcenter=0., vmax=limit), interpolation='none')
    contour_color = 'white' if args.cmap in {'berlin', 'managua'} else '#292929'
    ax.contour(x, y, maps['parametric'], levels=levels, colors=contour_color, linewidths=1.15, linestyles='--')
    contours = ax.contour(x, y, maps['total'], levels=levels, colors=contour_color, linewidths=1.4, linestyles='-')
    ax.clabel(contours, inline=True, fontsize=9, fmt='%g')
    ax.contour(x, y, truth, levels=levels, colors='#0066FF', linewidths=1.25, linestyles=':')
    ax.set(xlabel=r'$x$ (arcsec)', ylabel=r'$y$ (arcsec)', xlim=(bounds['xmin'], bounds['xmax']), ylim=(bounds['ymin'], bounds['ymax']))
    handles = [Line2D([], [], color='#292929', lw=1.15, ls='--', label=r'Parametric $\kappa$'),
               Line2D([], [], color='#292929', lw=1.4, label=r'Parametric + GRF'),
               Line2D([], [], color='#0066FF', lw=1.25, ls=':', label=r'True $\kappa$')]
    ax.legend(handles=handles, loc='upper left', frameon=True, facecolor='white', edgecolor='none', framealpha=.9, fontsize=11)
    ax.text(.97, .97, f'$z_s={header["ZSOURCE"]:g}$', transform=ax.transAxes, ha='right', va='top', fontsize=12, color=contour_color)
    cb = fig.colorbar(im, ax=ax, fraction=.048, pad=.025)
    cb.set_label(r'$\delta\kappa_{\rm GRF}/\kappa_{\rm total}$' if args.relative else r'$\delta\kappa_{\rm GRF}$')
    suffix = ('_relative' if args.relative else '') + args.maps_suffix
    if args.cmap != 'vik':
        suffix += '_' + args.cmap
    output = final/'figures'/f'{prefix}_grf_parametric_contours{suffix}.pdf'
    fig.savefig(output, bbox_inches='tight', pad_inches=.04, dpi=300)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(
        run=str(args.run.resolve()), inputs={k:str(p) for k,p in paths.items()},
        model_provenance=provenance, map_statistic=header.get('MAPSTAT', provenance['map_statistic']), quantity='signed conditional-mean convergence correction; not a random field draw',
        truth_source=str(truth_path), truth_alignment='native truth map bilinearly sampled through both WCSs; matching source redshift',
        contour_levels=levels, color_limits=[-limit, limit],
        colormap=args.cmap,
        display='GRF / (parametric + GRF) convergence' if args.relative else 'GRF convergence',
        display_minimum=float(values.min()), display_maximum=float(values.max()),
        minimum=float(maps['grf'].min()), maximum=float(maps['grf'].max()),
        validation='matching WCS and source redshift; total equals parametric plus GRF to atol=1e-12'), indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
