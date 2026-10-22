from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from image_catalog_mass import integrate_strips


def test_uniform_surface_density_partial_pixel_window():
    # Analytic integral of a constant sheet over a rectangle cutting all four edges.
    x, widths, profile, total = integrate_strips(
        np.full((2, 3), 7.0), [0, 1, 2, 3], [0, 2, 4], (0.25, 2.5, 0.5, 3.0),
    )
    np.testing.assert_allclose(widths, [0.75, 1, 0.5])
    np.testing.assert_allclose(x, [0.625, 1.5, 2.25])
    np.testing.assert_allclose(profile, 7 * 2.5)
    assert total == pytest.approx(7 * 2.25 * 2.5)


def test_nonuniform_strip_integral_preserves_mass():
    _, widths, profile, total = integrate_strips(
        [[1, 2], [4, 8]], [0, 2, 5], [0, 1, 3], (1, 4, 0.5, 2),
    )
    np.testing.assert_allclose(profile, [4.5, 9])
    np.testing.assert_allclose(widths, [1, 2])
    assert total == pytest.approx(22.5)
    assert np.dot(profile, widths) == pytest.approx(total)


def test_transposed_nonuniform_profile_matches_both_integrals():
    density = np.array([[1, 2], [4, 8]])
    xe, ye = [0, 2, 5], [0, 1, 3]
    _, _, _, x_total = integrate_strips(density, xe, ye, (1, 4, 0.5, 2))
    centers, widths, profile, y_total = integrate_strips(
        density.T, ye, xe, (0.5, 2, 1, 4))
    np.testing.assert_allclose(centers, [0.75, 1.5])
    np.testing.assert_allclose(widths, [0.5, 1])
    np.testing.assert_allclose(profile, [5, 20])
    assert np.dot(profile, widths) == pytest.approx(22.5)
    assert y_total == pytest.approx(x_total)


@pytest.mark.parametrize("bounds", [(-1, 1, 0, 1), (0, 2, 0, 3), (1, 0, 0, 1)])
def test_reject_incomplete_or_reversed_integration_window(bounds):
    with pytest.raises(ValueError, match="fully covered"):
        integrate_strips(np.ones((2, 2)), [0, 1, 2], [0, 1, 2], bounds)


from types import SimpleNamespace
import pandas as pd
from astropy.wcs import WCS
from astropy.io import fits
from lenscluster import image_catalog_mass as mass


def geometry_fixture(monkeypatch, reverse=False):
    reference=(3,342.18321,-44.530878)
    wcs=WCS(naxis=2)
    wcs.wcs.ctype=['RA---TAN','DEC--TAN']
    wcs.wcs.crval=reference[1:]
    wcs.wcs.crpix=[50,50]
    wcs.wcs.cdelt=[(1 if reverse else -1)/3600,1/3600]
    image=SimpleNamespace(wcs=wcs,pixel_scale_arcsec=1.)
    helpers=SimpleNamespace(_cutout_window=lambda *a,**k: SimpleNamespace(npix=100,x0=0,y0=0))
    monkeypatch.setattr(mass.plotting,'_image_catalog_cluster_overview_geometry',lambda *a:(0.,0.,100.))
    geometry=mass.overview_geometry(helpers,image,None,None,reference,resolution=16)
    return geometry,reference


@pytest.mark.parametrize('reverse',[False,True])
def test_mass_grid_wcs_and_marginal_alignment(monkeypatch,reverse):
    geometry,reference=geometry_fixture(monkeypatch,reverse)
    x,y=geometry['x'],geometry['y']
    ra,dec=mass.plotting._solver_arcsec_offsets_to_radec(x,y,reference)
    px,py=geometry['wcs'].world_to_pixel_values(ra,dec)
    gx,gy=mass.grid_pixels(geometry,x,y)
    np.testing.assert_allclose(px,gx,atol=1e-7)
    np.testing.assert_allclose(py,gy,atol=1e-7)
    assert (geometry['xlim'][1]<geometry['xlim'][0]) == reverse


@pytest.mark.parametrize('with_truth',[False,True])
@pytest.mark.parametrize('varying',[False,True])
def test_optional_truth_maps_profiles_and_empty_contours(tmp_path,monkeypatch,with_truth,varying):
    import matplotlib.pyplot as plt
    geometry,reference=geometry_fixture(monkeypatch)
    source=tmp_path/'truth.fits'
    # A constant, low-density sheet also tests the absence of contour crossings.
    header=geometry['wcs'].to_header();header['ZSOURCE']=9.
    fits.writeto(source,np.full((16,16),.001),header)
    evaluator=SimpleNamespace(state=SimpleNamespace(reference=reference,z_lens=.348,
        cosmo_config={'class':'FlatLambdaCDM','H0':70.,'Om0':.3}))
    values = np.broadcast_to(np.linspace(.01,3.,16),(16,16)) if varying else np.full((16,16),.001)
    monkeypatch.setattr(mass.plotting,'_lens_model_quantity_on_arcsec_grid',lambda *a:values)
    args=SimpleNamespace(caustic_source_redshift=9.,kappa_true_fits=str(source) if with_truth else None)
    paths=SimpleNamespace(tables=tmp_path/'tables',fits=tmp_path/'fits')
    catalog=pd.DataFrame({'x_obs_arcsec':[0.,20.],'y_obs_arcsec':[0.,-10.]})
    fig=plt.figure(figsize=(7.1,7.1));grid=fig.add_gridspec(2,2,width_ratios=[3.2,1],height_ratios=[3.2,1],left=.09,right=.95,bottom=.10,top=.96,wspace=.025,hspace=.025)
    ax=fig.add_subplot(grid[0,0]);ax.set_xlim(-.5,15.5);ax.set_ylim(-.5,15.5);ax.set_aspect('equal')
    try:
        provenance=mass.add_mass_panels(fig,paths,evaluator,np.array([]),args,geometry,catalog,{'reference_sample_kind':'maximum-likelihood'})
        assert len(fig.axes)==4
        assert fig.axes[1].get_yscale() == 'linear'
        assert fig.axes[2].get_xscale() == 'linear'
        assert provenance['truth_available'] == with_truth
        assert bool(provenance['contour_levels_msun_per_kpc2']) == varying
        x=pd.read_csv(paths.tables/'image_catalog_mass_profile.csv')
        y=pd.read_csv(paths.tables/'image_catalog_mass_profile_y.csv')
        assert ('truth_dMdx_msun_per_arcsec' in x)==with_truth
        assert 'subhalos_dMdx_msun_per_arcsec' not in x
        assert 'subhalos_dMdy_msun_per_arcsec' not in y
        np.testing.assert_allclose(np.dot(x.width_arcsec,x.model_dMdx_msun_per_arcsec),
                                   np.dot(y.width_arcsec,y.model_dMdy_msun_per_arcsec),rtol=1e-12)
        assert provenance['coordinate_alignment_max_fractional_error_x']<1e-7
    finally:
        plt.close(fig)


def test_configured_truth_errors_are_not_silently_omitted(tmp_path,monkeypatch):
    geometry,reference=geometry_fixture(monkeypatch)
    evaluator=SimpleNamespace(state=SimpleNamespace(reference=reference,z_lens=.348,
        cosmo_config={'class':'FlatLambdaCDM','H0':70.,'Om0':.3}))
    monkeypatch.setattr(mass.plotting,'_lens_model_quantity_on_arcsec_grid',lambda *a:np.ones((16,16)))
    args=SimpleNamespace(caustic_source_redshift=9.,kappa_true_fits=str(tmp_path/'missing.fits'))
    with pytest.raises(FileNotFoundError):
        mass.prepare_mass_maps(evaluator,np.array([]),args,geometry)
    header=geometry['wcs'].to_header();header['ZSOURCE']=2.
    fits.writeto(args.kappa_true_fits,np.ones((16,16)),header)
    with pytest.raises(ValueError,match='source redshift'):
        mass.prepare_mass_maps(evaluator,np.array([]),args,geometry)
