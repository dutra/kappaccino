import numpy as np

from kappaccino.plotting.zeta_masses import aperture_sums, pixel_weights, zeta_and_contrast


def test_pixel_weights_cover_the_circle_area():
    _, weights = pixel_weights((50.3, 49.7), 10.0, (100, 100))
    assert abs(weights.sum() - np.pi * 100.0) / (np.pi * 100.0) < 0.01


def test_zeta_removes_a_uniform_sheet_and_keeps_a_point_mass():
    shape = (101, 101)
    weights = [(pixel_weights((50.0, 50.0), 8.0, shape), pixel_weights((50.0, 50.0), 16.0, shape))]
    sheet = np.full(shape, 0.3)
    zeta, contrast = zeta_and_contrast(aperture_sums(sheet.ravel(), weights))
    assert abs(zeta[0]) < 1e-9 and abs(contrast[0]) < 1e-12
    clump = sheet.copy()
    clump[50, 50] += 5.0
    zeta, contrast = zeta_and_contrast(aperture_sums(clump.ravel(), weights))
    assert abs(zeta[0] - 5.0) < 1e-9 and contrast[0] > 0


def test_rebinned_grid_holds_block_means_at_block_centres():
    from astropy.wcs import WCS
    from kappaccino.plotting.zeta_masses import rebinned
    wcs = WCS(naxis=2); wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']; wcs.wcs.crval = [10., -5.]
    wcs.wcs.crpix = [101.3, 57.]; wcs.wcs.cdelt = [-.1/3600, .1/3600]
    values = np.random.default_rng(0).normal(size=(201, 200))
    binned, coarse = rebinned(values, wcs, 2)
    assert binned.shape == (100, 100)
    ra, dec = wcs.pixel_to_world_values(8.5, 6.5)          # the centre of the block of old pixels x 8-9, y 6-7
    np.testing.assert_allclose(coarse.world_to_pixel_values(ra, dec), [4., 3.], atol=1e-9)
    np.testing.assert_allclose(binned[3, 4], values[6:8, 8:10].mean())
    np.testing.assert_allclose(np.abs(coarse.wcs.cdelt), 2 * np.abs(wcs.wcs.cdelt))
