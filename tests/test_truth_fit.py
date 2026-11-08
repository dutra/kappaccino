from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from astropy.io import fits

from kappaccino.diagnostics.truth_fit import correlation_functions, image_plane_floor, load_truth_deflection


def _write_truth(tmp_path, n=64, pixel_deg=1.0 / 3600):
    """An FF-SIMS-like pair: deflection east = column index, north = 2 x row index, and a kappa map with the WCS."""
    rows, cols = np.mgrid[0:n, 0:n].astype(float)
    primary = fits.PrimaryHDU(cols.astype(np.float32))
    primary.header.update(XMIN=-n / 2 * 3600 * pixel_deg, XMAX=n / 2 * 3600 * pixel_deg, ZS=2.0, ZL=0.5)
    fits.HDUList([primary, fits.ImageHDU((2 * rows).astype(np.float32))]).writeto(tmp_path / "defl.fits")
    kappa = fits.PrimaryHDU(np.ones((n, n), dtype=np.float32))
    kappa.header.update(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRVAL1=0.0, CRVAL2=0.0, CDELT1=pixel_deg, CDELT2=pixel_deg,
                        CRPIX1=n / 2, CRPIX2=n / 2, CUNIT1="deg", CUNIT2="deg")
    kappa.writeto(tmp_path / "kappa.fits")
    return tmp_path / "defl.fits", tmp_path / "kappa.fits"


def test_truth_orientation_and_pixels(tmp_path):
    deflection, kappa = _write_truth(tmp_path)
    truth = load_truth_deflection(deflection, kappa)
    assert truth.z_source == 2.0 and truth.z_lens == 0.5 and np.isclose(truth.pixel_arcsec, 1.0)
    # x = 0 at the reference pixel (CRPIX - 1); x is positive west, so one arcsec west is one column lower
    alpha = truth.sample(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    column0, row0 = truth.column0, truth.row0
    np.testing.assert_allclose(alpha[0], [-column0, 2 * row0], atol=1e-4)      # alpha_west = -east
    np.testing.assert_allclose(alpha[1], [-(column0 - 1), 2 * row0], atol=1e-4)
    np.testing.assert_allclose(alpha[2], [-column0, 2 * (row0 + 1)], atol=1e-4)
    # d alpha / d theta: d(alpha_west)/dx = d(-col)/dx = +1, d(alpha_north)/dy = 2
    jac = truth.sample_jacobian(np.array([0.0]), np.array([0.0]))[0]
    np.testing.assert_allclose(jac, [[1.0, 0.0], [0.0, 2.0]], atol=1e-6)


def test_correlation_of_a_uniform_field():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-50, 50, 800), rng.uniform(-50, 50, 800)
    residual = np.tile([0.3, 0.0], (800, 1))
    corr = correlation_functions(x, y, residual, bins=5)
    assert np.isclose(corr["sigma_alpha"], 0.3 / np.sqrt(2))
    np.testing.assert_allclose(corr["c_l"] + corr["c_t"], 2.0, atol=1e-8)     # c_L + c_T = 2 for a uniform field


def test_image_plane_floor_is_zero_for_a_consistent_lens():
    x = np.array([10.0, -8.0, 3.0, 20.0, -15.0])
    y = np.array([2.0, 5.0, -12.0, 9.0, -4.0])
    family = np.array([0, 0, 0, 1, 1])
    source = np.array([[1.0, 0.5], [-2.0, 1.0]])
    images = SimpleNamespace(x_obs=x, y_obs=y, z_index_per_image=np.zeros(5, dtype=int), family_index_per_image=family,
                             image_labels=tuple(f"i{k}" for k in range(5)))
    state = SimpleNamespace(images=images, z_source_values=[2.0])
    lookup = {(xi, yi): source[f] for xi, yi, f in zip(x, y, family)}
    deflection = lambda xs, ys, z: np.array([[xi, yi] - lookup[(xi, yi)] for xi, yi in zip(xs, ys)])
    jacobian = lambda xs, ys, z: np.zeros((len(xs), 2, 2)) + 0.3 * np.eye(2)
    floor = image_plane_floor(state, deflection, jacobian)
    assert floor["summary"]["image_plane_p84_arcsec"] < 1e-12
    # move one image: its family's sources scatter, mapped back by A^-1 = I / 0.7
    shifted = lambda xs, ys, z: deflection(xs, ys, z) + np.where(np.isclose(xs, 10.0)[:, None], [0.7, 0.0], 0.0)
    floor = image_plane_floor(state, shifted, jacobian)
    np.testing.assert_allclose(floor["image_offsets_arcsec"][0], 2.0 / 3.0, rtol=1e-6)
