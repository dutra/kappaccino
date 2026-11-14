import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

from kappaccino.plotting.cumulative_source_area import (
    area_table, cumulative_area, shell_volume_per_arcsec2, write_cumulative_source_area,
)


class Draws:
    def __init__(self, maps, has_field=True):
        self.maps, self.has_field, self.calls = maps, has_field, []

    def __len__(self):
        return len(self.maps['detA'])

    def grid_maps(self, name, x, y, z_source, log=print):
        self.calls.append((name, float(z_source)))
        return self.maps


def test_area_sums_the_source_plane_pixels_above_each_threshold():
    det = np.array([[.1, -.1, .5], [np.nan, .02, 2.]])            # |mu| = 10, 10, 2, -, 50, 0.5
    area = cumulative_area(det, 4., mu=np.array([1., 5., 20., 100.]))
    np.testing.assert_allclose(area, 4 * np.array([.1 + .1 + .5 + .02, .1 + .1 + .02, .02, 0.]))
    stack = cumulative_area(np.stack([det, 2 * det]), 4., mu=np.array([5.]))
    np.testing.assert_allclose(stack[:, 0], [4 * .22, 4 * .04])  # doubled det A: only the mu = 25 pixel stays above 5


def test_shell_volume_matches_the_solid_angle_fraction():
    cosmo = FlatLambdaCDM(H0=70, Om0=.3)
    shell = (cosmo.comoving_volume(9.5) - cosmo.comoving_volume(8.5)).value
    arcsec2_on_sky = 4 * np.pi * (180 * 3600 / np.pi)**2
    np.testing.assert_allclose(shell_volume_per_arcsec2(cosmo, 9.), shell / arcsec2_on_sky, rtol=1e-12)


def test_table_has_no_grf_columns_without_a_shared_field():
    det = np.full((3, 4, 4), .1)
    table = area_table(det[0], dict(detA=det, detA_parametric=det), 1., with_grf=False)
    assert 'area_parametric_median_arcsec2' in table and 'area_model_median_arcsec2' not in table


def test_writer_skips_without_shear_truth(tmp_path):
    truth = dict(x=np.zeros((2, 2)), y=np.zeros((2, 2)), z_source=9., truth={'kappa': np.zeros((2, 2))})
    assert write_cumulative_source_area(truth, None, None, tmp_path, tmp_path, 'toy') is None
    assert write_cumulative_source_area(None, None, None, tmp_path, tmp_path, 'toy') is None
    assert not list(tmp_path.iterdir())


def test_writer_draws_truth_parametric_and_grf(tmp_path):
    x, y = np.meshgrid(np.linspace(-60, 60, 61), np.linspace(-60, 60, 61))
    radius = np.hypot(x, y)
    det = lambda einstein: 1 - einstein / np.maximum(radius, 1e-3)            # singular isothermal sphere
    rng = np.random.default_rng(0)
    parametric = np.stack([det(18 + rng.normal(0, .3)) for _ in range(6)])
    total = np.stack([det(20 + rng.normal(0, .3)) for _ in range(6)])
    draws = Draws(dict(detA=total, detA_parametric=parametric))
    truth = dict(x=x, y=y, z_source=9., truth={'detA': det(20.)})
    before = plt.rcParams['font.size']
    path = write_cumulative_source_area(truth, draws, FlatLambdaCDM(H0=70, Om0=.3), tmp_path, tmp_path, 'toy', log=lambda _: None)
    assert path.name == 'toy_cumulative_source_area.pdf' and path.read_bytes().startswith(b'%PDF')
    assert plt.rcParams['font.size'] == before and draws.calls == [('truth', 9.)]
    table = pd.read_csv(tmp_path / 'cumulative_source_area.csv')
    row = table.iloc[int(np.argmin(np.abs(table.mu - 5)))]
    assert abs(row.area_model_median_arcsec2 / row.area_true_arcsec2 - 1) < .1          # same Einstein radius as the truth
    assert row.area_parametric_median_arcsec2 < .9 * row.area_true_arcsec2             # smaller lens, smaller area
    assert row.area_model_q16_arcsec2 <= row.area_model_median_arcsec2 <= row.area_model_q84_arcsec2
    provenance = json.loads((tmp_path / 'cumulative_source_area_provenance.json').read_text())
    assert provenance['posterior_draws'] == 6 and provenance['pixel_arcsec'] == 2.
