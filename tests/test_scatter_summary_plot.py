import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kappaccino.plotting.scatter_summary import MU_REF, conditional_scatter, render_scatter_summary, run_scatter_table


class Draws:
    def __init__(self, maps, has_field=True):
        self.maps, self.has_field = maps, has_field

    def grid_maps(self, name, x, y, z_source, log=print):
        return self.maps


def test_scatter_is_half_the_68_per_cent_range_over_the_median():
    truth = np.full(200, 3.)
    ratio = np.linspace(.5, 1.5, 101)[:, None] * np.ones(200)             # draws spread uniformly about 1
    result = conditional_scatter(truth, truth * ratio, 3.)
    np.testing.assert_allclose([result['bias'], result['scatter']], [0., .34], atol=1e-12)
    assert result['n_pixels'] == 200
    biased = conditional_scatter(truth, .8 * truth * ratio, 3.)
    np.testing.assert_allclose([biased['bias'], biased['scatter']], [-.2, .34], atol=1e-12)   # scatter is relative to the median


def test_selection_uses_the_true_magnification_and_needs_enough_pixels():
    truth = np.r_[np.full(20, 3.), np.full(20, 10.)]
    draws = np.r_[np.full(20, 3.), np.full(20, 30.)][None] * np.array([[.9], [1.], [1.1]])
    assert abs(conditional_scatter(truth, draws, 10.)['bias'] - 2.) < 1e-12
    assert np.isnan(conditional_scatter(truth[:5], draws[:, :5], 3.)['scatter'])


def test_run_table_is_restricted_to_the_image_hull_and_drops_the_grf_without_a_field():
    x, y = np.meshgrid(np.linspace(-10, 10, 41), np.linspace(-10, 10, 41))
    true_mu = np.where(np.hypot(x, y) < 6, 3., 10.)
    det = np.stack([1 / (true_mu * f) for f in (.9, 1., 1.1)])
    truth = dict(x=x, y=y, z_source=9., truth={'abs_mu': true_mu})
    hull = np.array([[-4., -4.], [4., -4.], [4., 4.], [-4., 4.]])            # inside the mu = 3 disc
    table = run_scatter_table(truth, Draws(dict(detA=det, detA_parametric=2 * det)), hull)
    assert list(table.model) == ['parametric', 'parametric', 'model', 'model'] and list(table.mu_ref[:2]) == list(MU_REF)
    inside = table.set_index(['model', 'mu_ref'])
    assert inside.loc[('model', 3.), 'n_pixels'] == 17 * 17 and inside.loc[('model', 10.), 'n_pixels'] == 0
    np.testing.assert_allclose(inside.loc[('parametric', 3.), 'bias'], -.5)
    assert set(run_scatter_table(truth, Draws(dict(detA=det, detA_parametric=det), has_field=False), hull).model) == {'parametric'}


def test_summary_figure_has_one_row_per_cluster_model_and_keeps_the_style(tmp_path):
    table = pd.DataFrame(dict(model=['parametric', 'parametric', 'model', 'model'], mu_ref=[3., 10., 3., 10.],
                              bias=0., scatter=[.19, .41, .13, .27], n_pixels=100))
    before = plt.rcParams['font.size']
    path = render_scatter_summary({'Hera': table, 'Ares': table[table.model == 'parametric']}, tmp_path / 'scatter_summary.pdf')
    assert path.read_bytes().startswith(b'%PDF') and plt.rcParams['font.size'] == before
