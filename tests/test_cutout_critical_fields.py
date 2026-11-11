from types import SimpleNamespace as NS
import numpy as np
from kappaccino.plotting.critical_fields import median_cutout_determinants


class Draws:
    """Three reporting draws whose det A on a cutout grid at redshift z is x - z + k (draw k)."""
    indices = np.arange(3)
    key = 'test'
    has_field = False
    evaluator = NS(gp_knee=False)

    def __init__(self):
        self.calls = []

    def __len__(self):
        return 3

    def grid_maps(self, name, x, y, z, log=print):
        self.calls.append((name, z))
        return {'detA': np.stack([x - z + k for k in range(3)])}


def test_cutout_determinants_are_percentiles_over_the_draws_at_each_redshift(tmp_path):
    draws = Draws()
    grids, median, lower, upper = median_cutout_determinants(draws, [[0, 0], [0, 0]], [10, 10], [1, 2], tmp_path/'critical.h5',
                                                             resolution=16, log=lambda _: None)
    assert draws.calls == [('cutout 0', 1.), ('cutout 1', 2.)]
    np.testing.assert_allclose(median, grids[..., 0] - np.array([1, 2])[:, None, None] + 1, atol=1e-14)
    np.testing.assert_allclose(lower, median - .68, atol=1e-12)
    np.testing.assert_allclose(upper, median + .68, atol=1e-12)
    assert (tmp_path/'critical.h5').exists() and (tmp_path/'critical.json').exists()
