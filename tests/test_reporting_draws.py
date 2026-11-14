"""The reporting draws: one random set of posterior draws, each evaluated once and shared by every product."""
from types import SimpleNamespace

import numpy as np

from kappaccino.reporting import draws as reporting


def _run(monkeypatch, *, field=True):
    """A lens whose parametric Jacobian at draw theta is theta[0] I and whose GRF Jacobian at the reference redshift
    is theta[1] I (family zero weight 1), with a lens efficiency equal to the source redshift."""
    from kappaccino.lens import cosmology
    monkeypatch.setattr(cosmology, 'dpie_sigma0_factor_from_config', lambda zl, z, c: z)
    calls = []
    def total_fields_at(xy, packed, chunk=4096):
        calls.append(len(xy))
        return np.zeros((len(xy), 2)), np.broadcast_to(packed * np.eye(2), (len(xy), 2, 2)).copy()
    evaluator = SimpleNamespace(gp_knee=field, sigma_grf_index=0 if field else -1, gp_knee_spectrum='knee',
                                state=SimpleNamespace(gp_blocks='all', z_lens=.3, gp_reference_redshift=2., cosmo_config=None),
                                gp_weights_per_family=np.array([1.]), physical=lambda theta: np.asarray(theta),
                                packed_state_for_z=lambda physical, z: float(physical[0]), total_fields_at=total_fields_at)
    posterior = SimpleNamespace(samples=np.array([[.1, .0], [.2, .05], [.3, -.05], [.4, .1]]),
                                physical=np.array([[.1, .0], [.2, .05], [.3, -.05], [.4, .1]]))
    monkeypatch.setattr(reporting.ReportingDraws, 'reach', lambda self, xy: 1.)
    monkeypatch.setattr(reporting.ReportingDraws, 'correction', lambda self, k, reach: (
        lambda x, y, f: (0*x, 0*x, self.samples[k, 1] + 0*x, 0*x, 0*x, self.samples[k, 1] + 0*x)))
    return evaluator, posterior, calls


def test_draws_are_chosen_once_and_shared(monkeypatch):
    evaluator, posterior, _ = _run(monkeypatch)
    config = SimpleNamespace(reporting=SimpleNamespace(draws=3), runtime=SimpleNamespace(seed=7))
    first = reporting.reporting_draws(config, evaluator, posterior)
    assert len(first) == 3 and np.all(np.diff(first.indices) > 0)
    assert reporting.reporting_draws(config, evaluator, posterior) is first
    np.testing.assert_array_equal(first.samples, posterior.samples[first.indices])
    all_draws = reporting.ReportingDraws(evaluator, posterior, 50, 7)
    np.testing.assert_array_equal(all_draws.indices, np.arange(4))


def test_grid_maps_are_per_draw_evaluated_once_and_scale_the_field_with_redshift(monkeypatch):
    evaluator, posterior, calls = _run(monkeypatch)
    draws = reporting.ReportingDraws(evaluator, posterior, 4, 0)
    x, y = np.meshgrid(np.arange(3.), np.arange(2.))
    maps = draws.grid_maps('grid', x, y, 4., log=lambda _: None)
    # Parametric kappa theta[0]; GRF kappa theta[1] times the efficiency ratio 4/2.
    np.testing.assert_allclose(maps['kappa_parametric'][:, 0, 0], posterior.samples[:, 0], rtol=1e-6)
    np.testing.assert_allclose(maps['kappa_grf'][:, 0, 0], 2 * posterior.samples[:, 1], rtol=1e-6)
    np.testing.assert_allclose(maps['detA'][:, 0, 0], (1 - posterior.samples[:, 0] - 2 * posterior.samples[:, 1]) ** 2, rtol=1e-5)
    evaluated = len(calls)
    assert draws.grid_maps('grid', x, y, 4., log=lambda _: None) is maps and len(calls) == evaluated


def test_no_field_gives_zero_grf_maps(monkeypatch):
    evaluator, posterior, _ = _run(monkeypatch, field=False)
    draws = reporting.ReportingDraws(evaluator, posterior, 4, 0)
    assert not draws.has_field
    maps = draws.grid_maps('grid', np.zeros((1, 2)), np.zeros((1, 2)), 4., log=lambda _: None)
    np.testing.assert_array_equal(maps['kappa_grf'], 0.)


def test_product_draws_are_nested_in_the_image_draws(monkeypatch):
    evaluator, posterior, _ = _run(monkeypatch)
    posterior = SimpleNamespace(samples=np.arange(100.)[:, None], physical=np.arange(100.)[:, None])
    draws = reporting.ReportingDraws(evaluator, posterior, 8, 3, image_count=32)
    assert len(draws.indices) == 8 and len(draws.image_indices) == 32
    assert set(draws.indices) <= set(draws.image_indices)
    np.testing.assert_array_equal(draws.image_samples[:, 0], draws.image_indices)
    more = reporting.ReportingDraws(evaluator, posterior, 16, 3, image_count=32)
    assert set(draws.indices) <= set(more.indices)


def test_an_image_draw_gets_the_product_draws_field_or_its_own_seeded_one(monkeypatch):
    from kappaccino.plotting import grf
    evaluator, posterior, _ = _run(monkeypatch)
    evaluator._padded_evaluator = lambda: evaluator
    posterior = SimpleNamespace(samples=np.arange(100.)[:, None], physical=np.arange(100.)[:, None])
    monkeypatch.setattr(reporting, 'shared_theta', lambda evaluator, theta: theta)
    monkeypatch.setattr(grf, 'field_draw', lambda base, theta, rng: (float(theta[0]), float(rng.standard_normal())))
    monkeypatch.setattr(grf, 'field_realization', lambda base, draw, reach: (draw, reach))
    draws = reporting.ReportingDraws(evaluator, posterior, 8, 3, image_count=32)
    monkeypatch.setattr(reporting.ReportingDraws, 'correction', lambda self, k, reach: ('product', k, reach))
    products = [draws.image_correction(j, 5.) for j in range(32) if draws.image_indices[j] in draws.indices]
    assert products == [('product', k, 5.) for k in range(8)]
    j = next(j for j in range(32) if draws.image_indices[j] not in draws.indices)
    (row, noise), reach = draws.image_correction(j, 5.)
    assert row == draws.image_indices[j] and reach == 5.
    assert noise == np.random.default_rng([3, int(row)]).standard_normal()        # the seeding of a product draw


def test_a_rebuilt_shared_field_is_flagged(monkeypatch):
    evaluator, posterior, _ = _run(monkeypatch)
    assert not reporting.ReportingDraws(evaluator, posterior, 2, 0).field_is_rebuilt          # an all-block run samples its field
    monkeypatch.setattr(reporting, 'shared_field_evaluator', lambda e: SimpleNamespace())     # a system-block run gets a twin
    assert reporting.ReportingDraws(evaluator, posterior, 2, 0).field_is_rebuilt
    monkeypatch.setattr(reporting, 'shared_field_evaluator', lambda e: None)
    assert not reporting.ReportingDraws(evaluator, posterior, 2, 0).field_is_rebuilt
