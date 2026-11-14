"""Image recovery over the reporting draws: per-draw summaries, per-image medians, extra images, and the lens-only solve."""
import numpy as np
import pandas as pd
from types import SimpleNamespace

from kappaccino.diagnostics import image_recovery as diagnostics


def _rows(residuals):
    """Per-draw rows: draw d recovers image k with residual residuals[d][k] (NaN: not recovered)."""
    return pd.DataFrame([dict(draw_index=d, family_id='1', image_label=f'1.{k}', recovered=np.isfinite(r),
                              image_residual_arcsec=r, x_model_arcsec=k + .1 * d, y_model_arcsec=0.)
                         for d, values in enumerate(residuals) for k, r in enumerate(values)])


def test_draw_summary_and_lens_only_prefix():
    rows = _rows([[.3, .4, np.nan], [.1, .1, .1], [.2, .2, np.nan]])
    out = diagnostics.draw_summary(rows)
    np.testing.assert_allclose([out['recovered_image_count'][k] for k in ('q16', 'median', 'q84')], np.percentile([2, 3, 2], [16, 50, 84]))
    rms = [np.sqrt((.09 + .16) / 2), .1, .2]
    np.testing.assert_allclose(out['image_rms_arcsec']['median'], np.median(rms))
    lens_only = diagnostics.lens_only_summary(rows)
    assert set(lens_only) == {'lens_only_recovered_image_count', 'lens_only_image_rms_arcsec', 'lens_only_image_median_residual_arcsec'}
    assert diagnostics.lens_only_summary(None) == {}


def test_per_image_table_needs_half_the_draws_and_takes_medians_over_them():
    rows = _rows([[.3, .4, np.nan], [.1, .1, .1], [.2, .2, np.nan]])
    observed = pd.DataFrame(dict(image_label=['1.0', '1.1', '1.2'], x_obs_arcsec=[0., 1., 2.]))
    table = diagnostics.per_image_table(rows, observed, ('x_model_arcsec', 'image_residual_arcsec'))
    assert table.image_recovery_status.tolist() == ['recovered', 'recovered', 'not_recovered']
    np.testing.assert_allclose(table.recovered_fraction, [1., 1., 1/3])
    np.testing.assert_allclose(table.x_model_arcsec[:2], [.1, 1.1])
    np.testing.assert_allclose(table.image_residual_arcsec[0], .2)
    np.testing.assert_allclose(table.image_residual_q16[0], np.percentile([.3, .1, .2], 16))
    assert table.posterior_valid_draws.tolist() == [3, 3, 1]


def test_common_extra_images_need_half_the_draws():
    extra = pd.DataFrame(dict(draw_index=[0, 1, 2, 0], family_id=['1', '1', '1', '2'],
                              x_model_arcsec=[5., 5.2, 4.9, -3.], y_model_arcsec=[1., 1.1, .9, 0.]))
    common = diagnostics.common_extra_images(extra, 4)
    assert common.family_id.tolist() == ['1']
    np.testing.assert_allclose(common[['x_model_arcsec', 'y_model_arcsec', 'draw_fraction']].to_numpy()[0], [5., 1., .75])
    assert diagnostics.common_extra_images(pd.DataFrame(), 4).empty


def test_every_image_draw_is_solved_in_its_own_lens(monkeypatch):
    seen = []
    def fake_predictions(solver, state, evaluator, theta, *, adaptive=None, correction=None):
        seen.append((float(theta[0]), adaptive, correction))
        own = isinstance(correction, tuple)                    # the draw's realization moves the image and its magnification
        r = .1 * theta[0] + (.05 if own else 0.)
        return {"image_rows": [dict(family_id="1", image_label="1.a", x_obs_arcsec=0., y_obs_arcsec=0., z_source=2.,
                                    radius_arcsec=0., angle_deg=0., image_recovery_status="recovered", x_model_arcsec=r,
                                    y_model_arcsec=0., image_residual_arcsec=r, magnification_model=5. if own else 3.,
                                    magnification_model_at_model_position=6. if own else 3.)],
                "extra_rows": [dict(family_id="1", x_model_arcsec=4., y_model_arcsec=0.)] if theta[0] > 1 else [],
                "count_rows": [dict(family_id="1", z_source=2., observed_image_count=1, produced_image_count=1 + (theta[0] > 1),
                                    recovered_image_count=1, missing_image_count=0, extra_image_count=int(theta[0] > 1),
                                    multiplicity_failed=False)]}
    monkeypatch.setattr(diagnostics, "predictions_at", fake_predictions)
    monkeypatch.setattr(diagnostics, "ExactImageSolver", lambda evaluator, config: SimpleNamespace(timing={'exact_solver': 0.}))
    monkeypatch.setattr(diagnostics, "window_corners", lambda solver, state: np.array([[0., 0.], [30., 40.]]))
    monkeypatch.setattr(diagnostics, "image_plane_scales", lambda evaluator, theta: dict(sigma_gp=.1, ell_gp=1., sigma_meas=.01))
    class Draws(SimpleNamespace):
        has_field = True
        def __len__(self):
            return len(self.samples)
        def reach(self, xy):
            return float(np.hypot(*xy.max(0)))
        def image_correction(self, j, reach):
            return ('realization', int(self.image_indices[j]), reach)
    # Image draws 4, 9, 12, each in its own lens (local solve, then the parametric lens alone); the global search
    # for extra images runs on the product draws 9 and 12 only.
    draws = Draws(image_samples=np.array([[1.], [2.], [3.]]), image_indices=np.array([4, 9, 12]),
                  samples=np.array([[2.], [3.]]), indices=np.array([9, 12]))
    tables = diagnostics.image_recovery_tables(None, None, draws, None, log=lambda _: None)
    assert seen == [(1., False, ('realization', 4, 50.)), (1., False, None),
                    (2., False, ('realization', 9, 50.)), (2., None, ('realization', 9, 50.)), (2., False, None),
                    (3., False, ('realization', 12, 50.)), (3., None, ('realization', 12, 50.)), (3., False, None)]
    fit = tables['image_fit_quality']
    np.testing.assert_allclose(fit.x_model_arcsec, .25)                             # median of .15, .25, .35: the draws' own lenses
    np.testing.assert_allclose(fit.magnification_model, 5.)
    np.testing.assert_allclose(tables['model_magnification'].magnification_model_at_model_position_q50, 6.)
    rows = tables['posterior_image_residual_draws']
    assert rows.posterior_index.tolist() == [4, 9, 12] and rows.recovered.all()
    np.testing.assert_allclose(rows.x_model_arcsec, [.15, .25, .35])
    np.testing.assert_allclose(rows.image_residual_arcsec, [.15, .25, .35])
    np.testing.assert_allclose(tables['posterior_image_rms'].image_rms_arcsec, [.15, .25, .35])
    assert tables['posterior_image_rms'].posterior_index.tolist() == [4, 9, 12]
    np.testing.assert_allclose(tables['posterior_image_residual_draws_lens_only'].image_residual_arcsec, [.1, .2, .3])
    assert 'posterior_image_position_draws' not in tables
    extras = tables['posterior_extra_images']
    assert extras.posterior_index.tolist() == [9, 12] and extras.draw_index.tolist() == [0, 1]     # counted over the product draws
    for table, x in ((rows, [.15, .25, .35]), (extras, [4., 4.])):      # predicted observed: the lens position plus the scatter
        shift = np.c_[table.x_predicted_arcsec - x, table.y_predicted_arcsec]
        assert np.all(shift != 0) and np.abs(shift).max() < .6
    assert {'x_true_arcsec', 'y_true_arcsec'} <= set(rows.columns) & set(extras.columns)
    assert np.all(np.abs(rows.x_true_arcsec) < .05)       # catalog at 0: the true position is within the measurement error
    again = diagnostics.image_recovery_tables(None, None, draws, None, log=lambda _: None)['posterior_image_residual_draws']
    np.testing.assert_array_equal(again.x_predicted_arcsec, rows.x_predicted_arcsec)       # seeded by the draw's posterior row
    assert tables['image_recovery_extra_images'].draw_fraction.tolist() == [1.]
    assert tables['image_count_recovery'].extra_image_count.tolist() == [1.]
    assert list(tables['image_fit_quality_lens_only'].columns) == [c for c in diagnostics.LENS_ONLY_COLUMNS
                                                                  if c in tables['image_fit_quality_lens_only']]
    seen.clear()
    tables = diagnostics.image_recovery_tables(None, None, draws, None, extra_images=False, log=lambda _: None)
    assert all(adaptive is False for _, adaptive, _ in seen)
    assert tables['image_recovery_extra_images'].empty and 'extra_image_count' not in tables['image_count_recovery']
    # Without a mass field every solve is the parametric lens and there is no lens-only table.
    seen.clear()
    draws.has_field = False
    tables = diagnostics.image_recovery_tables(None, None, draws, None, log=lambda _: None)
    assert all(correction is None for _, _, correction in seen) and 'image_fit_quality_lens_only' not in tables


def test_image_plane_scatter_has_the_gp_and_measurement_covariance():
    xy = np.array([[0., 0.], [.5, 0.], [30., 0.]])
    scales = dict(sigma_gp=.2, ell_gp=1., sigma_meas=.05)
    rng = np.random.default_rng(4)
    d = np.stack([diagnostics.image_plane_scatter(xy, scales, rng) for _ in range(20000)])
    np.testing.assert_allclose(d.std(0), np.sqrt(.2**2 + .05**2), rtol=.03)
    covariance = np.einsum('nia,nja->ij', d, d) / (2 * len(d))
    np.testing.assert_allclose(covariance[0, 1], .2**2 * np.exp(-.125), rtol=.05)        # neighbours move together
    assert abs(covariance[0, 2]) < 2e-3 and abs(np.mean(d[:, 0, 0] * d[:, 0, 1])) < 2e-3  # far images and the two axes do not
    white = np.stack([diagnostics.image_plane_scatter(xy, dict(sigma_gp=0., ell_gp=np.nan, sigma_meas=.05), rng) for _ in range(5000)])
    np.testing.assert_allclose(white.std(0), .05, rtol=.05)
    assert diagnostics.image_plane_scatter(np.empty((0, 2)), scales, rng).shape == (0, 2)


def test_true_positions_are_the_catalog_minus_measurement_error_and_follow_neighbours():
    catalog = np.array([[0., 0.], [.3, 0.], [40., 0.], [80., 0.]])
    lens = catalog + np.array([[.2, .1], [.2, .1], [-.3, 0.], [np.nan, np.nan]])          # the last image is not recovered
    extra = np.array([[.15, 0.], [200., 0.]])                                             # one beside the first two, one far away
    scales = dict(sigma_gp=.2, ell_gp=1., sigma_meas=.02)
    rng = np.random.default_rng(7)
    draws = [diagnostics.true_positions(catalog, lens, extra, scales, rng) for _ in range(4000)]
    true, added = np.stack([d[0] for d in draws]), np.stack([d[1] for d in draws])
    assert np.isnan(true[:, 3]).all()
    error = catalog[:3] - true[:, :3]                                                      # the measurement error given the data
    assert np.abs(error.mean(0)).max() < .01 and error.std(0).max() < .02 and error.std(0).min() > .01
    # The extra image between the two neighbours shares their displacement; the far one keeps the prior.
    np.testing.assert_allclose((added[:, 0] - extra[0]).mean(0), [-.2, -.1], atol=.02)
    assert (added[:, 0] - extra[0]).std(0).max() < .06
    np.testing.assert_allclose((added[:, 1] - extra[1]).std(0), .2, rtol=.06)
    assert np.abs((added[:, 1] - extra[1]).mean(0)).max() < .015
    # Without a GP the model has no displacement: the true position is the lens position.
    exact, _ = diagnostics.true_positions(catalog, lens, extra, dict(sigma_gp=0., ell_gp=np.nan, sigma_meas=.02), rng)
    np.testing.assert_allclose(exact[:3], lens[:3])
