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


def test_tables_solve_image_draws_locally_and_product_draws_globally(monkeypatch):
    seen = []
    def fake_predictions(solver, state, evaluator, theta, *, adaptive=None, include_field=True):
        seen.append((float(theta[0]), adaptive, include_field))
        r = .1 * theta[0]
        return {"image_rows": [dict(family_id="1", image_label="1.a", x_obs_arcsec=0., y_obs_arcsec=0., z_source=2.,
                                    radius_arcsec=0., angle_deg=0., image_recovery_status="recovered", x_model_arcsec=r,
                                    y_model_arcsec=0., image_residual_arcsec=r, magnification_model=3.,
                                    magnification_model_at_model_position=3.)],
                "extra_rows": [dict(family_id="1", x_model_arcsec=4., y_model_arcsec=0.)] if theta[0] > 1 else [],
                "count_rows": [dict(family_id="1", z_source=2., observed_image_count=1, produced_image_count=1 + (theta[0] > 1),
                                    recovered_image_count=1, missing_image_count=0, extra_image_count=int(theta[0] > 1),
                                    multiplicity_failed=False)]}
    monkeypatch.setattr(diagnostics, "predictions_at", fake_predictions)
    monkeypatch.setattr(diagnostics, "ExactImageSolver", lambda evaluator, config: SimpleNamespace(timing={'exact_solver': 0.}))
    evaluator = SimpleNamespace(gp_knee=True, gp_deflection_correction=lambda theta: object())
    class Draws(SimpleNamespace):
        def __len__(self):
            return len(self.samples)
    # Image draws 4, 9, 12 (local solves); product draws 9, 12 (global search for extra images).
    draws = Draws(image_samples=np.array([[1.], [2.], [3.]]), image_indices=np.array([4, 9, 12]),
                  samples=np.array([[2.], [3.]]), indices=np.array([9, 12]))
    tables = diagnostics.image_recovery_tables(None, evaluator, draws, None, log=lambda _: None)
    assert seen == [(1., False, True), (2., False, True), (3., False, True), (2., None, True), (3., None, True),
                    (1., False, False), (2., False, False), (3., False, False)]
    np.testing.assert_allclose(tables['image_fit_quality'].x_model_arcsec, .2)
    assert tables['image_recovery_extra_images'].draw_fraction.tolist() == [1.]
    assert tables['image_count_recovery'].extra_image_count.tolist() == [1.]
    assert tables['posterior_extra_images'].posterior_index.tolist() == [9, 12]
    assert list(tables['image_fit_quality_lens_only'].columns) == [c for c in diagnostics.LENS_ONLY_COLUMNS
                                                                  if c in tables['image_fit_quality_lens_only']]
    assert tables['posterior_image_rms'].posterior_index.tolist() == [4, 9, 12]
    seen.clear()
    tables = diagnostics.image_recovery_tables(None, evaluator, draws, None, extra_images=False, log=lambda _: None)
    assert all(adaptive is False for _, adaptive, _ in seen)
    assert tables['image_recovery_extra_images'].empty and 'extra_image_count' not in tables['image_count_recovery']
