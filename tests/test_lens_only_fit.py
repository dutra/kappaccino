"""The lens-only residual summary and the switch that turns the GRF mean deflection off."""
import numpy as np
import pandas as pd
from types import SimpleNamespace

from kappaccino.diagnostics import image_recovery as diagnostics


def test_lens_only_summary_counts_recovered_images_only():
    frame = pd.DataFrame(dict(image_recovery_status=["recovered", "recovered", "missing"],
                              image_residual_arcsec=[0.3, 0.4, np.nan]))
    out = diagnostics.lens_only_summary(frame)
    assert out["lens_only_recovered_image_count"] == 2
    np.testing.assert_allclose(out["lens_only_image_rms_arcsec"], np.sqrt((0.09 + 0.16) / 2))
    np.testing.assert_allclose(out["lens_only_image_median_residual_arcsec"], 0.35)
    assert diagnostics.lens_only_summary(None) == {}


def test_no_lens_only_table_without_a_field():
    evaluator = SimpleNamespace(gp_knee=False)
    assert diagnostics.lens_only_table(None, None, evaluator, np.zeros(1)) is None
    evaluator = SimpleNamespace(gp_knee=True, gp_deflection_correction=lambda theta: None)
    assert diagnostics.lens_only_table(None, None, evaluator, np.zeros(1)) is None


def test_lens_only_table_solves_without_the_field(monkeypatch):
    seen = []
    def fake_predictions(solver, state, evaluator, theta, *, adaptive=None, include_field=True):
        seen.append(include_field)
        return {"image_rows": [dict(family_id="1", image_label="1.a", x_obs_arcsec=0., y_obs_arcsec=0., image_recovery_status="recovered",
                                    x_model_arcsec=.1, y_model_arcsec=0., image_residual_arcsec=.1, magnification_model=3.)]}
    monkeypatch.setattr(diagnostics, "predictions_at", fake_predictions)
    evaluator = SimpleNamespace(gp_knee=True, gp_deflection_correction=lambda theta: object())
    frame = diagnostics.lens_only_table(None, None, evaluator, np.zeros(1))
    assert seen == [False]
    assert list(frame.columns) == diagnostics.LENS_ONLY_COLUMNS
