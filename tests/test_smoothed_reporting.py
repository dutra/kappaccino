"""Astrometric-error provenance in image and magnification reports."""
from types import SimpleNamespace

import numpy as np

from lenscluster.image_diagnostics import image_prediction_for_family_latent


def test_unranked_smoothed_members_generate_scaling_relation_table(tmp_path):
    from test_smoothed_integration import make_evaluator
    from lenscluster import plotting
    _, evaluator, theta = make_evaluator(tmp_path)
    state = evaluator.state
    physical = np.asarray(evaluator._physical_parameter_vector(theta))
    table = plotting._scaling_relation_summary_table(
        evaluator.scaling_rank_df, state.parameter_specs, physical[None], physical, state.packed_lens_spec)
    assert len(table) == len(state.scaling_component_records) == 4
    assert table['rank'].tolist() == [-1]*4
    assert table['free_component_index'].tolist() == [-1]*4
    assert evaluator.scaling_rank_df['importance'].isna().all()


class ReportingEvaluator:
    def __init__(self, metadata=None):
        self.state = SimpleNamespace(smoothed_model=metadata)

    def _exact_family_prediction_details(self, params, family):
        return {
            "failed": False,
            "x_pred": family.x_obs + .6,
            "y_pred": family.y_obs,
            "exact_image_rms": .6,
        }

    def _get_exact_model_solver(self, redshift):
        return SimpleNamespace(magnification=lambda x, y, kwargs: np.array([2., -3.])), None

    def _build_packed_lens_state(self, params, redshift):
        return None

    def _packed_to_kwargs_lens(self, packed):
        return []


def _family():
    return SimpleNamespace(
        family_id="1", image_labels=["1.b", "1.a"], n_images=2,
        x_obs=np.array([1., 2.]), y_obs=np.array([3., 4.]),
        z_source=2., effective_z_source=2., sigma_arcsec=5.,
    )


def test_image_and_magnification_reports_use_smoothed_errors_by_label():
    evaluator = ReportingEvaluator({"image_metadata": {
        "1.a": {"sigma_obs_arcsec": 1.2}, "1.b": {"sigma_obs_arcsec": .3},
    }})
    result = image_prediction_for_family_latent(evaluator, _family(), np.empty(0), 0., 0.)
    for key in ("image_rows", "magnification_rows"):
        rows = result[key]
        assert [row["image_label"] for row in rows] == ["1.b", "1.a"]
        np.testing.assert_allclose([row["sigma_arcsec"] for row in rows], [.3, 1.2])
        np.testing.assert_allclose([row["image_sigma_eff_arcsec"] for row in rows], [.3, 1.2])
    np.testing.assert_allclose([
        row["image_residual_arcsec"] / row["image_sigma_eff_arcsec"]
        for row in result["image_rows"]
    ], [2., .5])


def test_legacy_reporting_retains_scalar_family_error():
    result = image_prediction_for_family_latent(ReportingEvaluator(), _family(), np.empty(0), 3., 2.)
    for key in ("image_rows", "magnification_rows"):
        np.testing.assert_allclose([row["sigma_arcsec"] for row in result[key]], [5., 5.])
        np.testing.assert_allclose([row["image_sigma_eff_arcsec"] for row in result[key]], [6., 6.])
