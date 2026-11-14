import json
import pytest

from kappaccino.mock_validation.aggregate_plots import load_campaign


def test_completed_selection_does_not_shrink_when_first_mock_converges(tmp_path):
    for seed, rhat in [(1, 1.18), (2, 1.13), (3, 1.05)]:
        directory = tmp_path / f'seed_{seed:08d}' / 'recovery'
        directory.mkdir(parents=True)
        payload = dict(seed=seed, max_r_hat=rhat, min_n_eff=100,
                       parameters=[], magnification=[], image_residual_draws=[], m2d=[], subhalos={})
        (directory/'recovery.json').write_text(json.dumps(payload))
        # An unfinished realization must never enter any aggregate table.
        (tmp_path/'seed_99999999').mkdir(exist_ok=True)
        assert load_campaign(tmp_path).seeds == list(range(1,seed+1))
        if seed < 3:
            with pytest.raises(ValueError, match='No converged realization'):
                load_campaign(tmp_path, converged_only=True)
    assert load_campaign(tmp_path, converged_only=True).seeds == [3]


def test_aggregate_rechecks_saved_verdict_at_current_cut(tmp_path):
    for seed, rhat, neff, saved_verdict in [(1, 1.099, 50, False), (2, 1.1, 100, True),
                                             (3, 1.14, 100, True), (4, 1.099, 49, True)]:
        directory = tmp_path / f"seed_{seed:08d}" / "recovery"
        directory.mkdir(parents=True)
        payload = dict(seed=seed, max_r_hat=rhat, min_n_eff=neff, converged=saved_verdict,
                       convergence_reasons=['old verdict'], parameters=[], magnification=[],
                       image_residual_draws=[], m2d=[], subhalos={})
        (directory / 'recovery.json').write_text(json.dumps(payload))
    campaign = load_campaign(tmp_path, converged_only=True)
    assert campaign.seeds == [1]
    assert campaign.summary.converged.tolist() == [True, False, False, False]
    assert '1.1' in campaign.summary.iloc[1].convergence_reasons
    assert load_campaign(tmp_path, converged_only=False).seeds == [1, 2, 3, 4]
    assert load_campaign(tmp_path).seeds == [1, 2, 3, 4]


def test_correlated_error_parameters_in_aggregate_figures(tmp_path):
    import pandas as pd
    from types import SimpleNamespace
    from kappaccino.mock_validation import aggregate_plots as plots

    names = ['images.sigma_grf', 'images.lambda_b', 'images.gp_knee_gamma',
             'images.sigma_gp', 'images.ell_gp', 'images.sigma_meas']
    rows = [dict(parameter=name, seed=seed, truth=value, median=value * 1.1,
                 q16=value * .8, q84=value * 1.3, covered_68=True, covered_95=True, scored=True)
            for name in names for seed, value in enumerate([.1, .2, .3])]
    campaign = SimpleNamespace(parameters=pd.DataFrame(rows))
    assert {p for p, _ in plots._standardized_parameter_panels(campaign)} == set(names)
    assert plots._parameter_component(names[0]) == 'grf'
    assert plots._parameter_component('images.ell_gp') == 'gp'
    assert 'arcsec' in plots._axis_label('images.ell_gp', 'truth')
    assert 'arcsec' not in plots._axis_label('images.gp_knee_gamma', 'truth')
    for render in [plots.plot_parameter_recovery, plots.plot_standardized_residuals, plots.plot_bias_and_coverage]:
        path = tmp_path / (render.__name__ + '.pdf')
        render(campaign, path)
        assert path.read_bytes().startswith(b'%PDF')
    legacy = SimpleNamespace(parameters=campaign.parameters.assign(parameter='halo.v_disp'))
    assert [p for p, _ in plots._standardized_parameter_panels(legacy)] == ['halo.v_disp']


def test_parameter_recovery_figure_fits_an_apj_page(tmp_path):
    """One row of six panels per component, centres pooled, the sources in the first free slot; a full paper
    campaign is 7.1 in wide and under 5.5 in tall."""
    import pandas as pd
    from types import SimpleNamespace
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: F401  (the PDF backend must be importable)
    from kappaccino.mock_validation import aggregate_plots as plots

    names = [n for n in plots.PAPER_PARAMETERS if not n.endswith(("grf2", "lambda_b2", "beta_low"))]
    names += ["source.1.beta_x", "source.1.beta_y", "source.2.beta_x", "source.2.beta_y"]
    rows = [dict(parameter=name, seed=seed, truth=value, median=value * 1.05, q16=value * .9, q84=value * 1.2)
            for name in names for seed, value in enumerate([1.0, 2.0, 3.0, 4.0])]
    campaign = SimpleNamespace(parameters=pd.DataFrame(rows))
    groups = plots.recovery_panels(campaign.parameters)
    assert [label for label, _, _ in groups] == ["Halo", "BCG", "GRF + GP", "Members", "Sources"]
    panels = {label: [name.split(".", 1)[1] for name, _ in entries] for label, _, entries in groups}
    assert panels["Halo"] == ["log10_mass_msun", "log10_aperture_mass_msun", "position_xy", "v_disp", "core_radius_kpc", "cut_radius_kpc"]
    assert panels["BCG"].count("position_xy") == 1 and panels["Sources"] == ["position_xy"]
    pooled = dict(groups[0][2])["halo.position_xy"]
    assert set(pooled.parameter) == {"halo.x_centre", "halo.y_centre"} and len(pooled) == 8
    layout = plots._recovery_rows(groups, 6)
    assert [len(row) for row in layout] == [6, 6, 6, 6]                     # the sources fill the BCG row
    assert [row[0][0] for row in layout] == ["Halo", "BCG", "GRF + GP", "Members"] and layout[1][-1][0] == "Sources"
    path = plots.plot_parameter_recovery(campaign, tmp_path / "recovery.pdf")
    import re
    box = re.search(rb"/MediaBox \[ *0 +0 +([0-9.]+) +([0-9.]+) *\]", path.read_bytes())
    width, height = (float(v) / 72.0 for v in box.groups())
    assert abs(width - plots.APJ_TEXT_WIDTH_IN) < 0.01 and height < 5.5
    # Fewer parameters still give a figure: one group, one row.
    few = SimpleNamespace(parameters=campaign.parameters[campaign.parameters.parameter.str.startswith("potfile.")])
    assert [len(row) for row in plots._recovery_rows(plots.recovery_panels(few.parameters), 6)] == [6]
    assert plots.plot_parameter_recovery(few, tmp_path / "few.pdf").exists()


def test_halo_aperture_uses_inside_radius_and_preserves_unknown_coverage():
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    from kappaccino.mock_validation.aggregate_plots import halo_aperture_parameters, coverage_table
    rows = [dict(seed=1, component='halo', radius_kpc=r, outer_image_radius_kpc=255,
                 truth_m2d_msun=1e14, model_q16=8e13, model_q50=1.01e14, model_q84=1.2e14)
            for r in (100, 230, 280)]
    result = halo_aperture_parameters(pd.DataFrame(rows))
    row = result.iloc[0]
    assert row.aperture_radius_kpc == 230
    assert row.truth == 14
    assert np.isclose(row["median"], np.log10(1.01e14))
    assert row.covered_68
    assert np.isnan(row.covered_95)
    result['truth_percentile'] = np.nan
    table = coverage_table(SimpleNamespace(parameters=result))
    assert np.isnan(table.iloc[0].coverage_95)
    rows[1].update(model_q025=7e13, model_q975=1.4e14)
    assert halo_aperture_parameters(pd.DataFrame(rows)).iloc[0].covered_95
    assert halo_aperture_parameters(pd.DataFrame()).empty
