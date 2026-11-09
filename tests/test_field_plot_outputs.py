"""Check that the normal GP output path includes both shared-field figures."""
from types import SimpleNamespace

import pytest
from unittest.mock import Mock

from kappaccino.plotting import gp, fields


def test_standard_gp_outputs_include_field_figures(monkeypatch, tmp_path):
    monkeypatch.setattr(gp, 'apply_style', lambda: None)
    monkeypatch.setattr(gp, 'plot_knee_kernel', lambda *args: args[-1])
    monkeypatch.setattr(gp, 'plot_scale_posteriors', lambda *args: args[-2])
    expected = [tmp_path / 'total_grf_dimensionless_power.pdf', tmp_path / 'gp_correction_correlation.pdf']
    writer = Mock(return_value=expected)
    monkeypatch.setattr(fields, 'write_field_plots', writer)
    config = SimpleNamespace(paths=SimpleNamespace(figure_prefix='test'))
    evaluator = SimpleNamespace(gp_mode=True, gp_knee=True)
    paths = SimpleNamespace(figures=tmp_path)
    outputs = gp.write_gp_plots(config, None, evaluator, None, paths, log=lambda _: None)
    assert outputs[-2:] == expected
    writer.assert_called_once()


def test_no_mass_field_skips_the_shared_maps():
    config = SimpleNamespace(likelihood=SimpleNamespace(gp_blocks='all'))
    evaluator = SimpleNamespace(gp_knee=True, sigma_grf_index=-1, gp_knee_spectrum='knee', state=SimpleNamespace(gp_blocks='all'))
    messages = []
    assert fields.write_field_plots(config, None, evaluator, None, None, log=messages.append) == []
    assert 'skipped' in messages[0]


def test_system_blocks_render_through_the_shared_twin(monkeypatch):
    """A system-block run's maps come from its all-block twin, with the draws converted to the twin's coordinates."""
    from kappaccino.likelihood import grf_fields as reporting_fields
    twin = SimpleNamespace(gp_knee=True, sigma_grf_index=0, gp_knee_spectrum='knee', state=SimpleNamespace(gp_blocks='all'))
    run = SimpleNamespace(gp_knee=True, sigma_grf_index=0, gp_knee_spectrum='knee', state=SimpleNamespace(gp_blocks='system'),
                          _shared_field_twin=twin)
    monkeypatch.setattr(reporting_fields, 'shared_posterior', lambda e, p: 'twin-posterior')
    seen = []
    def compute(config, state, evaluator, posterior):
        seen.append((evaluator, posterior))
        raise RuntimeError('stop after the swap')
    monkeypatch.setattr(fields, 'compute_fields', compute)
    config = SimpleNamespace(likelihood=SimpleNamespace(gp_blocks='system'), paths=SimpleNamespace(figure_prefix='x'))
    messages = []
    with pytest.raises(RuntimeError):
        fields.write_field_plots(config, None, run, 'posterior', None, log=messages.append)
    assert seen == [(twin, 'twin-posterior')]
    assert 'twin' in messages[0]
