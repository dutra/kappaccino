"""Model comparison (kappaccino.inference.model_comparison): the data log-likelihood and its tempered target, the data
count and temperature, BIC, the WBIC estimator and the comparison table.  The WBIC run itself is exercised by the
end-to-end test in test_pipeline.py."""
from __future__ import annotations

import math
from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from kappaccino.config import SamplingConfig
from kappaccino.core.state import build_state, with_member_classes
from kappaccino.inference import model_comparison as mc
from kappaccino.likelihood.evaluator import Evaluator
from test_gp_knee import knee_config

EXACT = np.asarray([0, 1])      # g1 and g2 lie among the images; g3 and g4 are far away


def _evaluator(config, likelihood="smoothed"):
    state = with_member_classes(build_state(config), EXACT)
    return state, Evaluator(state, likelihood)


@pytest.mark.parametrize("likelihood", ["smoothed", "knee"])
def test_data_and_prior_side_terms_add_up_to_the_likelihood(synthetic_config, likelihood):
    config = synthetic_config if likelihood == "smoothed" else knee_config(synthetic_config, "all")
    state, evaluator = _evaluator(config, "smoothed" if likelihood == "smoothed" else "gp_image_plane")
    theta = jnp.asarray(state.initial_theta)
    total, data = float(evaluator.log_likelihood(theta)), float(evaluator.data_log_likelihood(theta))
    physical = evaluator.physical(theta)
    stats = evaluator.statistics(theta)
    prior_side = float(stats["source_correction"] + evaluator.far_field_log_barrier(stats["fields"]["far_cut_radii"])
                       + evaluator.ellipticity_log_prior(evaluator.packed_state(physical)[0]))
    np.testing.assert_allclose(data, float(jnp.sum(stats["loglike"]) + evaluator.kinematics_log_likelihood(physical)), rtol=1e-12)
    np.testing.assert_allclose(total, data + prior_side, rtol=1e-12, atol=1e-9)
    # The tempered target scales the data terms only.
    np.testing.assert_allclose(float(evaluator.tempered_log_likelihood(theta, 1.0)), total, rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(float(evaluator.tempered_log_likelihood(theta, 0.25)), 0.25 * data + prior_side, rtol=1e-12, atol=1e-9)


def test_rejected_parameters_keep_the_rejection_values(synthetic_config):
    state, evaluator = _evaluator(synthetic_config)
    bad = jnp.asarray(state.initial_theta).at[0].set(jnp.nan)
    for value in (evaluator.log_likelihood(bad), evaluator.data_log_likelihood(bad), evaluator.tempered_log_likelihood(bad, 0.2)):
        assert float(value) == -1e30


def test_enumerated_members_refuse_the_data_likelihood(synthetic_config):
    config = replace(synthetic_config, members=replace(synthetic_config.members, membership="marginalize", membership_default_probability=0.6))
    state, evaluator = _evaluator(config)
    assert len(evaluator.membership_enumerated_positions)
    with pytest.raises(NotImplementedError, match="enumerated uncertain members"):
        evaluator.data_log_likelihood(jnp.asarray(state.initial_theta))
    with pytest.raises(NotImplementedError):
        mc.bic(state, evaluator, state.initial_theta)
    # Uncertain members by their expected mass only: separable.
    expected = replace(config, members=replace(config.members, membership_max_enumerated=0))
    state, evaluator = _evaluator(expected)
    assert np.isfinite(float(evaluator.data_log_likelihood(jnp.asarray(state.initial_theta))))


def test_data_count_temperature_and_bic(synthetic_config):
    state, evaluator = _evaluator(synthetic_config)
    counts = mc.n_data(state)
    assert counts == dict(image_coordinates=8, uncertain_image_coordinates=0, kinematics=0, n=8)
    assert mc.wbic_beta(8) == pytest.approx(1.0 / math.log(8.0))
    with pytest.raises(ValueError):
        mc.wbic_beta(2)
    record = mc.bic(state, evaluator, state.initial_theta)
    assert record["k"] == state.n_parameters and record["n"] == 8
    assert record["bic"] == pytest.approx(-2.0 * record["log_likelihood"] + state.n_parameters * math.log(8.0))
    assert sum(record["parameter_counts"].values()) == state.n_parameters
    assert record["data_fingerprint"] == mc.data_fingerprint(build_state(synthetic_config))   # the classes do not change the data


def test_wbic_estimator():
    rng = np.random.default_rng(0)
    log_like = rng.normal(-50.0, 3.0, size=(4, 500))
    result = mc.wbic_from_draws(log_like, beta=0.3, n=30)
    assert result["wbic"] == pytest.approx(100.0, abs=0.5)
    assert result["draws"] == 2000 and result["dropped_draws"] == 0 and result["ess"] > 1000
    assert result["wbic_error"] == pytest.approx(6.0 / math.sqrt(result["ess"]), rel=0.1)
    log_like[0, :10] = -np.inf                                      # rejected proposals are dropped
    log_like[1, :5] = -1e30
    assert mc.wbic_from_draws(log_like, beta=0.3, n=30)["dropped_draws"] == 15


def _record(run, bic=None, wbic=None, fingerprint="a", beta=0.2, error=1.0):
    return dict(run=run, bic=None if bic is None else dict(bic=bic, k=10, n=50, log_likelihood=-bic / 2, data_fingerprint=fingerprint),
                wbic=None if wbic is None else dict(wbic=wbic, wbic_error=error, n=50, beta=beta, data_fingerprint=fingerprint))


def test_compare_table_and_refusals():
    table = mc.compare([_record("nfw", 120.0, 130.0), _record("gnfw", 110.0, 124.0, error=2.0), _record("dpie", 125.0)])
    assert table["run"].tolist() == ["nfw", "gnfw", "dpie"]
    np.testing.assert_allclose(table["delta_bic"], [10.0, 0.0, 15.0])
    np.testing.assert_allclose(table["delta_wbic"].iloc[:2], [6.0, 0.0])
    np.testing.assert_allclose(table["delta_wbic_error"].iloc[:2], [math.hypot(1.0, 2.0), 0.0])
    assert np.isnan(table["wbic"].iloc[2])
    with pytest.raises(ValueError, match="different data"):
        mc.compare([_record("a", 1.0, 2.0), _record("b", 1.0, 2.0, fingerprint="b")])
    with pytest.raises(ValueError, match="temperatures"):
        mc.compare([_record("a", wbic=2.0), _record("b", wbic=2.0, beta=0.3)])


def test_wbic_settings_are_validated(synthetic_config):
    with pytest.raises(ValueError, match="wbic_samples"):
        replace(synthetic_config, sampling=SamplingConfig(wbic=True, wbic_samples=0)).validate()
