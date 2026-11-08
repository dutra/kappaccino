"""Saved run files: the configuration as JSON and arrays as HDF5, with nothing pickled."""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import numpyro.distributions as dist
import pytest

from kappaccino import priors
from kappaccino.config import DPIEHaloConfig, LensClusterSolverConfig, MassSheetConfig, ShearHaloConfig
from kappaccino.core.model import PosteriorResults
from kappaccino.io import artifacts
from kappaccino.io.hdf5 import load_arrays, save_arrays


def test_config_round_trips_through_json(synthetic_config, tmp_path):
    """A model with several halo classes (tagged in the JSON) and priors of several families reloads exactly."""
    model = replace(synthetic_config.model, large_halos=synthetic_config.model.large_halos + (
        ShearHaloConfig.free("S1", 0.5), MassSheetConfig("sheet", 0.01, 0.5, priors={"kappa": dist.Uniform(-0.1, 0.1)}),
        DPIEHaloConfig("2", 1.0, -1.0, 0.2, 0.0, 5.0, 900.0, 500.0, z_lens=0.5,
                       priors={"v_disp": dist.LogUniform(100.0, 1500.0), "x_centre": dist.Uniform(-5.0, 5.0)})))
    config = replace(synthetic_config, model=model)
    artifacts.save_config(tmp_path, config)
    assert not list(tmp_path.glob("*.pkl"))
    loaded = artifacts.load_config(tmp_path)
    assert loaded.to_json_dict() == json.loads((tmp_path / artifacts.CONFIG_FILE).read_text())
    assert [type(h) for h in loaded.model.large_halos] == [type(h) for h in config.model.large_halos]
    assert isinstance(loaded.model.large_halos, tuple)
    prior, again = config.model.large_halos[-1].priors["v_disp"], loaded.model.large_halos[-1].priors["v_disp"]
    x = np.array([150.0, 400.0, 1200.0])
    np.testing.assert_allclose(again.log_prob(x), prior.log_prob(x))
    assert priors.physical_bounds(again) == pytest.approx(priors.physical_bounds(prior))
    assert LensClusterSolverConfig.from_json_dict(loaded.to_json_dict()).to_json_dict() == loaded.to_json_dict()


def test_arrays_round_trip_and_objects_are_refused(tmp_path):
    path = save_arrays(tmp_path / "a.h5", x=np.arange(6.0).reshape(2, 3), flag=np.array([True, False]), n=7,
                       labels=np.array(["1.a", "1.b"]), fingerprint="abc", skipped=None)
    data = load_arrays(path)
    assert set(data) == {"x", "flag", "n", "labels", "fingerprint"}
    np.testing.assert_array_equal(data["x"], np.arange(6.0).reshape(2, 3))
    assert data["flag"].dtype == bool and int(data["n"]) == 7
    assert data["labels"].tolist() == ["1.a", "1.b"] and str(data["fingerprint"]) == "abc"
    with pytest.raises(TypeError, match="object arrays"):
        save_arrays(tmp_path / "b.h5", ragged=np.array([np.zeros(2), np.zeros(3)], dtype=object))


def test_posterior_round_trips_through_hdf5(tmp_path):
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(6, 3))
    posterior = PosteriorResults(samples=samples, physical=2 * samples, log_prob=rng.normal(size=6), accept_prob=np.full(6, 0.9),
                                 diverging=np.zeros(6, bool), num_steps=np.full(6, 7.0), warmup_steps=4, sample_steps=3,
                                 num_chains=2, energy=rng.normal(size=6), step_size=np.array([0.1, 0.2]),
                                 grouped_samples=samples.reshape(2, 3, 3), diagnostics={"note": "x"}, median_fit=None)
    artifacts.save_posterior(tmp_path, posterior)
    assert (tmp_path / artifacts.POSTERIOR_FILE).suffix == ".h5"
    loaded = artifacts.load_posterior(tmp_path)
    np.testing.assert_array_equal(loaded.samples, samples)
    np.testing.assert_array_equal(loaded.grouped_samples, posterior.grouped_samples)
    assert loaded.diverging.dtype == bool and loaded.num_chains == 2 and loaded.warmup_steps == 4
    assert loaded.median_fit is None and loaded.maximum_likelihood_fit is None and loaded.diagnostics == {"note": "x"}
