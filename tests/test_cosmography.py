"""Sampled flat wCDM cosmology (cosmography): Om0 and w0 as parameters of the lens model."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from kappaccino import pipeline
from kappaccino.config import PartitionConfig, ReportingConfig, RuntimeConfig, SamplingConfig, SearchConfig
from kappaccino.core.state import build_state
from kappaccino.lens.cosmology import flat_wcdm_kpc_per_arcsec, flat_wcdm_lensing_efficiency
from kappaccino.likelihood.evaluator import Evaluator
from kappaccino.run_files import load_priors, load_run

ROOT = Path(__file__).resolve().parents[1]
PRIORS = {"Om0": dist.Uniform(0.05, 0.95), "w0": dist.Uniform(-2.0, -0.2)}


def cosmography(config, priors=PRIORS, **fields):
    cosmology = replace(config.model.cosmology, Ode0=None, priors=dict(priors), **fields)
    return replace(config, model=replace(config.model, cosmology=cosmology))


def test_sampled_cosmology_adds_two_lens_parameters(synthetic_config):
    fixed = build_state(synthetic_config)
    assert (fixed.om0_param_index, fixed.w0_param_index) == (-1, -1) and not Evaluator(fixed).cosmology_sampled
    state = build_state(cosmography(synthetic_config).validate())
    om0, w0 = state.parameter_specs[state.om0_param_index], state.parameter_specs[state.w0_param_index]
    assert (om0.name, w0.name) == ("cosmology.Om0", "cosmology.w0") and om0.component_family == w0.component_family == "large"
    rest = [s.name for i, s in enumerate(state.parameter_specs) if i not in (state.om0_param_index, state.w0_param_index)]
    assert rest == [s.name for s in fixed.parameter_specs]
    evaluator = Evaluator(state)
    physical = evaluator.physical(jnp.asarray(state.initial_theta))
    assert (float(physical[state.om0_param_index]), float(physical[state.w0_param_index])) == (0.3, -1.0)   # the fiducial values
    assert [float(v) for v in evaluator.cosmology(physical)] == [70.0, 0.3, -1.0]
    only_om0 = build_state(cosmography(synthetic_config, {"Om0": PRIORS["Om0"]}).validate())
    assert only_om0.om0_param_index >= 0 and only_om0.w0_param_index == -1


def test_likelihood_at_the_fiducial_cosmology_equals_the_fixed_one(synthetic_config):
    fixed = build_state(synthetic_config)
    state = build_state(cosmography(synthetic_config))
    theta = jnp.asarray(state.initial_theta)
    reference, evaluator = Evaluator(fixed), Evaluator(state)
    np.testing.assert_allclose(float(evaluator.log_likelihood(theta)), float(reference.log_likelihood(jnp.asarray(fixed.initial_theta))),
                               rtol=0, atol=1e-9)
    for got, expected in zip(evaluator.lens_geometry(evaluator.physical(theta)),
                             reference.lens_geometry(reference.physical(jnp.asarray(fixed.initial_theta)))):
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    # The likelihood depends on both parameters, with gradients that match central differences.
    gradient = jax.grad(evaluator.log_likelihood)(theta)
    for index in (state.om0_param_index, state.w0_param_index):
        step = 1.0e-5
        difference = (float(evaluator.log_likelihood(theta.at[index].add(step))) - float(evaluator.log_likelihood(theta.at[index].add(-step)))) / (2 * step)
        assert abs(difference) > 1.0
        np.testing.assert_allclose(float(gradient[index]), difference, rtol=1e-5)


def test_geometry_follows_the_sampled_cosmology(synthetic_config):
    state = build_state(cosmography(synthetic_config))
    evaluator = Evaluator(state)
    theta = jnp.asarray(state.initial_theta).at[state.om0_param_index].set(0.6).at[state.w0_param_index].set(-0.5)
    physical = evaluator.physical(theta)
    kpc_per_arcsec, factor_by_z, rho_crit = evaluator.lens_geometry(physical)
    np.testing.assert_allclose(kpc_per_arcsec, flat_wcdm_kpc_per_arcsec(state.z_lens, 70.0, 0.6, -0.5), rtol=1e-12)
    efficiency = flat_wcdm_lensing_efficiency(state.z_lens, jnp.asarray(state.z_source_values), 70.0, 0.6, -0.5)
    np.testing.assert_allclose(factor_by_z / factor_by_z[0], efficiency / efficiency[0], rtol=1e-12)
    assert abs(float(kpc_per_arcsec) / state.kpc_per_arcsec - 1.0) > 0.01 and float(rho_crit) != evaluator.nfw_rho_crit
    # Maps and image solves at another redshift use the cosmology of the parameter vector, not the fiducial table.
    start = evaluator.physical(jnp.asarray(state.initial_theta))
    ratio = evaluator.packed_state_for_z(physical, 5.0).sigma0 / evaluator.packed_state_for_z(start, 5.0).sigma0
    expected = (flat_wcdm_lensing_efficiency(state.z_lens, 5.0, 70.0, 0.6, -0.5) / flat_wcdm_lensing_efficiency(state.z_lens, 5.0, 70.0, 0.3, -1.0)
                * flat_wcdm_kpc_per_arcsec(state.z_lens, 70.0, 0.6, -0.5) / flat_wcdm_kpc_per_arcsec(state.z_lens, 70.0, 0.3, -1.0))
    np.testing.assert_allclose(np.asarray(ratio)[np.asarray(ratio) > 0], float(expected), rtol=1e-10)


@pytest.mark.parametrize("change, message", [
    (dict(priors={"H0": dist.Uniform(50.0, 90.0)}), "only Om0 and w0"),
    (dict(priors={"Om0": dist.Uniform(0.0, 1.0)}), "support inside"),
    (dict(priors={"w0": dist.Uniform(-2.0, 0.5)}), "support inside"),
    (dict(priors={"Om0": dist.Normal(0.3, 0.1)}), "finite bounds"),
    (dict(priors=PRIORS, Om0=0.99), "lies outside its prior"),
    (dict(priors=PRIORS, Ode0=0.7), "Ode0 must be None"),
])
def test_invalid_cosmology_priors_raise(synthetic_config, change, message):
    cosmology = replace(synthetic_config.model.cosmology, **{"Ode0": None, **change})
    with pytest.raises(ValueError, match=message):
        replace(synthetic_config, model=replace(synthetic_config.model, cosmology=cosmology)).validate()


def test_uncertain_images_are_refused(synthetic_config):
    with pytest.raises(ValueError, match="not supported with a sampled cosmology"):
        replace(cosmography(synthetic_config), partition=PartitionConfig()).validate()


def test_priors_file_declares_the_cosmology(tmp_path):
    path = tmp_path / "priors.yaml"
    path.write_text("""
z_lens: 0.4
cosmology:
  H0: 70.0
  Om0:
    start: 0.27
    prior: uniform(0.05, 0.95)
  w0: normal(-1.0, 0.3) in [-2.0, -0.2]
""")
    cosmology = load_priors(path)[0].cosmology
    assert (cosmology.H0, cosmology.Om0, cosmology.w0, cosmology.Ode0) == (70.0, 0.27, -1.0, None) and set(cosmology.priors) == {"Om0", "w0"}
    path.write_text("z_lens: 0.4\ncosmology: {H0: 'uniform(50, 90)'}\n")
    with pytest.raises(ValueError, match="cannot be sampled"):
        load_priors(path)


@pytest.mark.parametrize("folder, om0", [("catalogs/hera", 0.24), ("catalogs/ares", 0.272), ("catalogs/as1063/bergamini2019", 0.3),
                                         ("catalogs/as1063/diego2026", 0.3), ("catalogs/m0416/bergamini2023", 0.3),
                                         ("catalogs/m0416/canucs2025", 0.3)])
def test_cluster_cosmography_profiles(folder, om0, monkeypatch):
    monkeypatch.chdir(ROOT)
    fixed = load_run(folder, overrides=["paths.run_name=t"]).validate()
    config = load_run(folder, ["cosmography"], ["paths.run_name=t"]).validate()
    cosmology = config.model.cosmology
    assert (cosmology.Om0, cosmology.w0, cosmology.Ode0) == (om0, -1.0, None) and set(cosmology.priors) == {"Om0", "w0"}
    assert not fixed.model.cosmology.priors and config.model.image_constraints.catalog_path == fixed.model.image_constraints.catalog_path
    # Everything else is the fixed-cosmology run of the same folder.
    a, b = fixed.to_json_dict(), config.to_json_dict()
    a["model"].pop("cosmology"), b["model"].pop("cosmology")
    assert a == b
    assert load_run(folder, ["fast", "cosmography"], ["paths.run_name=t"]).validate().likelihood.gp_blocks == "system"


def test_cosmography_end_to_end(synthetic_config):
    """Both stages with Om0 and w0 sampled: the search, NUTS, the outputs and resume."""
    config = replace(cosmography(synthetic_config), runtime=RuntimeConfig(seed=3, chains=2, nuts_chain_method="sequential", resume=True),
                     search=SearchConfig(n_starts=2, member_starts=1, lbfgs_max_iter=30, polish_max_iter=20),
                     sampling=SamplingConfig(warmup=10, samples=10, max_tree_depth=4),
                     reporting=ReportingConfig(draws=2, image_draws=2, extra_images=False))
    result = pipeline.run(config)
    state, posterior = result.state, result.posterior
    assert posterior.samples.shape == (20, state.n_parameters) and np.isfinite(posterior.log_prob).all()
    physical = np.asarray(jax.vmap(result.evaluator.physical)(jnp.asarray(posterior.samples)))
    om0, w0 = physical[:, state.om0_param_index], physical[:, state.w0_param_index]
    assert np.all((om0 > 0.05) & (om0 < 0.95)) and np.all((w0 > -2.0) & (w0 < -0.2)) and np.ptp(om0) > 0
    assert pipeline.run(config).resumed
