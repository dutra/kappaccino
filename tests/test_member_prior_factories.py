"""The member-relation prior factories: Bergamini 2019 and the neutral set, and how they reach the sampled specs."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from kappaccino import priors
from kappaccino.config import MemberPopulationConfig, ScalingModelConfig
from kappaccino.core.state import build_state


def _population(synthetic_config, factory):
    base = synthetic_config.model.member_populations[0]
    return factory("members", base.catalog_path, base.z_lens, mag0=20.0)


def test_neutral_factories_give_log_uniform_cut_and_scaling_slopes(synthetic_config):
    population = _population(synthetic_config, MemberPopulationConfig.neutral)
    config = replace(synthetic_config, model=replace(synthetic_config.model, member_populations=(population,)), scaling=ScalingModelConfig.neutral())
    state = build_state(config)
    specs = {spec.field: spec for spec in state.parameter_specs if spec.component_family == "scaling"}
    cut = specs["cutkpc"]
    assert cut.log and priors.family(cut.latent_prior) == "Uniform"                      # log-uniform
    lower, upper = priors.physical_bounds(cut.prior)                       # given at mag0, where they apply
    assert math.isclose(lower, 2.0, rel_tol=1e-6) and math.isclose(upper, 100.0, rel_tol=1e-6)
    sigma = specs["sigma"]
    assert sigma.log and priors.family(sigma.latent_prior) == "TruncatedNormal" and 0 < priors.physical_bounds(sigma.prior)[1] < np.inf
    slopes = {field: priors.parameters(specs[field].latent_prior) for field in ("alpha_sigma", "beta_radius")}
    assert (slopes["alpha_sigma"]["loc"], slopes["alpha_sigma"]["scale"], slopes["beta_radius"]["loc"], slopes["beta_radius"]["scale"]) == (0.25, 0.10, 0.5, 0.25)


def test_bergamini2019_factories_match_the_defaults(synthetic_config):
    population = _population(synthetic_config, MemberPopulationConfig.bergamini2019)
    mean = lambda prior: math.exp(priors.parameters(priors.split(prior)[0])["loc"] + priors.parameters(priors.split(prior)[0])["scale"] ** 2 / 2)
    assert math.isclose(mean(population.sigma_prior), 200.0) and math.isclose(mean(population.cutkpc_prior), 12.0)    # untruncated means
    assert math.isclose(priors.physical_bounds(population.cutkpc_prior)[1], 60.0)
    assert priors.to_dict(ScalingModelConfig.bergamini2019().alpha_sigma_prior) == priors.to_dict(ScalingModelConfig().alpha_sigma_prior)
    assert priors.to_dict(ScalingModelConfig.bergamini2019().beta_radius_prior) == priors.to_dict(ScalingModelConfig().beta_radius_prior)
    assert ScalingModelConfig.neutral(fit_corekpc=False).fit_corekpc is False
