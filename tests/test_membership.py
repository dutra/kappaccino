"""Uncertain membership (MemberConfig.membership="marginalize"): the catalog p_member and the prior rule, the exact sum
over the enumerated members' indicators, the expected-mass deflection covariance of the others, the calibration shift,
and that the fixed mode and certain members leave the likelihood unchanged."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.special import logsumexp

from kappaccino.evaluator import Evaluator
from kappaccino.image_catalog import CatalogError
from kappaccino.member_catalog import load_member_catalog, members_from_lenstool, write_member_catalog
from kappaccino.state import build_state, with_member_classes
from test_gp_knee import knee_config
from test_member_catalog import SOURCES, _reference, _yaml_config

EXACT = np.asarray([0, 1])      # g1 and g2 lie among the images; g3 and g4 are far away


def _evaluator(config, likelihood="smoothed"):
    state = with_member_classes(build_state(config), EXACT)
    return state, Evaluator(state, likelihood)


def _marginalize(config, p=0.6, **members):
    return replace(config, members=replace(config.members, membership="marginalize", membership_default_probability=p, **members))


def _brute_force(evaluator, theta, p):
    """log sum_z prior(z) L(z) over the enumerated members plus the terms that do not depend on z."""
    physical = evaluator.physical(theta)
    terms = []
    for z in np.asarray(evaluator.membership_configurations):
        stats = evaluator.statistics(theta, membership=jnp.asarray(z))
        terms.append(float(jnp.sum(stats["loglike"]) + stats["source_correction"]) + float(np.sum(np.where(z, np.log(p), np.log1p(-p)))))
    rest = (evaluator.far_field_log_barrier(evaluator.fields(physical)["far_cut_radii"])
            + evaluator.ellipticity_log_prior(evaluator.packed_state(physical)[0]) + evaluator.kinematics_log_likelihood(physical))
    return logsumexp(terms) + float(rest)


def test_catalog_reads_p_member(synthetic_config, tmp_path: Path):
    config = _yaml_config(synthetic_config, tmp_path, edit=lambda m: m["g3"].update(p_member=0.25))
    table = load_member_catalog(config.model.member_populations[0].catalog_path, _reference(config))
    p = dict(zip(table["id"], table["catalog_p_member"]))
    assert p["g3"] == 0.25 and np.isnan(p["g1"])
    potfile = load_member_catalog(synthetic_config.model.member_populations[0].catalog_path, _reference(synthetic_config))
    assert potfile["catalog_p_member"].isna().all()


@pytest.mark.parametrize("value", [0.0, 1.5, "high"])
def test_catalog_rejects_invalid_p_member(synthetic_config, tmp_path: Path, value):
    members = members_from_lenstool(synthetic_config.model.member_populations[0].catalog_path, _reference(synthetic_config), band="F160W")
    members["g1"]["p_member"] = value
    with pytest.raises(CatalogError, match="p_member"):
        write_member_catalog(tmp_path / "members.yaml", members, band="F160W")


def test_prior_rule_confirms_spectroscopic_members(synthetic_config, tmp_path: Path):
    def edit(members):
        members["g1"].update(z={"value": 0.3, "kind": "spec"})
        members["g2"].update(kinematics=[{"source": "K", "sigma": 150.0, "err": 5.0}])
        members["g3"].update(p_member=0.2)
        members["g4"].update(z={"value": 0.31, "kind": "phot"})
    config = _marginalize(_yaml_config(synthetic_config, tmp_path, edit=edit), p=0.7)
    np.testing.assert_allclose(build_state(config).members.membership_prior, [1.0, 1.0, 0.2, 0.7])
    assert build_state(replace(config, members=replace(config.members, membership="fixed"))).members.membership_prior.tolist() == [1.0] * 4


def test_fixed_mode_and_certain_members_leave_the_likelihood_unchanged(synthetic_config, tmp_path: Path):
    state, evaluator = _evaluator(synthetic_config)
    theta = jnp.asarray(state.initial_theta)
    baseline = float(evaluator.loglike(theta))
    uncertain = _yaml_config(synthetic_config, tmp_path, edit=lambda m: [m[k].update(p_member=0.5) for k in m])
    assert float(_evaluator(uncertain)[1].loglike(theta)) == baseline              # fixed: p_member ignored
    assert float(_evaluator(_marginalize(synthetic_config, p=1.0))[1].loglike(theta)) == baseline


@pytest.mark.parametrize("likelihood", ["smoothed", "image_plane", "knee"])
def test_enumerated_members_are_summed_exactly(synthetic_config, likelihood):
    config = {"smoothed": synthetic_config,
              "image_plane": replace(synthetic_config, likelihood=replace(synthetic_config.likelihood, likelihood_mode="gp_image_plane")),
              "knee": knee_config(synthetic_config, "all")}[likelihood]
    state, evaluator = _evaluator(_marginalize(config), "smoothed" if likelihood == "smoothed" else "gp_image_plane")
    assert evaluator.membership_enumerated_positions.tolist() == [0, 1]
    assert evaluator.membership_marginal_positions.tolist() == [2, 3]
    theta = jnp.asarray(state.initial_theta)
    value, gradient = evaluator.value_and_grad(theta)
    np.testing.assert_allclose(float(value), _brute_force(evaluator, theta, 0.6), rtol=1e-12)
    assert np.isfinite(np.asarray(gradient)).all()
    posterior = np.asarray(evaluator.membership_posterior(theta))
    assert posterior.shape == (2,) and np.all((posterior >= 0) & (posterior <= 1))


def test_enumeration_follows_the_classification_score(synthetic_config):
    state = with_member_classes(build_state(_marginalize(synthetic_config, membership_max_enumerated=1)), EXACT)
    _, nearest_first = state, Evaluator(state)
    nearest = np.asarray(nearest_first.geometry.separation).min(axis=1)
    assert nearest_first.membership_enumerated_positions.tolist() == [int(np.argmin(nearest[EXACT]))]   # no classification
    ids = state.members.member_ids
    lower = int(np.argmax(nearest[EXACT]))                                  # the farther exact member scores highest
    rankings = [{"member": ids[i], "score": 5.0 if i == lower else 2.0} for i in EXACT]
    ranked = Evaluator(with_member_classes(state, EXACT, {"rankings": rankings}))
    assert ranked.membership_enumerated_positions.tolist() == [lower]
    assert len(ranked.membership_configurations) == 2


def test_expected_mass_and_deflection_covariance(synthetic_config):
    p = 0.6
    state, full = _evaluator(synthetic_config)
    _, uncertain = _evaluator(_marginalize(synthetic_config, p=p, membership_max_enumerated=0))
    assert len(uncertain.membership_enumerated_positions) == 0
    theta = jnp.asarray(state.initial_theta)
    physical = full.physical(theta)
    alpha = np.asarray(full.fields(physical)["member_alpha"])                 # every member at full mass
    fields = uncertain.fields(physical)
    np.testing.assert_allclose(np.asarray(fields["member_alpha"]), p * alpha, rtol=1e-10, atol=1e-14)
    # beta = theta - alpha, and the members carry (1 - p) less deflection
    np.testing.assert_allclose(np.asarray(fields["backprojected"]),
                               np.asarray(full.fields(physical)["backprojected"]) + (1 - p) * alpha.sum(axis=0), rtol=1e-10)
    expected = p * (1 - p) * np.einsum("gni,gnj->nij", alpha, alpha)
    base = uncertain.per_image_covariance(physical, fields["jacobian"])["covariance"]     # the same lens without the term
    added = np.asarray(uncertain.statistics(theta)["covariance"]) - np.asarray(base)
    np.testing.assert_allclose(added, expected, rtol=1e-8, atol=1e-14)


def test_calibration_shift(synthetic_config):
    config = _marginalize(synthetic_config, membership_calibration_prior=dist.Normal(0.0, 1.0))
    state, evaluator = _evaluator(config)
    index = state.members.membership_calibration_param_index
    assert state.parameter_specs[index].name == "membership.logit_shift"
    theta = np.asarray(state.initial_theta)
    assert theta[index] == 0.0
    _, plain = _evaluator(_marginalize(synthetic_config))
    without = np.delete(theta, index)
    np.testing.assert_allclose(float(evaluator.loglike(jnp.asarray(theta))), float(plain.loglike(jnp.asarray(without))), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(evaluator.membership_probabilities(evaluator.physical(jnp.asarray(theta))))[:2], 0.6)
    # p -> 1: every member counts fully.  The toy images prefer dropping g1 and g2 by about 40 nats, so the shift must
    # beat that; log sigmoid keeps the configuration priors finite where p rounds to 1.
    theta[index] = 300.0
    base_state, base = _evaluator(synthetic_config)
    np.testing.assert_allclose(float(evaluator.loglike(jnp.asarray(theta))), float(base.loglike(jnp.asarray(without))), rtol=1e-9)


def test_config_validation(synthetic_config):
    for members in ({"membership": "soft"}, {"membership_default_probability": 0.0}, {"membership_max_enumerated": -1},
                    {"membership_calibration_prior": 1.0}):
        with pytest.raises(ValueError, match="membership"):
            replace(synthetic_config, members=replace(synthetic_config.members, **members)).validate()
