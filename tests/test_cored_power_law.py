"""Cored power-law members (profile 102): the dPIE with a free 3D slope at fixed total mass, core and cut.

The kernel is checked against the dPIE at slope 2, against the Poisson equation and finite differences at other slopes,
and against the circular closed form; the model wiring against a dPIE population and the configuration checks."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from kappaccino.core.model import CORED_POWER_LAW_PROFILE, DP_IE_PROFILE
from kappaccino.core.state import build_state, with_member_classes
from kappaccino.lens import cored_power_law as cpl, lensing
from kappaccino.lens.farfield import evaluate_farfield_per_member, member_farfield_geometry
from kappaccino.likelihood.evaluator import Evaluator

SHAPES = ((0.0, 0.0), (0.2, -0.1), (-0.35, 0.3), (0.45, 0.4))
RADII = ((0.005, 2.0), (0.3, 5.0))


def _points(n=300, seed=2, low=-2.5, high=1.6):
    rng = np.random.default_rng(seed)
    radius, angle = 10 ** rng.uniform(low, high, n), rng.uniform(0, 2 * np.pi, n)
    return jnp.asarray(radius * np.cos(angle)), jnp.asarray(radius * np.sin(angle))


def _relative(value, reference):
    return float(jnp.max(jnp.abs(value - reference)) / jnp.max(jnp.abs(reference)))


@pytest.mark.parametrize("e1,e2", SHAPES)
@pytest.mark.parametrize("a,s", RADII)
def test_slope_two_is_the_dpie(e1, e2, a, s):
    x, y = _points()
    expected = lensing.dpie_fields(x, y, 0.7, a, s, e1, e2, 0.1, -0.2)
    actual = cpl.cored_power_law_fields(x, y, 0.7, a, s, 2.0, e1, e2, 0.1, -0.2)
    for value, reference, tolerance in zip(actual, expected, (1e-4, 1e-4, 1e-3)):   # the quadrature budget (PANEL_NODES)
        assert _relative(value, reference) < tolerance


@pytest.mark.parametrize("slope", (1.1, 1.5, 2.4, 2.9))
@pytest.mark.parametrize("e1,e2", SHAPES)
@pytest.mark.parametrize("a,s", RADII)
def test_poisson_equation_and_symmetry(slope, e1, e2, a, s):
    """Half the trace of U is the convergence at the elliptical radius of the dPIE convention."""
    x, y = _points()
    _, u, du = cpl.cored_power_law_fields(x, y, 0.7, a, s, slope, e1, e2, 0.0, 0.0)
    phi, q = lensing.ellipticity_frame(e1, e2)
    xr, yr = x * jnp.cos(phi) + y * jnp.sin(phi), -x * jnp.sin(phi) + y * jnp.cos(phi)
    kappa = cpl.convergence(q * xr ** 2 + yr ** 2 / q, 0.7, a, s, slope)
    trace = 0.5 * (u[:, 0, 0] + u[:, 1, 1])
    assert _relative(trace, kappa) < 1e-4
    np.testing.assert_allclose(u[:, 0, 1], u[:, 1, 0], atol=1e-14)
    np.testing.assert_allclose(du[:, 0, 1], du[:, 1, 0], atol=1e-12)
    np.testing.assert_allclose(du[..., 0, 1], du[..., 1, 0], atol=1e-12)       # third derivatives commute


@pytest.mark.parametrize("slope", (1.3, 2.0, 2.7))
def test_derivatives_match_finite_differences(slope):
    x, y = _points(80, seed=5, low=-1.0, high=1.2)
    args = (0.7, 0.1, 3.0, slope, 0.25, 0.15, 0.1, -0.2)
    alpha, u, du = cpl.cored_power_law_fields(x, y, *args)
    h = 1e-5
    for axis, (dx, dy) in enumerate(((h, 0.0), (0.0, h))):
        plus = cpl.cored_power_law_fields(x + dx, y + dy, *args)
        minus = cpl.cored_power_law_fields(x - dx, y - dy, *args)
        assert _relative((plus[0] - minus[0]) / (2 * h), u[..., axis]) < 1e-6
        assert _relative((plus[1] - minus[1]) / (2 * h), du[..., axis]) < 1e-5


def test_circular_kernel_matches_the_closed_form():
    x, y = _points()
    for slope in (1.2, 2.0, 2.8):
        elliptical = cpl.cored_power_law_fields(x, y, 0.7, 0.05, 2.0, slope, 0.0, 0.0, 0.0, 0.0)
        circular = cpl.circular_fields(x, y, 0.7, 0.05, 2.0, slope)
        for value, reference in zip(elliptical, circular):
            assert _relative(value, reference) < 1e-4


@pytest.mark.parametrize("slope", (1.1, 1.7, 2.0, 2.6, 2.9))
def test_mass_inside_the_cut_is_the_dpie_mass(slope):
    """Every slope keeps the dPIE's projected mass inside the cut radius (R alpha(R) is that mass over pi)."""
    a, s = 0.05, 2.0
    dpie, _ = lensing.dpie_fields(jnp.asarray([s]), jnp.asarray([0.0]), 0.7, a, s, 0.0, 0.0, 0.0, 0.0, derivative=False)[:2]
    np.testing.assert_allclose(float(cpl.circular_deflection(s, 0.7, a, s, slope)), float(dpie[0, 0]), rtol=1e-12)


def test_dpie_total_mass_and_finite_far_fields():
    """At slope 2 the mass far outside the cut is the dPIE's total, 2 pi sigma0 a s; every slope stays finite there."""
    radius = jnp.asarray([1.0e5, 1.0e7])                                   # the dPIE misses s / R of its mass at R
    np.testing.assert_allclose(cpl.circular_deflection(radius, 0.7, 0.05, 2.0, 2.0) * radius, 2 * 0.7 * 0.05 * 2.0, rtol=1e-4)
    for slope in (1.05, 1.5, 2.9):
        assert np.all(np.isfinite(np.asarray(cpl.circular_fields(radius, 0.0 * radius, 0.7, 0.05, 2.0, slope)[2])))


def test_steeper_slopes_concentrate_the_mass():
    """At fixed total mass, the mass inside a small radius grows with the slope."""
    enclosed = [float(cpl.circular_deflection(0.3, 0.7, 0.02, 3.0, slope)) for slope in (1.3, 1.7, 2.0, 2.4, 2.8)]
    assert np.all(np.diff(enclosed) > 0)


def test_gradient_in_the_slope_matches_finite_differences():
    x, y = _points(50, seed=7, low=-1.0, high=1.0)

    def total(slope):
        alpha, u, _ = cpl.cored_power_law_fields(x, y, 0.7, 0.1, 3.0, slope, 0.2, 0.1, 0.0, 0.0, derivative=False)
        return jnp.sum(alpha ** 2) + jnp.sum(u ** 2)

    h = 1e-6
    numeric = (total(2.3 + h) - total(2.3 - h)) / (2 * h)
    np.testing.assert_allclose(float(jax.grad(total)(2.3)), float(numeric), rtol=1e-5)


def test_far_field_power_law_members_use_the_circular_profile():
    images = np.array([[30.0, 4.0], [-25.0, 18.0], [5.0, -40.0]])
    members = np.array([[0.0, 0.0], [2.0, -1.0]])
    geometry = member_farfield_geometry(images, members)
    sigma0, ra, rs = jnp.asarray([0.5, 0.3]), jnp.asarray([0.02, 0.03]), jnp.asarray([1.5, 2.0])
    series = evaluate_farfield_per_member(geometry, sigma0, ra, rs)
    exact = evaluate_farfield_per_member(geometry, sigma0, ra, rs, slope=jnp.asarray([2.0, 2.0]), power_law=np.array([True, True]))
    for value, reference in zip(exact, series):
        assert _relative(value, reference) < 1e-4                           # the dPIE series is accurate this far out
    steep = evaluate_farfield_per_member(geometry, sigma0, ra, rs, slope=jnp.asarray([2.6, 2.0]), power_law=np.array([True, False]))
    offset = images - members[0]
    reference = cpl.circular_fields(jnp.asarray(offset[:, 0]), jnp.asarray(offset[:, 1]), 0.5, 0.02, 1.5, 2.6)
    np.testing.assert_allclose(steep[0][0], reference[0], rtol=1e-12)
    np.testing.assert_allclose(steep[0][1], series[0][1], rtol=1e-12)        # the dPIE member keeps the series


# ----------------------------------------------------------------------------- the model
def _power_law_config(config, *, fixed_slope=None, slope_prior=None):
    population = config.model.member_populations[0]
    population = replace(population, profile_type=CORED_POWER_LAW_PROFILE,
                         **({"slope": fixed_slope} if fixed_slope is not None else {}),
                         **({"slope_prior": slope_prior} if slope_prior is not None else {}))
    config = replace(config, model=replace(config.model, member_populations=(population,)))
    if fixed_slope is not None:
        config = replace(config, scaling=replace(config.scaling, fixed_relation_fields=("slope",)))
    return config.validate()


def _exact_state(config):
    state = build_state(config)
    return with_member_classes(state, np.arange(2), dict(state.members.classification))   # the two members near the images


def test_a_population_at_slope_two_reproduces_the_dpie_population(synthetic_config):
    dpie = _exact_state(synthetic_config)
    power_law = _exact_state(_power_law_config(synthetic_config, fixed_slope=2.0))
    profiles = np.asarray(power_law.packed_lens_spec.profile_type)[np.asarray(power_law.members.component_indices)]
    assert np.all(profiles == CORED_POWER_LAW_PROFILE)
    assert [s.name for s in power_law.parameter_specs] == [s.name for s in dpie.parameter_specs]   # the slope is fixed
    a, b = Evaluator(dpie), Evaluator(power_law)
    theta = jnp.asarray(dpie.initial_theta)
    fields_a = a.fields(a.physical(theta), force_exact=True)
    fields_b = b.fields(b.physical(theta), force_exact=True)
    np.testing.assert_allclose(fields_b["backprojected"], fields_a["backprojected"], atol=1e-6)
    np.testing.assert_allclose(fields_b["jacobian"], fields_a["jacobian"], atol=1e-6)
    np.testing.assert_allclose(float(b.log_likelihood(theta)), float(a.log_likelihood(theta)), atol=1e-3)


def test_the_sampled_slope_is_a_population_parameter(synthetic_config):
    state = _exact_state(_power_law_config(synthetic_config, slope_prior=dist.Uniform(1.2, 2.8)))
    names = [s.name for s in state.parameter_specs]
    assert "members.slope" in names
    index = names.index("members.slope")
    spec = state.parameter_specs[index]
    assert spec.component_family == "scaling" and spec.profile_type == CORED_POWER_LAW_PROFILE and not spec.log
    assert float(state.initial_theta[index]) == pytest.approx(2.0)
    components = np.asarray(state.members.component_indices)
    np.testing.assert_array_equal(np.asarray(state.packed_lens_spec.slope_param_index)[components], index)
    evaluator = Evaluator(state)
    theta = np.array(state.initial_theta, dtype=float)
    packed, validity = evaluator.packed_state(evaluator.physical(jnp.asarray(theta)), factor_per_image=1.0)
    assert bool(validity.is_valid)
    np.testing.assert_allclose(np.asarray(packed.slope)[components], 2.0)
    assert np.all(np.asarray(packed.slope)[np.setdiff1d(np.arange(len(packed.slope)), components)] == 2.0)
    value, gradient = evaluator.value_and_grad(jnp.asarray(theta))
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(gradient)))
    assert float(gradient[index]) != 0.0
    theta[index] = 2.5
    assert float(evaluator.log_likelihood(jnp.asarray(theta))) != pytest.approx(float(value))


def test_far_field_members_follow_the_slope(synthetic_config):
    """Far-field members (the two distant ones) move with the slope through the circular profile."""
    state = _exact_state(_power_law_config(synthetic_config))
    evaluator = Evaluator(state)
    assert evaluator.far_power_law.all() and evaluator.far_power_law.size == 2
    index = [s.name for s in state.parameter_specs].index("members.slope")
    theta = np.array(state.initial_theta, dtype=float)
    far = []
    for slope in (1.5, 2.5):
        theta[index] = slope
        physical = evaluator.physical(jnp.asarray(theta))
        far.append(np.asarray(evaluator.fields(physical)["member_alpha"])[evaluator.far_positions])
    assert not np.allclose(far[0], far[1])


def test_configuration_checks(synthetic_config):
    population = synthetic_config.model.member_populations[0]

    def with_population(**changes):
        model = replace(synthetic_config.model, member_populations=(replace(population, **changes),))
        return replace(synthetic_config, model=model)

    with pytest.raises(ValueError, match="profile_type must be 81"):
        with_population(profile_type=12).validate()
    with pytest.raises(ValueError, match=r"inside \(1, 3\)"):
        with_population(profile_type=CORED_POWER_LAW_PROFILE, slope_prior=dist.Uniform(0.8, 2.5)).validate()
    with pytest.raises(ValueError, match=r"inside \(1, 3\)"):
        with_population(profile_type=CORED_POWER_LAW_PROFILE, slope=3.2).validate()
    config = with_population(profile_type=CORED_POWER_LAW_PROFILE)
    with pytest.raises(ValueError, match="lenstronomy"):
        replace(config, image_diagnostics=replace(config.image_diagnostics, exact_image_finder="lenstronomy")).validate()
    assert with_population(profile_type=DP_IE_PROFILE).validate() is not None


# ----------------------------------------------------------------------------- stellar kinematics (Jeans)
def _brute_aperture_dispersion_sq(slope, a, s, aperture):
    """The isotropic Jeans aperture dispersion by direct nested quadrature: M(r), the projected sigma^2 profile, the
    aperture average; in the units of ``aperture_dispersion_sq``."""
    from scipy.integrate import quad
    from scipy.special import gamma
    amp = float(cpl.amplitude(1.0, a, s, slope))
    rho0 = amp * gamma(slope / 2) / (np.sqrt(np.pi) * gamma((slope - 1) / 2))
    rho = lambda r: rho0 * ((r * r + a * a) ** (-slope / 2) - (r * r + s * s) ** (-slope / 2))
    mass = lambda r: quad(lambda x: 4 * np.pi * rho(x) * x * x, 0, r, limit=200)[0]
    projected_sq = lambda big_r: 2 * quad(lambda r: rho(r) * mass(r) / r ** 2 * np.sqrt(r * r - big_r * big_r),
                                          big_r, 200 * s, limit=200, points=[a, s])[0]
    numerator = quad(lambda big_r: 2 * np.pi * big_r * projected_sq(big_r), 0, aperture, limit=50)[0]
    return numerator / (np.pi * amp * float(cpl.enclosed(aperture ** 2, a, s, slope))) * 3 / (4 * np.pi)


@pytest.mark.parametrize("slope,a,s,aperture", ((2.0, 0.03, 2.0, 0.3), (1.5, 0.03, 2.0, 0.3), (2.6, 0.1, 3.0, 0.5)))
def test_aperture_dispersion_matches_brute_force_jeans(slope, a, s, aperture):
    np.testing.assert_allclose(float(cpl.aperture_dispersion_sq(slope, a, s, aperture)),
                               _brute_aperture_dispersion_sq(slope, a, s, aperture), rtol=1e-3)


def test_aperture_dispersion_ratio_is_one_at_slope_two_and_grows_with_the_slope():
    a, s, aperture = 0.03, 2.0, 0.3
    np.testing.assert_allclose(float(cpl.aperture_dispersion_ratio(2.0, a, s, aperture)), 1.0, rtol=1e-12)
    ratios = [float(cpl.aperture_dispersion_ratio(g, a, s, aperture)) for g in (1.3, 1.7, 2.0, 2.4, 2.8)]
    assert np.all(np.diff(ratios) > 0)
    gradient = jax.grad(lambda g: cpl.aperture_dispersion_ratio(g, a, s, aperture))(2.3)
    h = 1e-6
    numeric = (cpl.aperture_dispersion_ratio(2.3 + h, a, s, aperture) - cpl.aperture_dispersion_ratio(2.3 - h, a, s, aperture)) / (2 * h)
    np.testing.assert_allclose(float(gradient), float(numeric), rtol=1e-6)


APERTURES = {"g1": 0.35, "g2": 0.2}


def _with_apertures(config, tmp_path, *, apertures=APERTURES):
    """The kinematics test catalog (test_member_kinematics) with an aperture on each measurement."""
    from test_member_kinematics import _with_kinematics

    def extra(members):
        for name, radius in apertures.items():
            members[name]["kinematics"][0]["aperture_arcsec"] = radius
    return _with_kinematics(config, tmp_path, extra=extra)


def test_power_law_kinematics_use_the_jeans_ratio(synthetic_config, tmp_path):
    """At slope 2 the kinematics term is the dPIE term; at another slope each model dispersion carries the ratio at the
    member's core, cut and aperture (the measurement's catalog aperture, or the configured radius for all)."""
    from test_member_kinematics import MEASURED
    dpie_config = _with_apertures(synthetic_config, tmp_path / "dpie")
    power_law = _power_law_config(_with_apertures(synthetic_config, tmp_path / "cpl"))
    dpie_state, state = build_state(dpie_config), build_state(power_law)
    kinematics = state.member_kinematics
    ids = [member.split(":")[1] for member in kinematics["member_ids"]]
    np.testing.assert_allclose(kinematics["aperture_arcsec"], [APERTURES[i] for i in ids])
    a, b = Evaluator(dpie_state, "smoothed"), Evaluator(state, "smoothed")
    names = [s.name for s in state.parameter_specs]
    theta = np.asarray(state.initial_theta, dtype=float)
    theta_dpie = np.delete(theta, names.index("members.slope"))
    np.testing.assert_allclose(float(b.kinematics_log_likelihood(b.physical(jnp.asarray(theta)))),
                               float(a.kinematics_log_likelihood(a.physical(jnp.asarray(theta_dpie)))), rtol=1e-10)
    theta[names.index("members.slope")] = 2.5
    physical = b.physical(jnp.asarray(theta))
    packed, _ = b.packed_state(physical, factor_per_image=1.0)
    components = np.asarray(kinematics["component_index"])
    ratio = np.asarray(cpl.aperture_dispersion_ratio(2.5, np.asarray(packed.Ra)[components], np.asarray(packed.Rs)[components],
                                                     kinematics["aperture_arcsec"]))
    sigma = np.asarray(b.component_dispersions(physical))[components]
    c = power_law.model.member_populations[0].kinematics_aperture_factor
    z = (np.log(kinematics["sigma"]) - np.log(c * sigma * ratio)) / kinematics["log_err"]
    expected = -0.5 * np.sum(z ** 2) - np.sum(np.log(kinematics["log_err"])) - 0.5 * z.size * np.log(2 * np.pi)
    np.testing.assert_allclose(float(b.kinematics_log_likelihood(physical)), expected, rtol=1e-10)
    assert np.all(ratio > 1.0) and len(MEASURED) == len(components)
    fixed = replace(power_law, model=replace(power_law.model, member_populations=(
        replace(power_law.model.member_populations[0], kinematics_aperture_arcsec=0.4),)))
    np.testing.assert_allclose(build_state(fixed.validate()).member_kinematics["aperture_arcsec"], 0.4)


def test_power_law_kinematics_need_an_aperture(synthetic_config, tmp_path, monkeypatch):
    """A measurement without its own aperture takes its source's; with neither, a cored power-law population refuses its
    kinematics, while a dPIE population does not need one."""
    import test_member_catalog
    from test_member_kinematics import _with_kinematics
    with_source = _with_kinematics(synthetic_config, tmp_path / "source")
    source_aperture = test_member_catalog.SOURCES["K"]["aperture_arcsec"]
    np.testing.assert_allclose(build_state(_power_law_config(with_source)).member_kinematics["aperture_arcsec"], source_aperture)
    sources = {name: {"reference": entry["reference"], "quantity": "sigma_e"} for name, entry in test_member_catalog.SOURCES.items()}
    monkeypatch.setattr(test_member_catalog, "SOURCES", sources)
    plain = _with_kinematics(synthetic_config, tmp_path / "plain")
    assert np.all(np.isnan(build_state(plain).member_kinematics["aperture_arcsec"]))
    with pytest.raises(ValueError, match="no aperture_arcsec"):
        build_state(_power_law_config(plain))
