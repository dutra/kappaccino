"""Exact-kernel and independent derivative checks for fixed member classes."""
import json
from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lenscluster.member_tiers import (
    MemberTierRecord, MemberTierThresholds, member_farfield_geometry,
    evaluate_farfield, finite_logh_change, classify_member_tiers, audit_member_tiers,
)
from jaxtronomy.LensModel.Profiles.dpie_nie_e import DPIE_NIE_E

jax.config.update("jax_enable_x64", True)


def test_circular_kernel_normalization_and_order():
    positions = np.array([[16., 0.], [32., 0.], [64., 0.]])
    geometry = member_farfield_geometry(positions, np.array([[0., 0.]]))
    fields = evaluate_farfield(geometry, np.array([2.]), np.array([.2]), np.array([4.]))
    exact = np.stack(DPIE_NIE_E.derivatives(positions[:, 0], positions[:, 1], 2., .2, 4., 0., 0.), axis=-1)
    errors = np.linalg.norm(np.asarray(fields.alpha)[0] - exact, axis=-1)
    # Leading omitted circular deflection term is R^-6.
    np.testing.assert_allclose(errors[:-1] / errors[1:], 64., rtol=.06)
    far = member_farfield_geometry(np.array([[1e5, 0.]]), np.array([[0., 0.]]))
    np.testing.assert_allclose(evaluate_farfield(far, [2.], [.2], [4.]).alpha[0, 0, 0] * 1e5,
                               2 * 2 * .2 * 4, rtol=3e-5)
    # Values are also accurate in the elliptical exact kernel's true far limit.
    elliptical = np.stack(DPIE_NIE_E.derivatives(positions[:, 0], positions[:, 1], 2., .2, 4., .2, -.1), axis=-1)
    assert np.isfinite(elliptical).all()
    assert np.linalg.norm(elliptical[-1] - np.asarray(fields.alpha)[0, -1]) < .001


def test_geometry_spatial_derivatives_and_parameter_gradients():
    image = np.array([12., -7.])
    centers = np.array([[1., 2.], [-3., -2.]])
    params = jnp.array([2., 3., .2, .3, 3., 4.])
    def independent(point, p):
        delta = point[None] - jnp.asarray(centers)
        radius = jnp.linalg.norm(delta, axis=-1)
        kappa, core, cut = p[:2], p[2:4], p[4:6]
        amplitude = 2 * kappa * core * cut
        factor = (1 - (core + cut) / (2 * radius)
                  + (core + cut) * (core**2 + cut**2) / (8 * radius**3))
        return amplitude[:, None] * factor[:, None] * delta / radius[:, None]**2
    geometry = member_farfield_geometry(image[None], centers)
    fields = jax.jit(evaluate_farfield)(geometry, params[:2], params[2:4], params[4:6])
    np.testing.assert_allclose(fields.alpha[:, 0], independent(image, params), rtol=1e-14)
    np.testing.assert_allclose(fields.jacobian[:, 0], jax.jacfwd(independent)(image, params), rtol=1e-13)
    np.testing.assert_allclose(fields.jacobian_derivative[:, 0], jax.jacfwd(jax.jacfwd(independent))(image, params), rtol=1e-12)
    def actual(p):
        f = evaluate_farfield(geometry, p[:2], p[2:4], p[4:6])
        return jnp.sum(f.alpha**2) + jnp.sum(f.jacobian**2) + jnp.sum(f.jacobian_derivative**2)
    gradient = jax.jit(jax.grad(actual))(params)
    for index in range(len(params)):
        step = np.eye(len(params))[index] * 1e-5
        np.testing.assert_allclose(gradient[index], (actual(params + step) - actual(params - step)) / 2e-5, rtol=2e-8)


@pytest.mark.parametrize('members', [0, 1, 7, 300])
def test_sampling_farfield_matches_original_values_and_gradients(members):
    from lenscluster.member_tiers import evaluate_farfield_sampling
    rng = np.random.default_rng(19)
    geometry = member_farfield_geometry(rng.normal(size=(13,2))*5,
                                       rng.normal(size=(members,2))*100 + 500)
    params = jnp.asarray(np.stack((rng.uniform(.5,4.,members), rng.uniform(.1,.5,members), rng.uniform(2.,8.,members))))
    efficiencies = jnp.linspace(.2,.9,13)
    def original(p):
        f = evaluate_farfield(geometry, p[0,:,None]*efficiencies, p[1], p[2])
        return f.alpha, f.jacobian, f.jacobian_derivative.sum(axis=0)
    def optimized(p):
        return tuple(evaluate_farfield_sampling(geometry, p[0,:,None]*efficiencies, p[1], p[2]))
    for scale in (1., 1.7):
        values = params*scale
        for old, new in zip(jax.jit(original)(values), jax.jit(optimized)(values)):
            np.testing.assert_allclose(new, old, rtol=2e-12, atol=2e-14)
        # Nonuniform contractions exercise every output, including derivatives.
        def loss(fn, p):
            return sum(jnp.sum(v*jnp.arange(v.size).reshape(v.shape)) for v in fn(p))
        old_grad = jax.jit(jax.grad(lambda p: loss(original,p)))(values)
        new_grad = jax.jit(jax.grad(lambda p: loss(optimized,p)))(values)
        np.testing.assert_allclose(new_grad, old_grad, rtol=2e-12, atol=2e-12)


def test_efficiency_and_variable_effective_radii_are_current():
    geometry = member_farfield_geometry(np.array([[14., 2.], [17., -2.]]), np.array([[0., 0.]]))
    efficiency = np.array([.5, .8])
    explicit = evaluate_farfield(geometry, [2.], [.2], [4.], efficiency)
    packed = evaluate_farfield(geometry, 2 * efficiency[None], [.2], [4.])
    for actual, expected in zip(explicit, packed):
        np.testing.assert_allclose(actual, expected)
    def softened(s):
        # Holding physical velocity fixed means kappa0*core stays fixed.
        core = jnp.sqrt(.2**2 + s**2)
        return evaluate_farfield(geometry, jnp.array([.4 / core]), jnp.array([core]), jnp.array([4.])).alpha.sum()
    assert np.isfinite(jax.grad(softened)(.3))
    assert softened(.3) != softened(0.)


def references(amplitudes, errors=None, n_images=1, n_refs=1):
    m = len(amplitudes)
    alpha = np.zeros((n_refs, m, n_images, 2))
    alpha[..., 0] = np.asarray(amplitudes)[None, :, None]
    far = alpha.copy()
    if errors is not None:
        far[..., 0] -= np.asarray(errors)[None, :, None]
    u = np.zeros((n_refs, m, n_images, 2, 2))
    return dict(exact_alpha=alpha, exact_jacobian=u, template_alpha=far, template_jacobian=u.copy(),
                total_jacobian=np.tile(np.eye(2), (n_refs, n_images, 1, 1)),
                determinant_mean=np.ones((n_refs, n_images)), determinant_variance=np.full((n_refs, n_images), .01),
                sigma_src=np.full(n_refs, .02), s_sigma=np.full(n_refs, .02))


def classify(data, **kwargs):
    return classify_member_tiers(**data, member_ids=tuple(f"pop:{i}" for i in range(data["exact_alpha"].shape[1])), **kwargs)


def test_all_four_tiers_and_record_roundtrip():
    data = references([.3, .01, .3, .01], [.003, .003, 0., 0.])
    record, diag = classify(data, component_indices=(10, 11, 12, 13),
                            reference_parameter_names=("x",), reference_parameters=((1.,),))
    assert record.exact_sampled == (0,)
    assert record.exact_absorbed == (1,)
    assert record.approximate_sampled == (2,)
    assert record.approximate_absorbed == (3,)
    assert diag["approximation_validation"] == "passed"
    assert MemberTierRecord.from_dict(json.loads(json.dumps(record.to_dict()))) == record
    record.validate_identity(record.member_ids, record.component_indices)
    with pytest.raises(ValueError, match="identity"):
        record.validate_identity(record.member_ids[::-1], record.component_indices)
    json.dumps(diag)
    with pytest.raises(FrozenInstanceError):
        record.exact_sampled = ()
    with pytest.raises(ValueError, match="partition"):
        MemberTierRecord(("a",), (0,), exact_sampled=(0,), exact_absorbed=(0,))
    with pytest.raises(ValueError, match="version"):
        MemberTierRecord.from_dict({**record.to_dict(), "version": 999})


@pytest.mark.parametrize("sampled,exact", [(None, None), (0, 0), (2, 2), (99, 99), (1, None), (None, 1),
    ('all','all'), ('all',None), (None,'all'), ('all',0), (0,'all')])
def test_top_limits_and_automatic_axes(sampled, exact):
    data = references([.3, .01, .3, .01], [.003, .003, 0., 0.])
    automatic, _ = classify(data)
    record, diag = classify(data, top_sampled=sampled, top_exact=exact)
    assert len(record.sampled) == (len(automatic.sampled) if sampled is None else 4 if sampled == 'all' else min(sampled, 4))
    assert len(record.exact) == (len(automatic.exact) if exact is None else 4 if exact == 'all' else min(exact, 4))
    if sampled is None:
        assert record.sampled == automatic.sampled
    if exact is None:
        assert record.exact == automatic.exact
    if sampled == exact == 2:
        assert record.exact_sampled == (0,)
        assert record.exact_absorbed == (1,)
        assert record.approximate_sampled == (2,)
        assert record.approximate_absorbed == (3,)
        np.testing.assert_allclose(diag['rankings']['sampled']['scores'], np.array([.3,.3,.01,.01])*.04/(np.sqrt(.1)*.02))
        np.testing.assert_allclose(diag['rankings']['exact']['scores'], [1.5,1.5,0,0])
    if sampled == exact == 0:
        assert diag['approximation_validation'] == 'failed'
    json.dumps(diag, allow_nan=False)


def test_top_ties_missing_draws_and_empty_catalog():
    data = references([.1]*3, [.001]*3)
    for ids in [('c','a','b'), ('b','c','a')]:
        record, diag = classify_member_tiers(**data, member_ids=ids, top_sampled=1, top_exact=1,
                                            reference_uncertainty_available=False)
        assert {record.member_ids[i] for i in record.sampled} == {'a'}
        assert {record.member_ids[i] for i in record.exact} == {'a'}
        assert not diag['reference_uncertainty_available']
    record, _ = classify(data, top_sampled=1, reference_uncertainty_available=False)
    assert record.exact == (0,1,2)
    assert record.sampled == (0,)
    empty, _ = classify(references([]), top_sampled=2, top_exact=0)
    assert empty.sampled == empty.exact == ()


def test_top_determinant_and_worst_reference_scores():
    data = references([.01,.001], [0.,0.], n_refs=2, n_images=2)
    data['exact_jacobian'][1,1,1] = np.eye(2)*.5
    record, diag = classify(data, top_sampled=1, top_exact=1)
    assert record.sampled == record.exact == (1,)
    # Independently evaluate finite log h response using the scalar likelihood.
    from scipy.special import erf
    def log_smoothed_determinant(mean, variance):
        sd = np.sqrt(variance)
        return np.log(sd*np.sqrt(2/np.pi)*np.exp(-mean**2/(2*variance)) + mean*erf(mean/(np.sqrt(2)*sd)))
    # The trace response of .5 I against I is 1; scatter response is .04.
    baseline = float(log_smoothed_determinant(1., .01))
    for axis, delta, threshold in [('sampled', .04, np.sqrt(.1)), ('exact', 1., .1)]:
        expected = max(abs(float(log_smoothed_determinant(1.+delta,.01))-baseline),
                       abs(float(log_smoothed_determinant(1.-delta,.01))-baseline))/threshold
        assert diag['rankings'][axis]['scores'][0] == pytest.approx(expected)


def test_top_invalid_templates_and_references():
    data = references([.1,.1], [0.,0.])
    data['template_alpha'][0,1] = np.nan
    with pytest.raises(ValueError, match='cannot accommodate'):
        classify(data, top_exact=0)
    record, diag = classify(data, top_exact=1, top_sampled=0)
    assert record.exact == (1,)
    assert diag['rankings']['exact']['mandatory_exact_member_ids'] == ['pop:1']
    all_members, _ = classify(data, top_exact='all', top_sampled='all', reference_uncertainty_available=False)
    assert all_members.exact_sampled == (0,1)
    empty, _ = classify(references([]), top_exact='all', top_sampled='all')
    assert empty.exact == empty.sampled == ()
    json.dumps(diag, allow_nan=False)
    data['exact_alpha'][0,0] = np.nan
    with pytest.raises(ValueError, match='valid reference'):
        classify(data, top_sampled=1)


def test_required_counts_leave_fixed_top_limits_unchanged():
    from lenscluster.member_tiers import required_top_counts
    data = references([.15]*3, [.001]*3)
    record, _ = classify(data, top_sampled=0, top_exact=0)
    before = record.to_dict()
    report = required_top_counts(record, **data)
    assert report['required_counts'] == dict(sampled=1, exact=2)
    assert report['budgets']['approximation_validation'] == 'passed'
    assert report['required_member_ids'] == dict(sampled=['pop:0'], exact=['pop:0','pop:1'])
    assert record.to_dict() == before
    # Posterior audit scatters must not be capped/floored at selection priors.
    data['sigma_src'][:] = 1.
    data['s_sigma'][:] = .001
    assert required_top_counts(record, **data)['required_counts'] == dict(sampled=0, exact=0)
    data['exact_alpha'][0,0] = np.nan
    assert required_top_counts(record, **data)['status'] == 'unavailable'


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_top_rejects_invalid_limits(value):
    with pytest.raises(ValueError, match='nonnegative integer'):
        classify(references([.1]), top_sampled=value)


def test_collective_template_and_variance_promotions_are_deterministic():
    # Each scatter amplitude .006 is below sqrt(.1)*.02, but three exceed .02²/4 collectively.
    data = references([.15] * 3, [.001] * 3)
    record, diag = classify(data)
    assert record.exact == (0, 1)  # strict sum < .002; stable IDs break ties
    assert record.sampled == (0,)
    assert [row["reason"] for row in diag["promotions"]] == ["template_deflection", "template_deflection", "absorbed_variance"]
    assert classify(data)[0] == record
    assert diag["approximation_validation"] == "passed"


def test_contributor_ties_use_identity_after_catalog_reordering():
    data = references([.15] * 3, [.001] * 3)
    first, _ = classify_member_tiers(**data, member_ids=('c', 'a', 'b'))
    second, _ = classify_member_tiers(**data, member_ids=('b', 'c', 'a'))
    assert {first.member_ids[i] for i in first.exact} == {'a', 'b'}
    assert {second.member_ids[i] for i in second.exact} == {'a', 'b'}
    assert {first.member_ids[i] for i in first.sampled} == {'a'}
    assert {second.member_ids[i] for i in second.sampled} == {'a'}


def test_joint_reference_union_prior_guards_and_actual_audit():
    data = references([.01, .01], [0., 0.], n_refs=2)
    data["exact_alpha"][1, 1, 0, 0] = .3
    data["template_alpha"] = data["exact_alpha"].copy()
    data["sigma_src"][:] = .2
    data["s_sigma"][:] = .001
    record, diag = classify(data)
    assert record.sampled == (1,)
    np.testing.assert_allclose(diag["source_scatter_reference"], .02)
    np.testing.assert_allclose(diag["member_scatter_reference"], .02)
    assert audit_member_tiers(record, **data)["approximation_validation"] == "passed"
    data["sigma_src"][:] = 1e-7
    assert audit_member_tiers(record, **data)["approximation_validation"] == "failed"
    assert record.sampled == (1,)


def test_determinant_finite_change_near_zero_and_parity():
    delta = .1
    two = float(finite_logh_change(0., 1., delta, 0))
    one = float(finite_logh_change(0., 1., delta, 1))
    assert 0 < two < one
    np.testing.assert_allclose(two, delta**2 / 2, rtol=.004)
    data = references([0.], [0.])
    data["determinant_mean"][:] = 0.
    data["determinant_variance"][:] = 1.
    data["exact_jacobian"][..., 0, 0] = .12
    record, diag = classify(data, parity=np.array([1]))
    assert record.exact == (0,)  # finite logh error, although alpha error is zero
    assert diag["promotions"][0]["reason"] == "template_logh"


def test_empty_missing_uncertainty_invalid_geometry_and_failed_audit():
    data = references([])
    record, diag = classify(data)
    assert record.exact == record.sampled == ()
    assert diag["approximation_validation"] == "passed"
    fields = evaluate_farfield(member_farfield_geometry(np.empty((0, 2)), np.empty((0, 2))), [], [], [])
    assert fields.alpha.shape == (0, 0, 2)
    data = references([.01, .01])
    record, diag = classify(data, reference_uncertainty_available=False)
    assert record.exact_sampled == (0, 1)
    assert record.selection_reason == "all_exact_sampled_fallback"
    geometry = member_farfield_geometry(np.array([[0., 0.]]), np.array([[0., 0.]]))
    assert not geometry.valid_separation[0, 0]
    data = references([.01])
    data["template_alpha"][:] = np.nan
    record, diag = classify(data)
    assert record.exact_absorbed == (0,)
    assert diag["invalid_template_members"] == [0]
    data["exact_alpha"][:] = np.nan
    assert audit_member_tiers(record, **data)["approximation_validation"] == "failed"


def test_thresholds_and_input_shape_validation():
    with pytest.raises(ValueError, match="positive"):
        MemberTierThresholds(template_logh=0.)
    data = references([0.])
    data["template_alpha"] = np.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="shapes"):
        classify(data)


def test_audit_catches_individual_scatter_even_below_collective_cap():
    data = references([.01])
    record, _ = classify(data)
    assert record.approximate_absorbed == (0,)
    # .008 shift exceeds .00632 individual tolerance, below .01 collective bound.
    data["exact_alpha"][..., 0] = .2
    data["template_alpha"] = data["exact_alpha"].copy()
    audit = audit_member_tiers(record, **data)
    assert audit["absorbed_variance_max_ratio"] < 1
    assert audit["absorbed_individual_deflection_max_ratio"] > 1
    assert audit["approximation_validation"] == "failed"
