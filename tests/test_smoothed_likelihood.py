"""Independent references for the smoothed determinant and source transport."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr
from scipy.stats import multivariate_normal, norm

from lenscluster.smoothed_likelihood import (
    covariance_statistics, expected_absolute_normal, inverse_source_transport,
    log_expected_absolute_normal, source_whitening, statistics, transport_sources,
)


def fixture_arrays():
    rng = np.random.default_rng(92)
    n, g = 4, 3
    a = rng.normal(size=(n, 2, 2))
    a = (a + a.swapaxes(-1, -2)) / 2
    return dict(backprojected=rng.normal(size=(n, 2)) * .07,
                source=rng.normal(size=(n, 2)) * .03,
                jacobian=a, determinant_gradient=rng.normal(size=(n, 2)),
                sigma_obs=np.array([.03, .05, .1, .07]), sigma_src=.04,
                member_alpha=rng.normal(size=(g, n, 2)) * .12,
                member_jacobian=rng.normal(size=(g, n, 2, 2)) * .07,
                member_sigma=np.array([.05, .1, .13]), mask=np.ones(n, dtype=bool))


def test_conditioning_matches_independent_latent_gaussian():
    args = fixture_arrays()
    out = statistics(**args)
    for i in range(4):
        a, grad = args['jacobian'][i], args['determinant_gradient'][i]
        alpha, u = args['member_alpha'][:, i], args['member_jacobian'][:, i]
        adj = np.array([[a[1, 1], -a[0, 1]], [-a[1, 0], a[0, 0]]])
        q = np.array([np.trace(adj @ row) for row in u])
        so, ss = args['sigma_obs'][i], args['sigma_src']
        sm = 2 * args['member_sigma']
        b = np.concatenate((-so*a, (sm[:, None]*alpha).T, ss*np.eye(2)), axis=1)
        w = np.concatenate((so*grad, -sm*q, np.zeros(2)))
        c = b @ b.T
        cross = b @ w
        f = args['backprojected'][i] - args['source'][i]
        mean = np.linalg.det(a) + cross @ np.linalg.solve(c, f)
        var = w @ w - cross @ np.linalg.solve(c, cross)
        np.testing.assert_allclose(out['covariance'][i], c, rtol=2e-14, atol=1e-17)
        np.testing.assert_allclose(out['cross_covariance'][i], cross, rtol=2e-14, atol=1e-17)
        np.testing.assert_allclose(out['determinant_mean'][i], mean, rtol=2e-14)
        np.testing.assert_allclose(out['determinant_variance'][i], var, rtol=2e-13, atol=1e-17)
        expected_h = np.sqrt(var)*np.sqrt(2/np.pi)*np.exp(-mean**2/(2*var)) + mean*(2*ndtr(mean/np.sqrt(var))-1)
        expected_log = multivariate_normal.logpdf(f, mean=np.zeros(2), cov=c) + np.log(expected_h)
        np.testing.assert_allclose(out['loglike'][i], expected_log, rtol=2e-13, atol=1e-12)
    assert np.all(out['valid'])


@pytest.mark.parametrize('z', [-1000., -30., -8.01, -8., -7.99, -3., -.1, 0., .1, 3., 8., 30.])
def test_branch_moments_against_positive_integral(z):
    # A positive-integrand reference avoids subtracting two small Gaussian tails.
    if z < -5:
        a = -z
        integral = quad(lambda y: y*np.exp(-y-y*y/(2*a*a)), 0, np.inf,
                        epsabs=2e-13, epsrel=2e-13)[0]
        reference = norm.logpdf(z)-2*np.log(a)+np.log(integral)
    else:
        reference = np.log(quad(lambda y: y*norm.pdf(y-z), 0, max(z, 0)+40,
                                epsabs=2e-13, epsrel=2e-13)[0])
    actual = log_expected_absolute_normal(z, 1., 1)
    np.testing.assert_allclose(actual, reference, rtol=2e-14, atol=4e-12)
    np.testing.assert_allclose(log_expected_absolute_normal(-z, 1., -1), actual, atol=1e-14)
    plus = expected_absolute_normal(z, 1., 1)
    minus = expected_absolute_normal(z, 1., -1)
    np.testing.assert_allclose(expected_absolute_normal(z, 1.), plus+minus, rtol=2e-14)


def test_deterministic_limits_degeneracy_and_safe_gradients():
    for mean in [-2., 2.]:
        np.testing.assert_allclose(log_expected_absolute_normal(mean, 0.), np.log(2.))
        np.testing.assert_allclose(log_expected_absolute_normal(mean, 0., np.sign(mean)), np.log(2.))
        assert np.isneginf(log_expected_absolute_normal(mean, 0., -np.sign(mean)))
        value, grad = jax.value_and_grad(lambda x: log_expected_absolute_normal(x, 0.))(mean)
        assert np.isfinite(value) and np.isfinite(grad)
        np.testing.assert_allclose(grad, 1/mean)
    assert np.isneginf(log_expected_absolute_normal(0., 0.))
    assert np.isnan(log_expected_absolute_normal(1., -1.))
    assert np.isnan(log_expected_absolute_normal(1., 1., 2))
    out = statistics(jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.zeros((1, 2, 2)),
                     jnp.zeros((1, 2)), jnp.ones(1)*.1, .03)
    assert out['degenerate'][0] and not out['valid'][0]
    assert np.isneginf(out['loglike'][0])


@pytest.mark.parametrize('parity', [0, -1, 1])
def test_branch_log_gradient_in_tails_and_near_switch(parity):
    f = jax.jit(jax.value_and_grad(lambda z: log_expected_absolute_normal(z, .7**2, parity)))
    for z in [-40., -5.6001, -5.6, -5.5999, -.01, 0., .01, 4., 40.]:
        val, grad = f(z)
        assert np.isfinite(val) and np.isfinite(grad)
        step = 1e-5
        numeric = (log_expected_absolute_normal(z+step, .7**2, parity)
                   - log_expected_absolute_normal(z-step, .7**2, parity))/(2*step)
        np.testing.assert_allclose(grad, numeric, rtol=2e-6, atol=2e-7)


def test_fold_crossing_matches_displacement_quadrature():
    sigma_obs, sigma_src = .025, .02
    # The prescribed first-order map F + A u and D + g.u can be integrated
    # independently by conditioning each Cartesian displacement in the integral.
    for position in [-.03, 0., .03]:
        a = np.diag([1.2, position])
        grad = np.array([0., .8])
        f = np.array([.008, -.006])
        result = statistics(f[None], np.zeros((1, 2)), a[None], grad[None],
                            np.array([sigma_obs]), sigma_src)
        xfactor = norm.pdf(f[0], scale=np.sqrt(sigma_src**2+(1.2*sigma_obs)**2))
        integral = quad(lambda u: abs(np.linalg.det(a)+grad[1]*u)*norm.pdf(u, scale=sigma_obs)
                        *norm.pdf(f[1]+a[1, 1]*u, scale=sigma_src),
                        -.4, .4, points=[-np.linalg.det(a)/grad[1]], epsabs=1e-12)[0]
        np.testing.assert_allclose(np.exp(result['loglike'][0]), xfactor*integral, rtol=2e-10)


def test_current_members_and_full_likelihood_gradients():
    base = fixture_arrays()
    def objective(theta):
        args = dict(base)
        args['source'] = jnp.asarray(base['source']) + theta[0]
        args['jacobian'] = jnp.asarray(base['jacobian']) * jnp.exp(theta[1])
        args['member_alpha'] = jnp.asarray(base['member_alpha']) * jnp.exp(theta[2])
        args['member_jacobian'] = jnp.asarray(base['member_jacobian']) * jnp.exp(theta[2])
        args['member_sigma'] = jnp.asarray(base['member_sigma']) * jnp.exp(theta[3])
        args['sigma_src'] = .04*jnp.exp(theta[4])
        return statistics(**args)['loglike'].sum()
    theta = jnp.array([.004, .03, -.05, .1, .08])
    val, grad = jax.jit(jax.value_and_grad(objective))(theta)
    assert np.isfinite(val) and np.isfinite(grad).all()
    for i in range(5):
        step = np.eye(5)[i]*1e-5
        numerical = (objective(theta+step)-objective(theta-step))/(2e-5)
        np.testing.assert_allclose(grad[i], numerical, rtol=3e-6, atol=1e-7)
    assert np.isfinite(jax.jit(jax.hessian(objective))(theta)).all()


def test_masked_nan_rows_do_not_change_value_or_gradient():
    base = fixture_arrays()
    base['mask'] = np.array([True, True, False, True])
    clean = statistics(**base)
    for key in ['backprojected', 'source', 'jacobian', 'determinant_gradient', 'sigma_obs']:
        base[key][2] = np.nan
    base['member_alpha'][:, 2] = np.nan
    base['member_jacobian'][:, 2] = np.nan
    contaminated = statistics(**base)
    np.testing.assert_allclose(contaminated['loglike'], clean['loglike'], rtol=0, atol=0)
    assert contaminated['loglike'][2] == 0 and contaminated['valid'][2]
    fn = lambda shift: statistics(**dict(base, source=jnp.asarray(base['source'])+shift))['loglike'].sum()
    assert np.isfinite(jax.jit(jax.value_and_grad(fn))(.01)).all()


def test_source_transport_roundtrip_and_physical_target_identity():
    args = fixture_arrays()
    cov = covariance_statistics(args['jacobian'], args['sigma_obs'], args['sigma_src'],
                                args['member_alpha'], args['member_sigma'])['covariance']
    index = jnp.array([0, 1, 0, 1])
    whitening = source_whitening(args['backprojected'], cov, index, 2)
    offsets = jnp.array([[.3, -.7], [.1, 1.2]])
    prior_mean = jnp.array([[1., -.3], [-.4, .8]])
    prior_sigma = jnp.array([2., 3.])
    transformed = transport_sources(offsets, whitening, prior_mean, prior_sigma)
    np.testing.assert_allclose(inverse_source_transport(transformed['source'], whitening), offsets, rtol=2e-14, atol=2e-14)
    for f in range(2):
        rows = np.asarray(index) == f
        precision = np.linalg.inv(np.asarray(cov)[rows]).sum(0)
        expected_cov = np.linalg.inv(precision)
        expected_centroid = expected_cov @ np.einsum('nij,nj->i', np.linalg.inv(np.asarray(cov)[rows]), args['backprojected'][rows])
        np.testing.assert_allclose(whitening['centroid'][f], expected_centroid, rtol=2e-14)
        np.testing.assert_allclose(whitening['covariance'][f], expected_cov, rtol=2e-14)
        physical_prior = multivariate_normal.logpdf(transformed['source'][f], mean=prior_mean[f], cov=np.eye(2)*prior_sigma[f]**2)
        base_prior = multivariate_normal.logpdf(offsets[f], mean=np.zeros(2), cov=np.eye(2))
        jac = np.linalg.slogdet(np.asarray(whitening['cholesky'][f]))[1]
        np.testing.assert_allclose(base_prior+transformed['log_correction_per_family'][f], physical_prior+jac, rtol=2e-14)
    jacobian = jax.jacfwd(lambda u: transport_sources(u.reshape(2,2), whitening, prior_mean, prior_sigma)['source'].ravel())(offsets.ravel())
    np.testing.assert_allclose(np.linalg.slogdet(jacobian)[1], transformed['log_jacobian'].sum(), rtol=2e-14)


def test_source_transport_lens_gradient_and_empty_family_flag():
    base = fixture_arrays()
    family = jnp.array([0, 1, 0, 1])
    def target(t):
        beta = jnp.asarray(base['backprojected']) + t*.2
        cov = covariance_statistics(jnp.asarray(base['jacobian'])*jnp.exp(t), base['sigma_obs'], .04)['covariance']
        w = source_whitening(beta, cov, family, 2)
        return transport_sources(jnp.ones((2, 2))*.4, w, jnp.zeros((2, 2)), 2.)['log_correction']
    value, grad = jax.jit(jax.value_and_grad(target))(.03)
    assert np.isfinite([value, grad]).all()
    np.testing.assert_allclose(grad, (target(.03001)-target(.02999))/.00002, rtol=1e-7)
    w = source_whitening(base['backprojected'], jnp.tile(jnp.eye(2), (4,1,1)), family, 3)
    np.testing.assert_array_equal(w['valid'], [True, True, False])


def test_linearization_diagnostic_matches_conditional_astrometry():
    args = fixture_arrays()
    da = np.arange(32).reshape(4,2,2,2)*.01
    out = statistics(**args, jacobian_spatial_derivative=da)
    for i in range(4):
        a = args['jacobian'][i]
        c = np.asarray(out['covariance'][i])
        f = args['backprojected'][i]-args['source'][i]
        so = args['sigma_obs'][i]
        mean = -so**2*a.T@np.linalg.solve(c, f)
        cov = so**2*np.eye(2)-so**4*a.T@np.linalg.solve(c, a)
        epsilon = np.linalg.norm(da[i])*(mean@mean+np.trace(cov))/(2*args['sigma_src'])
        np.testing.assert_allclose(out['epsilon'][i], epsilon, rtol=2e-14)


def test_absorbed_derivatives_required_and_negative_scatter_flagged():
    args = fixture_arrays()
    with pytest.raises(ValueError, match='required'):
        statistics(**dict(args, member_jacobian=None))
    with pytest.raises(ValueError, match='member_jacobian must have shape'):
        statistics(**dict(args, member_jacobian=args['member_jacobian'][0]))
    out = statistics(**dict(args, member_sigma=-.1))
    assert not np.any(out['valid'])


@pytest.mark.parametrize('parity', [0, -1, 1])
def test_zero_mean_analytic_gradient_and_curvature(parity):
    variance=.09
    fn=lambda x: log_expected_absolute_normal(x[0],x[1],parity)
    value,gradient=jax.jit(jax.value_and_grad(fn))(jnp.array([0.,variance]))
    hessian=jax.jit(jax.hessian(fn))(jnp.array([0.,variance]))
    expected_log=.5*np.log(variance)+.5*np.log(2/np.pi)-(np.log(2) if parity else 0.)
    expected_mean_derivative=parity*np.sqrt(np.pi/(2*variance))
    expected_mean_curvature=(1.-np.pi/2)/variance if parity else 1./variance
    np.testing.assert_allclose(value,expected_log,rtol=1e-14)
    np.testing.assert_allclose(gradient,[expected_mean_derivative,.5/variance],rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(hessian[0,0],expected_mean_curvature,rtol=1e-14)
    assert np.isfinite(hessian).all()
