"""Per-image Gaussian likelihood with an astrometrically smoothed determinant.

The member arrays have shape ``(member, image, ...)``.  Their Jacobians are
positive deflection derivatives U = d alpha / d theta, not contributions to A.
All source and image coordinates, and positional scatters, use the same units.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc, log_ndtr, logsumexp

MODE = "smoothed-local-image-plane"
_LOG_2PI = float(np.log(2.0 * np.pi))
# The negative normal tail is an exponentially weighted, positive integral.
# Sixteen Gauss--Laguerre nodes give double precision for |mean / sigma| >= 8.
_LAGUERRE_NODES, _LAGUERRE_WEIGHTS = np.polynomial.laguerre.laggauss(16)


def _log_positive_normal_moment(z):
    """log(phi(z) + z Phi(z)), including the cancelling negative tail."""
    z = jnp.asarray(z)
    log_phi = -0.5 * z**2 - 0.5 * _LOG_2PI
    # Safe inputs are necessary even for branches discarded by jnp.where.
    positive_z = jnp.where(z > 0, z, 1.0)
    positive = jnp.logaddexp(log_phi, jnp.log(positive_z) + log_ndtr(z))
    positive = jnp.where(z > 0, positive, log_phi)
    central_z = jnp.where(jnp.abs(z) <= 8.0, z, 0.0)
    central_log_phi = -0.5 * central_z**2 - 0.5 * _LOG_2PI
    central = central_log_phi + jnp.log1p(
        central_z * jnp.exp(log_ndtr(central_z) - central_log_phi)
    )
    tail_a = jnp.maximum(-z, 8.0)
    nodes = jnp.asarray(_LAGUERRE_NODES, dtype=z.dtype)
    weights = jnp.asarray(_LAGUERRE_WEIGHTS, dtype=z.dtype)
    tail = log_phi - 2.0 * jnp.log(tail_a) + logsumexp(
        jnp.log(weights * nodes) - 0.5 * (nodes / tail_a[..., None])**2,
        axis=-1,
    )
    return jnp.where(jnp.abs(z) <= 8.0, central, jnp.where(z > 0, positive, tail))


@jax.custom_jvp
def _log_absolute_normal(mean, variance):
    """Closed-form log E|X| for finite mean and strictly positive variance.

    For |z| <= 8, use h/sd = 2 phi(z) + z erf(z/sqrt(2)). In the
    tail, use log|mean| + log1p(2 phi(|z|)/|z| - erfc(|z|/sqrt(2))).
    This retains tiny log moments even when h rounds to one. Selecting bounded
    inputs *before* division prevents overflow. The JVP below also retains
    small derivatives when the correction to the value rounds away.
    """
    sd = jnp.sqrt(variance)
    central = jnp.abs(mean) <= 8.0 * sd
    bounded = jnp.abs(mean) <= 128.0 * sd
    z = jnp.where(bounded, mean, 0.0) / sd
    scaled_h = jnp.sqrt(2.0 / jnp.pi) * jnp.exp(-0.5*z*z) + z*erf(z/jnp.sqrt(2.0))
    center = 0.5*jnp.log(variance) + jnp.log(scaled_h)
    tail_z = jnp.where(central | ~bounded, 8.0, jnp.abs(z))
    correction = (2.0*jnp.exp(-0.5*tail_z*tail_z - 0.5*_LOG_2PI)/tail_z
                  - erfc(tail_z/jnp.sqrt(2.0)))
    # The correction is nonnegative analytically; underflow/cancellation may
    # round its tiny tail to zero. This is not a floor on determinant variance.
    correction = jnp.where(bounded, jnp.maximum(correction, 0.0), 0.0)
    tail = jnp.log(jnp.where(central, 1.0, jnp.abs(mean))) + jnp.log1p(correction)
    return jnp.where(central, center, tail)


@_log_absolute_normal.defjvp
def _log_absolute_normal_jvp(primals, tangents):
    mean, variance = primals
    mean_dot, variance_dot = tangents
    value = _log_absolute_normal(mean, variance)
    sd = jnp.sqrt(variance)
    # 128 bounds z and z**2 without changing any representable first derivative
    # in float32/64. Do not form the unbounded standardized ratio first.
    bounded = jnp.abs(mean) <= 128.0 * sd
    z = jnp.where(bounded, mean, 0.0) / sd
    signed_probability = jnp.where(bounded, erf(z/jnp.sqrt(2.0)), jnp.sign(mean))
    # Exact Gaussian identities: d log(h)/dm = erf(z/sqrt(2))/h and
    # d log(h)/dv = phi(z)/(sd*h). The latter stays in log space so a tiny
    # phi can be combined with a tiny sd without underflowing first.
    d_mean = signed_probability * jnp.exp(-value)
    log_d_variance = -0.5*z*z - 0.5*_LOG_2PI - jnp.log(sd) - value
    d_variance = jnp.exp(jnp.where(bounded, log_d_variance, -jnp.inf))
    return value, d_mean*mean_dot + d_variance*variance_dot


def log_expected_absolute_normal(mean, variance, parity=None):
    """Log E[|X| 1(parity X > 0)] for X ~ N(mean, variance).

    ``parity=0`` or ``None`` sums both signs.  A zero variance uses the exact
    deterministic limit, including -inf for an impossible branch or X = 0.
    Negative variances and parity values other than 0, -1, +1 return NaN.
    """
    dtype = jnp.result_type(mean, variance, 1.0)
    mean, variance = jnp.broadcast_arrays(jnp.asarray(mean, dtype=dtype), jnp.asarray(variance, dtype=dtype))
    finite = jnp.isfinite(mean) & jnp.isfinite(variance)
    positive_variance = (variance > 0) & finite
    safe_mean = jnp.where(finite, mean, 0.0)
    safe_variance = jnp.where(positive_variance, variance, 1.0)
    if parity is None or (isinstance(parity, (int, np.integer)) and parity == 0):
        # Static dispatch: all-unknown sampling compiles no one-sided tail
        # quadrature. Mixed metadata uses the same closed form per unknown row.
        stochastic = _log_absolute_normal(safe_mean, safe_variance)
        deterministic_h = jnp.abs(safe_mean)
        valid = finite & (variance >= 0)
    else:
        parity = jnp.broadcast_to(jnp.asarray(parity), mean.shape)
        unknown = parity == 0
        two_sided = _log_absolute_normal(jnp.where(unknown, safe_mean, 0.0),
                                        jnp.where(unknown, safe_variance, 1.0))
        # Keep the supplied-parity moment, with safe inputs on discarded rows
        # so a large unknown-parity ratio cannot contaminate its derivatives.
        sd = jnp.sqrt(jnp.where(unknown, 1.0, safe_variance))
        z = jnp.where(unknown, 0.0, safe_mean) / sd
        log_plus = jnp.log(sd) + _log_positive_normal_moment(z)
        log_minus = jnp.log(sd) + _log_positive_normal_moment(-z)
        stochastic = jnp.where(unknown, two_sided, jnp.where(parity > 0, log_plus, log_minus))
        deterministic_h = jnp.where(unknown, jnp.abs(safe_mean), jnp.maximum(parity*safe_mean, 0.0))
        valid = finite & (variance >= 0) & (unknown | (jnp.abs(parity) == 1))
    deterministic = jnp.where(
        deterministic_h > 0, jnp.log(jnp.where(deterministic_h > 0, deterministic_h, 1.0)), -jnp.inf
    )
    return jnp.where(valid, jnp.where(positive_variance, stochastic, deterministic), jnp.nan)


def expected_absolute_normal(mean, variance, parity=None):
    """Expected absolute determinant; see log_expected_absolute_normal."""
    return jnp.exp(log_expected_absolute_normal(mean, variance, parity))


def _image_scale(value, n_images):
    return jnp.broadcast_to(jnp.asarray(value), (n_images,))


def _member_arrays(member_alpha, member_sigma, n_images, dtype, mask):
    if member_alpha is None:
        alpha = jnp.zeros((0, n_images, 2), dtype=dtype)
    else:
        alpha = jnp.asarray(member_alpha)
    if alpha.ndim != 3 or alpha.shape[1:] != (n_images, 2):
        raise ValueError("member_alpha must have shape (member, image, 2)")
    alpha = jnp.where(mask[None, :, None], alpha, 0.0)
    sigma = jnp.broadcast_to(jnp.asarray(member_sigma), (alpha.shape[0],))
    return alpha, sigma


def covariance_statistics(jacobian, sigma_obs, sigma_src, member_alpha=None,
                          member_sigma=0.0, mask=None):
    """Current source covariance and its Cholesky factor, with safe masked rows."""
    jacobian = jnp.asarray(jacobian)
    if jacobian.ndim != 3 or jacobian.shape[1:] != (2, 2):
        raise ValueError("jacobian must have shape (image, 2, 2)")
    n_images = jacobian.shape[0]
    mask = jnp.ones(n_images, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    eye = jnp.eye(2, dtype=jacobian.dtype)
    a = jnp.where(mask[:, None, None], jacobian, eye)
    obs = jnp.where(mask, _image_scale(sigma_obs, n_images), 0.0)
    src = jnp.where(mask, _image_scale(sigma_src, n_images), 1.0)
    alpha, member_sigma = _member_arrays(member_alpha, member_sigma, n_images, a.dtype, mask)
    response = 2.0 * member_sigma[:, None, None] * alpha
    member_covariance = jnp.einsum("gni,gnj->nij", response, response)
    covariance = src[:, None, None]**2 * eye + obs[:, None, None]**2 * (a @ jnp.swapaxes(a, -1, -2)) + member_covariance
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    chol = jnp.linalg.cholesky(covariance)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    valid = (jnp.all(jnp.isfinite(covariance), axis=(-2, -1))
             & jnp.all(jnp.isfinite(chol), axis=(-2, -1))
             & (obs >= 0) & (src >= 0)
             & jnp.all(jnp.isfinite(member_sigma) & (member_sigma >= 0))
             & jnp.all(jnp.isfinite(a), axis=(-2, -1)))
    return dict(jacobian=a, sigma_obs=obs, sigma_src=src, member_alpha=alpha,
                member_sigma=member_sigma, member_covariance=member_covariance,
                covariance=covariance, cholesky=chol, logdet=logdet,
                constrained=mask, valid=jnp.where(mask, valid, True))


def statistics(backprojected, source, jacobian, determinant_gradient, sigma_obs,
               sigma_src, member_alpha=None, member_jacobian=None,
               member_sigma=0.0, mask=None, *, parity=None,
               jacobian_spatial_derivative=None):
    """Compute the normalized local positional likelihood and diagnostics.

    ``member_jacobian`` is U_g (G,N,2,2), and ``determinant_gradient`` is
    grad_theta(det A) (N,2).  ``jacobian_spatial_derivative``, if supplied,
    has shape (N,2,2,2); its axis ordering is immaterial for the Frobenius norm.
    No image-plane inverse, determinant floor, or covariance floor is used.
    """
    result = covariance_statistics(jacobian, sigma_obs, sigma_src, member_alpha, member_sigma, mask)
    mask = result["constrained"]
    a, obs, src = (result[k] for k in ("jacobian", "sigma_obs", "sigma_src"))
    alpha, member_sigma = result["member_alpha"], result["member_sigma"]
    covariance, chol = result["covariance"], result["cholesky"]
    backprojected = jnp.where(mask[:, None], jnp.asarray(backprojected), 0.0)
    source = jnp.where(mask[:, None], jnp.asarray(source), 0.0)
    residual = backprojected - source
    g = jnp.where(mask[:, None], jnp.asarray(determinant_gradient), 0.0)
    if member_jacobian is None:
        if alpha.shape[0]:
            raise ValueError("member_jacobian is required when absorbed member deflections are supplied")
        u = jnp.zeros(alpha.shape[:2] + (2, 2), dtype=a.dtype)
    else:
        if jnp.shape(member_jacobian) != alpha.shape[:2] + (2, 2):
            raise ValueError("member_jacobian must have shape (member, image, 2, 2)")
        u = jnp.where(mask[None, :, None, None], jnp.asarray(member_jacobian), 0.0)
    adj = jnp.stack((jnp.stack((a[:, 1, 1], -a[:, 0, 1]), axis=-1),
                     jnp.stack((-a[:, 1, 0], a[:, 0, 0]), axis=-1)), axis=-2)
    determinant = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    q = jnp.einsum("nij,gnji->gn", adj, u)
    member_variance = (2.0 * member_sigma)**2
    member_cross_covariance = -jnp.einsum("g,gni,gn->ni", member_variance, alpha, q)
    member_determinant_variance = jnp.einsum("g,gn,gn->n", member_variance, q, q)
    cross_covariance = -obs[:, None]**2 * jnp.einsum("nij,nj->ni", a, g) + member_cross_covariance
    precision_residual = jnp.linalg.solve(covariance, residual[..., None])[..., 0]
    precision_cross = jnp.linalg.solve(covariance, cross_covariance[..., None])[..., 0]
    determinant_mean = determinant + jnp.sum(cross_covariance * precision_residual, axis=-1)
    # Residual latent coefficients give the Schur complement without subtraction.
    astrometric_remainder = g + jnp.einsum("nji,nj->ni", a, precision_cross)
    member_remainder = q + jnp.einsum("gni,ni->gn", alpha, precision_cross)
    determinant_variance = (obs**2 * jnp.sum(astrometric_remainder**2, axis=-1)
                            + jnp.einsum("g,gn,gn->n", member_variance, member_remainder, member_remainder)
                            + src**2 * jnp.sum(precision_cross**2, axis=-1))
    log_h = log_expected_absolute_normal(determinant_mean, determinant_variance, parity)
    whitened_residual = jnp.linalg.solve(chol, residual[..., None])[..., 0]
    quadratic = jnp.sum(whitened_residual**2, axis=-1)
    gaussian_loglike = -0.5 * (quadratic + result["logdet"] + 2.0 * _LOG_2PI)
    loglike = jnp.where(mask, gaussian_loglike + log_h, 0.0)
    degenerate = mask & (determinant_variance == 0) & (determinant_mean == 0)
    valid = (result["valid"] & jnp.all(jnp.isfinite(residual), axis=-1)
             & jnp.isfinite(determinant_mean) & jnp.isfinite(determinant_variance)
             & ~jnp.isnan(log_h) & ~degenerate)
    displacement_mean = -obs[:, None]**2 * jnp.einsum("nji,nj->ni", a, precision_residual)
    precision_a = jnp.linalg.solve(covariance, a)
    displacement_covariance = (obs[:, None, None]**2 * jnp.eye(2, dtype=a.dtype)
                               - obs[:, None, None]**4 * (jnp.swapaxes(a, -1, -2) @ precision_a))
    result.update(backprojected=backprojected, source=source, residual=residual,
                  determinant=determinant, determinant_gradient=g, member_q=q,
                  member_cross_covariance=member_cross_covariance,
                  member_determinant_variance=member_determinant_variance,
                  cross_covariance=cross_covariance, determinant_mean=determinant_mean,
                  determinant_variance=determinant_variance, log_h=log_h,
                  h=jnp.exp(log_h), quadratic=quadratic,
                  gaussian_loglike=jnp.where(mask, gaussian_loglike, 0.0),
                  loglike=loglike, degenerate=degenerate,
                  valid=jnp.where(mask, valid, True), displacement_mean=displacement_mean,
                  displacement_covariance=displacement_covariance)
    if jacobian_spatial_derivative is not None:
        da = jnp.where(mask[:, None, None, None], jnp.asarray(jacobian_spatial_derivative), 0.0)
        conditional_second_moment = jnp.sum(displacement_mean**2, axis=-1) + jnp.trace(displacement_covariance, axis1=-2, axis2=-1)
        safe_src = jnp.where(src > 0, src, 1.0)
        da_squared_norm = jnp.sum(da**2, axis=(-3, -2, -1))
        da_norm = jnp.where(da_squared_norm > 0,
                            jnp.sqrt(jnp.where(da_squared_norm > 0, da_squared_norm, 1.0)), 0.0)
        epsilon = da_norm * jnp.maximum(conditional_second_moment, 0.0) / (2.0 * safe_src)
        result["epsilon"] = jnp.where(mask, jnp.where(src > 0, epsilon, jnp.inf), 0.0)
    return result


def source_whitening(backprojected, covariance, family_index, n_families, mask=None):
    """Gaussian positional centroid and scale for a bijective source transport.

    This uses only the Gaussian positional factor.  The physical source prior
    and determinant factor remain in the target evaluated at transported beta.
    ``n_families`` is a static integer when the function is JIT compiled.
    """
    backprojected, covariance = jnp.asarray(backprojected), jnp.asarray(covariance)
    n_images = backprojected.shape[0]
    mask = jnp.ones(n_images, dtype=bool) if mask is None else jnp.asarray(mask, dtype=bool)
    family_index = jnp.asarray(family_index, dtype=int)
    safe_family_index = jnp.where(mask, family_index, 0)
    eye = jnp.eye(2, dtype=covariance.dtype)
    safe_covariance = jnp.where(mask[:, None, None], covariance, eye)
    safe_beta = jnp.where(mask[:, None], backprojected, 0.0)
    precision = jnp.linalg.solve(safe_covariance, jnp.broadcast_to(eye, safe_covariance.shape))
    precision = jnp.where(mask[:, None, None], precision, 0.0)
    family_precision = jnp.zeros((n_families, 2, 2), dtype=covariance.dtype).at[safe_family_index].add(precision)
    weighted_beta = jnp.einsum("nij,nj->ni", precision, safe_beta)
    family_weighted_beta = jnp.zeros((n_families, 2), dtype=backprojected.dtype).at[safe_family_index].add(weighted_beta)
    counts = jnp.zeros(n_families, dtype=int).at[safe_family_index].add(mask.astype(int))
    safe_family_precision = jnp.where((counts > 0)[:, None, None], family_precision, eye)
    source_covariance = jnp.linalg.solve(safe_family_precision, jnp.broadcast_to(eye, safe_family_precision.shape))
    source_covariance = 0.5 * (source_covariance + jnp.swapaxes(source_covariance, -1, -2))
    chol = jnp.linalg.cholesky(source_covariance)
    centroid = jnp.linalg.solve(safe_family_precision, family_weighted_beta[..., None])[..., 0]
    logdet_precision = -2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    valid = (counts > 0) & jnp.all(jnp.isfinite(chol), axis=(-2, -1)) & jnp.all(jnp.isfinite(centroid), axis=-1)
    return dict(centroid=centroid, covariance=source_covariance, cholesky=chol,
                precision=family_precision, logdet_precision=logdet_precision,
                counts=counts, valid=valid)


def transport_sources(offsets, whitening, prior_mean, prior_sigma):
    """Transport standard-normal sites and correct to the physical Gaussian prior.

    Returns the physical sources, per-family correction, and total correction.
    Add ``log_correction`` to a target whose source sites already contribute
    independent standard-normal log densities.  The correction contains the
    physical prior and transport Jacobian and subtracts those base densities.
    """
    offsets = jnp.asarray(offsets)
    source = whitening["centroid"] + jnp.einsum("fij,fj->fi", whitening["cholesky"], offsets)
    sigma = jnp.asarray(prior_sigma)
    if sigma.ndim == 1:
        sigma = sigma[:, None]
    safe_sigma = jnp.where(sigma > 0, sigma, 1.0)
    physical_log_prior = -0.5 * jnp.sum(((source - jnp.asarray(prior_mean)) / safe_sigma)**2
                                      + 2.0 * jnp.log(safe_sigma) + _LOG_2PI, axis=-1)
    base_log_prior = -0.5 * jnp.sum(offsets**2 + _LOG_2PI, axis=-1)
    log_jacobian = -0.5 * whitening["logdet_precision"]
    correction = physical_log_prior + log_jacobian - base_log_prior
    valid_sigma = jnp.all(jnp.isfinite(sigma) & (sigma > 0))
    valid = whitening["valid"] & valid_sigma & jnp.all(jnp.isfinite(source), axis=-1)
    return dict(source=source, log_correction_per_family=correction,
                log_correction=jnp.sum(correction), physical_log_prior=physical_log_prior,
                base_log_prior=base_log_prior, log_jacobian=log_jacobian, valid=valid)


def inverse_source_transport(source, whitening):
    """Return the standard-coordinate offsets corresponding to physical sources."""
    return jnp.linalg.solve(whitening["cholesky"],
                            (jnp.asarray(source) - whitening["centroid"])[..., None])[..., 0]
