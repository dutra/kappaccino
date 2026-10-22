"""Validate a branch-conditioned image-plane likelihood on analytic toy lenses.

This is an isolated research calculation, not a production likelihood. Run with
the lenstronomy interpreter. It does not enumerate global image roots or modify
inference settings. The model is

    q(x | beta) = h(x) K_T(B(x) - beta) / Z(beta), x in fixed D,
    p(y | beta) = integral_D N(y; x, S_obs) q(x | beta) dx.

Here D is fixed independently of the measured position y. On an injective
branch, h=abs(det A) corresponds to pushing a source discrepancy through the
branch and conditioning on branch existence. h=1 is a different image-measure
soft-constraint prior. Normalization alone does not establish physical adequacy
or global catalog calibration, and neither version models absent/extra images,
selection, branch assignment, or correlated displacements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm


jax.config.update("jax_enable_x64", True)


def _normal_logpdf(value, covariance):
    return -0.5 * (
        value @ jnp.linalg.solve(covariance, value)
        + jnp.linalg.slogdet(covariance)[1]
        + value.size * jnp.log(2.0 * jnp.pi)
    )


def _affine_source_logpdf(parameters, y, obs_cov, source_cov):
    a, c, beta = parameters[:4].reshape(2, 2), parameters[4:6], parameters[6:8]
    residual = a @ y + c - beta
    return _normal_logpdf(residual, a @ obs_cov @ a.T + source_cov)


def _affine_image_logpdf(parameters, y, obs_cov, source_cov):
    """Exact whole-plane affine convolution, including its image normalization."""
    a = parameters[:4].reshape(2, 2)
    return (
        jnp.linalg.slogdet(a)[1]
        + _affine_source_logpdf(parameters, y, obs_cov, source_cov)
    )


def _direct_affine_reference(parameters, y, obs_cov, source_cov):
    """Independent image-space Gaussian using NumPy linear algebra."""
    a, c, beta = parameters[:4].reshape(2, 2), parameters[4:6], parameters[6:8]
    mean = np.linalg.solve(a, beta - c)
    inverse = np.linalg.inv(a)
    covariance = obs_cov + inverse @ source_cov @ inverse.T
    delta = y - mean
    return float(-0.5 * (
        delta @ np.linalg.solve(covariance, delta)
        + np.linalg.slogdet(covariance)[1] + 2 * np.log(2 * np.pi)
    ))


def _finite_difference(function, parameters, step=1e-5):
    result = np.empty_like(parameters)
    for k in range(parameters.size):
        offset = np.zeros_like(parameters)
        offset[k] = step
        result[k] = (function(parameters + offset) - function(parameters - offset)) / (2 * step)
    return result


def check_affine():
    # Nonsymmetric A and non-diagonal SPD covariances prevent a diagonal-only
    # calculation from accidentally passing the covariance/gradient checks.
    parameters = np.array([0.25, 0.11, -0.04, 0.08, 0.03, -0.07, 0.08, -0.02])
    y = np.array([0.10, 0.35])
    obs_cov = np.array([[0.09, 0.012], [0.012, 0.04]])
    source_cov = np.array([[0.0004, 0.00008], [0.00008, 0.0009]])
    value = float(_affine_image_logpdf(parameters, y, obs_cov, source_cov))
    reference = _direct_affine_reference(parameters, y, obs_cov, source_cov)
    gradient = np.asarray(jax.grad(_affine_image_logpdf)(parameters, y, obs_cov, source_cov))
    fd = _finite_difference(
        lambda p: _direct_affine_reference(p, y, obs_cov, source_cov), parameters
    )
    np.testing.assert_allclose(value, reference, atol=2e-13, rtol=0)
    np.testing.assert_allclose(gradient, fd, atol=2e-6, rtol=2e-7)

    # A source-scale transformation scales all source-coordinate quantities.
    # Holding T fixed instead represents a genuinely different physical prior.
    factor = 0.5
    transformed = factor * parameters
    zero = np.zeros((2, 2))
    zero_t_change = float(
        _affine_image_logpdf(transformed, y, obs_cov, zero)
        - _affine_image_logpdf(parameters, y, obs_cov, zero)
    )
    legacy_change = float(
        _affine_source_logpdf(transformed, y, obs_cov, zero)
        - _affine_source_logpdf(parameters, y, obs_cov, zero)
    )
    scaled_t_change = float(
        _affine_image_logpdf(transformed, y, obs_cov, factor**2 * source_cov)
        - _affine_image_logpdf(parameters, y, obs_cov, source_cov)
    )
    fixed_t_change = float(
        _affine_image_logpdf(transformed, y, obs_cov, source_cov)
        - _affine_image_logpdf(parameters, y, obs_cov, source_cov)
    )
    np.testing.assert_allclose([zero_t_change, scaled_t_change], 0, atol=2e-13, rtol=0)
    np.testing.assert_allclose(legacy_change, -2 * np.log(factor), atol=2e-13, rtol=0)
    assert abs(fixed_t_change) > 0.01
    return dict(
        logpdf=value,
        direct_gaussian_logpdf=reference,
        logpdf_absolute_error=abs(value - reference),
        gradient_max_absolute_error=float(np.max(np.abs(gradient - fd))),
        source_scale_factor=factor,
        zero_source_covariance_logpdf_change=zero_t_change,
        scaled_source_covariance_logpdf_change=scaled_t_change,
        fixed_source_covariance_logpdf_change=fixed_t_change,
        legacy_source_density_logpdf_change=legacy_change,
    )


def _nonlinear_regular_root(parameters, initial, source_scale=1.0):
    """Unique root of a monotonic cubic; custom_root differentiates implicitly.

    The initial guess does not depend on the observed image position. The solve
    is intentionally a small scalar demonstration, not a global lens solver.
    """
    p, beta = parameters

    def equation(x):
        return source_scale * (x + p * x**3 - beta)

    def solve(function, guess):
        derivative = jax.grad(function)
        return jax.lax.fori_loop(
            0, 32, lambda _, x: x - function(x) / derivative(x), guess
        )

    def tangent_solve(linear_map, value):
        return value / linear_map(jnp.ones_like(value))

    return jax.lax.custom_root(equation, jnp.asarray(initial), solve, tangent_solve)


def check_implicit_regular_root():
    """Regular nonlinear branch with exact Gaussian image observation model."""
    parameters = np.array([0.35, 0.8])
    observed_y, sigma = 0.50, 0.30
    initial_guesses = (-2.0, 2.0)

    def logpdf(p, initial, source_scale=1.0):
        root = _nonlinear_regular_root(p, initial, source_scale)
        return -0.5 * ((observed_y - root) / sigma)**2 - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))

    def reference(p):
        root = brentq(lambda x: x + p[0] * x**3 - p[1], -8.0, 8.0,
                      xtol=1e-14, rtol=1e-14)
        return float(norm.logpdf(observed_y, loc=root, scale=sigma))

    roots = np.array([float(_nonlinear_regular_root(parameters, x)) for x in initial_guesses])
    values = np.array([float(logpdf(parameters, x)) for x in initial_guesses])
    gradients = np.array([np.asarray(jax.grad(logpdf)(parameters, x)) for x in initial_guesses])
    root_gradient = np.asarray(jax.grad(_nonlinear_regular_root)(parameters, initial_guesses[0]))
    reference_value = reference(parameters)
    reference_gradient = _finite_difference(reference, parameters)
    residuals = roots + parameters[0] * roots**3 - parameters[1]
    source_scale = 0.5
    scaled_value = float(logpdf(parameters, initial_guesses[0], source_scale))
    analytic_root_gradient = np.array([-roots[0]**3, 1.0]) / (1 + 3 * parameters[0] * roots[0]**2)
    np.testing.assert_allclose(residuals, 0, atol=2e-14, rtol=0)
    np.testing.assert_allclose(roots, roots[0], atol=2e-14, rtol=0)
    np.testing.assert_allclose(values, reference_value, atol=2e-13, rtol=0)
    np.testing.assert_allclose(gradients, np.tile(reference_gradient, (2, 1)), atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(root_gradient, analytic_root_gradient, atol=2e-13, rtol=0)
    np.testing.assert_allclose(scaled_value, values[0], atol=2e-13, rtol=0)
    return dict(
        map="B(x;p)=x+p*x^3, globally monotonic for p>0",
        parameters=dict(p=float(parameters[0]), beta=float(parameters[1])),
        observed_y=observed_y, observation_sigma=sigma,
        root_solver="32 Newton steps with jax.lax.custom_root implicit derivative",
        initial_guesses=list(initial_guesses), roots=roots.tolist(),
        maximum_root_residual=float(np.max(np.abs(residuals))),
        maximum_initialization_root_difference=float(np.ptp(roots)),
        maximum_initialization_gradient_difference=float(np.max(np.ptp(gradients, axis=0))),
        logpdf=float(values[0]), scipy_brentq_logpdf=reference_value,
        logpdf_absolute_error=float(np.max(np.abs(values - reference_value))),
        gradient_p_beta=gradients[0].tolist(),
        scipy_brentq_finite_difference_gradient=reference_gradient.tolist(),
        gradient_max_absolute_error=float(np.max(np.abs(gradients - reference_gradient))),
        implicit_root_gradient_error=float(np.max(np.abs(root_gradient - analytic_root_gradient))),
        source_scale_factor=source_scale,
        source_scale_logpdf_change=scaled_value - values[0],
        notes=[
            "The observation model is a normalized image-plane Gaussian centered at a deterministic regular root.",
            "Both initial guesses are fixed independently of observed_y; no nearest-observation root switching occurs.",
            "Jointly rescaling B and beta leaves the deterministic image root and its image likelihood unchanged.",
            "This monotonic scalar case validates implicit differentiation only; it does not validate general lens branches, fold crossings, or image completeness.",
            "The implicit derivative requires a nonzero mapping derivative; critical roots require separate treatment.",
        ],
    )


def _nodes_weights(count, lower, upper):
    nodes, weights = leggauss(count)
    return (
        jnp.asarray(lower + (nodes + 1) * (upper - lower) / 2),
        jnp.asarray(weights * (upper - lower) / 2),
    )


def _fold_logpdf(y, beta, nodes, weights, tau, sigma, coarea, map_scale=1.0):
    """Fixed positive branch of B(x)=x**2/2, with fixed quadrature nodes."""
    log_h = jnp.log(map_scale * nodes) if coarea else jnp.zeros_like(nodes)
    log_kernel = -0.5 * ((map_scale * nodes**2 / 2 - beta) / tau)**2 - jnp.log(tau * jnp.sqrt(2 * jnp.pi))
    log_weight = jnp.log(weights) + log_h + log_kernel
    log_obs = -0.5 * ((y - nodes) / sigma)**2 - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
    return jax.scipy.special.logsumexp(log_obs + log_weight) - jax.scipy.special.logsumexp(log_weight)


def _fold_adaptive_reference(y, beta, length, tau, sigma, coarea, map_scale=1.0):
    def weight(x):
        return (map_scale * x if coarea else 1.0) * norm.pdf(map_scale * x**2 / 2, loc=beta, scale=tau)

    denominator = quad(weight, 0, length, epsabs=2e-12, epsrel=2e-12)[0]
    numerator = quad(
        lambda x: norm.pdf(y, loc=x, scale=sigma) * weight(x),
        0, length, epsabs=2e-12, epsrel=2e-12,
    )[0]
    return float(np.log(numerator) - np.log(denominator)), denominator


def check_fold():
    length, tau, sigma = 4.0, 0.08, 0.15
    resolutions = (128, 256, 512)
    nodes_by_resolution = {n: _nodes_weights(n, 0, length) for n in resolutions}
    # A fixed observation range captures all but a bounded <2e-23 Gaussian
    # observation tail for every x in D. It is independent of beta and y.
    ys, yweights = _nodes_weights(512, -10 * sigma, length + 10 * sigma)
    rows = []
    for coarea in (False, True):
        for beta, y in ((1.0, 1.35), (0.05, 0.30), (0.0, 0.12), (-0.05, 0.08)):
            values, gradients = [], []
            for count in resolutions:
                nodes, weights = nodes_by_resolution[count]
                function = lambda b: _fold_logpdf(y, b, nodes, weights, tau, sigma, coarea)
                value, gradient = jax.value_and_grad(function)(jnp.asarray(beta))
                values.append(float(value))
                gradients.append(float(gradient))
            nodes, weights = nodes_by_resolution[resolutions[-1]]
            density = jnp.exp(jax.vmap(
                lambda obs: _fold_logpdf(obs, beta, nodes, weights, tau, sigma, coarea)
            )(ys))
            normalization = float(density @ yweights)
            reference, partition = _fold_adaptive_reference(y, beta, length, tau, sigma, coarea)
            source_scale = 0.5
            scaled_value = float(_fold_logpdf(
                y, source_scale * beta, nodes, weights, source_scale * tau,
                sigma, coarea, map_scale=source_scale,
            ))
            scaled_reference, scaled_partition = _fold_adaptive_reference(
                y, source_scale * beta, length, source_scale * tau,
                sigma, coarea, map_scale=source_scale,
            )
            expected_partition_ratio = 1.0 if coarea else 1.0 / source_scale
            np.testing.assert_allclose(scaled_partition / partition, expected_partition_ratio, atol=2e-12, rtol=0)
            np.testing.assert_allclose([scaled_value, scaled_reference], reference, atol=2e-12, rtol=0)
            step = 1e-5
            fd = (
                _fold_adaptive_reference(y, beta + step, length, tau, sigma, coarea)[0]
                - _fold_adaptive_reference(y, beta - step, length, tau, sigma, coarea)[0]
            ) / (2 * step)
            log_weights = jnp.log(weights) - 0.5 * ((nodes**2 / 2 - beta) / tau)**2
            if coarea:
                log_weights = log_weights + jnp.log(nodes)
                analytic_partition = ndtr((length**2 / 2 - beta) / tau) - ndtr(-beta / tau)
                np.testing.assert_allclose(partition, analytic_partition, atol=2e-12, rtol=0)
            latent_weights = jax.nn.softmax(log_weights)
            mean = float(latent_weights @ nodes)
            variance = float(latent_weights @ (nodes - mean)**2)
            np.testing.assert_allclose(normalization, 1.0, atol=2e-10, rtol=0)
            np.testing.assert_allclose(values, reference, atol=2e-9, rtol=0)
            np.testing.assert_allclose(gradients, fd, atol=2e-6, rtol=2e-7)
            rows.append(dict(
                measure="coarea" if coarea else "image_measure",
                beta=beta, observed_y=y, logpdf=values[-1],
                integrated_observation_density=normalization,
                partition=partition, latent_mean=mean, latent_variance=variance,
                observation_variance=variance + sigma**2,
                logpdf_adaptive_error=abs(values[-1] - reference),
                beta_gradient=gradients[-1], beta_gradient_adaptive_fd=fd,
                gradient_absolute_error=abs(gradients[-1] - fd),
                resolution_logpdf_spread=max(values) - min(values),
                resolution_gradient_spread=max(gradients) - min(gradients),
                source_scale_factor=source_scale,
                scaled_source_model_logpdf_change=scaled_value - values[-1],
                scaled_source_model_partition_ratio=scaled_partition / partition,
            ))
    return dict(
        branch_bounds=[0, length], source_discrepancy_sigma=tau,
        observation_sigma=sigma, quadrature_resolutions=list(resolutions),
        observation_normalization_bounds=[float(ys.min()), float(ys.max())],
        notes=[
            "Domain and quadrature nodes are independent of measured position.",
            "Both measures normalize but encode different latent-position priors.",
            "Positive T permits a conditional distribution at beta<=0; it does not assert existence of a deterministic fold root there.",
            "At T=0 and beta<0 the positive-branch event has probability zero, so its conditional likelihood is undefined.",
            "The coarea partition is the probability that a Gaussian-displaced source lies in this branch's source footprint.",
            "Scaling B, beta, and tau together preserves both nonlinear observation densities on the same fixed image domain.",
        ], rows=rows,
    )


def check_observed_centered_patch():
    """Counterexample: re-centering D on each tested observation breaks p(y)."""
    half_width, tau, sigma = 0.2, 0.15, 0.3
    offsets, weights = _nodes_weights(256, -half_width, half_width)

    def density(y):
        # Identity map and beta=0. D(y)=[y-a,y+a] varies inside the supposed
        # density. The source normal prefactor cancels in this local ratio.
        log_weights = jnp.log(weights) - 0.5 * ((y + offsets) / tau)**2
        log_obs = -0.5 * (offsets / sigma)**2 - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
        return jnp.exp(jax.scipy.special.logsumexp(log_weights + log_obs)
                       - jax.scipy.special.logsumexp(log_weights))

    points = np.array([0.0, 1.0, 5.0, 30.0, 100.0])
    values = np.asarray(jax.vmap(density)(jnp.asarray(points)))
    ys, yweights = _nodes_weights(512, -5.0, 5.0)
    integral = float(jax.vmap(density)(ys) @ yweights)
    tail_limit = float(norm.pdf(half_width, scale=sigma))
    assert integral > 10
    np.testing.assert_allclose(values[-1], tail_limit, rtol=1e-3)
    return dict(
        map="identity", source_beta=0, source_sigma=tau,
        observation_sigma=sigma, observed_centered_patch_half_width=half_width,
        observed_positions=points.tolist(), apparent_density=values.tolist(),
        integral_over_minus5_plus5=integral,
        analytic_nonzero_tail_limit=tail_limit,
        note="An observation-dependent domain creates a nondecaying pseudo-density; a proper selection model or independent fixed domain is required.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON report; omitted by default.")
    args = parser.parse_args()
    report = dict(
        purpose="Isolated numerical validation of normalized branch-conditioned image-plane observation models",
        status="toy_validation_only",
        jax_version=jax.__version__, dtype="float64",
        assumptions=[
            "The image branch/domain is conditioned on and fixed independently of the measured image position.",
            "The latent true image position has an explicitly normalized distribution on that domain.",
            "Observations are additive Gaussian image-plane errors with known covariance.",
            "h=1 and h=abs(det A) are distinct generative choices for nonlinear maps.",
            "No claim of global image-catalog normalization, selection calibration, lens recovery, or production readiness is made.",
        ],
        affine=check_affine(), implicit_regular_root=check_implicit_regular_root(),
        fold=check_fold(),
        observed_centered_patch_counterexample=check_observed_centered_patch(),
    )
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
