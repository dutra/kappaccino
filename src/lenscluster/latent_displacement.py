"""Local source-residual Gaussian with analytically marginalized displacements.

This is not a normalized image-catalog likelihood. Family sources are explicit;
image displacements are integrated out. Member errors are independent between
images in this approximation, even when a physical member affects many images.
"""
import math


MODE = "isotropic-latent-displacement"


def validate_settings(sigma_beta, engine, source_parameterization):
    if sigma_beta is None or not math.isfinite(float(sigma_beta)) or float(sigma_beta) <= 0:
        raise ValueError("latent_displacement_sigma_beta_arcsec must be explicitly set, finite and positive.")
    if engine != "full_flat":
        raise ValueError("isotropic-latent-displacement requires full_flat.")
    if source_parameterization not in {"direct", "prior-whitened"}:
        raise ValueError("isotropic-latent-displacement supports direct or prior-whitened sources only.")


def statistics(backprojected, source, jacobian, sigma_obs, sigma_beta, member_covariance, mask):
    """Batched Gaussian algebra; mask unsafe rows before factorization.

    Positions have shape (N, 2), matrices (N, 2, 2), scales and mask (N,).
    Validation of active inputs belongs outside the compiled computation.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    residual = jnp.where(mask[:, None], backprojected - source, 0.0)
    jacobian = jnp.where(mask[:, None, None], jacobian, 0.0)
    member_covariance = jnp.where(mask[:, None, None], member_covariance, 0.0)
    sigma_obs = jnp.where(mask, sigma_obs, 1.0)
    covariance = (sigma_beta**2 * jnp.eye(2)
                  + sigma_obs[:, None, None]**2 * (jacobian @ jnp.swapaxes(jacobian, -1, -2))
                  + member_covariance)
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -1, -2))
    chol = jnp.linalg.cholesky(covariance)
    whitened = jax.vmap(lambda l, r: solve_triangular(l, r, lower=True))(chol, residual)
    quadratic = jnp.sum(whitened**2, axis=-1)
    logdet = 2 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
    loglike = jnp.where(mask, -0.5 * (quadratic + logdet + 2 * jnp.log(2 * jnp.pi)), 0.0)
    return dict(backprojected=backprojected, source=source, residual=residual,
                jacobian=jacobian, member_covariance=member_covariance, covariance=covariance,
                quadratic=quadratic, logdet=logdet, loglike=loglike, constrained=mask)
