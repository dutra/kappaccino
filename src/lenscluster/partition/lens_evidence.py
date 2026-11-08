"""Lens terms of uncertain images: the per-image position term at any source redshift, and the evidence of a group of
images as the images of one source with the source position integrated out (an unrelated image is a group of one;
``slots`` also uses groups to seed new slots).

Each image i ray-shoots to beta_i(z) = x_i - alpha(x_i; z) with Jacobian A_i(z).  As for the trusted families
(``smoothed_likelihood``), the image-plane density of the observed position given a source beta is, to first order,

    p(x_i | beta, z, Theta) = N(beta | beta_i, C_i) h_i,    C_i = sigma_meas^2 A_i A_i^T + sigma_def^2 I + sigma_sel^2 R_i

with h_i the measurement-smoothed |det A_i| (``log_expected_absolute_normal`` with the catalog parity).  With the
Gaussian source prior N(beta | mu_0, s_0^2 I) the source integral is closed form:

    E_lens(F, z) = prod_i [N(m | beta_i, C_i) h_i(m)] N(m | mu_0, s_0^2 I) 2 pi |P|^-1/2,
    P = I / s_0^2 + sum_i C_i^-1,   m = P^-1 (mu_0 / s_0^2 + sum_i C_i^-1 beta_i)

The determinant factor is taken at the source's posterior mean m (it varies slowly with the source on the scale of
P^-1/2).  The prior width s_0 sets the evidence cost of an extra family, log(2 pi s_0^2) - log(2 pi |P|^-1/2) up to the
fit: it is the source-plane area a family's source could have come from.

Every lens component's deflection is proportional to the dPIE lensing-efficiency factor f(z) except the external
shear, which ``Evaluator.packed_state`` holds fixed across redshifts.  So alpha(x; z) = r(z) alpha_ref(x) +
alpha_shear(x) with r = f(z) / f_ref, and one field evaluation per image (deflection, Jacobian and its spatial
derivative at f_ref) serves the whole redshift grid.  Every member is evaluated with its exact kernel at the uncertain
images (no far-field series).  Under ``gp_image_plane`` an uncertain image gets the diagonal block of the family
covariance and no correlation with other images; with the knee kernel its deflection floor carries the GRF amplitudes
weighted by the lensing efficiency at its source redshift, sigma_def^2 = w^2 sigma_alpha^2 + w_2^2 sigma_alpha2^2 +
sigma_fl^2 (the ``knee_background`` spectrum is not supported).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .. import smoothed_likelihood as sl
from ..model import SHEAR_PROFILE

LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class LensImages:
    """Uncertain images and the fixed quantities of their lens evidence.

    Attributes
    ----------
    xy : jax.numpy.ndarray
        Positions in arcsec offsets from the state's reference (x positive west), shape (image, 2).
    sigma_inst : jax.numpy.ndarray
        Astrometric uncertainty in arcsec, shape (image,).
    parity : jax.numpy.ndarray
        Catalog parity, -1, +1 or 0 when unknown, int32 of shape (image,).
    z_grid : numpy.ndarray
        Source redshifts of the quadrature, shape (grid,).
    ratio : jax.numpy.ndarray
        Lensing-efficiency factor at each grid redshift over ``factor_reference``, shape (grid,).
    factor_reference : float
        The dPIE lensing-efficiency factor the reference fields are evaluated at.
    prior_mean : jax.numpy.ndarray
        Mean of the Gaussian source prior in arcsec, shape (2,).
    prior_sigma : float
        Width of the source prior per axis in arcsec.
    gp_reference_factor : float
        The lensing-efficiency factor at the knee kernel's reference redshift (its GRF weights are relative to it).
    """
    xy: jnp.ndarray
    sigma_inst: jnp.ndarray
    parity: jnp.ndarray
    z_grid: np.ndarray
    ratio: jnp.ndarray
    factor_reference: float
    prior_mean: jnp.ndarray
    prior_sigma: float
    gp_reference_factor: float = 1.0

    @property
    def n_images(self) -> int:
        return int(self.xy.shape[0])


def lens_images(state, xy, sigma_inst, parity, z_grid, *, prior_mean=None, prior_sigma: float | None = None) -> LensImages:
    """``LensImages`` for a model state; the source prior defaults to the trusted families' prior."""
    from ..jax_cosmology import dpie_sigma0_factor_from_config
    z_grid = np.asarray(z_grid, dtype=float)
    if np.any(z_grid <= float(state.z_lens)):
        raise ValueError(f"the redshift grid must lie behind the lens (z_lens = {state.z_lens}).")
    factors = np.asarray([dpie_sigma0_factor_from_config(state.z_lens, float(z), state.cosmo_config) for z in z_grid])
    reference = float(factors[len(factors) // 2])
    gp_reference = dpie_sigma0_factor_from_config(state.z_lens, float(getattr(state, "gp_reference_redshift", 2.0)), state.cosmo_config)
    members = state.members
    return LensImages(xy=jnp.asarray(xy, dtype=jnp.float64).reshape(-1, 2), sigma_inst=jnp.asarray(sigma_inst, dtype=jnp.float64),
                      parity=jnp.asarray(parity, dtype=jnp.int32), z_grid=z_grid, ratio=jnp.asarray(factors / reference),
                      factor_reference=reference,
                      prior_mean=jnp.asarray(members.source_prior_mean if prior_mean is None else prior_mean, dtype=jnp.float64),
                      prior_sigma=float(members.source_prior_sigma if prior_sigma is None else prior_sigma), gp_reference_factor=float(gp_reference))


def reference_fields(evaluator, physical: jnp.ndarray, images: LensImages) -> dict[str, jnp.ndarray]:
    """Deflection, Jacobian U = d alpha / d x and its spatial derivative at the images: the part proportional to the
    lensing efficiency (at ``factor_reference``) and the fixed shear part."""
    packed = evaluator.packed_state(physical, factor_per_image=jnp.asarray(images.factor_reference, dtype=jnp.float64))[0]
    components = np.arange(evaluator.n_components, dtype=np.int32)
    alpha, u, du = evaluator.component_rows(images.xy, packed, components, derivative=True)
    shear = np.asarray(evaluator.state.packed_lens_spec.profile_type)[components] == SHEAR_PROFILE
    scaled, fixed = np.flatnonzero(~shear), np.flatnonzero(shear)
    return dict(alpha=jnp.sum(alpha[scaled], axis=0), u=jnp.sum(u[scaled], axis=0), du=jnp.sum(du[scaled], axis=0),
                alpha_fixed=jnp.sum(alpha[fixed], axis=0), u_fixed=jnp.sum(u[fixed], axis=0))


def grid_fields(reference: dict[str, jnp.ndarray], images: LensImages, ratio: jnp.ndarray | None = None) -> dict[str, jnp.ndarray]:
    """Source positions, Jacobian A = I - U and grad det A of every image at every redshift, shape (redshift, image, ...).

    The redshifts are the grid's by default, or any given by their lensing-efficiency ``ratio`` to ``factor_reference``."""
    r = images.ratio if ratio is None else ratio
    alpha = r[:, None, None] * reference["alpha"][None] + reference["alpha_fixed"][None]
    a = jnp.eye(2) - (r[:, None, None, None] * reference["u"][None] + reference["u_fixed"][None])
    da = -r[:, None, None, None, None] * reference["du"][None]
    g = (a[..., 1, 1, None] * da[..., 0, 0, :] + a[..., 0, 0, None] * da[..., 1, 1, :]
         - a[..., 1, 0, None] * da[..., 0, 1, :] - a[..., 0, 1, None] * da[..., 1, 0, :])
    return dict(beta=images.xy[None] - alpha, jacobian=a, determinant_gradient=g)


def error_scales(evaluator, physical: jnp.ndarray, images: LensImages, ratio: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """The error-model scales of the trusted images applied to the uncertain ones at the redshifts of ``ratio`` (their
    lensing efficiency over ``factor_reference``): sigma_meas per image, sigma_def per (redshift, image), sigma_sel and
    lambda_sel."""
    scales = evaluator.error_scales(physical)
    sigma_meas = jnp.sqrt(jnp.square(images.sigma_inst) + jnp.square(scales["sigma_extra"]))
    shape = (jnp.shape(ratio)[0], images.n_images)
    if getattr(evaluator, "gp_knee", False):
        if evaluator.gp_knee_spectrum == "knee_background":
            raise NotImplementedError("uncertain images do not support the knee_background spectrum")
        w = jnp.asarray(ratio) * images.factor_reference / images.gp_reference_factor
        w2 = jnp.ones_like(w) if evaluator.gp_knee_spectrum == "knee_foreground" else w
        sigma_def = jnp.sqrt(jnp.square(w * scales["gp_sigma_alpha"]) + jnp.square(w2 * scales["gp_sigma_alpha2"]) + evaluator.gp_floor ** 2)
        sigma_def = jnp.broadcast_to(sigma_def[:, None], shape)
    else:
        sigma_def = jnp.broadcast_to(scales["sigma_def"], shape)
    return dict(sigma_meas=sigma_meas, sigma_def=sigma_def, sigma_sel=scales["sigma_sel"], lambda_sel=scales["lambda_sel"])


def image_terms(fields: dict[str, jnp.ndarray], scales: dict[str, jnp.ndarray], images: LensImages) -> dict[str, jnp.ndarray]:
    """The per-image parts of the evidence that do not depend on the family, shape (grid, image, ...).

    As in ``smoothed_likelihood.statistics``: the covariance C, its precision and log det, det A, p = C^-1 c with the
    measurement cross-covariance c = -sigma_meas^2 A grad(det A), and the variance of the smoothed determinant (free of
    the residual r).  Its mean is det A + p . r with r = beta_i - source.
    """
    n_grid, n_image = fields["beta"].shape[:2]
    flat = lambda x: x.reshape((n_grid * n_image,) + x.shape[2:])
    a, g = flat(fields["jacobian"]), flat(fields["determinant_gradient"])
    meas = jnp.broadcast_to(scales["sigma_meas"][None], (n_grid, n_image)).reshape(-1)
    result = sl.source_covariance(a, meas, jnp.broadcast_to(scales["sigma_def"], (n_grid, n_image)).reshape(-1), scales["sigma_sel"],
                                  scales["lambda_sel"])
    covariance = result["covariance"]
    precision = jnp.linalg.inv(covariance)
    cross = -meas[:, None] ** 2 * jnp.einsum("nij,nj->ni", a, g)
    p = jnp.einsum("nij,nj->ni", precision, cross)
    remainder = g + jnp.einsum("nji,nj->ni", a, p)
    variance = meas ** 2 * jnp.sum(remainder ** 2, axis=-1) + jnp.einsum("ni,nij,nj->n", p, covariance - result["measurement"], p)
    determinant = a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]
    unflat = lambda x: x.reshape((n_grid, n_image) + x.shape[1:])
    return dict(beta=fields["beta"], precision=unflat(precision), logdet=unflat(result["logdet"]), determinant=unflat(determinant),
                p=unflat(p), determinant_variance=unflat(variance), valid=unflat(result["valid"] & jnp.isfinite(variance)))


def family_lens_evidence(terms: dict[str, jnp.ndarray], images: LensImages, member_index: jnp.ndarray,
                         member_mask: jnp.ndarray) -> dict[str, Any]:
    """log E_lens(F, z_g) for padded families, shape (family, grid), with the source posterior of each.

    ``terms`` from ``image_terms``; ``member_index`` (family, n_max) holds image indices (any valid index in unused
    slots), ``member_mask`` which slots are used.  Returns ``log_evidence`` (-inf where any term is invalid),
    ``source_mean`` (family, grid, 2) and ``source_covariance`` (family, grid, 2, 2), the Gaussian posterior of the source.
    """
    index, mask = jnp.asarray(member_index), jnp.asarray(member_mask, dtype=bool)

    def take(array):      # (grid, image, ...) -> (family, grid, slot, ...)
        return jnp.moveaxis(array[:, index], 0, 1)

    slot = mask[:, None, :]
    beta, precision = take(terms["beta"]), jnp.where(slot[..., None, None], take(terms["precision"]), 0.0)
    prior_precision = 1.0 / images.prior_sigma ** 2
    total_precision = prior_precision * jnp.eye(2) + jnp.sum(precision, axis=2)
    weighted = prior_precision * images.prior_mean + jnp.einsum("fgsij,fgsj->fgi", precision, beta)
    posterior_covariance = jnp.linalg.inv(total_precision)
    mean = jnp.einsum("fgij,fgj->fgi", posterior_covariance, weighted)
    residual = beta - mean[:, :, None, :]
    quadratic = jnp.einsum("fgsi,fgsij,fgsj->fgs", residual, precision, residual)
    determinant_mean = take(terms["determinant"]) + jnp.einsum("fgsi,fgsi->fgs", take(terms["p"]), residual)
    parity = jnp.broadcast_to(images.parity[index][:, None, :], determinant_mean.shape)
    log_h = sl.log_expected_absolute_normal(determinant_mean, take(terms["determinant_variance"]), parity)
    per_image = -0.5 * (quadratic + take(terms["logdet"]) + 2.0 * LOG_2PI) + log_h
    image_valid = take(terms["valid"]) & jnp.isfinite(per_image)
    prior_residual = mean - images.prior_mean
    log_prior = -0.5 * jnp.sum(prior_residual ** 2, axis=-1) * prior_precision - jnp.log(2.0 * jnp.pi * images.prior_sigma ** 2)
    _, logdet = jnp.linalg.slogdet(total_precision)
    value = jnp.sum(jnp.where(slot & image_valid, per_image, 0.0), axis=-1) + log_prior + LOG_2PI - 0.5 * logdet
    valid = jnp.all(image_valid | ~slot, axis=-1) & jnp.isfinite(value)
    return dict(log_evidence=jnp.where(valid, value, -jnp.inf), source_mean=mean, source_covariance=posterior_covariance)
