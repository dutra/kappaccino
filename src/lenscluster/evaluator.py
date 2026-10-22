"""Lens fields and the smoothed-local-image-plane log-likelihood for one model state."""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import jax_lensing, smoothed_likelihood as sl
from .farfield import evaluate_farfield, member_farfield_geometry, slice_geometry
from .model import (COMPONENT_FAMILY_SCALING, DP_IE_PROFILE, SAFE_RADIUS_MARGIN_ARCSEC, SAFE_RADIUS_MARGIN_KPC,
                    SHEAR_PROFILE, ModelState, PackedLensState, PackedLensValidity, apply_parameter_transforms_jax)

BAD_LOG_LIKE = -1.0e30


class Evaluator:
    """Evaluates deflections, Jacobians and the log-likelihood for a fixed member classification."""

    def __init__(self, state: ModelState) -> None:
        self.state = state
        specs = state.parameter_specs
        kinds = np.asarray([spec.transform_kind for spec in specs], dtype=object)
        self.log_positive_mask = jnp.asarray(kinds == "log_positive")
        self.log_offset_positive_mask = jnp.asarray(kinds == "log_offset_positive")
        self.affine_mask = jnp.asarray(kinds == "affine")
        self.transform_offset = jnp.asarray([float(spec.transform_offset) for spec in specs], dtype=jnp.float64)
        self.transform_scale = jnp.asarray([float(spec.transform_scale) for spec in specs], dtype=jnp.float64)
        packed = state.packed_lens_spec
        self.spec = {name: jnp.asarray(getattr(packed, name)) for name in vars(packed)}
        self.n_components = int(np.asarray(packed.profile_type).size)
        self.factor_by_z = jnp.asarray([state.dpie_sigma0_factor_by_z[z] for z in state.z_source_values], dtype=jnp.float64)
        images = state.images
        self.xy = jnp.stack((images.x_obs, images.y_obs), -1)
        members = state.members
        self.member_components = np.asarray(members.component_indices, dtype=np.int32)
        self.exact_positions = np.asarray(members.exact_positions, dtype=np.int32)
        self.far_positions = np.asarray(members.far_positions, dtype=np.int32)
        far_components = set(self.member_components[self.far_positions].tolist())
        self.exact_components = np.asarray([i for i in range(self.n_components) if i not in far_components], dtype=np.int32)
        self.member_xy = np.stack((np.asarray(packed.x_center_base)[self.member_components],
                                   np.asarray(packed.y_center_base)[self.member_components]), -1)
        self.geometry = member_farfield_geometry(np.asarray(self.xy), self.member_xy)
        self.far_geometry = slice_geometry(self.geometry, self.far_positions)
        self.far_components = self.member_components[self.far_positions]
        self.far_min_separation = jnp.asarray(np.asarray(self.geometry.separation)[self.far_positions].min(axis=1)) if len(self.far_positions) else jnp.zeros(0)
        self.guard_cut_radii = float(members.far_field_guard_cut_radii)
        self.scale_indices = jnp.asarray(sorted({int(members.sigma_src_index)} | {int(i) for i in np.asarray(members.scatter_param_index) if i >= 0}), dtype=jnp.int32)
        self.loglike = jax.jit(self.log_likelihood)
        self.value_and_grad = jax.jit(jax.value_and_grad(self.log_likelihood))

    # ------------------------------------------------------------------ parameters
    def physical(self, params: jnp.ndarray) -> jnp.ndarray:
        return apply_parameter_transforms_jax(params, self.log_positive_mask, self.log_offset_positive_mask,
                                              self.transform_offset, self.affine_mask, self.transform_scale)

    def _apply(self, base: jnp.ndarray, index: jnp.ndarray, physical: jnp.ndarray) -> jnp.ndarray:
        safe = jnp.where(index >= 0, index, 0)
        return jnp.where(index >= 0, jnp.take(physical, safe), base)

    # ------------------------------------------------------------------ packed lens state
    def packed_state(self, physical: jnp.ndarray, factor_per_image: jnp.ndarray | None = None) -> tuple[PackedLensState, PackedLensValidity]:
        """Current dPIE and shear parameters.  sigma0 has shape (component, image) unless a scalar factor is given."""
        spec = self.spec
        x_center = self._apply(spec["x_center_base"], spec["x_center_param_index"], physical)
        y_center = self._apply(spec["y_center_base"], spec["y_center_param_index"], physical)
        e1 = self._apply(spec["e1_base"], spec["e1_param_index"], physical)
        e2 = self._apply(spec["e2_base"], spec["e2_param_index"], physical)
        core_radius_kpc = self._apply(spec["core_radius_kpc_base"], spec["core_radius_param_index"], physical)
        cut_radius_kpc = self._apply(spec["cut_radius_kpc_base"], spec["cut_radius_param_index"], physical)
        v_disp = self._apply(spec["v_disp_base"], spec["v_disp_param_index"], physical)
        gamma1 = self._apply(spec["gamma1_base"], spec["gamma1_param_index"], physical)
        gamma2 = self._apply(spec["gamma2_base"], spec["gamma2_param_index"], physical)
        profile_type = spec["profile_type"]
        is_dpie = profile_type == DP_IE_PROFILE
        is_shear = profile_type == SHEAR_PROFILE
        is_scaling = spec["component_family"] == COMPONENT_FAMILY_SCALING
        luminosity_ratio = spec["luminosity_ratio"]
        sigma_ref = self._apply(spec["sigma_ref_base"], spec["sigma_ref_param_index"], physical)
        cut_ref = self._apply(spec["cut_ref_base"], spec["cut_ref_param_index"], physical)
        core_ref = self._apply(spec["core_ref_base"], spec["core_ref_param_index"], physical)
        alpha_sigma = self._apply(spec["alpha_sigma_base"], spec["alpha_sigma_param_index"], physical)
        beta_radius = self._apply(spec["beta_radius_base"], spec["beta_radius_param_index"], physical)
        scaled_vdisp = sigma_ref * jnp.power(luminosity_ratio, alpha_sigma)
        size_scale = jnp.power(luminosity_ratio, beta_radius)
        v_disp = jnp.where(is_scaling, scaled_vdisp, v_disp)
        # Sampled member deviation: non-centered offset times the population scatter.
        offset_index, scatter_index = spec["velocity_offset_param_index"], spec["velocity_scatter_param_index"]
        delta = jnp.where(offset_index >= 0, physical[jnp.maximum(offset_index, 0)] * physical[jnp.maximum(scatter_index, 0)], 0.0)
        v_disp = v_disp * jnp.exp(delta)
        core_radius_kpc = jnp.where(is_scaling, core_ref * size_scale, core_radius_kpc)
        cut_radius_kpc = jnp.where(is_scaling, cut_ref * size_scale, cut_radius_kpc)
        kpc_per_arcsec = jnp.asarray(self.state.kpc_per_arcsec, dtype=jnp.float64)
        softening_index = int(self.state.log_softening_length_param_index)
        if softening_index >= 0:
            softening_kpc = jnp.exp(physical[softening_index])
            core_effective_kpc = jnp.sqrt(jnp.square(core_radius_kpc) + jnp.square(softening_kpc))
        else:
            core_effective_kpc = core_radius_kpc
        cut_radius_kpc = jnp.where(spec["cut_radius_excess_above_effective_core"],
                                   core_effective_kpc + jnp.maximum(cut_radius_kpc, SAFE_RADIUS_MARGIN_KPC), cut_radius_kpc)
        ra_raw = core_effective_kpc / kpc_per_arcsec
        rs_raw = cut_radius_kpc / kpc_per_arcsec
        ra = jnp.maximum(ra_raw, SAFE_RADIUS_MARGIN_ARCSEC)
        rs = jnp.maximum(rs_raw, ra + SAFE_RADIUS_MARGIN_ARCSEC)
        if factor_per_image is None:
            factor_per_image = jnp.take(self.factor_by_z, self.state.images.z_index_per_image)
            sigma0 = (v_disp[:, None] ** 2) * factor_per_image[None, :] / ra[:, None]
            sigma0 = jnp.where(is_dpie[:, None], sigma0, 0.0)
        else:
            sigma0 = jnp.where(is_dpie, (v_disp ** 2) * jnp.asarray(factor_per_image, dtype=jnp.float64) / ra, 0.0)
        packed = PackedLensState(
            profile_type=profile_type, sigma0=sigma0, Ra=jnp.where(is_dpie, ra, 0.0), Rs=jnp.where(is_dpie, rs, 0.0),
            e1=jnp.where(is_dpie, e1, 0.0), e2=jnp.where(is_dpie, e2, 0.0), center_x=x_center, center_y=y_center,
            gamma1=jnp.where(is_shear, gamma1, 0.0), gamma2=jnp.where(is_shear, gamma2, 0.0))
        flags = jnp.stack([
            jnp.any(~jnp.isfinite(sigma0)),
            jnp.any(is_dpie & (ra_raw <= 0.0)),
            jnp.any(is_dpie & (rs_raw <= ra_raw)),
            jnp.any(is_dpie & ~jnp.isfinite(rs_raw)),
            jnp.any(is_dpie & ~jnp.isfinite(v_disp)),
            jnp.any(is_scaling & (~jnp.isfinite(alpha_sigma) | ~jnp.isfinite(beta_radius))),
            jnp.any(~jnp.isfinite(x_center) | ~jnp.isfinite(y_center)),
            jnp.any(is_shear & (~jnp.isfinite(gamma1) | ~jnp.isfinite(gamma2))),
            jnp.any(is_dpie & (~jnp.isfinite(e1) | ~jnp.isfinite(e2) | ((jnp.square(e1) + jnp.square(e2)) >= 1.0))),
        ])
        return packed, PackedLensValidity(is_valid=~jnp.any(flags), reason_flags=flags)

    def packed_state_for_z(self, physical: jnp.ndarray, z_source: float) -> PackedLensState:
        factor = self.state.dpie_sigma0_factor_by_z.get(float(z_source))
        if factor is None:
            from .jax_cosmology import dpie_sigma0_factor_from_config
            factor = float(dpie_sigma0_factor_from_config(self.state.z_lens, float(z_source), self.state.cosmo_config))
            self.state.dpie_sigma0_factor_by_z[float(z_source)] = factor
        return self.packed_state(physical, factor_per_image=jnp.asarray(factor, dtype=jnp.float64))[0]

    # ------------------------------------------------------------------ lens fields
    def component_rows(self, xy: jnp.ndarray, packed: PackedLensState, components: np.ndarray):
        """Per-component deflection rows (C, N, 2) and Jacobians U = d alpha / d theta (C, N, 2, 2)."""
        x, y = xy[:, 0], xy[:, 1]
        components = np.asarray(components, dtype=np.int32)
        if components.size == 0:
            return jnp.zeros((0, x.shape[0], 2)), jnp.zeros((0, x.shape[0], 2, 2))
        profile = np.asarray(self.state.packed_lens_spec.profile_type)[components]
        rows = [jnp.zeros((components.size, x.shape[0]), dtype=jnp.float64) for _ in range(6)]
        dpie_mask = profile == DP_IE_PROFILE
        if np.any(dpie_mask):
            idx = jnp.asarray(components[dpie_mask])
            take = lambda value: jnp.take(jnp.asarray(value, dtype=jnp.float64), idx, axis=0)
            dpie_rows = jax_lensing.grouped_dpie_rows_derivatives_and_hessian(
                x, y, take(packed.sigma0), take(packed.Ra), take(packed.Rs), take(packed.e1), take(packed.e2),
                take(packed.center_x), take(packed.center_y))
            positions = jnp.asarray(np.flatnonzero(dpie_mask), dtype=jnp.int32)
            rows = [row.at[positions].set(value) for row, value in zip(rows, dpie_rows)]
        shear_mask = profile == SHEAR_PROFILE
        if np.any(shear_mask):
            idx = jnp.asarray(components[shear_mask])
            gamma1 = jnp.take(packed.gamma1, idx)[:, None]
            gamma2 = jnp.take(packed.gamma2, idx)[:, None]
            ones = jnp.ones_like(x)[None, :]
            shear_rows = (gamma1 * x[None] + gamma2 * y[None], gamma2 * x[None] - gamma1 * y[None],
                          -gamma1 * ones, -gamma2 * ones, -gamma2 * ones, gamma1 * ones)
            positions = jnp.asarray(np.flatnonzero(shear_mask), dtype=jnp.int32)
            rows = [row.at[positions].set(value) for row, value in zip(rows, shear_rows)]
        alpha = jnp.stack(rows[:2], -1)
        # Component rows store delta A = -U; the determinant conditioning needs U.
        u = -jnp.stack((jnp.stack(rows[2:4], -1), jnp.stack(rows[4:6], -1)), -2)
        return alpha, u

    def fields(self, physical: jnp.ndarray, packed: PackedLensState | None = None, *, force_exact: bool = False) -> dict[str, Any]:
        """Backprojected images, total Jacobian A, grad det A, and per-member deflections."""
        if packed is None:
            packed = self.packed_state(physical)[0]
        xy = self.xy
        components = np.arange(self.n_components, dtype=np.int32) if force_exact else self.exact_components
        alpha_rows, u_rows = self.component_rows(xy, packed, components)
        alpha, u = jnp.sum(alpha_rows, axis=0), jnp.sum(u_rows, axis=0)

        def total_u(position):
            return jnp.sum(self.component_rows(position, packed, components)[1], axis=0)

        dx = jnp.broadcast_to(jnp.array([1.0, 0.0]), xy.shape)
        dy = jnp.broadcast_to(jnp.array([0.0, 1.0]), xy.shape)
        du = jnp.stack((jax.jvp(total_u, (xy,), (dx,))[1], jax.jvp(total_u, (xy,), (dy,))[1]), -1)
        n_members, n_images = len(self.member_components), xy.shape[0]
        member_alpha = jnp.zeros((n_members, n_images, 2), dtype=xy.dtype)
        member_u = jnp.zeros((n_members, n_images, 2, 2), dtype=xy.dtype)
        row_lookup = {int(c): i for i, c in enumerate(components)}
        exact_member = np.asarray([i for i, c in enumerate(self.member_components) if int(c) in row_lookup], dtype=int)
        row_indices = np.asarray([row_lookup[int(self.member_components[i])] for i in exact_member], dtype=int)
        member_alpha = member_alpha.at[exact_member].set(alpha_rows[row_indices])
        member_u = member_u.at[exact_member].set(u_rows[row_indices])
        far_cut_radii = None
        if not force_exact and len(self.far_positions):
            ci = jnp.asarray(self.far_components)
            far = evaluate_farfield(self.far_geometry, jnp.take(packed.sigma0, ci, axis=0), jnp.take(packed.Ra, ci), jnp.take(packed.Rs, ci))
            member_alpha = member_alpha.at[self.far_positions].set(far.alpha)
            member_u = member_u.at[self.far_positions].set(far.jacobian)
            alpha = alpha + jnp.sum(far.alpha, axis=0)
            u = u + jnp.sum(far.jacobian, axis=0)
            du = du + far.total_jacobian_derivative
            far_cut_radii = jnp.take(packed.Rs, ci)
        a = jnp.eye(2) - u
        da = -du
        g = (a[:, 1, 1, None] * da[:, 0, 0, :] + a[:, 0, 0, None] * da[:, 1, 1, :]
             - a[:, 1, 0, None] * da[:, 0, 1, :] - a[:, 0, 1, None] * da[:, 1, 0, :])
        return dict(backprojected=xy - alpha, jacobian=a, determinant_gradient=g, jacobian_spatial_derivative=da,
                    member_alpha=member_alpha, member_jacobian=member_u, far_cut_radii=far_cut_radii, packed=packed)

    def far_field_valid(self, far_cut_radii) -> jnp.ndarray:
        """True when every far-field member keeps all images beyond the guard distance in cut radii."""
        if far_cut_radii is None or len(self.far_positions) == 0:
            return jnp.asarray(True)
        return jnp.all(self.far_min_separation >= self.guard_cut_radii * far_cut_radii)

    # ------------------------------------------------------------------ likelihood
    def statistics(self, params: jnp.ndarray, *, force_exact: bool = False) -> dict[str, Any]:
        physical = self.physical(params)
        images, members = self.state.images, self.state.members
        fields = self.fields(physical, force_exact=force_exact)
        sigma_src = physical[members.sigma_src_index]
        cov = sl.source_covariance(fields["jacobian"], images.sigma_per_image, sigma_src)
        whitening = sl.source_whitening(fields["backprojected"], cov["covariance"], images.family_index_per_image, images.n_families)
        coordinates = jnp.stack((params[images.source_x_param_index], params[images.source_y_param_index]), -1)
        transported = sl.transport_sources(coordinates, whitening, jnp.asarray(members.source_prior_mean), members.source_prior_sigma)
        source = transported["source"]
        stats = sl.statistics(fields["backprojected"], source[images.family_index_per_image], fields["jacobian"],
                              fields["determinant_gradient"], images.sigma_per_image, sigma_src, parity=images.parity)
        stats.update(source_by_family=source, source_correction=transported["log_correction"], whitening=whitening,
                     transport_valid=transported["valid"], sigma_src=sigma_src, fields=fields, physical=physical)
        return stats

    def log_likelihood(self, params: jnp.ndarray) -> jnp.ndarray:
        """Log-likelihood plus the source transport correction, or BAD_LOG_LIKE / -inf when rejected.

        Rejected proposals are evaluated at the known-valid initial model so that every
        branch stays finite; the selected value is then the rejection constant with zero gradient.
        """
        reference = jnp.asarray(self.state.initial_theta, dtype=jnp.float64)
        physical = self.physical(params)
        _, validity = self.packed_state(physical)
        ok_model = validity.is_valid & jnp.all(jnp.isfinite(params))
        scales = physical[self.scale_indices]
        ok_scales = jnp.all(jnp.isfinite(scales ** 2) & (scales > 0.0))
        ok = ok_model & ok_scales
        safe_params = jnp.where(ok, params, reference)
        stats = self.statistics(safe_params)
        value = jnp.sum(stats["loglike"]) + stats["source_correction"]
        valid = jnp.all(stats["valid"]) & jnp.all(stats["transport_valid"]) & ~jnp.isnan(value) & self.far_field_valid(stats["fields"]["far_cut_radii"])
        value = jnp.where(valid, value, -jnp.inf)
        return jnp.where(ok_model, jnp.where(ok_scales, value, -jnp.inf), jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))

    # ------------------------------------------------------------------ reporting
    def reported_physical(self, params: jnp.ndarray) -> jnp.ndarray:
        """Physical parameters with the sampled source offsets replaced by physical source positions."""
        physical = self.physical(params)
        source = self.statistics(params)["source_by_family"]
        images = self.state.images
        return physical.at[images.source_x_param_index].set(source[:, 0]).at[images.source_y_param_index].set(source[:, 1])

    def latent_from_reported_physical(self, physical: np.ndarray) -> np.ndarray:
        """Invert reported_physical: whitened source offsets from physical source positions."""
        from .model import convert_theta_to_latent
        physical = np.asarray(physical, dtype=float)
        latent = convert_theta_to_latent(physical, self.state.parameter_specs)
        images = self.state.images
        ix, iy = np.asarray(images.source_x_param_index), np.asarray(images.source_y_param_index)
        latent[ix] = 0.0
        latent[iy] = 0.0
        whitening = self.statistics(jnp.asarray(latent))["whitening"]
        offsets = np.asarray(sl.inverse_source_transport(np.stack((physical[ix], physical[iy]), -1), whitening))
        latent[ix], latent[iy] = offsets[:, 0], offsets[:, 1]
        return latent

    def source_positions(self, params: jnp.ndarray) -> jnp.ndarray:
        """Physical source positions per family, (F, 2); traceable under jit and vmap."""
        return self.statistics(params)["source_by_family"]
