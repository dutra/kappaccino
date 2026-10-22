"""Shared lens physics and exact-image reporting, without inference engines.

These kernels are shared by the legacy evaluator and the tiered evaluator.
There are no likelihoods, member-selection policies, or surrogate caches here.
"""
from __future__ import annotations
from typing import Any, Protocol
import json
import time
import numpy as np
import jax.numpy as jnp
from scipy.optimize import linear_sum_assignment
from lenstronomy.LensModel.lens_model import LensModel as NPLensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver as NPLensEquationSolver
from . import jax_lensing
from .model import (FamilyData, PackedLensState, PackedLensValidity,
    apply_parameter_transforms_jax as _apply_parameter_transforms_jax)
from .jax_cosmology import (
    dpie_sigma0_factor_from_config as _dpie_sigma0_factor_from_config,
    flat_wcdm_kpc_per_arcsec as _jax_flat_wcdm_kpc_per_arcsec,
    flat_wcdm_lens_geometry_factors as _jax_flat_wcdm_lens_geometry_factors)
import jax

DP_IE_PROFILE = 81
SHEAR_PROFILE = 14
DEFAULT_MATCH_TOLERANCE = 2.0
DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC = 0.1
DEFAULT_EXACT_IMAGE_PRECISION_LIMIT = 1.0e-8
DEFAULT_EXACT_IMAGE_NUM_ITER_MAX = 200
EXACT_IMAGE_FINDER_LENSTRONOMY = "lenstronomy"
EXACT_IMAGE_FINDER_LOCAL_LM = "local-lm"
EXACT_IMAGE_FINDER_LOCAL_LM_ADAPTIVE = "local-lm-adaptive"
DEFAULT_EXACT_IMAGE_FINDER = EXACT_IMAGE_FINDER_LENSTRONOMY
DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC = 0.0
DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC = 0.3
DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE = 1.0e-3
DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE = 1.0e-6
MIN_JACOBIAN_NORMAL_DETERMINANT_GUARD = 1.0e-30
SAFE_RADIUS_MARGIN_ARCSEC = 1.0e-3
SAFE_RADIUS_MARGIN_KPC = 1.0e-3

def _smooth_residual_cap(
    residual_x: jnp.ndarray,
    residual_y: jnp.ndarray,
    max_residual_arcsec: float = DEFAULT_LIKELIHOOD_STABILIZER_MAX_RESIDUAL_ARCSEC,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if float(max_residual_arcsec) <= 0.0:
        return residual_x, residual_y
    radius = jnp.sqrt(jnp.square(residual_x) + jnp.square(residual_y) + jnp.asarray(1.0e-30, dtype=jnp.float64))
    scaled_radius = radius / jnp.asarray(float(max_residual_arcsec), dtype=jnp.float64)
    scale = jnp.tanh(scaled_radius) / jnp.maximum(scaled_radius, jnp.asarray(1.0e-12, dtype=jnp.float64))
    return residual_x * scale, residual_y * scale

def _anchored_solved_image_plane_step_from_jacobian(
    f_x: jnp.ndarray,
    f_y: jnp.ndarray,
    jac_a00: jnp.ndarray,
    jac_a01: jnp.ndarray,
    jac_a10: jnp.ndarray,
    jac_a11: jnp.ndarray,
    *,
    trust_radius_arcsec: float = DEFAULT_ANCHORED_IMAGE_PLANE_TRUST_RADIUS_ARCSEC,
    lm_damping_relative: float = DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
    lm_damping_absolute: float = DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    finite_entries = (
        jnp.isfinite(f_x)
        & jnp.isfinite(f_y)
        & jnp.isfinite(jac_a00)
        & jnp.isfinite(jac_a01)
        & jnp.isfinite(jac_a10)
        & jnp.isfinite(jac_a11)
    )
    normal00 = jnp.square(jac_a00) + jnp.square(jac_a10)
    normal01 = jac_a00 * jac_a01 + jac_a10 * jac_a11
    normal11 = jnp.square(jac_a01) + jnp.square(jac_a11)
    trace = jnp.maximum(normal00 + normal11, jnp.asarray(0.0, dtype=jnp.float64))
    lam = (
        jnp.asarray(float(lm_damping_absolute), dtype=jnp.float64)
        + jnp.asarray(float(lm_damping_relative), dtype=jnp.float64) * trace
    )
    h00 = normal00 + lam
    h11 = normal11 + lam
    h01 = normal01
    rhs0 = -(jac_a00 * f_x + jac_a10 * f_y)
    rhs1 = -(jac_a01 * f_x + jac_a11 * f_y)
    det_h = h00 * h11 - jnp.square(h01)
    det_h_safe = jnp.maximum(det_h, jnp.asarray(MIN_JACOBIAN_NORMAL_DETERMINANT_GUARD, dtype=jnp.float64))
    delta_x = (h11 * rhs0 - h01 * rhs1) / det_h_safe
    delta_y = (-h01 * rhs0 + h00 * rhs1) / det_h_safe
    delta_x, delta_y = _smooth_residual_cap(delta_x, delta_y, trust_radius_arcsec)
    finite = (
        finite_entries
        & jnp.isfinite(lam)
        & jnp.isfinite(det_h)
        & jnp.isfinite(delta_x)
        & jnp.isfinite(delta_y)
    )
    return delta_x, delta_y, finite

def _json_arcsec_array(values: Any) -> str:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        array = np.asarray([], dtype=float)
    finite_values = [float(value) for value in array if np.isfinite(value)]
    return json.dumps(finite_values, separators=(",", ":"))

class ImageRedshiftData(Protocol):
    effective_z_index_per_image: jax.Array


class LensPhysicsMixin:
    def _apply_param_updates(
        self,
        base: jnp.ndarray,
        param_index: np.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:
        if params.shape[0] == 0:
            return base
        param_index_jax = jnp.asarray(param_index, dtype=jnp.int32)
        safe_index = jnp.where(param_index_jax >= 0, param_index_jax, 0)
        updated = jnp.take(params, safe_index)
        return jnp.where(param_index_jax >= 0, updated, base)

    def _physical_parameter_vector(self, params: jnp.ndarray) -> jnp.ndarray:
        return _apply_parameter_transforms_jax(
            params,
            self.transform_kind_log_positive_mask,
            self.transform_kind_log_offset_positive_mask,
            self.transform_offset_array,
            self.transform_kind_affine_mask,
            self.transform_scale_array,
        )

    def _cosmology_parameters_from_physical(self, physical_params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not self.fit_cosmology_flat_wcdm:
            return (
                jnp.asarray(self.cosmology_fiducial_om0, dtype=jnp.float64),
                jnp.asarray(self.cosmology_fiducial_w0, dtype=jnp.float64),
            )
        om0 = jnp.take(physical_params, jnp.asarray(self.cosmology_om0_param_index, dtype=jnp.int32))
        w0 = jnp.take(physical_params, jnp.asarray(self.cosmology_w0_param_index, dtype=jnp.int32))
        return om0, w0

    def _sampled_cosmology_geometry_for_physical(
        self,
        physical_params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        om0, w0 = self._cosmology_parameters_from_physical(physical_params)
        kpc_per_arcsec, _efficiency, dpie_sigma0_factors = _jax_flat_wcdm_lens_geometry_factors(
            self.state.z_lens,
            self.cosmology_effective_z_values_jax,
            self.cosmology_h0,
            om0,
            w0,
        )
        return kpc_per_arcsec, dpie_sigma0_factors

    def _kpc_per_arcsec_for_physical(self, physical_params: jnp.ndarray) -> jnp.ndarray:
        if not self.fit_cosmology_flat_wcdm:
            return jnp.asarray(self.kpc_per_arcsec, dtype=jnp.float64)
        om0, w0 = self._cosmology_parameters_from_physical(physical_params)
        return _jax_flat_wcdm_kpc_per_arcsec(
            self.state.z_lens,
            self.cosmology_h0,
            om0,
            w0,
        )

    def _softening_length_kpc_from_physical(self, physical_params: jnp.ndarray) -> jnp.ndarray:
        index = int(getattr(self, "log_softening_length_param_index", -1))
        if index < 0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        log_softening = jnp.take(jnp.asarray(physical_params, dtype=jnp.float64), jnp.asarray(index, dtype=jnp.int32))
        return jnp.exp(log_softening)

    def _effective_core_radius_kpc_from_physical(
        self,
        core_radius_kpc: jnp.ndarray,
        physical_params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if int(getattr(self, "log_softening_length_param_index", -1)) < 0:
            return (
                core_radius_kpc,
                jnp.asarray(0.0, dtype=jnp.float64),
                jnp.asarray(jnp.nan, dtype=jnp.float64),
            )
        softening_length_kpc = self._softening_length_kpc_from_physical(physical_params)
        core_radius_effective_kpc = jnp.sqrt(jnp.square(core_radius_kpc) + jnp.square(softening_length_kpc))
        return core_radius_effective_kpc, softening_length_kpc, jnp.log(softening_length_kpc)

    def _total_cut_radius_kpc_from_excess_modes(
        self,
        cut_radius_value_kpc: jnp.ndarray,
        core_radius_effective_kpc: jnp.ndarray,
        spec_arrays: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        excess_mode = jnp.asarray(spec_arrays["cut_radius_excess_above_effective_core"], dtype=bool)
        return jnp.where(
            excess_mode,
            core_radius_effective_kpc + jnp.maximum(cut_radius_value_kpc, SAFE_RADIUS_MARGIN_KPC),
            cut_radius_value_kpc,
        )

    def _effective_scaling_exponents_from_physical(
        self,
        spec_jax: dict[str, Any],
        physical_params: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        alpha_sigma = self._apply_param_updates(
            spec_jax["alpha_sigma_base"], spec_jax["alpha_sigma_param_index"], physical_params
        )
        beta_radius = self._apply_param_updates(
            spec_jax["beta_radius_base"], spec_jax["beta_radius_param_index"], physical_params
        )
        gamma_ml = 2.0 * alpha_sigma + beta_radius - 1.0
        return alpha_sigma, beta_radius, gamma_ml

    def _dpie_sigma0_factor_for_z_source(self, z_source: float) -> float:
        factor = self.dpie_sigma0_factors.get(float(z_source))
        if factor is not None:
            return float(factor)
        factor = self.exact_dpie_sigma0_factors.get(float(z_source))
        if factor is not None:
            return float(factor)
        return _dpie_sigma0_factor_from_config(self.state.z_lens, float(z_source), self.cosmo_config)

    def _build_flat_packed_lens_state_with_validity_from_physical(
        self,
        physical_params: jnp.ndarray,
        flat_data: ImageRedshiftData,
        *,
        stop_gradient: bool,
        kpc_per_arcsec: jnp.ndarray | None = None,
        dpie_sigma0_factors: jnp.ndarray | None = None,
        sigma_log_offset: float | jnp.ndarray = 0.0,
        core_log_offset: float | jnp.ndarray = 0.0,
        cut_log_offset: float | jnp.ndarray = 0.0,
    ) -> tuple[PackedLensState, PackedLensValidity]:
        spec_jax = self.packed_spec_jax
        x_center = self._apply_param_updates(spec_jax["x_center_base"], spec_jax["x_center_param_index"], physical_params)
        y_center = self._apply_param_updates(spec_jax["y_center_base"], spec_jax["y_center_param_index"], physical_params)
        e1 = self._apply_param_updates(spec_jax["e1_base"], spec_jax["e1_param_index"], physical_params)
        e2 = self._apply_param_updates(spec_jax["e2_base"], spec_jax["e2_param_index"], physical_params)
        core_radius_kpc = self._apply_param_updates(
            spec_jax["core_radius_kpc_base"], spec_jax["core_radius_param_index"], physical_params
        )
        cut_radius_kpc = self._apply_param_updates(
            spec_jax["cut_radius_kpc_base"], spec_jax["cut_radius_param_index"], physical_params
        )
        v_disp = self._apply_param_updates(spec_jax["v_disp_base"], spec_jax["v_disp_param_index"], physical_params)
        gamma1 = self._apply_param_updates(spec_jax["gamma1_base"], spec_jax["gamma1_param_index"], physical_params)
        gamma2 = self._apply_param_updates(spec_jax["gamma2_base"], spec_jax["gamma2_param_index"], physical_params)

        profile_type = spec_jax["profile_type"]
        component_family = spec_jax["component_family"]
        is_dpie = profile_type == DP_IE_PROFILE
        is_shear = profile_type == SHEAR_PROFILE
        is_scaling = component_family == 1

        luminosity_ratio = spec_jax["luminosity_ratio"]
        sigma_ref = self._apply_param_updates(
            spec_jax["sigma_ref_base"], spec_jax["sigma_ref_param_index"], physical_params
        )
        cut_ref = self._apply_param_updates(
            spec_jax["cut_ref_base"], spec_jax["cut_ref_param_index"], physical_params
        )
        core_ref = self._apply_param_updates(
            spec_jax["core_ref_base"], spec_jax["core_ref_param_index"], physical_params
        )
        alpha_sigma, beta_radius, _gamma_ml = self._effective_scaling_exponents_from_physical(spec_jax, physical_params)
        scaled_vdisp = sigma_ref * jnp.power(luminosity_ratio, alpha_sigma)
        size_luminosity_scale = jnp.power(luminosity_ratio, beta_radius)
        scaled_core = core_ref * size_luminosity_scale
        scaled_cut = cut_ref * size_luminosity_scale
        scaled_vdisp = scaled_vdisp * jnp.exp(jnp.asarray(sigma_log_offset, dtype=jnp.float64))
        scaled_core = scaled_core * jnp.exp(jnp.asarray(core_log_offset, dtype=jnp.float64))
        scaled_cut = scaled_cut * jnp.exp(jnp.asarray(cut_log_offset, dtype=jnp.float64))
        v_disp, core_radius_kpc, cut_radius_kpc = self._apply_independent_free_log_displacements(
            physical_params,
            scaled_vdisp,
            scaled_core,
            scaled_cut,
            v_disp,
            core_radius_kpc,
            cut_radius_kpc,
            spec_jax,
        )
        v_disp = jnp.where(is_scaling, scaled_vdisp, v_disp)
        v_disp = self._smoothed_member_vdisp(v_disp, physical_params)
        core_radius_kpc = jnp.where(is_scaling, scaled_core, core_radius_kpc)
        cut_radius_kpc = jnp.where(is_scaling, scaled_cut, cut_radius_kpc)

        if kpc_per_arcsec is None:
            kpc_per_arcsec = self._kpc_per_arcsec_for_physical(physical_params)
        kpc_per_arcsec = jnp.asarray(kpc_per_arcsec, dtype=jnp.float64)
        core_radius_effective_kpc, _softening_length_kpc, _log_softening_length_kpc = (
            self._effective_core_radius_kpc_from_physical(core_radius_kpc, physical_params)
        )
        cut_radius_kpc = self._total_cut_radius_kpc_from_excess_modes(
            cut_radius_kpc,
            core_radius_effective_kpc,
            spec_jax,
        )
        ra_raw = core_radius_effective_kpc / kpc_per_arcsec
        rs_raw = cut_radius_kpc / kpc_per_arcsec
        ra = jnp.maximum(ra_raw, SAFE_RADIUS_MARGIN_ARCSEC)
        rs = jnp.maximum(rs_raw, ra + SAFE_RADIUS_MARGIN_ARCSEC)
        if dpie_sigma0_factors is None:
            if bool(getattr(self, "fit_cosmology_flat_wcdm", False)):
                _kpc_per_arcsec, dpie_sigma0_factors = self._sampled_cosmology_geometry_for_physical(physical_params)
            else:
                dpie_sigma0_factors = jnp.asarray(
                    [self._dpie_sigma0_factor_for_z_source(float(z_source)) for z_source in self.cosmology_effective_z_values],
                    dtype=jnp.float64,
                )
        factor_per_image = jnp.take(
            jnp.asarray(dpie_sigma0_factors, dtype=jnp.float64),
            flat_data.effective_z_index_per_image,
        )
        safe_factor = jnp.where(jnp.isfinite(factor_per_image), factor_per_image, 0.0)
        branch_weight = self._independent_branch_weight_from_physical(
            physical_params,
        )
        sigma0 = branch_weight[:, None] * (v_disp[:, None] ** 2) * safe_factor[None, :] / ra[:, None]

        packed_state = PackedLensState(
            profile_type=profile_type,
            sigma0=jnp.where(is_dpie[:, None], sigma0, 0.0),
            Ra=jnp.where(is_dpie, ra, 0.0),
            Rs=jnp.where(is_dpie, rs, 0.0),
            e1=jnp.where(is_dpie, e1, 0.0),
            e2=jnp.where(is_dpie, e2, 0.0),
            center_x=x_center,
            center_y=y_center,
            gamma1=jnp.where(is_shear, gamma1, 0.0),
            gamma2=jnp.where(is_shear, gamma2, 0.0),
        )
        reason_flags = jnp.stack(
            [
                jnp.any(is_dpie[:, None] & ~jnp.isfinite(sigma0)),
                jnp.any(is_dpie & (ra_raw <= 0.0)),
                jnp.any(is_dpie & (rs_raw <= ra_raw)),
                jnp.any(is_dpie & ~jnp.isfinite(rs_raw)),
                jnp.any(is_dpie & ~jnp.isfinite(v_disp)),
                jnp.any(
                    is_scaling
                    & (
                        ~jnp.isfinite(alpha_sigma)
                        | ~jnp.isfinite(beta_radius)
                    )
                ),
                jnp.any(~jnp.isfinite(x_center) | ~jnp.isfinite(y_center)),
                jnp.any(is_shear & (~jnp.isfinite(gamma1) | ~jnp.isfinite(gamma2))),
                jnp.any(
                    is_dpie
                    & (
                        ~jnp.isfinite(e1)
                        | ~jnp.isfinite(e2)
                        | ((jnp.square(e1) + jnp.square(e2)) >= 1.0)
                    )
                ),
                jnp.any(~jnp.isfinite(factor_per_image)),
            ],
            axis=0,
        )
        if stop_gradient:
            reason_flags = jax.lax.stop_gradient(reason_flags)
        return packed_state, PackedLensValidity(is_valid=~jnp.any(reason_flags), reason_flags=reason_flags)


class ExactImageReportingMixin:
    def _component_indices_np(self, component_indices: np.ndarray | None = None) -> np.ndarray:
        if component_indices is None:
            return np.arange(len(self.state.lens_model_list), dtype=np.int32)
        return np.asarray(component_indices, dtype=np.int32).reshape(-1)

    def _split_grouped_component_indices(self, component_indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        indices = self._component_indices_np(component_indices)
        if indices.size == 0:
            return indices, indices
        profile_type = np.asarray(self.state.packed_lens_spec.profile_type, dtype=np.int32)
        selected_profiles = profile_type[indices]
        unsupported = sorted(set(int(value) for value in selected_profiles.tolist()) - {DP_IE_PROFILE, SHEAR_PROFILE})
        if unsupported:
            raise RuntimeError(
                "Mandatory grouped lensing backend supports only DPIE_NIE_E and SHEAR profiles; "
                f"unsupported profile codes={unsupported}."
            )
        dpie_indices = indices[selected_profiles == DP_IE_PROFILE].astype(np.int32)
        shear_indices = indices[selected_profiles == SHEAR_PROFILE].astype(np.int32)
        return dpie_indices, shear_indices

    def _take_packed_components(
        self,
        value: jnp.ndarray,
        component_indices: np.ndarray,
    ) -> jnp.ndarray:
        return jnp.take(
            jnp.asarray(value, dtype=jnp.float64),
            jnp.asarray(component_indices, dtype=jnp.int32),
            axis=0,
        )

    def _grouped_dpie_params(
        self,
        packed_state: PackedLensState,
        component_indices: np.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (
            self._take_packed_components(packed_state.sigma0, component_indices),
            self._take_packed_components(packed_state.Ra, component_indices),
            self._take_packed_components(packed_state.Rs, component_indices),
            self._take_packed_components(packed_state.e1, component_indices),
            self._take_packed_components(packed_state.e2, component_indices),
            self._take_packed_components(packed_state.center_x, component_indices),
            self._take_packed_components(packed_state.center_y, component_indices),
        )

    def _grouped_shear_alpha_and_hessian(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if component_indices.size == 0:
            zeros = jnp.zeros_like(x, dtype=jnp.float64)
            return zeros, zeros, zeros, zeros, zeros, zeros
        gamma1 = self._take_packed_components(packed_state.gamma1, component_indices)
        gamma2 = self._take_packed_components(packed_state.gamma2, component_indices)
        gamma1_sum = jnp.sum(gamma1)
        gamma2_sum = jnp.sum(gamma2)
        alpha_x = gamma1_sum * x + gamma2_sum * y
        alpha_y = gamma2_sum * x - gamma1_sum * y
        ones = jnp.ones_like(x, dtype=jnp.float64)
        h_xx = gamma1_sum * ones
        h_xy = gamma2_sum * ones
        h_yx = gamma2_sum * ones
        h_yy = -gamma1_sum * ones
        return alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy

    def _grouped_alpha_and_hessian_for_components(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        dpie_indices, shear_indices = self._split_grouped_component_indices(component_indices)
        zeros = jnp.zeros_like(x, dtype=jnp.float64)
        alpha_x = zeros
        alpha_y = zeros
        h_xx = zeros
        h_xy = zeros
        h_yx = zeros
        h_yy = zeros
        if dpie_indices.size:
            dpie_values = jax_lensing.grouped_dpie_derivatives_and_hessian(
                x,
                y,
                *self._grouped_dpie_params(packed_state, dpie_indices),
            )
            alpha_x = alpha_x + dpie_values[0]
            alpha_y = alpha_y + dpie_values[1]
            h_xx = h_xx + dpie_values[2]
            h_xy = h_xy + dpie_values[3]
            h_yx = h_yx + dpie_values[4]
            h_yy = h_yy + dpie_values[5]
        if shear_indices.size:
            shear_values = self._grouped_shear_alpha_and_hessian(x, y, packed_state, shear_indices)
            alpha_x = alpha_x + shear_values[0]
            alpha_y = alpha_y + shear_values[1]
            h_xx = h_xx + shear_values[2]
            h_xy = h_xy + shear_values[3]
            h_yx = h_yx + shear_values[4]
            h_yy = h_yy + shear_values[5]
        return alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy

    def _lensing_jacobian_for_components(
        self,
        z_source: float,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del z_source
        _, _, h_xx, h_xy, h_yx, h_yy = self._grouped_alpha_and_hessian_for_components(
            x,
            y,
            packed_state,
            component_indices,
        )
        return 1.0 - h_xx, -h_xy, -h_yx, 1.0 - h_yy

    def _flat_lensing_jacobian_for_components(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        _, _, h_xx, h_xy, h_yx, h_yy = self._grouped_alpha_and_hessian_for_components(
            x,
            y,
            packed_state,
            component_indices,
        )
        return 1.0 - h_xx, -h_xy, -h_yx, 1.0 - h_yy

    def _ray_shooting_for_components(
        self,
        z_source: float,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del z_source
        alpha_x, alpha_y, *_ = self._grouped_alpha_and_hessian_for_components(x, y, packed_state, component_indices)
        return x - alpha_x, y - alpha_y

    def _flat_ray_shooting_for_components(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha_x, alpha_y, *_ = self._grouped_alpha_and_hessian_for_components(x, y, packed_state, component_indices)
        return x - alpha_x, y - alpha_y

    def _flat_ray_shooting_and_lensing_jacobian_for_components(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        alpha_x, alpha_y, h_xx, h_xy, h_yx, h_yy = self._grouped_alpha_and_hessian_for_components(
            x,
            y,
            packed_state,
            component_indices,
        )
        return x - alpha_x, y - alpha_y, 1.0 - h_xx, -h_xy, -h_yx, 1.0 - h_yy

    def _exact_source_ray_shooting(
        self,
        family: FamilyData,
        packed_state: PackedLensState,
    ) -> tuple[np.ndarray, np.ndarray]:
        model, _solver = self._get_exact_model_solver(family.z_source)
        kwargs_lens = self._packed_to_kwargs_lens(packed_state)
        beta_x, beta_y = model.ray_shooting(
            jnp.asarray(family.x_obs, dtype=jnp.float64),
            jnp.asarray(family.y_obs, dtype=jnp.float64),
            kwargs_lens,
        )
        return np.asarray(beta_x, dtype=float), np.asarray(beta_y, dtype=float)

    def _exact_ray_shooting_and_jacobian_numpy(
        self,
        family: FamilyData,
        packed_state: PackedLensState,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_jax = jnp.asarray(x, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)
        beta_x, beta_y = self._ray_shooting_for_components(family.z_source, x_jax, y_jax, packed_state)
        jac_a00, jac_a01, jac_a10, jac_a11 = self._lensing_jacobian_for_components(
            family.z_source,
            x_jax,
            y_jax,
            packed_state,
        )
        return (
            np.asarray(beta_x, dtype=float),
            np.asarray(beta_y, dtype=float),
            np.asarray(jac_a00, dtype=float),
            np.asarray(jac_a01, dtype=float),
            np.asarray(jac_a10, dtype=float),
            np.asarray(jac_a11, dtype=float),
        )

    def _solve_exact_images(
        self,
        family: FamilyData,
        packed_state: PackedLensState,
        source_x: float,
        source_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        finder = str(getattr(self, "exact_image_finder", DEFAULT_EXACT_IMAGE_FINDER))
        self._last_exact_image_solver_diagnostics = {
            "exact_image_finder": finder,
            "exact_image_lm_converged": np.full(int(getattr(family, "n_images", 0)), np.nan),
            "exact_image_lm_iterations": np.full(int(getattr(family, "n_images", 0)), np.nan),
            "exact_image_adaptive_used": np.zeros(int(getattr(family, "n_images", 0)), dtype=bool),
            "exact_image_final_step_arcsec": np.full(int(getattr(family, "n_images", 0)), np.nan),
            "exact_image_final_source_residual_arcsec": np.full(int(getattr(family, "n_images", 0)), np.nan),
            "exact_image_identification_pass": np.full(int(getattr(family, "n_images", 0)), False, dtype=bool),
            "exact_image_displacement_from_observed_arcsec": np.full(int(getattr(family, "n_images", 0)), np.nan),
        }
        if finder == EXACT_IMAGE_FINDER_LENSTRONOMY:
            x_pred, y_pred = self._solve_exact_images_lenstronomy(family, packed_state, source_x, source_y)
            self._last_exact_image_solver_diagnostics.update(
                {
                    "exact_image_identification_pass": np.ones(min(len(x_pred), int(getattr(family, "n_images", 0))), dtype=bool),
                }
            )
            return x_pred, y_pred
        if finder == EXACT_IMAGE_FINDER_LOCAL_LM:
            return self._solve_exact_images_local_lm(family, packed_state, source_x, source_y, adaptive=False)
        if finder == EXACT_IMAGE_FINDER_LOCAL_LM_ADAPTIVE:
            return self._solve_exact_images_local_lm(family, packed_state, source_x, source_y, adaptive=True)
        raise RuntimeError(f"Unsupported exact image finder: {finder}")

    def _solve_exact_images_local_lm(
        self,
        family: FamilyData,
        packed_state: PackedLensState,
        source_x: float,
        source_y: float,
        *,
        adaptive: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        exact_start = time.perf_counter()
        try:
            start_x = np.asarray(family.x_obs, dtype=float)
            start_y = np.asarray(family.y_obs, dtype=float)
            result = self._refine_exact_images_local_lm(family, packed_state, source_x, source_y, start_x, start_y)
            adaptive_used = np.zeros(int(start_x.size), dtype=bool)
            ok = np.asarray(result["identification_pass"], dtype=bool)
            if adaptive and not bool(np.all(ok)):
                failed = np.flatnonzero(~ok)
                radius = min(float(self.exact_image_lm_trust_radius_arcsec), max(float(self.exact_image_min_distance_arcsec), 1.0e-3))
                for idx in failed:
                    best_result: dict[str, Any] | None = None
                    best_score = np.inf
                    for level in range(int(self.exact_image_adaptive_max_levels)):
                        offsets = np.asarray([-radius, 0.0, radius], dtype=float)
                        grid_x, grid_y = np.meshgrid(offsets, offsets)
                        candidate_x = float(start_x[idx]) + grid_x.reshape(-1)
                        candidate_y = float(start_y[idx]) + grid_y.reshape(-1)
                        refined = self._refine_exact_images_local_lm(
                            family,
                            packed_state,
                            source_x,
                            source_y,
                            candidate_x,
                            candidate_y,
                        )
                        scores = np.asarray(refined["final_step_arcsec"], dtype=float)
                        source_scores = np.asarray(refined["final_source_residual_arcsec"], dtype=float)
                        combined = np.where(
                            np.asarray(refined["finite"], dtype=bool),
                            scores + 0.01 * source_scores,
                            np.inf,
                        )
                        best_idx = int(np.argmin(combined))
                        if np.isfinite(combined[best_idx]) and combined[best_idx] < best_score:
                            best_score = float(combined[best_idx])
                            best_result = {key: np.asarray(value)[best_idx] for key, value in refined.items()}
                        if best_result is not None and bool(best_result["identification_pass"]):
                            break
                        radius *= 0.5
                    if best_result is not None and bool(best_result["identification_pass"]):
                        for key in ("x", "y", "converged", "finite", "identification_pass", "iterations", "final_step_arcsec", "final_source_residual_arcsec"):
                            result[key][idx] = best_result[key]
                        adaptive_used[idx] = True
                ok = np.asarray(result["identification_pass"], dtype=bool)
            diagnostics = {
                "exact_image_finder": EXACT_IMAGE_FINDER_LOCAL_LM_ADAPTIVE if adaptive else EXACT_IMAGE_FINDER_LOCAL_LM,
                "exact_image_lm_converged": np.asarray(result["converged"], dtype=bool),
                "exact_image_lm_iterations": np.asarray(result["iterations"], dtype=int),
                "exact_image_adaptive_used": adaptive_used,
                "exact_image_final_step_arcsec": np.asarray(result["final_step_arcsec"], dtype=float),
                "exact_image_final_source_residual_arcsec": np.asarray(result["final_source_residual_arcsec"], dtype=float),
                "exact_image_identification_pass": np.asarray(result["identification_pass"], dtype=bool),
                "exact_image_displacement_from_observed_arcsec": np.sqrt(
                    np.square(np.asarray(result["x"], dtype=float) - start_x)
                    + np.square(np.asarray(result["y"], dtype=float) - start_y)
                ),
            }
            self._last_exact_image_solver_diagnostics = diagnostics
            if not bool(np.any(ok)):
                raise RuntimeError("local exact image solver failed identification tolerance")
            # Keep converged roots for partial matching. Failed anchors are not
            # model images, but their per-anchor diagnostics remain available.
            return np.asarray(result["x"], dtype=float)[ok], np.asarray(result["y"], dtype=float)[ok]
        finally:
            elapsed = time.perf_counter() - exact_start
            self.timing_totals["exact_solver"] = self.timing_totals.get("exact_solver", 0.0) + elapsed
            key = "exact_solver_local_lm_adaptive" if adaptive else "exact_solver_local_lm"
            self.timing_totals[key] = self.timing_totals.get(key, 0.0) + elapsed

    def _refine_exact_images_local_lm(
        self,
        family: FamilyData,
        packed_state: PackedLensState,
        source_x: float,
        source_y: float,
        start_x: np.ndarray,
        start_y: np.ndarray,
    ) -> dict[str, Any]:
        current_x = np.asarray(start_x, dtype=float).reshape(-1).copy()
        current_y = np.asarray(start_y, dtype=float).reshape(-1).copy()
        n_images = int(current_x.size)
        converged = np.zeros(n_images, dtype=bool)
        finite = np.isfinite(current_x) & np.isfinite(current_y)
        iterations = np.zeros(n_images, dtype=int)
        final_step = np.full(n_images, np.inf, dtype=float)
        final_source_residual = np.full(n_images, np.inf, dtype=float)
        for step_index in range(int(self.exact_image_lm_max_iter)):
            beta_x, beta_y, jac_a00, jac_a01, jac_a10, jac_a11 = self._exact_ray_shooting_and_jacobian_numpy(
                family,
                packed_state,
                current_x,
                current_y,
            )
            residual_x = beta_x - float(source_x)
            residual_y = beta_y - float(source_y)
            delta_x, delta_y, step_finite = _anchored_solved_image_plane_step_from_jacobian(
                jnp.asarray(residual_x, dtype=jnp.float64),
                jnp.asarray(residual_y, dtype=jnp.float64),
                jnp.asarray(jac_a00, dtype=jnp.float64),
                jnp.asarray(jac_a01, dtype=jnp.float64),
                jnp.asarray(jac_a10, dtype=jnp.float64),
                jnp.asarray(jac_a11, dtype=jnp.float64),
                trust_radius_arcsec=float(self.exact_image_lm_trust_radius_arcsec),
                lm_damping_relative=float(self.anchored_image_plane_lm_damping_relative),
                lm_damping_absolute=float(self.anchored_image_plane_lm_damping_absolute),
            )
            delta_x_np = np.asarray(delta_x, dtype=float)
            delta_y_np = np.asarray(delta_y, dtype=float)
            step_norm = np.sqrt(np.square(delta_x_np) + np.square(delta_y_np))
            source_norm = np.sqrt(np.square(residual_x) + np.square(residual_y))
            step_finite_np = np.asarray(step_finite, dtype=bool)
            active = finite & step_finite_np & ~converged
            current_x = np.where(active, current_x + delta_x_np, current_x)
            current_y = np.where(active, current_y + delta_y_np, current_y)
            final_step = np.where(active | converged, step_norm, final_step)
            final_source_residual = np.where(active | converged, source_norm, final_source_residual)
            iterations = np.where(active, step_index + 1, iterations)
            newly_converged = (
                active
                & np.isfinite(step_norm)
                & np.isfinite(source_norm)
                & (step_norm <= float(self.exact_image_displacement_tol_arcsec))
                & (source_norm <= max(float(self.exact_image_identification_tol_arcsec), float(self.exact_image_precision_limit)))
            )
            converged = converged | newly_converged
            finite = finite & step_finite_np & np.isfinite(current_x) & np.isfinite(current_y)
            if bool(np.all(converged | ~finite)):
                break
        beta_x, beta_y, jac_a00, jac_a01, jac_a10, jac_a11 = self._exact_ray_shooting_and_jacobian_numpy(
            family,
            packed_state,
            current_x,
            current_y,
        )
        residual_x = beta_x - float(source_x)
        residual_y = beta_y - float(source_y)
        check_dx, check_dy, check_finite = _anchored_solved_image_plane_step_from_jacobian(
            jnp.asarray(residual_x, dtype=jnp.float64),
            jnp.asarray(residual_y, dtype=jnp.float64),
            jnp.asarray(jac_a00, dtype=jnp.float64),
            jnp.asarray(jac_a01, dtype=jnp.float64),
            jnp.asarray(jac_a10, dtype=jnp.float64),
            jnp.asarray(jac_a11, dtype=jnp.float64),
            trust_radius_arcsec=float(self.exact_image_lm_trust_radius_arcsec),
            lm_damping_relative=float(self.anchored_image_plane_lm_damping_relative),
            lm_damping_absolute=float(self.anchored_image_plane_lm_damping_absolute),
        )
        root_error = np.sqrt(np.square(np.asarray(check_dx, dtype=float)) + np.square(np.asarray(check_dy, dtype=float)))
        source_norm = np.sqrt(np.square(residual_x) + np.square(residual_y))
        finite_final = (
            finite
            & np.asarray(check_finite, dtype=bool)
            & np.isfinite(root_error)
            & np.isfinite(source_norm)
            & np.isfinite(current_x)
            & np.isfinite(current_y)
        )
        identification_pass = finite_final & (root_error <= float(self.exact_image_identification_tol_arcsec))
        converged = converged | (identification_pass & (root_error <= float(self.exact_image_displacement_tol_arcsec)))
        return {
            "x": current_x,
            "y": current_y,
            "converged": converged,
            "finite": finite_final,
            "identification_pass": identification_pass,
            "iterations": iterations,
            "final_step_arcsec": root_error,
            "final_source_residual_arcsec": source_norm,
        }

    def _solve_exact_images_lenstronomy(
        self,
        family: FamilyData,
        packed_state: PackedLensState,
        source_x: float,
        source_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        model, solver = self._get_exact_model_solver(family.z_source)
        kwargs_lens = self._packed_to_kwargs_lens(packed_state)
        exact_start = time.perf_counter()
        try:
            x_pred, y_pred = solver.image_position_from_source(
                source_x,
                source_y,
                kwargs_lens,
                solver="lenstronomy",
                min_distance=float(getattr(self, "exact_image_min_distance_arcsec", DEFAULT_EXACT_IMAGE_MIN_DISTANCE_ARCSEC)),
                search_window=family.search_window,
                x_center=family.x_center,
                y_center=family.y_center,
                num_iter_max=int(getattr(self, "exact_image_num_iter_max", DEFAULT_EXACT_IMAGE_NUM_ITER_MAX)),
                precision_limit=float(getattr(self, "exact_image_precision_limit", DEFAULT_EXACT_IMAGE_PRECISION_LIMIT)),
            )
            self.exact_lenstronomy_count = int(getattr(self, "exact_lenstronomy_count", 0)) + 1
            return np.asarray(x_pred, dtype=float), np.asarray(y_pred, dtype=float)
        finally:
            elapsed = time.perf_counter() - exact_start
            self.timing_totals["exact_solver"] = self.timing_totals.get("exact_solver", 0.0) + elapsed
            self.timing_totals["exact_solver_lenstronomy"] = self.timing_totals.get("exact_solver_lenstronomy", 0.0) + elapsed

    def _get_exact_model_solver(self, z_source: float) -> tuple[NPLensModel, NPLensEquationSolver]:
        model = self.exact_models_by_z.get(z_source)
        if model is None:
            exact_model_setup_start = time.perf_counter()
            model = NPLensModel(
                lens_model_list=self.state.lens_model_list,
                z_lens=self.state.z_lens,
                z_source=z_source,
                cosmo=self.cosmo,
            )
            self.exact_models_by_z[z_source] = model
            self.exact_solvers_by_z[z_source] = NPLensEquationSolver(model)
            self.timing_totals["exact_model_cache_setup"] += time.perf_counter() - exact_model_setup_start
        return model, self.exact_solvers_by_z[z_source]

    def _image_match_diagnostics(self, x_pred: np.ndarray, y_pred: np.ndarray, family: FamilyData) -> dict[str, Any]:
        x_array = np.asarray(x_pred, dtype=float).reshape(-1)
        y_array = np.asarray(y_pred, dtype=float).reshape(-1)
        n_observed = int(family.n_images)
        n_produced = int(min(x_array.size, y_array.size))
        matched_x = np.full(n_observed, np.nan, dtype=float)
        matched_y = np.full(n_observed, np.nan, dtype=float)
        recovered_mask = np.zeros(n_observed, dtype=bool)
        diagnostics: dict[str, Any] = {
            "produced_image_count": n_produced,
            "recovered_image_count": 0,
            "missing_image_count": n_observed,
            "extra_image_count": n_produced,
            "multiplicity_failed": True,
            "multiplicity_failure_reason": "no_model_images" if n_produced == 0 else "no_matches",
            "matched_model_x_arcsec": matched_x,
            "matched_model_y_arcsec": matched_y,
            "recovered_image_mask": recovered_mask,
            "extra_model_x_arcsec": np.asarray([], dtype=float),
            "extra_model_y_arcsec": np.asarray([], dtype=float),
            "_matched_x": matched_x,
            "_matched_y": matched_y,
        }
        if x_array.size != y_array.size:
            n_extra = int(max(x_array.size, y_array.size))
            extra_x = np.full(n_extra, np.nan, dtype=float)
            extra_y = np.full(n_extra, np.nan, dtype=float)
            extra_x[: x_array.size] = x_array
            extra_y[: y_array.size] = y_array
            diagnostics.update(
                {
                    "produced_image_count": n_extra,
                    "extra_image_count": n_extra,
                    "multiplicity_failure_reason": "prediction_shape_mismatch",
                    "extra_model_x_arcsec": extra_x,
                    "extra_model_y_arcsec": extra_y,
                }
            )
            return diagnostics
        if n_observed == 0:
            diagnostics.update(
                {
                    "recovered_image_count": 0,
                    "missing_image_count": 0,
                    "extra_image_count": n_produced,
                    "multiplicity_failed": bool(n_produced != 0),
                    "multiplicity_failure_reason": "extra_model_images" if n_produced else "",
                    "extra_model_x_arcsec": x_array.copy(),
                    "extra_model_y_arcsec": y_array.copy(),
                }
            )
            return diagnostics
        if n_produced == 0:
            return diagnostics
        pred = np.column_stack([x_array, y_array])
        obs = np.column_stack([np.asarray(family.x_obs, dtype=float), np.asarray(family.y_obs, dtype=float)])
        if pred.shape[0] != n_produced or obs.shape[0] != n_observed:
            diagnostics.update(
                {
                    "multiplicity_failure_reason": "prediction_shape_mismatch",
                    "extra_model_x_arcsec": x_array.copy(),
                    "extra_model_y_arcsec": y_array.copy(),
                }
            )
            return diagnostics
        finite_pred = np.isfinite(pred).all(axis=1)
        finite_obs = np.isfinite(obs).all(axis=1)
        cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
        cost = np.where(finite_pred[:, None] & finite_obs[None, :], cost, np.inf)
        finite_cost = np.isfinite(cost)
        if not np.any(finite_cost):
            diagnostics.update(
                {
                    "multiplicity_failure_reason": "nonfinite_prediction",
                    "extra_model_x_arcsec": x_array.copy(),
                    "extra_model_y_arcsec": y_array.copy(),
                }
            )
            return diagnostics
        finite_max = float(np.max(cost[finite_cost]))
        match_tolerance = float(getattr(self, "match_tolerance_arcsec", DEFAULT_MATCH_TOLERANCE))
        assignment_cost = np.where(finite_cost, cost, finite_max + match_tolerance + 1.0)
        row_ind, col_ind = linear_sum_assignment(assignment_cost)
        accepted = np.isfinite(cost[row_ind, col_ind]) & (cost[row_ind, col_ind] <= match_tolerance)
        recovered_count = int(np.sum(accepted))
        accepted_pred_indices = np.asarray(row_ind[accepted], dtype=int)
        for pred_idx, obs_idx in zip(row_ind[accepted], col_ind[accepted]):
            matched_x[int(obs_idx)] = x_array[int(pred_idx)]
            matched_y[int(obs_idx)] = y_array[int(pred_idx)]
            recovered_mask[int(obs_idx)] = True
        extra_mask = np.ones(n_produced, dtype=bool)
        extra_mask[accepted_pred_indices] = False
        missing_count = int(max(0, n_observed - recovered_count))
        extra_count = int(max(0, n_produced - recovered_count))
        if missing_count == 0 and extra_count == 0:
            reason = ""
        elif missing_count > 0 and extra_count > 0:
            reason = "missing_and_extra_model_images" if n_produced != n_observed else "match_tolerance_exceeded"
        elif missing_count > 0:
            reason = "missing_model_images"
        else:
            reason = "extra_model_images"
        diagnostics.update(
            {
                "recovered_image_count": recovered_count,
                "missing_image_count": missing_count,
                "extra_image_count": extra_count,
                "multiplicity_failed": bool(missing_count > 0 or extra_count > 0),
                "multiplicity_failure_reason": reason,
                "matched_model_x_arcsec": matched_x,
                "matched_model_y_arcsec": matched_y,
                "recovered_image_mask": recovered_mask,
                "extra_model_x_arcsec": x_array[extra_mask],
                "extra_model_y_arcsec": y_array[extra_mask],
                "_matched_x": matched_x,
                "_matched_y": matched_y,
            }
        )
        return diagnostics

    def _match_images(self, x_pred: np.ndarray, y_pred: np.ndarray, family: FamilyData) -> tuple[np.ndarray, np.ndarray] | None:
        match_details = self._image_match_diagnostics(x_pred, y_pred, family)
        if (
            int(match_details["produced_image_count"]) != family.n_images
            or int(match_details["recovered_image_count"]) != family.n_images
        ):
            return None
        ordered_x = np.asarray(match_details["_matched_x"], dtype=float)
        ordered_y = np.asarray(match_details["_matched_y"], dtype=float)
        if ordered_x.shape != (family.n_images,) or ordered_y.shape != (family.n_images,):
            return None
        if not np.all(np.isfinite(ordered_x) & np.isfinite(ordered_y)):
            return None
        return ordered_x, ordered_y

    def _record_exact_prediction_details(self, family_id: str, details: dict[str, Any]) -> None:
        if not hasattr(self, "_last_exact_prediction_details"):
            self._last_exact_prediction_details = {}
        self._last_exact_prediction_details[str(family_id)] = details

    def _packed_to_kwargs_lens(self, packed_state: PackedLensState) -> list[dict[str, float]]:
        convert_start = time.perf_counter()
        kwargs_lens: list[dict[str, float]] = []
        profile_type = np.asarray(packed_state.profile_type, dtype=int)
        for idx, profile in enumerate(profile_type.tolist()):
            if profile == DP_IE_PROFILE:
                kwargs_lens.append(
                    {
                        "sigma0": float(np.asarray(packed_state.sigma0[idx])),
                        "Ra": float(np.asarray(packed_state.Ra[idx])),
                        "Rs": float(np.asarray(packed_state.Rs[idx])),
                        "e1": float(np.asarray(packed_state.e1[idx])),
                        "e2": float(np.asarray(packed_state.e2[idx])),
                        "center_x": float(np.asarray(packed_state.center_x[idx])),
                        "center_y": float(np.asarray(packed_state.center_y[idx])),
                    }
                )
            elif profile == SHEAR_PROFILE:
                kwargs_lens.append(
                    {
                        "gamma1": float(np.asarray(packed_state.gamma1[idx])),
                        "gamma2": float(np.asarray(packed_state.gamma2[idx])),
                    }
                )
            else:  # pragma: no cover
                raise ValueError(f"Unsupported profile type {profile}.")
        self.timing_totals["validation_conversion"] += time.perf_counter() - convert_start
        return kwargs_lens

    def _empty_arc_aware_image_support_details(
        self,
        family: FamilyData,
        match_details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        n_images = int(getattr(family, "n_images", 0))
        point_recovered = np.zeros(n_images, dtype=bool)
        point_residual = np.full(n_images, np.nan, dtype=float)
        if isinstance(match_details, dict):
            recovered = np.asarray(match_details.get("recovered_image_mask", point_recovered), dtype=bool).reshape(-1)
            matched_x = np.asarray(match_details.get("matched_model_x_arcsec", point_residual), dtype=float).reshape(-1)
            matched_y = np.asarray(match_details.get("matched_model_y_arcsec", point_residual), dtype=float).reshape(-1)
            if recovered.shape == (n_images,) and matched_x.shape == (n_images,) and matched_y.shape == (n_images,):
                point_recovered = recovered
                point_residual = np.sqrt(
                    np.square(matched_x - np.asarray(family.x_obs, dtype=float))
                    + np.square(matched_y - np.asarray(family.y_obs, dtype=float))
                )
                point_residual = np.where(point_recovered, point_residual, np.nan)
        point_valid = point_recovered & np.isfinite(point_residual)
        point_residual = np.where(point_valid, point_residual, np.nan)
        supported_or_recovered = np.isfinite(point_residual)
        arc_aware_rms = (
            float(np.sqrt(np.mean(np.square(point_residual[supported_or_recovered]))))
            if np.any(supported_or_recovered)
            else np.nan
        )
        status = np.where(point_valid, "point_recovered", "not_recovered").astype(object)
        return {
            "point_image_residual_arcsec": point_residual,
            "arc_candidate_supported": np.zeros(n_images, dtype=bool),
            "arc_candidate_image_residual_arcsec": np.full(n_images, np.nan, dtype=float),
            "preferred_recovery_status": status,
            "preferred_image_residual_arcsec": point_residual,
            "arc_recovery_status": status,
            "arc_aware_image_residual_arcsec": point_residual,
            "arc_noncritical_direction_residual_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_critical_direction_residual_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_critical_direction_x": np.full(n_images, np.nan, dtype=float),
            "arc_critical_direction_y": np.full(n_images, np.nan, dtype=float),
            "arc_noncritical_direction_x": np.full(n_images, np.nan, dtype=float),
            "arc_noncritical_direction_y": np.full(n_images, np.nan, dtype=float),
            "arc_s_min": np.full(n_images, np.nan, dtype=float),
            "arc_s_max": np.full(n_images, np.nan, dtype=float),
            "arc_detA": np.full(n_images, np.nan, dtype=float),
            "arc_prior_probability": np.full(n_images, np.nan, dtype=float),
            "p_arc": np.full(n_images, np.nan, dtype=float),
            "arc_log_odds": np.full(n_images, np.nan, dtype=float),
            "arc_curve_distance_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_curve_arclength_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_curve_finite": np.zeros(n_images, dtype=bool),
            "arc_support_anchor_x_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_support_anchor_y_arcsec": np.full(n_images, np.nan, dtype=float),
            "arc_support_curve_x_arcsec": np.asarray([_json_arcsec_array([]) for _ in range(n_images)], dtype=object),
            "arc_support_curve_y_arcsec": np.asarray([_json_arcsec_array([]) for _ in range(n_images)], dtype=object),
            "arc_supported_mask": np.zeros(n_images, dtype=bool),
            "arc_supported": np.zeros(n_images, dtype=bool),
            "arc_support_finite_mask": np.zeros(n_images, dtype=bool),
            "arc_aware_image_rms_arcsec": arc_aware_rms,
            "arc_aware_recovered_image_count": int(np.sum(supported_or_recovered)),
            "arc_aware_missing_image_count": int(max(0, n_images - int(np.sum(supported_or_recovered)))),
            "arc_supported_image_count": 0,
            "arc_candidate_supported_image_count": 0,
        }



    def _grouped_component_rows_for_components(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        indices = self._component_indices_np(component_indices)
        if indices.size == 0:
            empty = jnp.zeros((0, x.shape[0]), dtype=jnp.float64)
            return empty, empty, empty, empty, empty, empty
        profile_type = np.asarray(self.state.packed_lens_spec.profile_type, dtype=np.int32)
        selected_profiles = profile_type[indices]
        unsupported = sorted(set(int(value) for value in selected_profiles.tolist()) - {DP_IE_PROFILE, SHEAR_PROFILE})
        if unsupported:
            raise RuntimeError(
                "Mandatory grouped lensing backend supports only DPIE_NIE_E and SHEAR profiles; "
                f"unsupported profile codes={unsupported}."
            )
        rows = [jnp.zeros((indices.size, x.shape[0]), dtype=jnp.float64) for _ in range(6)]
        dpie_mask_np = selected_profiles == DP_IE_PROFILE
        if np.any(dpie_mask_np):
            dpie_rows = jax_lensing.grouped_dpie_rows_derivatives_and_hessian(
                x,
                y,
                *self._grouped_dpie_params(packed_state, indices[dpie_mask_np].astype(np.int32)),
            )
            dpie_positions = jnp.asarray(np.where(dpie_mask_np)[0], dtype=jnp.int32)
            rows = [row.at[dpie_positions].set(value) for row, value in zip(rows, dpie_rows)]
        shear_mask_np = selected_profiles == SHEAR_PROFILE
        if np.any(shear_mask_np):
            shear_indices = indices[shear_mask_np].astype(np.int32)
            gamma1 = self._take_packed_components(packed_state.gamma1, shear_indices)[:, jnp.newaxis]
            gamma2 = self._take_packed_components(packed_state.gamma2, shear_indices)[:, jnp.newaxis]
            x_row = x[jnp.newaxis, :]
            y_row = y[jnp.newaxis, :]
            ones = jnp.ones_like(x_row, dtype=jnp.float64)
            shear_rows = (
                gamma1 * x_row + gamma2 * y_row,
                gamma2 * x_row - gamma1 * y_row,
                -gamma1 * ones,
                -gamma2 * ones,
                -gamma2 * ones,
                gamma1 * ones,
            )
            shear_positions = jnp.asarray(np.where(shear_mask_np)[0], dtype=jnp.int32)
            rows = [row.at[shear_positions].set(value) for row, value in zip(rows, shear_rows)]
        return tuple(rows)  # type: ignore[return-value]


    def _flat_component_alpha_and_jacobian_delta_rows_for_components(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return self._grouped_component_rows_for_components(x, y, packed_state, component_indices)


    def _flat_component_alpha_rows_for_components(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        packed_state: PackedLensState,
        component_indices: np.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha_x, alpha_y, *_ = self._grouped_component_rows_for_components(x, y, packed_state, component_indices)
        return alpha_x, alpha_y

