"""Dedicated runtime for sampled sources and four immutable member classes.

The engine owns no surrogate, active-selection, metric, or scatter caches.
Lens physics and exact-image reporting are shared through neutral services.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from collections import defaultdict
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from .model import FamilyValidationCache, EvaluationResult, convert_theta_to_latent
from .jax_cosmology import (kpc_per_arcsec_from_config, h0_from_config,
    om0_from_config, w0_from_config, flat_wcdm_lens_geometry_factors, dpie_sigma0_factor)
from .evaluator_support import _build_geometry_cache, _build_cosmology_from_config
from .lens_physics import LensPhysicsMixin, ExactImageReportingMixin
from .smoothed_integration import SmoothedEvaluatorMixin, MODE
from .sampling_engines import canonical_sampling_engine, TIERED_MEMBERS


@dataclass(frozen=True)
class SmoothedImageData:
    family_ids: tuple[str, ...]
    n_families: int
    x_obs: jax.Array
    y_obs: jax.Array
    sigma_per_image: jax.Array
    reliability_per_image: jax.Array
    image_has_constraint: jax.Array
    effective_z_index_per_image: jax.Array
    global_family_index_per_image: jax.Array
    global_family_source_x_param_index: jax.Array
    global_family_source_y_param_index: jax.Array
    bin_index_per_image: jax.Array
    local_image_index_per_image: jax.Array


class TieredMembersEvaluator(SmoothedEvaluatorMixin, LensPhysicsMixin, ExactImageReportingMixin):
    sampling_engine = TIERED_MEMBERS
    sample_likelihood_mode = MODE
    use_magnitude_likelihood = False
    surrogate_enabled = False
    active_inference_enabled = False
    flat_source_metric_cache = None
    use_bulk_ray_shooting = True

    def __init__(self, state, match_tolerance_arcsec=2., *, sampling_engine=TIERED_MEMBERS,
                 sample_likelihood_mode=MODE, source_plane_covariance_floor=0.,
                 quick_diagnostics=False, use_magnitude_likelihood=False, **options):
        if sample_likelihood_mode != MODE or canonical_sampling_engine(sampling_engine, MODE) != TIERED_MEMBERS:
            raise ValueError('tiered_members requires smoothed-local-image-plane.')
        if use_magnitude_likelihood:
            raise ValueError('tiered_members supports positions only.')
        if state.smoothed_model is None:
            raise ValueError('Missing smoothed likelihood metadata.')
        self.state = state
        self.match_tolerance_arcsec = float(match_tolerance_arcsec)
        self.source_plane_covariance_floor = float(source_plane_covariance_floor or 0.)
        self.source_plane_covariance_mode = 'not_used'
        self.source_position_parameterization = state.source_position_parameterization
        self.source_position_conditional = self.source_position_parameterization == 'likelihood-whitened'
        self.quick_diagnostics = bool(quick_diagnostics)
        self.cosmo_config = dict(state.cosmo_config)
        geometry = state.geometry_cache
        if geometry is None:
            geometry = _build_geometry_cache(self.cosmo_config, state.z_lens, state.family_data, state.bin_data)
            state.geometry_cache = geometry
        self.kpc_per_arcsec = kpc_per_arcsec_from_config(state.z_lens, self.cosmo_config)
        self.dpie_sigma0_factors = {float(z):float(value) for z,value in geometry.dpie_sigma0_factor_by_effective_z.items()}
        self.exact_dpie_sigma0_factors = {float(z):float(value) for z,value in geometry.dpie_sigma0_factor_by_exact_z.items()}
        self.cosmology_h0 = h0_from_config(self.cosmo_config)
        self.cosmology_fiducial_om0 = om0_from_config(self.cosmo_config)
        self.cosmology_fiducial_w0 = w0_from_config(self.cosmo_config)
        self.cosmology_effective_z_values = np.asarray(geometry.effective_z_source_values, dtype=float)
        self.cosmology_effective_z_values_jax = jnp.asarray(self.cosmology_effective_z_values)
        self.cosmology_effective_z_to_index = {float(z):i for i,z in enumerate(self.cosmology_effective_z_values)}
        names = {spec.sample_name:i for i,spec in enumerate(state.parameter_specs)}
        self.cosmology_om0_param_index = names.get('cosmology_Om0', -1)
        self.cosmology_w0_param_index = names.get('cosmology_w0', -1)
        self.fit_cosmology_flat_wcdm = bool(getattr(state,'fit_cosmology_flat_wcdm',False)
            and self.cosmology_om0_param_index >= 0 and self.cosmology_w0_param_index >= 0)
        self.log_softening_length_param_index = names.get('log_softening_length_kpc', -1)
        self.softening_length_sampled = self.log_softening_length_param_index >= 0
        specs = state.parameter_specs
        kinds = np.asarray([spec.transform_kind for spec in specs])
        self.transform_kind_log_positive_mask = jnp.asarray(kinds == 'log_positive')
        self.transform_kind_log_offset_positive_mask = jnp.asarray(kinds == 'log_offset_positive')
        self.transform_kind_affine_mask = jnp.asarray(kinds == 'affine')
        self.transform_offset_array = jnp.asarray([spec.transform_offset for spec in specs], dtype=jnp.float64)
        self.transform_scale_array = jnp.asarray([spec.transform_scale for spec in specs], dtype=jnp.float64)
        self.source_position_param_indices_by_family = {
            str(f.family_id):tuple(next((i for i,spec in enumerate(specs) if spec.component_family=='source_position'
                and spec.potential_id==str(f.family_id) and spec.field==field),-1) for field in ('beta_x','beta_y'))
            for f in state.family_data}
        self.packed_spec_jax = self._prepare_packed_spec_arrays()
        self.traced_bin_data = tuple(self._trace_bin(b) for b in state.bin_data)
        self.traced_bin_data_by_z = {b.effective_z_source:b for b in self.traced_bin_data}
        self.flat_critical_arc_data = self._build_smoothed_image_data()
        self.timing_totals = defaultdict(float)
        self.eval_wall_times = []
        self.approximate_eval_count = self.full_refresh_count = self.invalid_state_rejection_count = 0
        self.invalid_state_reason_counts = {}
        self.validation_cache = {f.family_id:FamilyValidationCache() for f in state.family_data}
        self.exact_models_by_z, self.exact_solvers_by_z = {}, {}
        self.exact_lenstronomy_count = 0
        self._cosmo = None
        self._initialize_reporting_options(options)
        self._initialize_smoothed_model()
        self.scaling_rank_df = self._smoothed_scaling_diagnostics()
        # Catalog identities below are reporting metadata, never a selection policy.
        family = np.asarray(state.packed_lens_spec.component_family)
        self.scaling_component_indices = np.flatnonzero(family == 1).astype(np.int32)
        self.large_component_indices = np.flatnonzero(family == 0).astype(np.int32)
        self.exact_scaling_component_indices = np.intersect1d(self.scaling_component_indices,self.smoothed_exact_components)
        self.active_scaling_component_indices = self.exact_scaling_component_indices
        self.cached_scaling_component_indices = np.setdiff1d(self.scaling_component_indices,self.smoothed_exact_components)
        self.inactive_scaling_component_indices = self.cached_scaling_component_indices
        self.scaling_param_indices = np.asarray([i for i,s in enumerate(specs) if s.component_family=='scaling'],dtype=int)
        self.scaling_scatter_param_indices = np.array([],dtype=int)
        self.grouped_lensing_backend = 'grouped_dpie_nie_e'
        self.grouped_dpie_component_count = int(np.sum(np.asarray(state.packed_lens_spec.profile_type)==81))
        self.grouped_shear_component_count = int(np.sum(np.asarray(state.packed_lens_spec.profile_type)==14))
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    @property
    def cosmo(self):
        if self._cosmo is None:
            self._cosmo = _build_cosmology_from_config(self.cosmo_config)
        return self._cosmo

    def _initialize_reporting_options(self, options):
        from . import lens_physics as defaults
        values = dict(exact_image_min_distance_arcsec=.1, exact_image_precision_limit=1e-8,
            exact_image_num_iter_max=200, exact_image_finder=defaults.DEFAULT_EXACT_IMAGE_FINDER,
            exact_image_displacement_tol_arcsec=.0001, exact_image_identification_tol_arcsec=.001,
            exact_image_lm_max_iter=30, exact_image_lm_trust_radius_arcsec=1., exact_image_adaptive_max_levels=8,
            anchored_image_plane_lm_damping_relative=defaults.DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_RELATIVE,
            anchored_image_plane_lm_damping_absolute=defaults.DEFAULT_ANCHORED_IMAGE_PLANE_LM_DAMPING_ABSOLUTE)
        for name,default in values.items():
            setattr(self,name,options.get(name,default))
        for name in ('exact_image_min_distance_arcsec', 'exact_image_precision_limit',
                     'exact_image_displacement_tol_arcsec', 'exact_image_identification_tol_arcsec',
                     'exact_image_lm_trust_radius_arcsec'):
            value=float(getattr(self,name))
            if not np.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be finite and positive.')
            setattr(self,name,value)
        for name in ('exact_image_num_iter_max','exact_image_lm_max_iter','exact_image_adaptive_max_levels'):
            value=int(getattr(self,name))
            if value<=0:
                raise ValueError(f'{name} must be positive.')
            setattr(self,name,value)
        if self.exact_image_finder not in ('lenstronomy','local-lm','local-lm-adaptive'):
            raise ValueError('Unsupported exact image finder.')

    def _prepare_packed_spec_arrays(self):
        spec = self.state.packed_lens_spec
        fields = ('x_center','y_center','e1','e2','core_radius_kpc','cut_radius_kpc','v_disp','gamma1','gamma2',
                  'sigma_ref','cut_ref','core_ref','alpha_sigma','beta_radius')
        result = {name:jnp.asarray(getattr(spec,name),dtype=jnp.int32 if name.endswith('param_index') else jnp.float64)
                  for field in fields for name in (field+'_base',field.replace('_kpc','')+'_param_index')}
        for name in ('profile_type','component_family','luminosity_ratio','cut_radius_excess_above_effective_core'):
            result[name]=jnp.asarray(getattr(spec,name))
        return result

    def _trace_bin(self, b):
        indices = np.asarray(b.family_index_per_image,dtype=int)
        counts = np.bincount(indices,minlength=len(b.family_ids))
        return SimpleNamespace(effective_z_source=float(b.effective_z_source), family_ids=tuple(b.family_ids),
            n_families=len(b.family_ids), x_obs=jnp.asarray(b.x_obs,dtype=jnp.float64), y_obs=jnp.asarray(b.y_obs,dtype=jnp.float64),
            family_index_per_image=jnp.asarray(b.family_index_per_image,dtype=jnp.int32), sigma_per_image=jnp.asarray(b.sigma_per_image,dtype=jnp.float64),
            reliability_per_image=jnp.asarray(b.reliability_per_image,dtype=jnp.float64), image_has_constraint=jnp.asarray(counts[indices]>1),
            effective_z_index=self.cosmology_effective_z_to_index[float(b.effective_z_source)])

    def _build_smoothed_image_data(self):
        bs=self.traced_bin_data
        cat=lambda values,dtype: jnp.asarray(np.concatenate(values) if values else [],
            dtype=jnp.int32 if dtype is int else jnp.float64 if dtype is float else dtype)
        ids=tuple(str(f) for b in bs for f in b.family_ids)
        offsets=np.cumsum([0]+[b.n_families for b in bs[:-1]])
        return SmoothedImageData(ids,len(ids),*(cat([np.asarray(getattr(b,name)) for b in bs],dtype)
            for name,dtype in [('x_obs',float),('y_obs',float),('sigma_per_image',float),
                              ('reliability_per_image',float),('image_has_constraint',bool)]),
            cat([np.full(len(b.x_obs),b.effective_z_index) for b in bs],int),
            cat([np.asarray(b.family_index_per_image)+offset for b,offset in zip(bs,offsets)],int),
            jnp.asarray([self.source_position_param_indices_by_family[f][0] for f in ids],dtype=jnp.int32),
            jnp.asarray([self.source_position_param_indices_by_family[f][1] for f in ids],dtype=jnp.int32),
            cat([np.full(len(b.x_obs),i) for i,b in enumerate(bs)],int),
            cat([np.arange(len(b.x_obs)) for b in bs],int))

    def _apply_independent_free_log_displacements(self, physical, sv, sc, sr, v, c, r, spec):
        return v,c,r

    def _independent_branch_weight_from_physical(self, physical):
        return jnp.ones(len(self.state.base_components),dtype=jnp.float64)

    def _source_loglike_impl(self, params):
        if self.sampling_engine != TIERED_MEMBERS or self.sample_likelihood_mode != MODE:
            raise ValueError('tiered_members requires smoothed-local-image-plane; legacy engines are disabled.')
        return self._flat_smoothed_loglike_impl(params)

    def source_loglike(self, params):
        started=time.perf_counter()
        value=float(self._source_loglike_fn(jnp.asarray(params,dtype=jnp.float64)))
        self.eval_wall_times.append(time.perf_counter()-started)
        self.approximate_eval_count += 1
        return value

    def _reported_physical_parameter_vector(self, params):
        return self._smoothed_reported_physical(params)

    def reported_physical_parameter_vector(self, params):
        return np.asarray(self._smoothed_reported_physical(params))

    def reported_physical_to_latent_parameter_vector(self, physical):
        from .smoothed_likelihood import inverse_source_transport
        physical=np.asarray(physical,dtype=float)
        latent=convert_theta_to_latent(physical,self.state.parameter_specs)
        if self.source_position_conditional:
            data=self.flat_critical_arc_data
            ix,iy=np.asarray(data.global_family_source_x_param_index),np.asarray(data.global_family_source_y_param_index)
            latent[ix]=latent[iy]=0.
            whitening=self._smoothed_statistics(jnp.asarray(latent))['whitening']
            offsets=np.asarray(inverse_source_transport(np.stack((physical[ix],physical[iy]),-1),whitening))
            latent[ix],latent[iy]=offsets[:,0],offsets[:,1]
        return latent

    def _source_position_for_family_numpy(self, params, family_id):
        physical=self.reported_physical_parameter_vector(params)
        ix,iy=self.source_position_param_indices_by_family[str(family_id)]
        return float(physical[ix]),float(physical[iy])

    def _family_source_summary(self, params):
        return self._smoothed_source_summary(params)

    def evaluate(self, params, validate_all_families=False):
        return EvaluationResult(loglike=float(jnp.sum(self._smoothed_statistics(jnp.asarray(params))['loglike'])),
            family_predictions=self._smoothed_source_summary(params))

    def _build_packed_lens_state(self, params, z_source, **options):
        physical=self._physical_parameter_vector(params)
        scale=self._kpc_per_arcsec_for_physical(physical)
        if self.fit_cosmology_flat_wcdm:
            om0,w0=self._cosmology_parameters_from_physical(physical)
            _,_,factors=flat_wcdm_lens_geometry_factors(self.state.z_lens,jnp.asarray([z_source]),self.cosmology_h0,om0,w0)
        else:
            factor=self.dpie_sigma0_factors.get(float(z_source),self.exact_dpie_sigma0_factors.get(float(z_source)))
            factors=(jnp.asarray([factor]) if factor is not None else jnp.atleast_1d(dpie_sigma0_factor(
                self.state.z_lens,z_source,self.cosmology_h0,self.cosmology_fiducial_om0,self.cosmology_fiducial_w0)))
        packed,_=self._build_flat_packed_lens_state_with_validity_from_physical(physical,
            SimpleNamespace(effective_z_index_per_image=jnp.zeros(1,dtype=int)),stop_gradient=False,
            kpc_per_arcsec=scale,dpie_sigma0_factors=factors,**options)
        return packed._replace(sigma0=packed.sigma0[:,0])

    def _build_truth_grid_packed_lens_state(self, params, z_source):
        return self._build_packed_lens_state(params,z_source)

    def _arc_aware_image_support_details(self, params, family, sx, sy, match_details=None):
        return self._empty_arc_aware_image_support_details(family)

    def refresh_scaling_scatter_cache(self, *args, **kwargs):
        return

    def refresh_source_metric_cache(self, *args, **kwargs):
        return

    def release_runtime_caches(self):
        self.exact_models_by_z.clear()
        self.exact_solvers_by_z.clear()
        for name in ('_source_loglike_fn', '_smoothed_reported_physical_batch', '_smoothed_physical_loglike_batch'):
            clear = getattr(getattr(self, name, None), 'clear_cache', None)
            if clear is not None:
                clear()

    def _exact_family_prediction_details(self, params: np.ndarray, family: FamilyData) -> dict[str, Any]:
        cache = self.validation_cache[family.family_id]
        base_details: dict[str, Any] = {
            "produced_image_count": np.nan,
            "recovered_image_count": np.nan,
            "missing_image_count": np.nan,
            "extra_image_count": np.nan,
            "multiplicity_failed": True,
            "multiplicity_failure_reason": "",
            "matched_model_x_arcsec": np.full(int(getattr(family, "n_images", 0)), np.nan, dtype=float),
            "matched_model_y_arcsec": np.full(int(getattr(family, "n_images", 0)), np.nan, dtype=float),
            "recovered_image_mask": np.zeros(int(getattr(family, "n_images", 0)), dtype=bool),
            "extra_model_x_arcsec": np.asarray([], dtype=float),
            "extra_model_y_arcsec": np.asarray([], dtype=float),
            "failed": True,
        }
        base_details.update(self._empty_arc_aware_image_support_details(family))
        packed_state = self._build_packed_lens_state(jnp.asarray(params, dtype=jnp.float64), family.z_source)
        try:
            beta_x, beta_y = self._exact_source_ray_shooting(family, packed_state)
        except Exception:
            cache.multiplicity_mismatch_count += 1
            details = {**base_details, "multiplicity_failure_reason": "source_ray_shooting_failed"}
            self._record_exact_prediction_details(family.family_id, details)
            return details
        sampled_source = self._source_position_for_family_numpy(params, family.family_id)
        if sampled_source is None:
            raise ValueError('Exact image diagnostics require sampled physical source coordinates.')
        source_x,source_y=sampled_source
        source_residuals = np.sqrt((np.asarray(beta_x, dtype=float) - source_x) ** 2 + (np.asarray(beta_y, dtype=float) - source_y) ** 2)
        cache.source_plane_rms = float(np.sqrt(np.mean(source_residuals**2)))
        cache.last_source_x = source_x
        cache.last_source_y = source_y
        cache.exact_validation_count += 1
        x_pred: np.ndarray
        y_pred: np.ndarray
        try:
            x_pred, y_pred = self._solve_exact_images(family, packed_state, source_x, source_y)
        except Exception:
            arc_details = self._arc_aware_image_support_details(params, family, source_x, source_y)
            cache.multiplicity_mismatch_count += 1
            solver_details = dict(getattr(self, "_last_exact_image_solver_diagnostics", {}))
            details = {
                **base_details,
                **arc_details,
                **solver_details,
                "multiplicity_failure_reason": "exact_image_prediction_failed",
            }
            self._record_exact_prediction_details(family.family_id, details)
            return details

        match_details = self._image_match_diagnostics(np.asarray(x_pred), np.asarray(y_pred), family)
        arc_details = self._arc_aware_image_support_details(params, family, source_x, source_y, match_details)
        solver_details = dict(getattr(self, "_last_exact_image_solver_diagnostics", {}))
        matched = self._match_images(np.asarray(x_pred), np.asarray(y_pred), family)
        if matched is None:
            if len(x_pred) != family.n_images:
                cache.multiplicity_mismatch_count += 1
            else:
                cache.match_failure_count += 1
            details = {
                **base_details,
                **{key: value for key, value in match_details.items() if not str(key).startswith("_")},
                **arc_details,
                **solver_details,
                "failed": True,
            }
            self._record_exact_prediction_details(family.family_id, details)
            return details
        residuals = np.sqrt((matched[0] - family.x_obs) ** 2 + (matched[1] - family.y_obs) ** 2)
        rms = float(np.sqrt(np.mean(residuals**2)))
        cache.exact_image_rms = rms
        details = {
            **base_details,
            **{key: value for key, value in match_details.items() if not str(key).startswith("_")},
            **arc_details,
            **solver_details,
            "failed": False,
            "x_pred": matched[0],
            "y_pred": matched[1],
            "exact_image_rms": rms,
        }
        self._record_exact_prediction_details(family.family_id, details)
        return details

    def _exact_family_prediction(self, params: np.ndarray, family: FamilyData) -> tuple[np.ndarray, np.ndarray, float] | None:
        details = self._exact_family_prediction_details(params, family)
        if bool(details.get("failed", True)):
            return None
        return (
            np.asarray(details["x_pred"], dtype=float),
            np.asarray(details["y_pred"], dtype=float),
            float(details["exact_image_rms"]),
        )
