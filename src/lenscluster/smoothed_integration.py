"""Solver integration for the sampled-source local image-plane likelihood.

Only fixed geometry is cached. Lens fields, absorbed scatter and source transport
are functions of the current proposal. Legacy model state has no payload here.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import numpy as np
import jax
import jax.numpy as jnp

from .model import ParameterSpec

MODE = "smoothed-local-image-plane"
REVISION = 1


def add_model_parameters(args, specs, records, n_components, images):
    """Append the new hierarchy without changing legacy packed-spec fields."""
    from .member_tiers import MemberTierRecord

    identities = tuple(f"{r['potfile_id']}:{r['catalog_id']}" for r in records)
    components = tuple(int(r['component_index']) for r in records)
    if len(set(identities)) != len(identities) or len(set(components)) != len(components):
        raise ValueError("Population member identities and component indices must be unique.")
    payload = getattr(args, "member_tiers_payload", None)
    phase = str(getattr(args, "member_tier_phase", "initializer"))
    if phase == "fixed":
        if payload is None:
            raise ValueError("Fixed smoothed member classes require saved member_tiers_payload.")
        tiers = MemberTierRecord.from_dict(payload)
        if tuple(tiers.member_ids) != identities or tuple(tiers.component_indices) != components:
            raise ValueError("Saved member identities/order differ from the current population.")
        sampled = set(tiers.exact_sampled) | set(tiers.approximate_sampled)
    elif phase == "initializer":
        sampled = set()
    else:
        raise ValueError(f"Unknown member_tier_phase={phase!r}.")

    scatter_by_population = {}
    scatter_indices = np.full(n_components, -1, dtype=int)
    offset_indices = np.full(n_components, -1, dtype=int)
    median = float(getattr(args, "member_velocity_scatter_prior_median", .02))
    width = float(getattr(args, "member_velocity_scatter_prior_log_sigma", 1.))
    for r in records:
        population = str(r['potfile_id'])
        if population not in scatter_by_population:
            scatter_by_population[population] = len(specs)
            specs.append(ParameterSpec(
                name=f"{population}.velocity_log_scatter", sample_name=f"{population}_velocity_log_scatter",
                potential_id=population, profile_type=81, field="velocity_log_scatter",
                prior_kind="normal", lower=-np.inf, upper=np.inf, step=.1,
                mean=float(np.log(median)), std=width, component_family="member_velocity_scatter",
                transform_kind="log_positive", physical_lower=0., physical_mean=median))
        scatter_indices[int(r['component_index'])] = scatter_by_population[population]
    for site_index, member in enumerate(sorted(sampled)):
        r = records[member]
        name = f"{r['potfile_id']}_{r['catalog_id']}_velocity_offset"
        offset_indices[int(r['component_index'])] = len(specs)
        specs.append(ParameterSpec(
            name=f"{r['potfile_id']}.{r['catalog_id']}.velocity_offset", sample_name=name,
            potential_id=str(r['potfile_id']), profile_type=81, field="velocity_offset",
            prior_kind="normal", lower=-np.inf, upper=np.inf, step=.1, mean=0., std=1.,
            component_family="member_scatter_offset", transform_kind="identity",
            physical_mean=0., physical_std=1., sample_site_name="member_velocity_offsets",
            sample_site_index=site_index))
    sigma_index = len(specs)
    specs.append(ParameterSpec(
        name="source.sigma_src", sample_name="source_sigma_src", potential_id="source",
        profile_type=0, field="sigma_src", prior_kind="normal", lower=-np.inf, upper=np.inf,
        step=.1, mean=float(np.log(getattr(args, "source_scatter_prior_median_arcsec", .02))),
        std=float(getattr(args, "source_scatter_prior_log_sigma", .7)),
        component_family="source_scatter", transform_kind="log_positive", physical_lower=0.,
        physical_mean=float(getattr(args, "source_scatter_prior_median_arcsec", .02))))
    metadata = {}
    for row in images.itertuples():
        metadata[str(row.image_label)] = {
            "sigma_obs_arcsec": float(getattr(row, 'sigma_obs_arcsec', getattr(args, 'pos_sigma_arcsec', None) or .5)),
            "parity": int(getattr(row, 'parity', 0)),
        }
    return dict(revision=REVISION, phase=phase, tiers=payload,
                member_ids=list(identities), member_components=list(components),
                scatter_indices=scatter_indices.tolist(), offset_indices=offset_indices.tolist(),
                sigma_src_index=sigma_index, image_metadata=metadata,
                image_metadata_fingerprint=hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                source_prior_mean=[float(getattr(args, 'source_position_prior_mean_x_arcsec', 0.)),
                                   float(getattr(args, 'source_position_prior_mean_y_arcsec', 0.))],
                source_prior_sigma=float(getattr(args, 'source_position_prior_sigma_arcsec', 100.)),
                source_scatter_prior_median_arcsec=float(getattr(args, 'source_scatter_prior_median_arcsec', .02)),
                member_velocity_scatter_prior_median=median)


class SmoothedEvaluatorMixin:
    def _smoothed_member_vdisp(self, v_disp, physical):
        metadata = getattr(self.state, "smoothed_model", None)
        if metadata is None:
            return v_disp
        offsets = jnp.asarray(metadata['offset_indices'], dtype=int)
        scatters = jnp.asarray(metadata['scatter_indices'], dtype=int)
        delta = jnp.where(offsets >= 0, physical[jnp.maximum(offsets, 0)] * physical[jnp.maximum(scatters, 0)], 0.)
        return v_disp * jnp.exp(delta)

    def _initialize_smoothed_model(self):
        from .member_tiers import MemberTierRecord, member_farfield_geometry
        meta = self.state.smoothed_model
        if not isinstance(meta, dict) or meta.get('revision') != REVISION:
            raise ValueError("Missing or incompatible smoothed likelihood metadata.")
        fingerprint = hashlib.sha256(json.dumps(meta['image_metadata'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if meta.get('image_metadata_fingerprint', fingerprint) != fingerprint:
            raise ValueError("Saved image metadata fingerprint does not match its values.")
        if self.sampling_engine != 'tiered_members' or self.use_magnitude_likelihood:
            raise ValueError("Smoothed likelihood requires tiered_members and positions only.")
        if bool(getattr(self.state, 'perturbation_discovery_stage0', False)):
            raise ValueError("Smoothed likelihood cannot use legacy perturbation-discovery warm-up.")
        if self.source_position_parameterization not in {'likelihood-whitened', 'direct'}:
            raise ValueError("Smoothed likelihood requires likelihood-whitened or direct sources.")
        if self.source_plane_covariance_floor != 0.:
            raise ValueError("Smoothed likelihood does not permit a covariance floor.")
        self.smoothed_member_components = np.asarray(meta['member_components'], dtype=int)
        if meta['phase'] == 'initializer':
            far, sampled = (), ()
        elif meta['phase'] == 'fixed':
            tiers = MemberTierRecord.from_dict(meta['tiers'])
            tiers.validate_identity(meta['member_ids'], meta['member_components'])
            far = tuple(sorted(tiers.approximate_sampled + tiers.approximate_absorbed))
            sampled = tuple(sorted(tiers.exact_sampled + tiers.approximate_sampled))
        else:
            raise ValueError("Unrecognized saved smoothed member phase.")
        self.smoothed_far_positions = np.asarray(far, dtype=int)
        self.smoothed_absorbed_positions = np.asarray(sorted(set(range(len(self.smoothed_member_components))) - set(sampled)), dtype=int)
        far_components = set(self.smoothed_member_components[self.smoothed_far_positions].tolist())
        self.smoothed_exact_components = np.asarray([i for i in range(len(self.state.base_components)) if i not in far_components], dtype=int)
        data = self.flat_critical_arc_data
        # Mask before any kernels/JVPs: a later likelihood mask cannot prevent
        # NaN derivatives from an inactive image entering a parameter gradient.
        active = np.asarray(data.image_has_constraint, dtype=bool)
        if not np.all(np.isfinite(np.asarray(data.x_obs)[active])) or not np.all(np.isfinite(np.asarray(data.y_obs)[active])):
            raise ValueError("Active image positions must be finite.")
        data = replace(data, x_obs=jnp.where(data.image_has_constraint, data.x_obs, .123),
                       y_obs=jnp.where(data.image_has_constraint, data.y_obs, -.456))
        labels = np.empty(len(data.x_obs), dtype=object)
        family_map = {str(f.family_id): f for f in self.state.family_data}
        image_families = np.asarray(data.global_family_index_per_image)
        # Flat image indices are local to a redshift bin, which can contain
        # multiple families. Within each family, _build_bin_data preserves its
        # image order, so use the global family membership to place its labels.
        for family_index, family_id in enumerate(data.family_ids):
            rows = np.flatnonzero(image_families == family_index)
            family_labels = family_map[str(family_id)].image_labels
            if len(rows) != len(family_labels):
                raise ValueError(f"Flat image count differs from family {family_id!r} metadata.")
            labels[rows] = family_labels
        errors = np.asarray([meta['image_metadata'][label]['sigma_obs_arcsec'] for label in labels])
        self.smoothed_parity = jnp.asarray([meta['image_metadata'][label]['parity'] for label in labels], dtype=int)
        if not np.all(np.isin(np.asarray(self.smoothed_parity), [-1, 0, 1])):
            raise ValueError("Saved image parity must be -1, 0, or +1.")
        # Parity metadata is immutable. None statically removes one-sided
        # quadrature from the common all-unknown sampling executable.
        self._smoothed_likelihood_parity = (None if np.all(np.asarray(self.smoothed_parity) == 0)
                                             else self.smoothed_parity)
        if not np.all(np.isfinite(errors[active]) & (errors[active] > 0.)):
            raise ValueError("Active astrometric uncertainties must be positive and finite.")
        self.flat_critical_arc_data = replace(data, sigma_per_image=jnp.asarray(np.where(active, errors, 1.)))
        spec = self.state.packed_lens_spec
        member_xy = np.stack((spec.x_center_base[self.smoothed_member_components], spec.y_center_base[self.smoothed_member_components]), -1)
        image_xy = np.stack((np.asarray(data.x_obs), np.asarray(data.y_obs)), -1)
        self.smoothed_geometry = member_farfield_geometry(image_xy, member_xy)
        # Keep compiled batches on this evaluator so repeated reporting calls
        # reuse executables. Reinitializing geometry also invalidates them.
        self._smoothed_reported_physical_batch = jax.jit(
            jax.vmap(self._reported_physical_parameter_vector))
        # Maximum-likelihood reporting excludes source-coordinate prior and
        # Jacobian corrections, which belong only to the sampling target.
        self._smoothed_physical_loglike_batch = jax.jit(jax.vmap(
            lambda theta: jnp.sum(self._smoothed_statistics(theta)['loglike'])))
        self._initialize_smoothed_guard_reference()

    def _smoothed_scaling_diagnostics(self):
        """Catalog bookkeeping only; never run the legacy brightness ranking."""
        import pandas as pd
        columns = ('potfile_id', 'potfile_order', 'component_index', 'catalog_id', 'rank',
                   'free_component_index', 'selected_active', 'selected_independent',
                   'requested_active_count', 'selection_mode', 'importance', 'min_distance_arcsec',
                   'brightness', 'catalog_mag', 'catalog_color', 'x_centre', 'y_centre')
        rows = []
        exact = set(self.smoothed_exact_components.tolist())
        for record in self.state.scaling_component_records:
            row = {key: record.get(key, np.nan) for key in columns}
            component = int(record['component_index'])
            # Reporting uses -1 for unranked rows and expects integer metadata.
            row.update(rank=-1, free_component_index=-1, selected_active=component in exact,
                       selected_independent=False, requested_active_count=0,
                       selection_mode='smoothed_exact' if component in exact else 'smoothed_far_field')
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _smoothed_packed_with_validity(self, physical):
        geometry = {}
        if self.fit_cosmology_flat_wcdm:
            scale, factors = self._sampled_cosmology_geometry_for_physical(physical)
            geometry = dict(kpc_per_arcsec=scale, dpie_sigma0_factors=factors)
        return self._build_flat_packed_lens_state_with_validity_from_physical(
            physical, self.flat_critical_arc_data, stop_gradient=False, **geometry)

    def _smoothed_packed(self, physical):
        return self._smoothed_packed_with_validity(physical)[0]

    def _smoothed_exact_rows(self, xy, packed, components, *, spatial_derivative=False):
        """Batched exact fields, retaining constant catalog geometry separately."""
        from .dpie_derivatives import grouped_dpie_fields_and_third_derivatives
        from .lens_physics import DP_IE_PROFILE, SHEAR_PROFILE

        components = np.asarray(components, dtype=int)
        spec = self.state.packed_lens_spec
        profiles = np.asarray(spec.profile_type)[components]
        unsupported = set(profiles.tolist()) - {DP_IE_PROFILE, SHEAR_PROFILE}
        if unsupported:
            raise ValueError(f"Unsupported smoothed exact profile codes: {sorted(unsupported)}")
        fixed = np.isin(components, self.smoothed_member_components)
        for field in ('x_center', 'y_center', 'e1', 'e2'):
            fixed &= np.asarray(getattr(spec, field + '_param_index'))[components] < 0
        batches, positions = [], []
        for fixed_geometry in (True, False):
            rows = np.flatnonzero((profiles == DP_IE_PROFILE) & (fixed == fixed_geometry))
            if not rows.size:
                continue
            ci = components[rows]
            if fixed_geometry:
                # Keep these constants visible to XLA, while amplitudes and
                # radii retain offsets, scaling relations and redshift factors.
                geometry = tuple(jnp.asarray(getattr(spec, field + '_base')[ci])
                                 for field in ('e1', 'e2', 'x_center', 'y_center'))
            else:
                geometry = (packed.e1[ci], packed.e2[ci], packed.center_x[ci], packed.center_y[ci])
            batches.append(grouped_dpie_fields_and_third_derivatives(
                xy[:, 0], xy[:, 1], packed.sigma0[ci], packed.Ra[ci], packed.Rs[ci], *geometry))
            positions.append(rows)
        rows = np.flatnonzero(profiles == SHEAR_PROFILE)
        if rows.size:
            ci = components[rows]
            g1, g2 = packed.gamma1[ci, None], packed.gamma2[ci, None]
            x, y = xy[:, 0], xy[:, 1]
            alpha = jnp.stack((g1*x + g2*y, g2*x - g1*y), -1)
            g1, g2 = jnp.broadcast_arrays(g1, g2, jnp.ones_like(x)[None, :])[:2]
            u = jnp.stack((jnp.stack((g1, g2), -1), jnp.stack((g2, -g1), -1)), -2)
            batches.append((alpha, u, jnp.zeros(u.shape + (2,), dtype=xy.dtype)))
            positions.append(rows)
        if batches:
            order = np.argsort(np.concatenate(positions))
            fields = tuple(jnp.concatenate(items, axis=0)[order] for items in zip(*batches))
        else:
            fields = tuple(jnp.zeros((0, xy.shape[0]) + shape, dtype=xy.dtype)
                           for shape in ((2,), (2, 2), (2, 2, 2)))
        return fields if spatial_derivative else fields[:2]

    def _smoothed_fields(self, physical, *, force_exact=False):
        from .member_tiers import evaluate_farfield_sampling
        packed = self._smoothed_packed(physical)
        data = self.flat_critical_arc_data
        xy = jnp.stack((data.x_obs, data.y_obs), -1)
        components = np.arange(len(self.state.base_components), dtype=int) if force_exact else self.smoothed_exact_components
        alpha_rows, u_rows, du_rows = self._smoothed_exact_rows(
            xy, packed, components, spatial_derivative=True)
        alpha, u = jnp.sum(alpha_rows, axis=0), jnp.sum(u_rows, axis=0)
        du = jnp.sum(du_rows, axis=0)
        members = self.smoothed_member_components
        n_members, n_images = len(members), xy.shape[0]
        ma = jnp.zeros((n_members, n_images, 2), dtype=xy.dtype)
        mu = jnp.zeros((n_members, n_images, 2, 2), dtype=xy.dtype)
        row_lookup = {int(component): i for i, component in enumerate(components)}
        exact_positions = np.asarray([i for i, c in enumerate(members) if int(c) in row_lookup], dtype=int)
        row_indices = np.asarray([row_lookup[int(members[i])] for i in exact_positions], dtype=int)
        ma = ma.at[exact_positions].set(alpha_rows[row_indices])
        mu = mu.at[exact_positions].set(u_rows[row_indices])
        if not force_exact and len(self.smoothed_far_positions):
            far = self.smoothed_far_positions
            # Geometry contains every population member; slice before contraction.
            geom = jax.tree.map(lambda a: a[far], self.smoothed_geometry)
            ci = members[far]
            fields = evaluate_farfield_sampling(geom, packed.sigma0[ci], packed.Ra[ci], packed.Rs[ci])
            ma = ma.at[far].set(fields.alpha)
            mu = mu.at[far].set(fields.jacobian)
            alpha = alpha + jnp.sum(fields.alpha, axis=0)
            u = u + jnp.sum(fields.jacobian, axis=0)
            du = du + fields.total_jacobian_derivative
        a = jnp.eye(2) - u
        da = -du
        g = (a[:, 1, 1, None] * da[:, 0, 0, :] + a[:, 0, 0, None] * da[:, 1, 1, :]
             - a[:, 1, 0, None] * da[:, 0, 1, :] - a[:, 0, 1, None] * da[:, 1, 0, :])
        scatter_indices = jnp.asarray(self.state.smoothed_model['scatter_indices'], dtype=int)[members]
        scatter = physical[scatter_indices]
        return dict(backprojected=xy-alpha, jacobian=a, determinant_gradient=g,
                    jacobian_spatial_derivative=da, member_alpha=ma, member_jacobian=mu,
                    member_sigma=scatter)

    def _smoothed_statistics(self, params, *, force_exact=False, omit_absorbed=False):
        from . import smoothed_likelihood as sl
        physical = self._physical_parameter_vector(params)
        data = self.flat_critical_arc_data
        fields = self._smoothed_fields(physical, force_exact=force_exact)
        absorbed = np.asarray([], dtype=int) if omit_absorbed else self.smoothed_absorbed_positions
        ma, mu = fields['member_alpha'][absorbed], fields['member_jacobian'][absorbed]
        ms = fields['member_sigma'][absorbed]
        sigma = physical[self.state.smoothed_model['sigma_src_index']]
        cov = sl.covariance_statistics(fields['jacobian'], data.sigma_per_image, sigma,
                                      ma, ms, data.image_has_constraint)
        whitening = sl.source_whitening(fields['backprojected'], cov['covariance'],
            data.global_family_index_per_image, data.n_families, data.image_has_constraint)
        ix, iy = data.global_family_source_x_param_index, data.global_family_source_y_param_index
        coordinates = jnp.stack((params[ix], params[iy]), -1)
        meta = self.state.smoothed_model
        if self.source_position_parameterization == 'likelihood-whitened':
            transported = sl.transport_sources(coordinates, whitening,
                jnp.asarray(meta['source_prior_mean']), meta['source_prior_sigma'])
            source, correction = transported['source'], transported['log_correction']
        else:
            source, correction = jnp.stack((physical[ix], physical[iy]), -1), jnp.asarray(0.)
        stats = sl.statistics(fields['backprojected'], source[data.global_family_index_per_image],
            fields['jacobian'], fields['determinant_gradient'], data.sigma_per_image, sigma,
            ma, mu, ms, data.image_has_constraint, parity=self._smoothed_likelihood_parity,
            jacobian_spatial_derivative=fields['jacobian_spatial_derivative'])
        stats.update(source_by_family=source, source_correction=correction, whitening=whitening,
                     sigma_src=sigma, **{f'all_{key}': fields[key] for key in ('member_alpha','member_jacobian','member_sigma')})
        return stats

    def _smoothed_input_validity(self, params):
        physical = self._physical_parameter_vector(params)
        _, validity = self._smoothed_packed_with_validity(physical)
        outer = validity.is_valid & jnp.all(jnp.isfinite(params))
        scales = physical[self._smoothed_scale_indices]
        safe = jnp.all(jnp.isfinite(scales**2) & (scales > 0.))
        return outer, safe

    def _smoothed_loglike_value(self, params):
        """Complete target contribution; caller must supply safe parameters."""
        stats = self._smoothed_statistics(params)
        value = jnp.sum(stats['loglike']) + stats['source_correction']
        valid = jnp.all(stats['valid']) & jnp.all(stats['whitening']['valid']) & ~jnp.isnan(value)
        return jnp.where(valid, value, -jnp.inf)

    def _initialize_smoothed_guard_reference(self):
        """Validate a fixed fallback in this evaluator's latent coordinates.

        This changes neither priors nor valid-proposal values. Rebuilding the
        evaluator (including after a stage/class change) revalidates the point.
        Never reuse a differentiated packed state from the rejected proposal.
        """
        from .evaluator_support import _default_theta, _reference_theta_from_init_values
        meta = self.state.smoothed_model
        self._smoothed_scale_indices = jnp.asarray(sorted(set(
            [meta['sigma_src_index']] + [i for i in meta['scatter_indices'] if i >= 0])), dtype=int)
        reference = jnp.asarray(_reference_theta_from_init_values(
            self.state.parameter_specs, getattr(self.state, 'svi_init_values', None),
            _default_theta(self.state.parameter_specs)), dtype=jnp.float64)
        def check(theta):
            return self._smoothed_loglike_value(theta), self._smoothed_input_validity(theta)
        (value, flags), gradient = jax.jit(jax.value_and_grad(check, has_aux=True))(reference)
        if not (all(bool(flag) for flag in flags) and np.isfinite(float(value))
                and np.isfinite(np.asarray(gradient)).all()):
            raise ValueError("Cannot initialize the smoothed safe-reference guard: starting values "
                             "must have a valid lens, representable scatter, and finite likelihood "
                             "and gradient. Supply valid stage initialization values.")
        self._smoothed_guard_reference = reference
        # Invalidate executables when geometry/metadata is explicitly rebuilt.
        self._source_loglike_fn = jax.jit(self._source_loglike_impl)

    def _flat_smoothed_loglike_impl(self, params):
        from .sampling_engines import BAD_LOG_LIKE
        outer, safe = self._smoothed_input_validity(params)
        current = jnp.where(outer & safe, params, self._smoothed_guard_reference)
        # Differentiate the physics only at the selected safe vector. Selecting
        # just the output, or reusing the packed rejected state, leaks NaNs.
        value = self._smoothed_loglike_value(current)
        return jnp.where(outer, jnp.where(safe, value, -jnp.inf),
                         jnp.asarray(BAD_LOG_LIKE, dtype=jnp.float64))

    def _smoothed_reported_physical(self, params):
        physical = self._physical_parameter_vector(params)
        if self.source_position_parameterization != 'likelihood-whitened':
            return physical
        source = self._smoothed_statistics(params)['source_by_family']
        data = self.flat_critical_arc_data
        return physical.at[data.global_family_source_x_param_index].set(source[:, 0]).at[data.global_family_source_y_param_index].set(source[:, 1])

    def _smoothed_source_summary(self, params):
        stats = jax.device_get(self._smoothed_statistics(jnp.asarray(params)))
        data = self.flat_critical_arc_data
        output = {}
        for f, family in enumerate(data.family_ids):
            mask = np.asarray(data.global_family_index_per_image) == f
            residual = stats['residual'][mask]
            source = stats['source_by_family'][f]
            radius = np.linalg.norm(residual, axis=-1)
            output[str(family)] = dict(source_x=float(source[0]), source_y=float(source[1]),
                source_beta_x=stats['backprojected'][mask, 0], source_beta_y=stats['backprojected'][mask, 1],
                source_sigma_int_arcsec=float(stats['sigma_src']),
                source_sigma_eff_arcsec=float(np.sqrt(np.mean(np.trace(stats['covariance'][mask],axis1=-2,axis2=-1))/2)),
                source_covariance_arcsec2=stats['covariance'][mask], image_sigma_int_arcsec=0.,
                source_plane_rms=float(np.sqrt(np.mean(radius**2))), residual_max=float(radius.max()),
                x_pred=np.full(mask.sum(), np.nan), y_pred=np.full(mask.sum(), np.nan),
                exact_image_rms=np.nan, failed=not np.all(np.isfinite(residual)))
        return output
