"""Build the model state: parameter specs, lens components, families, images, member sites."""
from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from .config import LensClusterSolverConfig
from .inputs import (catalog_shape_to_ellipticity, declared_model_inputs, filter_image_families,
                     lenstool_ellipticite_to_axis_ratio, lenstool_shape_to_e1e2, radec_to_offsets_arcsec,
                     shear_polar_to_gamma1gamma2)
from .jax_cosmology import dpie_sigma0_factor_from_config, kpc_per_arcsec_from_config
from .model import (COMPONENT_FAMILY_LARGE, COMPONENT_FAMILY_SCALING, DP_IE_PROFILE, DPIE_PROFILE_NAME,
                    SHEAR_PROFILE, SHEAR_PROFILE_NAME, FamilyData, ImageData,
                    MemberModel, ModelState, PackedLensSpec, ParameterSpec, default_theta, physical_to_latent)
from .priors import bounds, physical_center, split

from .profiles import SUPPORTED_PROFILES, profile as lens_profile
SOFTENING_LENGTH_SAMPLE_NAME = "softening_length_kpc"
SOURCE_POSITION_SITE_NAME = "source_position_beta"
MEMBER_OFFSET_SITE_NAME = "member_velocity_offsets"
MEMBER_RADIUS_OFFSET_SITE_NAME = "member_radius_offsets"


# ----------------------------------------------------------------------------- parsing
def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value_f) else value_f


def _coerce_numeric(value: Any, context: str) -> float:
    direct = _safe_float(value)
    if direct is not None:
        return direct
    if isinstance(value, str):
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))
    raise ValueError(f"Could not parse numeric value for {context}: {value!r}")


def sample_name(potential_id: str, field: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", f"{potential_id}_{field}")


def _prior_spec(name: str, sample_name_: str, potential_id: str, field: str, prior, component_family: str, *,
                profile_type: int = 0, **site) -> ParameterSpec:
    return ParameterSpec(name=name, sample_name=sample_name_, potential_id=potential_id, profile_type=profile_type, field=field,
                         prior=prior, component_family=component_family, **site)


def _error_scale_spec(field: str, prior) -> ParameterSpec:
    """A sampled scale of the error model (images.<field>)."""
    return _prior_spec(f"images.{field}", f"images_{field}", "images", field, prior, "error_scale")


def _append_spec(specs, assignments, *, potential_id, profile_type, field_name, prior, component_family="large"):
    assignments.append((field_name, len(specs)))
    specs.append(_prior_spec(f"{potential_id}.{field_name}", sample_name(potential_id, field_name), potential_id, field_name, prior,
                             component_family, profile_type=profile_type))


def _sampled_fields(info) -> set[str]:
    """The fields of a profile that take a prior: the required ones with the shape as its Cartesian pair and, for the
    dPIE, the cut radius as its excess above the effective core."""
    fields = set(info.required_fields) - set(info.polar_shape or ()) | set(info.cartesian_shape or ())
    if "cut_radius_kpc" in fields:
        fields = fields - {"cut_radius_kpc"} | {"cut_excess_kpc"}
    return fields


def build_halo_parameter_specs(potentials: list[dict[str, Any]]):
    """Specs, per-component (field, index) assignments, and lens model names for the large halos."""
    specs: list[ParameterSpec] = []
    assignments_by_component: list[list[tuple[str, int]]] = []
    lens_model_list: list[str] = []
    for potential in potentials:
        potential_id = str(potential["id"])
        profile_type = int(potential["profil"])
        info = lens_profile(profile_type)
        lens_model_list.append(info.name)
        assignments: list[tuple[str, int]] = []
        priors = dict(potential.get("priors", {}) or {})
        unknown = set(priors) - _sampled_fields(info)
        if unknown:
            raise ValueError(f"Potential {potential_id}: no sampled field {sorted(unknown)}; priors take {sorted(_sampled_fields(info))} "
                             "(shapes as their Cartesian pair, see priors.ellipticity_box and priors.shear_box; the dPIE cut radius as "
                             "cut_excess_kpc, above the effective core)")
        shape = info.cartesian_shape or ()
        if shape and len(set(shape) & set(priors)) == 1:
            raise ValueError(f"Potential {potential_id} samples only part of {shape}; both are required.")
        for field_name, prior in priors.items():
            _append_spec(specs, assignments, potential_id=potential_id, profile_type=profile_type, field_name=field_name, prior=prior)
        assignments_by_component.append(assignments)
    return specs, assignments_by_component, lens_model_list


# ----------------------------------------------------------------------------- member scaling
def build_scaling_parameter_specs(potfiles: list[dict[str, Any]], scaling: Any, *, start_index: int = 0):
    """Reference scaling parameters per population: sigma, cutkpc, corekpc, alpha_sigma, beta_radius."""
    specs: list[ParameterSpec] = []
    indices_by_potfile: list[dict[str, int]] = []
    lens_model_list: list[str] = []
    for potfile in potfiles:
        potfile_id = str(potfile["id"])
        if int(potfile["type"]) != DP_IE_PROFILE:
            raise ValueError(f"Unsupported potfile type {potfile['type']} for {potfile_id}.")
        lens_model_list.extend([DPIE_PROFILE_NAME] * len(potfile["catalog_df"]))
        priors = {"sigma": potfile["sigma_prior"], "cutkpc": potfile["cutkpc_prior"], "corekpc": scaling.corekpc_prior,
                  "alpha_sigma": scaling.alpha_sigma_prior, "beta_radius": scaling.beta_radius_prior}
        field_index: dict[str, int] = {}
        fields = ("sigma", "cutkpc", "corekpc", "alpha_sigma", "beta_radius") if getattr(scaling, "fit_corekpc", True) else ("sigma", "cutkpc", "alpha_sigma", "beta_radius")
        fields = tuple(f for f in fields if f not in set(getattr(scaling, "fixed_relation_fields", ())))   # held at the nominal / prior centre
        for field_name in fields:
            field_index[field_name] = start_index + len(specs)
            specs.append(_prior_spec(f"{potfile_id}.{field_name}", sample_name(potfile_id, field_name), potfile_id, field_name,
                                     priors[field_name], "scaling", profile_type=DP_IE_PROFILE))
        indices_by_potfile.append(field_index)
    return specs, indices_by_potfile, lens_model_list


def build_member_components(potfiles, reference, scaling_param_indices, *, start_component_index, scaling):
    """Member dPIE components on the scaling relations, with their packed assignments and records."""
    _, ra0_deg, dec0_deg = reference
    components, assignments, scaling_assignments, records = [], [], [], []
    for potfile_order, (potfile, param_index) in enumerate(zip(potfiles, scaling_param_indices)):
        catalog_df = potfile["catalog_df"]
        if catalog_df.empty:
            continue
        core_radius_kpc = float(potfile.get("corekpc_nominal", potfile["corekpc"]))   # at the pivot, like sigma and cutkpc
        cut_radius_kpc = float(potfile["cutkpc_nominal"])
        x_offsets, y_offsets = radec_to_offsets_arcsec(catalog_df["ra"].to_numpy(dtype=float),
                                                       catalog_df["dec"].to_numpy(dtype=float), ra0_deg, dec0_deg)
        magnitudes = catalog_df["catalog_mag"].to_numpy(dtype=float)
        luminosity_ratio = np.power(10.0, -0.4 * (magnitudes - float(potfile["mag0"])))
        for row_index, row in enumerate(catalog_df.itertuples(index=False)):
            ellipticite, angle_pos = catalog_shape_to_ellipticity(row.catalog_a, row.catalog_b, row.catalog_theta)
            component_index = start_component_index + len(components)
            components.append({
                "id": f"{potfile['id']}.{row.id}", "profil": DP_IE_PROFILE,
                "x_centre": float(x_offsets[row_index]), "y_centre": float(y_offsets[row_index]),
                "ellipticite": ellipticite, "angle_pos": angle_pos,
                "core_radius_kpc": core_radius_kpc, "cut_radius_kpc": cut_radius_kpc,
                "v_disp": float(potfile["sigma_nominal"]), "z_lens": float(potfile["zlens"]),
                "component_family_code": COMPONENT_FAMILY_SCALING,
            })
            assignments.append([])
            scaling_assignments.append({
                "component_index": component_index, "luminosity_ratio": float(luminosity_ratio[row_index]),
                "sigma_ref_base": float(potfile["sigma_nominal"]), "cut_ref_base": cut_radius_kpc, "core_ref_base": core_radius_kpc,
                "alpha_sigma_base": physical_center(scaling.alpha_sigma_prior), "beta_radius_base": physical_center(scaling.beta_radius_prior),
                "sigma_ref_param_index": int(param_index.get("sigma", -1)), "cut_ref_param_index": int(param_index.get("cutkpc", -1)),
                "core_ref_param_index": int(param_index.get("corekpc", -1)),
                "alpha_sigma_param_index": int(param_index.get("alpha_sigma", -1)),
                "beta_radius_param_index": int(param_index.get("beta_radius", -1)),
            })
            records.append({
                "potfile_id": str(potfile["id"]), "potfile_order": int(potfile_order), "catalog_row_index": int(row_index),
                "component_index": int(component_index), "catalog_id": str(row.id), "catalog_mag": float(row.catalog_mag),
                "keep": bool(getattr(row, "catalog_keep", False)), "role": str(getattr(row, "catalog_role", "member")),
                "kinematics": list(getattr(row, "catalog_kinematics", None) or []),
                "luminosity_ratio": float(luminosity_ratio[row_index]),
                "x_centre": float(x_offsets[row_index]), "y_centre": float(y_offsets[row_index]),
                "ellipticite": float(ellipticite), "angle_pos": float(angle_pos),
            })
    return components, assignments, scaling_assignments, records


def _field_param_index(assignments: list[tuple[str, int]], field: str) -> int:
    for name, index in assignments:
        if name == field:
            return int(index)
    return -1


def build_packed_lens_spec(base_components, component_assignments, scaling_assignments, scaling) -> PackedLensSpec:
    n = len(base_components)
    ints = lambda: np.full(n, -1, dtype=np.int32)
    floats = lambda value=0.0: np.full(n, float(value), dtype=float)
    spec = PackedLensSpec(
        profile_type=np.zeros(n, dtype=np.int32), component_family=np.zeros(n, dtype=np.int32),
        x_center_base=floats(), y_center_base=floats(), e1_base=floats(), e2_base=floats(),
        core_radius_kpc_base=floats(), cut_radius_kpc_base=floats(), v_disp_base=floats(),
        gamma1_base=floats(), gamma2_base=floats(),
        x_center_param_index=ints(), y_center_param_index=ints(), e1_param_index=ints(), e2_param_index=ints(),
        core_radius_param_index=ints(), cut_radius_param_index=ints(), v_disp_param_index=ints(),
        gamma1_param_index=ints(), gamma2_param_index=ints(),
        luminosity_ratio=np.ones(n, dtype=float), sigma_ref_base=floats(), cut_ref_base=floats(), core_ref_base=floats(),
        alpha_sigma_base=floats(physical_center(scaling.alpha_sigma_prior)), beta_radius_base=floats(physical_center(scaling.beta_radius_prior)),
        sigma_ref_param_index=ints(), cut_ref_param_index=ints(), core_ref_param_index=ints(),
        alpha_sigma_param_index=ints(), beta_radius_param_index=ints(),
        cut_radius_excess_above_effective_core=np.zeros(n, dtype=bool),
        velocity_offset_param_index=ints(), velocity_scatter_param_index=ints(),
        scale_radius_kpc_base=floats(), log10_m200_base=floats(), scale_radius_param_index=ints(), log10_m200_param_index=ints(),
        kappa_base=floats(), kappa_param_index=ints())
    spec.ellipticity_soft_cap, spec.ellipticity_soft_width = floats(np.inf), floats(1.0)
    for idx, (component, assignments) in enumerate(zip(base_components, component_assignments)):
        if component.get("ellipticity_soft_cap") is not None:
            q = float(lenstool_ellipticite_to_axis_ratio(float(component["ellipticity_soft_cap"])))
            spec.ellipticity_soft_cap[idx] = (1.0 - q) / (1.0 + q)                       # Lenstool ellipticite -> modulus |e|
            spec.ellipticity_soft_width[idx] = float(component.get("ellipticity_soft_width", 0.02))
        spec.profile_type[idx] = int(component["profil"])
        spec.component_family[idx] = int(component.get("component_family_code", COMPONENT_FAMILY_LARGE))
        spec.x_center_base[idx] = float(component.get("x_centre", 0.0))
        spec.y_center_base[idx] = float(component.get("y_centre", 0.0))
        angle_pos = float(component.get("angle_pos", 0.0))
        spec.e1_base[idx], spec.e2_base[idx] = lenstool_shape_to_e1e2(float(component.get("ellipticite", 0.0)), angle_pos)
        spec.gamma1_base[idx], spec.gamma2_base[idx] = shear_polar_to_gamma1gamma2(float(component.get("gamma", 0.0)), angle_pos)
        spec.core_radius_kpc_base[idx] = float(component.get("core_radius_kpc", 0.0))
        spec.cut_radius_kpc_base[idx] = float(component.get("cut_radius_kpc", 0.0))
        spec.v_disp_base[idx] = float(component.get("v_disp", 0.0))
        spec.scale_radius_kpc_base[idx] = float(component.get("scale_radius_kpc", 0.0))
        spec.log10_m200_base[idx] = float(component.get("log10_m200", 0.0))
        spec.kappa_base[idx] = float(component.get("kappa", 0.0))
        for field_name, array in (("x_centre", spec.x_center_param_index), ("y_centre", spec.y_center_param_index),
                                  ("e1", spec.e1_param_index), ("e2", spec.e2_param_index),
                                  ("core_radius_kpc", spec.core_radius_param_index), ("cut_excess_kpc", spec.cut_radius_param_index),
                                  ("v_disp", spec.v_disp_param_index), ("gamma1", spec.gamma1_param_index), ("gamma2", spec.gamma2_param_index),
                                  ("scale_radius_kpc", spec.scale_radius_param_index), ("log10_m200", spec.log10_m200_param_index),
                                  ("kappa", spec.kappa_param_index)):
            array[idx] = _field_param_index(assignments, field_name)
    for item in scaling_assignments:
        idx = int(item["component_index"])
        spec.component_family[idx] = COMPONENT_FAMILY_SCALING
        spec.luminosity_ratio[idx] = float(item["luminosity_ratio"])
        for key in ("sigma_ref_base", "cut_ref_base", "core_ref_base", "alpha_sigma_base", "beta_radius_base",
                    "sigma_ref_param_index", "cut_ref_param_index", "core_ref_param_index",
                    "alpha_sigma_param_index", "beta_radius_param_index"):
            getattr(spec, key)[idx] = item[key]
    spec.cut_radius_excess_above_effective_core[:] = (
        (spec.profile_type == DP_IE_PROFILE) & (spec.component_family != COMPONENT_FAMILY_SCALING) & (spec.cut_radius_param_index >= 0))
    return spec


# ----------------------------------------------------------------------------- families and images
def build_family_data(images_df: pd.DataFrame, sigma_inst_arcsec: float, reference: tuple[int, float, float]) -> list[FamilyData]:
    _, ra0_deg, dec0_deg = reference
    families: list[FamilyData] = []
    for family_id in sorted(images_df["family_id"].astype(str).unique().tolist()):
        family_df = images_df[images_df["family_id"].astype(str) == family_id].copy()
        z_values = pd.to_numeric(family_df["catalog_z"], errors="coerce")
        if z_values.isna().any():
            raise ValueError(f"Family {family_id} has missing or invalid catalog_z values.")
        unique_z = np.unique(np.round(z_values.to_numpy(dtype=float), 8))
        if len(unique_z) != 1:
            raise ValueError(f"Family {family_id} has inconsistent catalog_z values {unique_z.tolist()}.")
        z_source = float(unique_z[0])
        x_obs, y_obs = radec_to_offsets_arcsec(family_df["ra"].to_numpy(), family_df["dec"].to_numpy(), ra0_deg, dec0_deg)
        families.append(FamilyData(
            family_id=family_id, z_source=z_source, sigma_inst_arcsec=float(sigma_inst_arcsec),
            image_labels=family_df["image_label"].astype(str).tolist(),
            x_obs=np.asarray(x_obs, dtype=float), y_obs=np.asarray(y_obs, dtype=float),
            reliability=np.asarray(pd.to_numeric(family_df.get("family_reliability", 1.0), errors="coerce")
                                   .fillna(1.0).clip(0.0, 1.0).to_numpy(dtype=float))))
    return families


def build_image_data(families: list[FamilyData], images_df: pd.DataFrame, z_values: list[float],
                     source_param_indices: dict[str, tuple[int, int]]) -> ImageData:
    """Flatten the images grouped by source redshift, then family order, then catalog order."""
    by_label = images_df.set_index(images_df["image_label"].astype(str))
    ordered = sorted(range(len(families)), key=lambda i: (families[i].z_source, i))
    xs, ys, sigmas, parities, zi, fi, labels = [], [], [], [], [], [], []
    family_ids, sx, sy = [], [], []
    system_of_family = {str(r["family_id"]): str(r["system_id"]) for _, r in images_df.iterrows()} if "system_id" in images_df.columns else {}
    system_ids: list[str] = []
    si: list[int] = []
    z_index = {float(z): i for i, z in enumerate(z_values)}
    for new_index, family_index in enumerate(ordered):
        family = families[family_index]
        family_ids.append(family.family_id)
        ix, iy = source_param_indices[family.family_id]
        sx.append(int(ix)); sy.append(int(iy))
        system = system_of_family.get(family.family_id, family.family_id)
        if system not in system_ids:
            system_ids.append(system)
        for label, x, y in zip(family.image_labels, family.x_obs, family.y_obs):
            si.append(system_ids.index(system))
            row = by_label.loc[label]
            sigma = float(row["sigma_inst_arcsec"]) if "sigma_inst_arcsec" in by_label.columns else float(family.sigma_inst_arcsec)
            parity = int(row["parity"]) if "parity" in by_label.columns and np.isfinite(float(row["parity"])) else 0
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"Image {label} has a nonpositive astrometric uncertainty.")
            if parity not in (-1, 0, 1):
                raise ValueError(f"Image {label} parity must be -1, 0, or +1.")
            xs.append(float(x)); ys.append(float(y)); sigmas.append(sigma); parities.append(parity)
            zi.append(z_index[float(family.z_source)]); fi.append(new_index); labels.append(str(label))
    return ImageData(
        family_ids=tuple(family_ids), n_families=len(family_ids),
        x_obs=jnp.asarray(xs, dtype=jnp.float64), y_obs=jnp.asarray(ys, dtype=jnp.float64),
        sigma_inst_per_image=jnp.asarray(sigmas, dtype=jnp.float64), parity=jnp.asarray(parities, dtype=jnp.int32),
        z_index_per_image=jnp.asarray(zi, dtype=jnp.int32), family_index_per_image=jnp.asarray(fi, dtype=jnp.int32),
        source_x_param_index=jnp.asarray(sx, dtype=jnp.int32), source_y_param_index=jnp.asarray(sy, dtype=jnp.int32),
        image_labels=tuple(labels), system_ids=tuple(system_ids), system_index_per_image=jnp.asarray(si, dtype=jnp.int32))


def source_sample_name(family_id: str, axis: str) -> str:
    safe = str(family_id).replace(".", "_").replace("-", "_")
    return f"source_{safe}_beta_{axis}"


def build_member_kinematics(records: list[dict[str, Any]], populations) -> dict[str, Any]:
    """The measured dispersions of the members on the scaling relations, from the sources each population turns on:
    component index, ln sigma_meas, the total log error e_i = sqrt((err_i / sigma_meas)^2 + systematic^2) and the
    aperture factor per measurement.  A member may have one measurement among the sources turned on (two would count the
    same galaxy twice); a source turned on but found on no member is an error.  Empty when no source is on."""
    by_id = {str(p.id): p for p in populations}
    rows, found = [], {pid: set() for pid in by_id}
    for record in records:
        population = by_id[record["potfile_id"]]
        used = [m for m in record.get("kinematics", []) if m["source"] in population.kinematics_sources]
        if len(used) > 1:
            raise ValueError(f"member {record['potfile_id']}:{record['catalog_id']} has {len(used)} measurements among "
                             f"kinematics_sources {list(population.kinematics_sources)}; turn on one source per galaxy.")
        for m in used:
            found[record["potfile_id"]].add(m["source"])
            rows.append((record, m, population))
    for pid, population in by_id.items():
        missing = sorted(set(population.kinematics_sources) - found[pid])
        if missing:
            raise ValueError(f"member population {pid!r}: kinematics_sources {missing} are on no member on the scaling relation.")
    if not rows:
        return {}
    return dict(member_ids=tuple(f"{r['potfile_id']}:{r['catalog_id']}" for r, _, _ in rows),
                sources=tuple(m["source"] for _, m, _ in rows),
                component_index=np.asarray([int(r["component_index"]) for r, _, _ in rows], dtype=np.int32),
                sigma=np.asarray([float(m["sigma"]) for _, m, _ in rows]),
                err=np.asarray([float(m["err"]) for _, m, _ in rows]),
                log_err=np.asarray([math.hypot(float(m["err"]) / float(m["sigma"]), float(p.kinematics_systematic)) for _, m, p in rows]),
                aperture_factor=np.asarray([float(p.kinematics_aperture_factor) for _, _, p in rows]))


# ----------------------------------------------------------------------------- state assembly
def build_state(config: LensClusterSolverConfig, *, exact_member_positions: np.ndarray | None = None, drop_member_ids=(),
                classification: dict[str, Any] | None = None) -> ModelState:
    """Build the full model state. Every member is sampled; classes default to all exact."""
    model_config = config.model
    potentials, potfiles, images_df, cosmo_config, reference = declared_model_inputs(model_config, config.scaling)
    # Members removed after the pivot and the priors were set from the full catalog (members.min_deflection_arcsec).
    dropped = tuple(str(i) for i in drop_member_ids)
    if dropped:
        for potfile in potfiles:
            catalog_df = potfile["catalog_df"]
            keep = ~catalog_df["id"].astype(str).isin({d.split(":", 1)[1] if ":" in d else d for d in dropped if d.split(":", 1)[0] in (str(potfile["id"]), d)})
            potfile["catalog_df"] = catalog_df.loc[keep].copy().reset_index(drop=True)
    images_df, _ = filter_image_families(images_df)
    z_lens = float(next(float(p["z_lens"]) for p in potentials if p.get("z_lens") is not None))
    kpc_per_arcsec = kpc_per_arcsec_from_config(z_lens, cosmo_config)
    scaling = config.scaling
    sigma_inst_arcsec = float(config.likelihood.sigma_inst_arcsec if config.likelihood.sigma_inst_arcsec is not None
                              else model_config.image_constraints.sigma_inst_arcsec)
    if config.likelihood.sigma_inst_arcsec is not None:
        # The likelihood override replaces every image's astrometric error, the catalog's included (floor still applies).
        images_df["sigma_inst_arcsec"] = max(sigma_inst_arcsec, float(model_config.image_constraints.sigma_floor_arcsec))

    specs, component_assignments, lens_model_list = build_halo_parameter_specs(potentials)
    base_components = [{k: v for k, v in potential.items() if k != "priors"} for potential in potentials]
    scaling_specs, scaling_param_indices, scaling_models = build_scaling_parameter_specs(potfiles, scaling, start_index=len(specs))
    for spec_item in scaling_specs:
        specs.append(spec_item)
    components, assignments, scaling_assignments, records = build_member_components(
        potfiles, reference, scaling_param_indices, start_component_index=len(base_components), scaling=scaling)
    base_components.extend(components)
    component_assignments.extend(assignments)
    lens_model_list.extend(scaling_models)

    # Member hierarchy: population scatter(s), one sampled offset per member (and per radius in sigma_rcut mode), source scatter.
    n_components = len(base_components)
    deviation_mode = str(scaling.member_deviation)
    scatter_index = np.full(n_components, -1, dtype=np.int32)
    offset_index = np.full(n_components, -1, dtype=np.int32)
    radius_scatter_index = np.full(n_components, -1, dtype=np.int32)
    radius_offset_index = np.full(n_components, -1, dtype=np.int32)
    scatter_specs = [("velocity_log_scatter", scaling.member_velocity_scatter_prior, scatter_index)]
    if deviation_mode in ("sigma_rcut", "mass_rcut"):
        scatter_specs.append(("radius_log_scatter", scaling.member_radius_scatter_prior, radius_scatter_index))
    for field_name, member_prior, index_array in scatter_specs:
        scatter_by_population: dict[str, int] = {}
        for record in records:
            population = str(record["potfile_id"])
            if population not in scatter_by_population:
                scatter_by_population[population] = len(specs)
                specs.append(_prior_spec(f"{population}.{field_name}", f"{population}_{field_name}", population, field_name,
                                         member_prior, "member_velocity_scatter", profile_type=DP_IE_PROFILE))
            index_array[int(record["component_index"])] = scatter_by_population[population]
    offset_specs = [("velocity_offset", MEMBER_OFFSET_SITE_NAME, offset_index, scatter_index)]
    if deviation_mode in ("sigma_rcut", "mass_rcut"):
        offset_specs.append(("radius_offset", MEMBER_RADIUS_OFFSET_SITE_NAME, radius_offset_index, radius_scatter_index))
    heavy_tail = str(getattr(scaling, "member_deviation_tail", "student_t")) == "student_t"
    import numpyro.distributions as dist
    unit_prior = dist.StudentT(float(getattr(scaling, "member_deviation_dof", 4.0)), 0.0, 1.0) if heavy_tail else dist.Normal(0.0, 1.0)
    for field_name, site_name, index_array, scale_index in offset_specs:
        for site_index, record in enumerate(records):
            component = int(record["component_index"])
            index_array[component] = len(specs)
            specs.append(_prior_spec(f"{record['potfile_id']}.{record['catalog_id']}.{field_name}",
                                     f"{record['potfile_id']}_{record['catalog_id']}_{field_name}", str(record["potfile_id"]), field_name,
                                     unit_prior, "member_scatter_offset", profile_type=DP_IE_PROFILE, sample_site_name=site_name,
                                     sample_site_index=site_index, centered=False, scale_site_name=specs[int(scale_index[component])].sample_name))
    # Member shapes: one unit offset per member and component of the ellipticity (ScalingModelConfig.member_shape_scatter).
    shape_e1_index = np.full(n_components, -1, dtype=np.int32)
    shape_e2_index = np.full(n_components, -1, dtype=np.int32)
    shape_scatter = scaling.member_shape_scatter
    if shape_scatter is not None:
        for field_name, index_array in (("shape_e1_offset", shape_e1_index), ("shape_e2_offset", shape_e2_index)):
            for site_index, record in enumerate(records):
                component = int(record["component_index"])
                index_array[component] = len(specs)
                specs.append(_prior_spec(f"{record['potfile_id']}.{record['catalog_id']}.{field_name}",
                                         f"{record['potfile_id']}_{record['catalog_id']}_{field_name}", str(record["potfile_id"]),
                                         field_name, dist.Normal(0.0, 1.0), "member_shape_offset", profile_type=DP_IE_PROFILE,
                                         sample_site_name=f"member_{field_name}", sample_site_index=site_index))
    sigma_def_index = sigma_extra_index = sigma_sel_index = lambda_sel_index = sigma_shared_index = sigma_white_index = gp_ell_index = -1
    gp_sigma_alpha_index = gp_knee_index = gp_knee_gamma_index = -1
    gp_sigma_alpha2_index = gp_knee2_index = gp_knee_beta_low_index = -1
    likelihood = config.likelihood
    spectrum = str(getattr(likelihood, "gp_knee_spectrum", "knee"))
    if likelihood.likelihood_mode == "gp_image_plane" and likelihood.gp_kernel == "knee":
        # Knee kernel: the mass field (sigma_alpha, lambda_t; optional) and the image-plane GP (sigma_shared, ell, sigma_white).
        if likelihood.gp_sigma_alpha_prior is not None:
            gp_sigma_alpha_index = len(specs)
            specs.append(_error_scale_spec("gp_sigma_alpha", likelihood.gp_sigma_alpha_prior))
            if spectrum != "power_law" and likelihood.gp_knee_fixed_arcsec is None:   # a power law has no knee; or it is fixed
                gp_knee_index = len(specs)
                specs.append(_error_scale_spec("gp_knee", likelihood.gp_knee_prior))
            if spectrum in ("two_knee", "knee_foreground", "knee_background"):
                gp_sigma_alpha2_index = len(specs)
                specs.append(_error_scale_spec("gp_sigma_alpha2", likelihood.gp_knee2_sigma_alpha_prior))
                gp_knee2_index = len(specs)
                specs.append(_error_scale_spec("gp_knee2", likelihood.gp_knee2_prior))
            if spectrum == "broken_power_law":
                gp_knee_beta_low_index = len(specs)
                specs.append(_error_scale_spec("gp_knee_beta_low", likelihood.gp_knee_beta_low_prior))
            if likelihood.gp_knee_gamma_prior is not None:
                gp_knee_gamma_index = len(specs)
                specs.append(_error_scale_spec("gp_knee_gamma", likelihood.gp_knee_gamma_prior))
        sigma_shared_index = len(specs)
        specs.append(_error_scale_spec("sigma_shared", likelihood.gp_knee_sigma_shared_prior))
        if likelihood.gp_knee_ell_fixed_arcsec is None:
            gp_ell_index = len(specs)
            specs.append(_error_scale_spec("gp_ell", likelihood.gp_knee_ell_prior))
        sigma_white_index = len(specs)
        specs.append(_error_scale_spec("sigma_white", likelihood.gp_knee_sigma_white_prior))
    elif likelihood.likelihood_mode == "gp_image_plane":
        # Two sampled image-plane scales; the source-plane floors and the selection term do not exist in this model.
        sigma_shared_index = len(specs)
        specs.append(_error_scale_spec("sigma_shared", config.likelihood.sigma_shared_prior))
        sigma_white_index = len(specs)
        specs.append(_error_scale_spec("sigma_white", likelihood.sigma_white_prior))
        if config.likelihood.gp_correlation_length_prior is not None:
            gp_ell_index = len(specs)
            specs.append(_error_scale_spec("gp_ell", config.likelihood.gp_correlation_length_prior))
    else:
        sigma_def_index = len(specs)
        specs.append(_error_scale_spec("sigma_def", config.likelihood.sigma_def_prior))
        if config.likelihood.sigma_extra_prior is not None:
            sigma_extra_index = len(specs)
            specs.append(_error_scale_spec("sigma_extra", config.likelihood.sigma_extra_prior))
        if config.likelihood.sigma_sel_prior is not None:
            sigma_sel_index = len(specs)
            specs.append(_error_scale_spec("sigma_sel", config.likelihood.sigma_sel_prior))
            lambda_sel_index = len(specs)
            specs.append(_error_scale_spec("lambda_sel", config.likelihood.lambda_sel_prior))

    families = build_family_data(images_df, sigma_inst_arcsec, reference)
    source_param_indices: dict[str, tuple[int, int]] = {}
    prior_mean = tuple(float(v) for v in config.likelihood.source_position_prior_mean_arcsec)
    prior_sigma = float(config.likelihood.source_position_prior_sigma_arcsec)
    site_index = 0
    for family in families:
        pair = []
        for axis, center in zip(("x", "y"), prior_mean):
            pair.append(len(specs))
            # The whitened offset of the source from its conditional mean (smoothed_likelihood.transport_sources).
            specs.append(_prior_spec(f"source.{family.family_id}.beta_{axis}", source_sample_name(family.family_id, axis),
                                     str(family.family_id), f"beta_{axis}", dist.Normal(0.0, 1.0), "source_position",
                                     sample_site_name=SOURCE_POSITION_SITE_NAME, sample_site_index=site_index))
            site_index += 1
        source_param_indices[family.family_id] = (pair[0], pair[1])

    softening_index = -1
    if scaling.softening_length_prior is not None:
        softening_index = len(specs)
        specs.append(_prior_spec(f"softening_length.{SOFTENING_LENGTH_SAMPLE_NAME}", SOFTENING_LENGTH_SAMPLE_NAME, "softening_length",
                                 SOFTENING_LENGTH_SAMPLE_NAME, scaling.softening_length_prior, "softening_length"))

    packed = build_packed_lens_spec(base_components, component_assignments, scaling_assignments, scaling)
    packed.velocity_offset_param_index = offset_index
    packed.velocity_scatter_param_index = scatter_index
    packed.radius_offset_param_index = radius_offset_index
    packed.radius_scatter_param_index = radius_scatter_index
    packed.deviation_centered = np.zeros(n_components, dtype=bool)
    packed.deviation_mode = deviation_mode
    packed.radius_deviation_members = str(scaling.radius_deviation_members)
    packed.radius_deviation_active = np.ones(n_components, dtype=bool)
    packed.shape_e1_offset_param_index = shape_e1_index
    packed.shape_e2_offset_param_index = shape_e2_index
    packed.shape_scatter = float(shape_scatter or 0.0)
    packed.shape_active = shape_e1_index >= 0

    z_values = sorted({float(family.z_source) for family in families})
    factor_by_z = {float(z): float(dpie_sigma0_factor_from_config(z_lens, float(z), cosmo_config)) for z in z_values}
    images = build_image_data(families, images_df, z_values, source_param_indices)

    member_components = np.asarray([int(r["component_index"]) for r in records], dtype=np.int32)
    n_members = len(records)
    exact = np.arange(n_members, dtype=np.int32) if exact_member_positions is None else np.asarray(sorted(set(int(i) for i in exact_member_positions)), dtype=np.int32)
    far = np.asarray(sorted(set(range(n_members)) - set(exact.tolist())), dtype=np.int32)
    members = MemberModel(
        member_ids=tuple(f"{r['potfile_id']}:{r['catalog_id']}" for r in records),
        component_indices=member_components,
        scatter_param_index=scatter_index[member_components] if n_members else np.asarray([], dtype=np.int32),
        offset_param_index=offset_index[member_components] if n_members else np.asarray([], dtype=np.int32),
        radius_scatter_param_index=radius_scatter_index[member_components] if n_members else np.asarray([], dtype=np.int32),
        radius_offset_param_index=radius_offset_index[member_components] if n_members else np.asarray([], dtype=np.int32),
        centered=np.zeros(n_members, dtype=bool),
        exact_positions=exact, far_positions=far,
        source_prior_mean=prior_mean, source_prior_sigma=prior_sigma,
        far_field_guard_cut_radii=float(config.members.far_field_cut_radii),
        classification=dict(classification or {}),
        far_field_barrier_width=float(config.members.far_field_barrier_width),
        far_field_barrier_strength=float(config.members.far_field_barrier_strength))
    state = ModelState(
        run_name=str(config.paths.run_name or "cluster"), cosmo_config=cosmo_config, z_lens=z_lens,
        kpc_per_arcsec=float(kpc_per_arcsec), reference=reference, parameter_specs=specs,
        base_components=base_components, packed_lens_spec=packed, lens_model_list=lens_model_list,
        family_data=families, images=images, potfiles=potfiles, scaling_component_records=records,
        members=members, z_source_values=z_values, dpie_sigma0_factor_by_z=factor_by_z,
        softening_length_param_index=int(softening_index), sigma_def_param_index=int(sigma_def_index),
        sigma_extra_param_index=int(sigma_extra_index), sigma_sel_param_index=int(sigma_sel_index),
        lambda_sel_param_index=int(lambda_sel_index), sigma_shared_param_index=int(sigma_shared_index), sigma_white_param_index=int(sigma_white_index),
        gp_ell_param_index=int(gp_ell_index), gp_correlation_length_arcsec=float(config.likelihood.gp_correlation_length_arcsec),
        gp_floor_arcsec=float(config.likelihood.gp_floor_arcsec),
        gp_kernel=str(config.likelihood.gp_kernel), gp_reference_redshift=float(config.likelihood.gp_reference_redshift),
        gp_blocks=str(config.likelihood.gp_blocks),
        gp_knee=dict(gamma=float(config.likelihood.gp_knee_gamma), project=tuple(str(m) for m in config.likelihood.gp_knee_project_modes),
                     spectrum=spectrum, plane_redshift=float(getattr(config.likelihood, "gp_knee2_plane_redshift", 0.7)), knee_fixed=config.likelihood.gp_knee_fixed_arcsec, ell_fixed=config.likelihood.gp_knee_ell_fixed_arcsec,
                     beta_low_grid=_table_grid(config.likelihood.gp_knee_beta_low_prior, config.likelihood.gp_knee_beta_low_step),
                     gamma_grid=(None if config.likelihood.gp_knee_gamma_prior is None
                                 else _table_grid(config.likelihood.gp_knee_gamma_prior, config.likelihood.gp_knee_gamma_step)),
                     scales=tuple(float(v) for v in config.likelihood.gp_knee_scales_arcsec),
                     lambda_grid=(float(config.likelihood.gp_knee_lambda_grid[0]), float(config.likelihood.gp_knee_lambda_grid[1]),
                                  int(config.likelihood.gp_knee_lambda_grid[2])), r_step=float(config.likelihood.gp_knee_r_step_arcsec)),
        gp_sigma_alpha_param_index=int(gp_sigma_alpha_index), gp_knee_param_index=int(gp_knee_index),
        gp_knee_gamma_param_index=int(gp_knee_gamma_index), gp_sigma_alpha2_param_index=int(gp_sigma_alpha2_index),
        gp_knee2_param_index=int(gp_knee2_index), gp_knee_beta_low_param_index=int(gp_knee_beta_low_index),
        freeze_determinant_gradient=bool(config.likelihood.freeze_determinant_gradient),
        dropped_member_ids=dropped,
        member_kinematics=build_member_kinematics(records, model_config.member_populations))
    # Starts: the prior centres, and the configured starts of the error model (physical values).
    likelihood = config.likelihood
    starts = {sigma_shared_index: likelihood.sigma_shared_start_arcsec, sigma_white_index: likelihood.sigma_white_start_arcsec,
              gp_ell_index: likelihood.gp_correlation_length_arcsec}
    if likelihood.likelihood_mode == "gp_image_plane" and likelihood.gp_kernel == "knee":
        starts = {gp_sigma_alpha_index: likelihood.gp_sigma_alpha_start_arcsec, gp_knee_index: likelihood.gp_knee_start_arcsec,
                  gp_sigma_alpha2_index: likelihood.gp_knee2_sigma_alpha_start_arcsec, gp_knee2_index: likelihood.gp_knee2_start_arcsec,
                  gp_knee_beta_low_index: likelihood.gp_knee_beta_low_start, gp_knee_gamma_index: likelihood.gp_knee_gamma,
                  sigma_shared_index: likelihood.gp_knee_sigma_shared_start_arcsec, gp_ell_index: likelihood.gp_knee_ell_start_arcsec,
                  sigma_white_index: likelihood.gp_knee_sigma_white_start_arcsec}
    state.initial_theta = default_theta(specs)
    for index, value in starts.items():
        if index >= 0:
            state.initial_theta[index] = physical_to_latent(float(value), specs[index])
    return with_member_classes(state, exact, classification, centered=np.zeros(n_members, dtype=bool))


def _table_grid(prior, step: float) -> tuple[float, float, int]:
    """(low, high, count): a tabulation grid over the bounds of a prior at spacing ``step``."""
    low, high = bounds(split(prior)[0])
    return float(low), float(high), int(round((float(high) - float(low)) / float(step))) + 1


def _current_centering(state: ModelState) -> np.ndarray:
    n_members = len(state.members.member_ids)
    return np.asarray(state.members.centered, dtype=bool) if len(state.members.centered) else np.zeros(n_members, dtype=bool)


def convert_member_centering(state: ModelState, theta: np.ndarray, centered: np.ndarray) -> np.ndarray:
    """Latent vector(s) ``theta`` (shape (n,) or (k, n)) moved from the state's current member centering to ``centered``,
    so that every member's physical deviation, and so the model, is unchanged: a unit offset z becomes the deviation
    s z when the member becomes centered, and back.  Each row uses its own sampled scatter s."""
    old_centered = _current_centering(state)
    centered = np.asarray(centered, dtype=bool)
    out = np.array(theta, dtype=float, copy=True)
    for position in np.flatnonzero(centered != old_centered):
        for offset_idx, scatter_idx in ((state.members.offset_param_index[position], state.members.scatter_param_index[position]),
                                        (state.members.radius_offset_param_index[position], state.members.radius_scatter_param_index[position])):
            if int(offset_idx) < 0:
                continue
            scatter = np.exp(out[..., int(scatter_idx)])              # latent log scatter -> physical scatter
            out[..., int(offset_idx)] = out[..., int(offset_idx)] * scatter if centered[position] else out[..., int(offset_idx)] / scatter
    return out


def with_member_classes(state: ModelState, exact_positions: np.ndarray, classification: dict[str, Any] | None = None, *,
                        centered: np.ndarray | None = None) -> ModelState:
    """Set the exact/far-field classes and, if given, which members use centered deviations; the initial theta
    is converted between centered and unit-offset form so that the physical model is unchanged."""
    n_members = len(state.members.member_ids)
    exact = np.asarray(sorted(set(int(i) for i in exact_positions)), dtype=np.int32)
    far = np.asarray(sorted(set(range(n_members)) - set(exact.tolist())), dtype=np.int32)
    old_centered = _current_centering(state)
    centered = old_centered.copy() if centered is None else np.asarray(centered, dtype=bool).copy()
    specs = list(state.parameter_specs)
    theta = convert_member_centering(state, state.initial_theta, centered)
    packed_centered = np.asarray(state.packed_lens_spec.deviation_centered, dtype=bool).copy()
    for position in range(n_members):
        packed_centered[int(state.members.component_indices[position])] = centered[position]
        if centered[position] == old_centered[position]:
            continue
        for offset_idx in (state.members.offset_param_index[position], state.members.radius_offset_param_index[position]):
            if int(offset_idx) >= 0:
                specs[int(offset_idx)] = replace(specs[int(offset_idx)], centered=bool(centered[position]))
    active = np.asarray(state.packed_lens_spec.radius_deviation_active, dtype=bool).copy()
    if state.packed_lens_spec.radius_deviation_members == "exact" and active.size:
        active[np.asarray(state.members.component_indices)] = False
        active[np.asarray(state.members.component_indices)[exact]] = True
    shape_active = np.asarray(state.packed_lens_spec.shape_active, dtype=bool).copy()
    if shape_active.size:
        # Only exact members carry a sampled shape: the far-field series is circular, so a far-field shape would be prior-only.
        components = np.asarray(state.members.component_indices)
        has_offset = np.asarray(state.packed_lens_spec.shape_e1_offset_param_index) >= 0
        shape_active[components] = False
        shape_active[components[exact]] = has_offset[components[exact]]
    packed = replace(state.packed_lens_spec, deviation_centered=packed_centered, radius_deviation_active=active, shape_active=shape_active)
    info = dict(classification if classification is not None else state.members.classification)
    info["centered_count"] = int(centered.sum())
    info["centered_members"] = [state.members.member_ids[i] for i in np.flatnonzero(centered)]
    members = replace(state.members, exact_positions=exact, far_positions=far, centered=centered, classification=info)
    new_state = replace(state, members=members, parameter_specs=specs, packed_lens_spec=packed)
    new_state.initial_theta = theta
    return new_state


def deviation_constraint_ratio(state: ModelState, evaluator, theta: np.ndarray, scatter: float) -> np.ndarray:
    """Per member, the data curvature of the log-likelihood in the ln-sigma deviation divided by the prior curvature 1/s^2.

    Above one the data constrain the deviation more tightly than the population prior at scatter ``s``: such a
    member should sample its deviation directly (centered); below one a unit offset is the better coordinate.
    """
    import jax
    import jax.numpy as jnp
    theta = np.asarray(theta, dtype=float)
    offsets = np.asarray(state.members.offset_param_index)
    if offsets.size == 0:
        return np.zeros(0)
    centered = np.asarray(state.members.centered, dtype=bool)
    scatter_now = np.exp(theta[np.asarray(state.members.scatter_param_index)])
    grad = jax.grad(evaluator.log_likelihood)

    def curvature(i):
        direction = jnp.zeros(theta.size).at[i].set(1.0)
        return -jax.jvp(grad, (jnp.asarray(theta),), (direction,))[1][i]

    h = np.asarray(jax.jit(jax.vmap(curvature))(jnp.asarray(offsets)))
    h_delta = np.where(centered, h, h / scatter_now ** 2)            # unit-offset coordinates carry a factor s^2
    return np.maximum(h_delta, 0.0) * float(scatter) ** 2
