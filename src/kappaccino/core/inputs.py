"""Declared model inputs: halos, member populations, and image catalogs from the config."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..config import IndependentMemberHaloConfig, LensModelConfig, MemberPopulationConfig
from ..io.image_catalog import load_image_catalog
from ..io.member_catalog import load_member_catalog
from ..priors import cut_excess_from_moments, ellipticity_box, lognormal_from_moments, physical_center, scaled
from ..lens.profiles import profile as lens_profile
from ..io.sky import radec_to_offsets_arcsec


def check_potential_fields(potential: dict[str, Any]) -> dict[str, Any]:
    """The potential with its profile's required fields and z_lens checked to be finite numbers."""
    info = lens_profile(int(potential["profil"]))
    for name in (*info.required_fields, "z_lens"):
        value = potential.get(name)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Potential {potential.get('id', '?')} is missing a finite {name}.")
    return potential


def lenstool_ellipticite_to_axis_ratio(ellipticite: float | np.ndarray) -> np.ndarray:
    safe_e = np.clip(np.asarray(ellipticite, dtype=float), 0.0, 1.0 - 1.0e-9)
    q = np.sqrt((1.0 - safe_e) / (1.0 + safe_e))
    return np.clip(q, 1.0e-3, 1.0)


def lenstool_shape_to_e1e2(ellipticite: float, angle_pos_deg: float) -> tuple[float, float]:
    q = lenstool_ellipticite_to_axis_ratio(float(ellipticite))
    modulus = (1.0 - q) / (1.0 + q)
    phi = np.deg2rad(float(angle_pos_deg))
    return float(modulus * np.cos(2.0 * phi)), float(modulus * np.sin(2.0 * phi))


def shear_polar_to_gamma1gamma2(gamma: float, angle_pos_deg: float) -> tuple[float, float]:
    phi = np.deg2rad(float(angle_pos_deg))
    return float(gamma * np.cos(2.0 * phi)), float(gamma * np.sin(2.0 * phi))


def axis_ratio_to_lenstool_ellipticite(q: float) -> float:
    safe_q = max(1.0e-3, min(1.0, float(q)))
    return float((1.0 - safe_q * safe_q) / (1.0 + safe_q * safe_q))


def catalog_shape_to_ellipticity(a_value: float, b_value: float, theta_value: float) -> tuple[float, float]:
    a_axis = max(float(a_value), 1.0e-3)
    b_axis = max(min(float(b_value), a_axis), 1.0e-3)
    q = max(1.0e-3, min(1.0, b_axis / a_axis))
    return axis_ratio_to_lenstool_ellipticite(q), float(theta_value)


def cosmology_config_from_model_config(model_config: LensModelConfig) -> dict[str, Any]:
    cosmo = model_config.cosmology
    om0 = float(cosmo.Om0)
    ode0 = float(1.0 - om0 if cosmo.Ode0 is None else cosmo.Ode0)
    return {"class": str(cosmo.class_name), "H0": float(cosmo.H0), "Om0": om0, "Ode0": ode0, "w0": float(cosmo.w0)}


def _independent_member_default_positive_std(mean: float, *, dex_sigma: float = 0.5) -> float:
    safe_mean = max(float(mean), 1.0e-12)
    log_sigma = float(np.log(10.0) * dex_sigma)
    return float(safe_mean * np.sqrt(np.expm1(log_sigma * log_sigma)))


def _independent_member_default_priors(*, x_centre: float, y_centre: float, population: MemberPopulationConfig) -> dict[str, Any]:
    import numpyro.distributions as dist
    core_mean = float(population.corekpc)
    cut_mean = float(population.cutkpc)
    sigma_mean = float(population.sigma)
    return {
        # The light fixes a galaxy's centroid to a fraction of an arcsecond; a loose position prior makes the posterior
        # multi-well as the halo hops across nearby images.
        "x_centre": dist.TruncatedNormal(x_centre, 0.3, low=x_centre - 2.0, high=x_centre + 2.0),
        "y_centre": dist.TruncatedNormal(y_centre, 0.3, low=y_centre - 2.0, high=y_centre + 2.0),
        **ellipticity_box(0.8),
        "core_radius_kpc": lognormal_from_moments(core_mean, _independent_member_default_positive_std(core_mean), low=0.01, high=50.0),
        "cut_excess_kpc": cut_excess_from_moments(cut_mean, _independent_member_default_positive_std(cut_mean), low=1.0, high=1000.0,
                                                  core=core_mean),
        "v_disp": lognormal_from_moments(sigma_mean, max(0.5 * sigma_mean, 1.0e-6), low=30.0, high=1000.0),
    }


def _independent_member_halo_potential(independent: IndependentMemberHaloConfig, population: MemberPopulationConfig,
                                       catalog_row: pd.Series, reference: tuple[int, float, float],
                                       cosmo_config: dict[str, Any]) -> dict[str, Any]:
    _, ra0_deg, dec0_deg = reference
    x_offsets, y_offsets = radec_to_offsets_arcsec(np.asarray([float(catalog_row["ra"])]), np.asarray([float(catalog_row["dec"])]), ra0_deg, dec0_deg)
    x_centre, y_centre = float(x_offsets[0]), float(y_offsets[0])
    if {"catalog_a", "catalog_b", "catalog_theta"}.issubset(catalog_row.index):
        ellipticite, angle_pos = catalog_shape_to_ellipticity(
            float(catalog_row["catalog_a"]), float(catalog_row["catalog_b"]), float(catalog_row["catalog_theta"]))
    else:
        ellipticite, angle_pos = 0.0, 0.0
    catalog_id = str(independent.catalog_id)
    potential = {
        "id": str(independent.id or f"independent_member_{population.id}_{catalog_id}"),
        "profil": int(population.profile_type),
        "x_centre": x_centre, "y_centre": y_centre,
        "ellipticite": float(ellipticite), "angle_pos": float(angle_pos),
        "core_radius_kpc": float(population.corekpc), "cut_radius_kpc": float(population.cutkpc),
        "v_disp": float(population.sigma), "z_lens": float(population.z_lens),
    }
    normalized = check_potential_fields(dict(potential))
    merged = {**_independent_member_default_priors(x_centre=x_centre, y_centre=y_centre, population=population), **(independent.priors or {})}
    normalized["priors"] = {name: prior for name, prior in merged.items() if prior is not None}      # None fixes a default
    normalized["independent_member_population_id"] = str(population.id)
    normalized["independent_member_catalog_id"] = catalog_id
    return normalized


def _apply_catalog_exclusions(catalog_df: pd.DataFrame, population: MemberPopulationConfig) -> pd.DataFrame:
    exclude_ids = tuple(str(value).strip() for value in population.exclude_catalog_ids or ())
    if not exclude_ids:
        return catalog_df
    catalog_ids = catalog_df["id"].astype(str)
    missing = [value for value in exclude_ids if value not in set(catalog_ids)]
    if missing:
        raise ValueError(f"member population {population.id!r} exclude_catalog_ids not found in "
                         f"{population.catalog_path}: {', '.join(missing)}")
    filtered = catalog_df.loc[~catalog_ids.isin(set(exclude_ids))].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"member population {population.id!r} has no catalog rows after applying exclude_catalog_ids.")
    return filtered


def automatic_pivot(catalog_df: pd.DataFrame, images_df: pd.DataFrame, reference: tuple[int, float, float], *,
                    floor_arcsec: float = 1.0) -> dict[str, Any]:
    """Image-leverage-weighted member magnitude, w_i = sum_j 1 / max(R_ij, floor)^2, rounded to the nearest magnitude."""
    _, ra0, dec0 = reference
    mx, my = radec_to_offsets_arcsec(catalog_df["ra"].to_numpy(dtype=float), catalog_df["dec"].to_numpy(dtype=float), ra0, dec0)
    ix, iy = radec_to_offsets_arcsec(images_df["ra"].to_numpy(dtype=float), images_df["dec"].to_numpy(dtype=float), ra0, dec0)
    distance = np.hypot(mx[:, None] - ix[None, :], my[:, None] - iy[None, :])
    weights = np.sum(1.0 / np.maximum(distance, float(floor_arcsec)) ** 2, axis=1)
    magnitudes = catalog_df["catalog_mag"].to_numpy(dtype=float)
    weighted = float(np.sum(weights * magnitudes) / np.sum(weights))
    return {"mag0": float(np.round(weighted)), "weighted_mean_mag": weighted, "member_median_mag": float(np.median(magnitudes)),
            "n_members": int(len(magnitudes))}


def declared_model_inputs(model_config: LensModelConfig, scaling: Any = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, dict[str, Any], tuple[int, float, float]]:
    """Return (potentials with priors, potfiles, images, cosmology config, reference).

    ``scaling`` (a ScalingModelConfig) supplies the prior alpha and beta used to move a population's sigma and cutkpc
    priors to an automatic pivot.
    """
    reference = (int(model_config.reference.reference), float(model_config.reference.ra0_deg), float(model_config.reference.dec0_deg))
    cosmo_config = cosmology_config_from_model_config(model_config)
    if model_config.image_constraints is None:
        raise ValueError("model.image_constraints is required.")
    constraints = model_config.image_constraints
    images_df = load_image_catalog(constraints.catalog_path, sigma_inst_arcsec=constraints.sigma_inst_arcsec,
                                   sigma_floor_arcsec=constraints.sigma_floor_arcsec)
    if images_df.empty:
        raise ValueError("No multiple-image constraints found in the image catalog.")

    potentials: list[dict[str, Any]] = []
    for halo in model_config.large_halos:
        info = lens_profile(int(halo.profile_type))
        potential = {"id": str(halo.id), "profil": int(halo.profile_type), "z_lens": float(halo.z_lens),
                     **{name: float(getattr(halo, name)) for name in info.required_fields}}
        normalized = check_potential_fields(potential)
        normalized["priors"] = {name: prior for name, prior in (halo.priors or {}).items() if prior is not None}
        if getattr(halo, "ellipticity_soft_cap", None) is not None:
            normalized["ellipticity_soft_cap"] = float(halo.ellipticity_soft_cap)
            normalized["ellipticity_soft_width"] = float(halo.ellipticity_soft_width)
        potentials.append(normalized)

    independent_by_population: dict[str, list[IndependentMemberHaloConfig]] = {}
    for item in model_config.independent_member_halos or ():
        independent_by_population.setdefault(str(item.population_id), []).append(item)
    potfiles: list[dict[str, Any]] = []
    for population in model_config.member_populations:
        catalog_df = load_member_catalog(population.catalog_path, reference, band=population.band)
        treatment = catalog_df["catalog_treatment"].astype(str)
        # Catalog treatments: "exclude" leaves the model; "independent" becomes an independent member halo with default
        # priors unless the config declares that member (the config's priors then apply).
        catalog_df = catalog_df.loc[treatment != "exclude"].copy().reset_index(drop=True)
        independent_items = list(independent_by_population.get(str(population.id), []))
        declared = {str(item.catalog_id) for item in independent_items}
        independent_items += [IndependentMemberHaloConfig(population_id=str(population.id), catalog_id=str(cid))
                              for cid in catalog_df.loc[catalog_df["catalog_treatment"] == "independent", "id"].astype(str)
                              if cid not in declared]
        if independent_items:
            catalog_ids = catalog_df["id"].astype(str)
            for independent in independent_items:
                match = catalog_df.loc[catalog_ids == str(independent.catalog_id)]
                if match.empty:
                    raise ValueError(f"independent member halo population_id={population.id!r} "
                                     f"catalog_id={independent.catalog_id!r} was not found in {population.catalog_path}.")
                potentials.append(_independent_member_halo_potential(independent, population, match.iloc[0], reference, cosmo_config))
            catalog_df = catalog_df.loc[~catalog_ids.isin({str(item.catalog_id) for item in independent_items})].copy().reset_index(drop=True)
        catalog_df = _apply_catalog_exclusions(catalog_df, population)
        sigma_prior, cutkpc_prior = population.sigma_prior, population.cutkpc_prior
        sigma_nominal, cutkpc_nominal, corekpc_nominal, pivot = float(population.sigma), float(population.cutkpc), float(population.corekpc), None
        if isinstance(population.mag0, str):
            if scaling is None:
                raise ValueError("mag0='auto' needs the scaling configuration to move the priors to the pivot.")
            pivot = automatic_pivot(catalog_df, images_df, reference)
            pivot["prior_mag0"] = float(population.prior_mag0)
            ratio = 10.0 ** (-0.4 * (pivot["mag0"] - float(population.prior_mag0)))     # L0_new / L0_old
            f_sigma, f_cut = ratio ** physical_center(scaling.alpha_sigma_prior), ratio ** physical_center(scaling.beta_radius_prior)
            pivot.update(luminosity_ratio=ratio, sigma_factor=f_sigma, cutkpc_factor=f_cut)
            sigma_prior, cutkpc_prior = scaled(sigma_prior, f_sigma), scaled(cutkpc_prior, f_cut)
            sigma_nominal, cutkpc_nominal, corekpc_nominal = sigma_nominal * f_sigma, cutkpc_nominal * f_cut, corekpc_nominal * f_cut
            mag0 = pivot["mag0"]
        else:
            mag0 = float(population.mag0)
        potfiles.append({
            "id": str(population.id), "type": int(population.profile_type),
            "catalog_path": str(population.catalog_path), "catalog_df": catalog_df,
            "mag0": mag0, "pivot": pivot, "corekpc": float(population.corekpc), "corekpc_nominal": corekpc_nominal, "zlens": float(population.z_lens),
            "sigma_prior": sigma_prior, "sigma_nominal": sigma_nominal, "cutkpc_prior": cutkpc_prior, "cutkpc_nominal": cutkpc_nominal,
        })
    return potentials, potfiles, images_df, cosmo_config, reference


def filter_image_families(images_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop families without a positive redshift and single-image families."""
    family_ids = images_df["family_id"].astype(str)
    z_values = pd.to_numeric(images_df["catalog_z"], errors="coerce").to_numpy(dtype=float)
    bad_z_families = sorted(family_ids[np.isfinite(z_values) & (z_values <= 0.0)].unique().tolist())
    kept = images_df.loc[~family_ids.isin(bad_z_families)].copy()
    counts = kept["family_id"].astype(str).value_counts()
    singleton_families = sorted(counts[counts <= 1].index.astype(str).tolist())
    kept = kept.loc[~kept["family_id"].astype(str).isin(singleton_families)].copy().reset_index(drop=True)
    if kept.empty:
        raise ValueError("No multi-image families with positive redshift remain in the image catalog.")
    return kept, {"dropped_nonpositive_redshift_families": bad_z_families, "dropped_singleton_families": singleton_families}
