"""Shared parameter initialization and cosmology construction services."""
from __future__ import annotations
from typing import Any
from .model import BinData
from .jax_cosmology import DEFAULT_JAX_COSMO_DISTANCE_STEPS
from .model import FamilyData
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import FlatwCDM
from .model import GeometryCache
from astropy.cosmology import LambdaCDM
from .model import ParameterSpec
from .jax_cosmology import dpie_sigma0_factor_from_config as _dpie_sigma0_factor_from_config
from .jax_cosmology import h0_from_config as _h0_from_config
from .jax_cosmology import ode0_from_config as _ode0_from_config
from .jax_cosmology import om0_from_config as _om0_from_config
from .jax_cosmology import w0_from_config as _w0_from_config
import numpy as np
import time
from astropy.cosmology import wCDM

DEFAULT_NUTS_INIT_BOUNDARY_FRAC = 0.02


SAFE_SCALING_EXPONENT_ABS_MIN = 1.0e-3


def _clip_value_to_safe_bounds(value: float, spec: ParameterSpec, boundary_frac: float = 0.0) -> float:
    lower = float(spec.lower)
    upper = float(spec.upper)
    clipped = float(np.clip(float(value), lower, upper))
    if spec.prior_kind == "uniform":
        width = float(upper - lower)
        margin = min(max(float(boundary_frac) * width, 0.0), 0.5 * width - 1.0e-12) if width > 0.0 else 0.0
        if margin > 0.0:
            clipped = float(np.clip(clipped, lower + margin, upper - margin))
    elif spec.prior_kind == "truncated_normal" and np.isfinite(lower):
        margin = max(float(boundary_frac) * max(abs(lower), 1.0e-12), 0.0)
        if spec.field == "beta_radius":
            # Only initialization moves inside the bijection; beta=0 remains physical.
            margin = max(margin, float(boundary_frac) * float(spec.std))
        if np.isfinite(upper):
            width = float(upper - lower)
            margin = min(margin, 0.5 * width - 1.0e-12) if width > 0.0 else 0.0
        if margin > 0.0:
            clipped = max(clipped, lower + margin)
    if spec.field in {"vdslope", "slope"} and lower < SAFE_SCALING_EXPONENT_ABS_MIN and upper > -SAFE_SCALING_EXPONENT_ABS_MIN:
        if abs(clipped) < SAFE_SCALING_EXPONENT_ABS_MIN:
            pos_candidate = max(SAFE_SCALING_EXPONENT_ABS_MIN, lower)
            neg_candidate = min(-SAFE_SCALING_EXPONENT_ABS_MIN, upper)
            if clipped >= 0.0 and pos_candidate <= upper:
                clipped = pos_candidate
            elif clipped < 0.0 and neg_candidate >= lower:
                clipped = neg_candidate
            elif pos_candidate <= upper:
                clipped = pos_candidate
            elif neg_candidate >= lower:
                clipped = neg_candidate
    return float(np.clip(clipped, lower, upper))


def _clip_theta_to_support(
    theta: np.ndarray,
    parameter_specs: list[ParameterSpec],
    boundary_frac: float = 0.0,
) -> np.ndarray:
    clipped = np.asarray(theta, dtype=float).copy()
    for idx, spec in enumerate(parameter_specs):
        clipped[idx] = _clip_value_to_safe_bounds(clipped[idx], spec, boundary_frac=boundary_frac)
    return clipped


def _default_theta(parameter_specs: list[ParameterSpec]) -> np.ndarray:
    default = np.asarray(
        [
            0.5 * (spec.lower + spec.upper) if spec.prior_kind == "uniform" else float(spec.mean)
            for spec in parameter_specs
        ],
        dtype=float,
    )
    return _clip_theta_to_support(default, parameter_specs, boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC)


def _reference_theta_from_init_values(
    parameter_specs: list[ParameterSpec],
    init_values: dict[str, float] | None,
    fallback: np.ndarray,
) -> np.ndarray:
    reference = np.asarray(fallback, dtype=float).copy()
    if init_values:
        for idx, spec in enumerate(parameter_specs):
            if spec.sample_name in init_values:
                reference[idx] = float(init_values[spec.sample_name])
    return _clip_theta_to_support(reference, parameter_specs, boundary_frac=DEFAULT_NUTS_INIT_BOUNDARY_FRAC)


def _build_cosmology_from_config(cosmo_config: dict[str, Any]):
    if not isinstance(cosmo_config, dict):
        return FlatLambdaCDM(H0=70.0, Om0=0.3)
    class_name = str(cosmo_config.get("class", "FlatLambdaCDM"))
    h0 = _h0_from_config(cosmo_config)
    om0 = _om0_from_config(cosmo_config)
    ode0 = _ode0_from_config(cosmo_config)
    w0 = _w0_from_config(cosmo_config)
    if class_name == "FlatLambdaCDM":
        return FlatLambdaCDM(H0=h0, Om0=om0)
    if class_name == "FlatwCDM":
        return FlatwCDM(H0=h0, Om0=om0, w0=w0)
    if class_name == "LambdaCDM":
        return LambdaCDM(H0=h0, Om0=om0, Ode0=ode0)
    if class_name == "wCDM":
        return wCDM(H0=h0, Om0=om0, Ode0=ode0, w0=w0)
    return FlatLambdaCDM(H0=h0, Om0=om0)


def _build_geometry_cache(
    cosmo_config: dict[str, Any],
    z_lens: float,
    family_data: list[FamilyData],
    bin_data: list[BinData],
    *,
    family_redshift_binning_sec: float = 0.0,
) -> GeometryCache:
    geometry_start = time.perf_counter()
    effective_z_source_values = sorted({float(bin_item.effective_z_source) for bin_item in bin_data})
    exact_z_source_values = sorted({float(family.z_source) for family in family_data})
    family_z_source_map = {str(family.family_id): float(family.z_source) for family in family_data}
    family_effective_z_source_map = {
        str(family.family_id): float(family.effective_z_source) for family in family_data
    }
    dpie_sigma0_factor_by_effective_z = {
        float(z_source): _dpie_sigma0_factor_from_config(z_lens, float(z_source), cosmo_config)
        for z_source in effective_z_source_values
    }
    dpie_sigma0_factor_by_exact_z = {
        float(z_source): _dpie_sigma0_factor_from_config(z_lens, float(z_source), cosmo_config)
        for z_source in exact_z_source_values
    }
    return GeometryCache(
        effective_z_source_values=[float(value) for value in effective_z_source_values],
        exact_z_source_values=[float(value) for value in exact_z_source_values],
        family_z_source_map=family_z_source_map,
        family_effective_z_source_map=family_effective_z_source_map,
        dpie_sigma0_factor_by_effective_z=dpie_sigma0_factor_by_effective_z,
        dpie_sigma0_factor_by_exact_z=dpie_sigma0_factor_by_exact_z,
        family_redshift_binning_sec=float(family_redshift_binning_sec),
        geometry_cache_build_sec=float(time.perf_counter() - geometry_start),
        flat_wcdm_quadrature_order=DEFAULT_JAX_COSMO_DISTANCE_STEPS,
        lens_quadrature_z=None,
        lens_quadrature_weights=None,
        effective_z_quadrature_z=None,
        effective_z_quadrature_weights=None,
        exact_z_quadrature_z=None,
        exact_z_quadrature_weights=None,
    )
