"""Configuration of a mock-validation campaign: one mock generator config, one solver template."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                      MemberPopulationConfig, PriorConfig, ReferenceFrameConfig, RunPathsConfig)
from ..jax_cosmology import flat_wcdm_config, kpc_per_arcsec_from_config
from .generation import MockClusterPaths, SingleBCGMockConfig, mock_potfile_cut_prior_parameters


@dataclass(frozen=True)
class MockValidationPathsConfig:
    output_dir: str | Path = "validation_runs"
    campaign_name: str = "mock_validation"
    run_name: str = "single_bcg"


@dataclass(frozen=True)
class MockValidationRuntimeConfig:
    realizations: int = 1
    seed: int = 12345
    resume: bool = True
    quiet: bool = False


@dataclass(frozen=True)
class MockValidationConfig:
    mock: SingleBCGMockConfig = field(default_factory=SingleBCGMockConfig)
    paths: MockValidationPathsConfig = field(default_factory=MockValidationPathsConfig)
    runtime: MockValidationRuntimeConfig = field(default_factory=MockValidationRuntimeConfig)
    solver: LensClusterSolverConfig = field(default_factory=LensClusterSolverConfig)

    def with_updates(self, **updates: Any) -> "MockValidationConfig":
        return replace(self, **updates)

    def validate(self) -> "MockValidationConfig":
        if int(self.runtime.realizations) <= 0:
            raise ValueError("runtime.realizations must be positive.")
        if int(self.runtime.seed) < 0:
            raise ValueError("runtime.seed must be nonnegative.")
        if self.solver.likelihood.pos_sigma_arcsec is None:
            raise ValueError("solver.likelihood.pos_sigma_arcsec is required for mock image constraints.")
        self.mock.validate()
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return _jsonable({"mock": asdict(self.mock), "paths": asdict(self.paths), "runtime": asdict(self.runtime),
                          "solver": self.solver.to_json_dict()})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _dpie_halo_config(component: Any, mock_config: SingleBCGMockConfig, kpc_per_arcsec: float, *,
                      position_half_width_arcsec: float, cut_radius_prior_fractional_half_width: float) -> DPIEHaloConfig:
    core_kpc = float(component.core_radius_arcsec) * kpc_per_arcsec
    cut_kpc = float(component.cut_radius_arcsec) * kpc_per_arcsec
    half = float(cut_radius_prior_fractional_half_width)
    return DPIEHaloConfig(
        id=str(component.potential_id), x_centre=float(component.x_centre), y_centre=float(component.y_centre),
        ellipticite=float(component.ellipticite), angle_pos=float(component.angle_pos), core_radius_kpc=core_kpc,
        cut_radius_kpc=cut_kpc, v_disp=float(component.v_disp), z_lens=float(mock_config.z_lens),
        priors={
            "x_centre": PriorConfig("uniform", lower=float(component.x_centre) - position_half_width_arcsec,
                                    upper=float(component.x_centre) + position_half_width_arcsec, step=0.05),
            "y_centre": PriorConfig("uniform", lower=float(component.y_centre) - position_half_width_arcsec,
                                    upper=float(component.y_centre) + position_half_width_arcsec, step=0.05),
            "ellipticite": PriorConfig("uniform", lower=0.0, upper=0.75, step=0.02),
            "angle_pos": PriorConfig("uniform", lower=-90.0, upper=90.0, step=0.5),
            "core_radius_kpc": PriorConfig("uniform", lower=max(0.001, float(component.core_radius_arcsec) * 0.2) * kpc_per_arcsec,
                                           upper=float(component.core_radius_arcsec) * 3.0 * kpc_per_arcsec, step=0.02 * kpc_per_arcsec),
            "cut_radius_kpc": PriorConfig("uniform", lower=cut_kpc * (1.0 - half), upper=cut_kpc * (1.0 + half), step=0.5 * kpc_per_arcsec),
            "v_disp": PriorConfig("uniform", lower=float(component.v_disp) * 0.55, upper=float(component.v_disp) * 1.45, step=1.0),
        })


def single_bcg_mock_lens_model_config(mock_config: SingleBCGMockConfig, paths: MockClusterPaths, *,
                                      image_constraints_sigma_arcsec: float) -> LensModelConfig:
    sigma_arcsec = float(image_constraints_sigma_arcsec)
    if not math.isfinite(sigma_arcsec) or sigma_arcsec <= 0.0:
        raise ValueError("image_constraints_sigma_arcsec must be positive and finite.")
    kpc = float(kpc_per_arcsec_from_config(mock_config.z_lens, flat_wcdm_config(h0=70.0, om0=0.3)))
    large_halos = (
        _dpie_halo_config(mock_config.halo, mock_config, kpc, position_half_width_arcsec=8.0,
                          cut_radius_prior_fractional_half_width=float(mock_config.halo_cut_radius_prior_fractional_half_width)),
        _dpie_halo_config(mock_config.bcg, mock_config, kpc, position_half_width_arcsec=float(mock_config.bcg_position_prior_half_width_arcsec),
                          cut_radius_prior_fractional_half_width=float(mock_config.bcg_cut_radius_prior_fractional_half_width)),
    )
    members: tuple[MemberPopulationConfig, ...] = ()
    member_catalog = paths.root / "members.cat"
    if int(mock_config.n_subhalos) > 0 and member_catalog.is_file():
        cutkpc = float(mock_config.subhalo_cut_radius_arcsec) * kpc
        cut_lower, cut_upper, cut_std = mock_potfile_cut_prior_parameters(cutkpc, float(mock_config.subhalo_cut_radius_prior_fractional_half_width))
        members = (MemberPopulationConfig(
            id="potfile", catalog_path=member_catalog, mag0=float(mock_config.subhalo_mag0),
            corekpc=float(mock_config.subhalo_core_radius_arcsec) * kpc, sigma=float(mock_config.subhalo_sigma_ref), cutkpc=cutkpc,
            z_lens=float(mock_config.z_lens),
            sigma_prior=PriorConfig("normal", mean=float(mock_config.subhalo_sigma_ref), std=float(mock_config.subhalo_sigma_ref_std)),
            cutkpc_prior=PriorConfig("normal", mean=cutkpc, std=cut_std, lower=cut_lower, upper=cut_upper)),)
    return LensModelConfig(
        reference=ReferenceFrameConfig(reference=3, ra0_deg=float(mock_config.reference_ra_deg), dec0_deg=float(mock_config.reference_dec_deg)),
        cosmology=CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7), large_halos=large_halos, member_populations=members,
        image_constraints=ImageConstraintsConfig(catalog_path=paths.image_catalog_path, sigma_arcsec=sigma_arcsec))


def solver_config_for_mock(config: MockValidationConfig, *, paths: MockClusterPaths, seed: int, output_dir: str | Path) -> LensClusterSolverConfig:
    template = config.solver
    model = single_bcg_mock_lens_model_config(config.mock, paths, image_constraints_sigma_arcsec=float(template.likelihood.pos_sigma_arcsec))
    return replace(template, model=model, paths=RunPathsConfig(output_dir=output_dir, run_name="solver"),
                   runtime=replace(template.runtime, seed=int(seed), resume=bool(config.runtime.resume), quiet=bool(config.runtime.quiet)))
