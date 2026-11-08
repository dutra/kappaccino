"""Data model for the single-stage smoothed-local-image-plane solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from . import priors

DP_IE_PROFILE = 81          # Lenstool profile codes
SHEAR_PROFILE = 14
NFW_PROFILE = 12
MASS_SHEET_PROFILE = 101    # not a Lenstool profile: uniform external convergence at the lens redshift
DPIE_PROFILE_NAME = "DPIE_NIE_E"        # lenstronomy names (exact-image solver)
SHEAR_PROFILE_NAME = "SHEAR"
NFW_PROFILE_NAME = "NFW_ELLIPSE_CSE"
MASS_SHEET_PROFILE_NAME = "CONVERGENCE"
COMPONENT_FAMILY_LARGE = 0
COMPONENT_FAMILY_SCALING = 1
DEFAULT_SEARCH_PADDING = 8.0
SAFE_RADIUS_MARGIN_ARCSEC = 1.0e-3
SAFE_RADIUS_MARGIN_KPC = 1.0e-3


@dataclass(frozen=True)
class ParameterSpec:
    """One sampled parameter.  ``prior`` is the NumPyro distribution of its physical value; the parameter is sampled in
    that distribution's base coordinate (its log for an exp-transformed prior, see priors.py), the "latent" value."""
    name: str
    sample_name: str
    potential_id: str
    profile_type: int
    field: str
    prior: Any
    component_family: str = "large"
    sample_site_name: str | None = None
    sample_site_index: int | None = None
    centered: bool = False                   # deviation sampled directly with prior N(loc, scatter) instead of a unit offset
    scale_site_name: str | None = None       # sample site whose exp(value) is that scatter

    @cached_property
    def latent_prior(self) -> Any:
        """The distribution of the latent value."""
        return priors.split(self.prior)[0]

    @cached_property
    def log(self) -> bool:
        """True when the physical value is exp(latent)."""
        return priors.split(self.prior)[1]


@dataclass
class PackedLensSpec:
    """Fixed per-component arrays plus the parameter index that overrides each field."""
    profile_type: np.ndarray
    component_family: np.ndarray
    x_center_base: np.ndarray
    y_center_base: np.ndarray
    e1_base: np.ndarray
    e2_base: np.ndarray
    core_radius_kpc_base: np.ndarray
    cut_radius_kpc_base: np.ndarray
    v_disp_base: np.ndarray
    gamma1_base: np.ndarray
    gamma2_base: np.ndarray
    x_center_param_index: np.ndarray
    y_center_param_index: np.ndarray
    e1_param_index: np.ndarray
    e2_param_index: np.ndarray
    core_radius_param_index: np.ndarray
    cut_radius_param_index: np.ndarray
    v_disp_param_index: np.ndarray
    gamma1_param_index: np.ndarray
    gamma2_param_index: np.ndarray
    luminosity_ratio: np.ndarray
    sigma_ref_base: np.ndarray
    cut_ref_base: np.ndarray
    core_ref_base: np.ndarray
    alpha_sigma_base: np.ndarray
    beta_radius_base: np.ndarray
    sigma_ref_param_index: np.ndarray
    cut_ref_param_index: np.ndarray
    core_ref_param_index: np.ndarray
    alpha_sigma_param_index: np.ndarray
    beta_radius_param_index: np.ndarray
    cut_radius_excess_above_effective_core: np.ndarray
    # Sampled member deviations (per component) and their population scatters; -1 where absent.
    velocity_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    velocity_scatter_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    radius_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    radius_scatter_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    deviation_centered: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    deviation_mode: str = "sigma"
    # Components whose radius deviation acts on the model; far-field members keep the relation radius when
    # radius_deviation_members is "exact" (their offsets are then prior-only).
    radius_deviation_active: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    radius_deviation_members: str = "exact"
    # Sampled member shapes (ScalingModelConfig.member_shape_scatter): unit offsets per component, -1 where absent; the
    # offsets move e1, e2 by shape_scatter times their value on the components in shape_active (the exact members).
    shape_e1_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    shape_e2_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    shape_active: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))
    shape_scatter: float = 0.0
    # Smooth ellipticity cap per component: the modulus |e| above which a smooth penalty -softplus((|e| - cap) / width)
    # takes over from the flat prior (inf: none; DPIEHaloConfig.ellipticity_soft_cap).
    ellipticity_soft_cap: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    ellipticity_soft_width: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    # Elliptical NFW halos: M200 and scale radius.
    scale_radius_kpc_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    log10_m200_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    scale_radius_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    log10_m200_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    # External convergence: the sheet's kappa for a source at infinite redshift (D_ls / D_s = 1).
    kappa_base: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    kappa_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))


class PackedLensState(NamedTuple):
    """Per-component kernel parameters.  Normalizations (sigma0, alpha_rs) carry the lensing efficiency of the source
    and have shape (component, image) in the likelihood or (component,) for a single source redshift."""
    profile_type: jnp.ndarray
    sigma0: jnp.ndarray          # dPIE
    Ra: jnp.ndarray
    Rs: jnp.ndarray
    e1: jnp.ndarray              # dPIE and NFW
    e2: jnp.ndarray
    center_x: jnp.ndarray
    center_y: jnp.ndarray
    gamma1: jnp.ndarray          # external shear
    gamma2: jnp.ndarray
    nfw_rs: jnp.ndarray          # NFW: scale radius (arcsec) and deflection at the scale radius (arcsec)
    alpha_rs: jnp.ndarray
    kappa: jnp.ndarray           # mass sheet: convergence seen by each source (efficiency applied)


class PackedLensValidity(NamedTuple):
    is_valid: jnp.ndarray
    reason_flags: jnp.ndarray


@dataclass
class FamilyData:
    family_id: str
    z_source: float
    sigma_inst_arcsec: float
    image_labels: list[str]
    x_obs: np.ndarray
    y_obs: np.ndarray
    reliability: np.ndarray

    @property
    def effective_z_source(self) -> float:
        return self.z_source

    @property
    def n_images(self) -> int:
        return len(self.image_labels)

    @property
    def x_center(self) -> float:
        if self.n_images == 0:
            return float("nan")
        return float(0.5 * (np.min(self.x_obs) + np.max(self.x_obs)))

    @property
    def y_center(self) -> float:
        if self.n_images == 0:
            return float("nan")
        return float(0.5 * (np.min(self.y_obs) + np.max(self.y_obs)))

    @property
    def search_window(self) -> float:
        span_x = np.ptp(self.x_obs) if self.n_images > 1 else 0.0
        span_y = np.ptp(self.y_obs) if self.n_images > 1 else 0.0
        return float(max(span_x, span_y) + DEFAULT_SEARCH_PADDING)


@dataclass(frozen=True)
class ImageData:
    """All constrained images of every family, flattened, with per-family bookkeeping."""
    family_ids: tuple[str, ...]
    n_families: int
    x_obs: jnp.ndarray
    y_obs: jnp.ndarray
    sigma_inst_per_image: jnp.ndarray
    parity: jnp.ndarray
    z_index_per_image: jnp.ndarray
    family_index_per_image: jnp.ndarray
    source_x_param_index: jnp.ndarray
    source_y_param_index: jnp.ndarray
    image_labels: tuple[str, ...]
    system_ids: tuple[str, ...] = ()                 # galaxies: a system holds the knots (families) of one source galaxy
    system_index_per_image: jnp.ndarray | None = None

    @property
    def n_images(self) -> int:
        return int(self.x_obs.shape[0])


@dataclass(frozen=True)
class MemberModel:
    """Population members: identities, component rows, parameter indices, and classes."""
    member_ids: tuple[str, ...]
    component_indices: np.ndarray
    scatter_param_index: np.ndarray          # per member -> population scatter parameter
    offset_param_index: np.ndarray           # per member -> sampled velocity offset parameter
    exact_positions: np.ndarray              # positions into member arrays evaluated exactly
    far_positions: np.ndarray                # positions evaluated by the far-field series
    source_prior_mean: tuple[float, float]
    source_prior_sigma: float
    far_field_guard_cut_radii: float
    classification: dict[str, Any] = field(default_factory=dict)
    far_field_barrier_width: float = 0.05     # width of the soft guard barrier in ln(R / (guard r_cut))
    far_field_barrier_strength: float = 25.0  # log-likelihood penalty scale
    radius_scatter_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    radius_offset_param_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    centered: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=bool))


@dataclass
class ModelState:
    run_name: str
    cosmo_config: dict[str, Any]
    z_lens: float
    kpc_per_arcsec: float
    reference: tuple[int, float, float]
    parameter_specs: list[ParameterSpec]
    base_components: list[dict[str, Any]]
    packed_lens_spec: PackedLensSpec
    lens_model_list: list[str]
    family_data: list[FamilyData]
    images: ImageData
    potfiles: list[dict[str, Any]]
    scaling_component_records: list[dict[str, Any]]
    members: MemberModel
    z_source_values: list[float]
    dpie_sigma0_factor_by_z: dict[float, float]
    softening_length_param_index: int = -1
    sigma_def_param_index: int = -1              # residual deflection scatter, isotropic in the source plane
    sigma_extra_param_index: int = -1            # extra measurement scatter added in quadrature to sigma_inst; -1 when absent
    sigma_sel_param_index: int = -1              # feature-selection scatter; -1 when the term is disabled
    lambda_sel_param_index: int = -1             # stretching scale of the selection weight R = (I + A A^T / lambda_sel^2)^-1; -1 with the term
    sigma_shared_param_index: int = -1           # gp_image_plane: the correlated image-plane scale; -1 under the other likelihoods
    sigma_white_param_index: int = -1            # gp_image_plane: the independent image-plane scale
    gp_ell_param_index: int = -1                 # gp_image_plane: the sampled correlation length; -1 when ell is fixed
    gp_correlation_length_arcsec: float = 7.0    # gp_image_plane: ell (fixed value or start) and sigma_fl (LikelihoodConfig)
    gp_floor_arcsec: float = 0.0015
    gp_kernel: str = "image_plane"               # gp_image_plane: "image_plane" (B S B^T) or "knee" (mass field + B S B^T, see LikelihoodConfig)
    gp_reference_redshift: float = 2.0           # knee kernel: z_ref of the redshift weights
    gp_blocks: str = "all"                       # knee kernel: "all" | "system" | "family" covariance blocks
    gp_knee: dict[str, Any] = field(default_factory=dict)   # knee kernel: gamma, scales, lambda grid, r step
    gp_sigma_alpha_param_index: int = -1         # knee kernel: sigma_alpha; -1 when the mass field is dropped
    gp_knee_param_index: int = -1                # knee kernel: lambda_t
    gp_knee_gamma_param_index: int = -1          # knee kernel: the sampled slope gamma; -1 when fixed
    gp_sigma_alpha2_param_index: int = -1        # knee kernel, two_knee / knee_foreground spectra: the second field's sigma_alpha
    gp_knee2_param_index: int = -1               # ... and its knee
    gp_knee_beta_low_param_index: int = -1       # knee kernel, broken_power_law spectrum: the low-k slope
    freeze_determinant_gradient: bool = False    # hold grad det A fixed in the backward pass (LikelihoodConfig)
    dropped_member_ids: tuple[str, ...] = ()  # catalog members removed from the lens model (members.min_deflection_arcsec)
    # Measured member dispersions in the likelihood (state.build_member_kinematics); empty when no source is on.
    member_kinematics: dict[str, Any] = field(default_factory=dict)
    initial_theta: np.ndarray | None = None

    @property
    def n_parameters(self) -> int:
        return len(self.parameter_specs)


# Plotting code written against the previous solver uses this name.
BuildState = ModelState


@dataclass
class PosteriorResults:
    samples: np.ndarray                      # (draws, n_params) latent
    physical: np.ndarray                     # (draws, n_params) reported physical
    log_prob: np.ndarray
    accept_prob: np.ndarray
    diverging: np.ndarray
    num_steps: np.ndarray
    warmup_steps: int
    sample_steps: int
    num_chains: int
    energy: np.ndarray                       # (draws,) Hamiltonian energy of each transition (E-BFMI)
    step_size: np.ndarray                    # (chains,) step size at the end of warmup
    grouped_samples: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    median_fit: np.ndarray | None = None
    maximum_likelihood_fit: np.ndarray | None = None


def physical_to_latent(value: float, spec: ParameterSpec) -> float:
    return float(np.log(max(float(value), 1.0e-12))) if spec.log else float(value)


def latent_to_physical(value: float, spec: ParameterSpec) -> float:
    return float(np.exp(float(value))) if spec.log else float(value)


def latent_array_to_physical(values: np.ndarray, spec: ParameterSpec) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.exp(array) if spec.log else array


def convert_theta_to_physical(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    return np.asarray([latent_to_physical(v, spec) for v, spec in zip(theta_array, parameter_specs)], dtype=float).reshape(theta_array.shape)


def convert_theta_to_latent(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    theta_array = np.asarray(theta, dtype=float)
    return np.asarray([physical_to_latent(v, spec) for v, spec in zip(theta_array, parameter_specs)], dtype=float).reshape(theta_array.shape)


def convert_sample_matrix_to_physical(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    array = np.asarray(samples, dtype=float)
    converted = array.copy()
    for idx, spec in enumerate(parameter_specs):
        converted[..., idx] = latent_array_to_physical(array[..., idx], spec)
    return converted


def log_mask(parameter_specs: list[ParameterSpec]) -> np.ndarray:
    """True for the parameters whose physical value is exp(latent)."""
    return np.asarray([spec.log for spec in parameter_specs], dtype=bool)


def apply_parameter_transforms_jax(params: jnp.ndarray, log_positive_mask: jnp.ndarray) -> jnp.ndarray:
    """Physical values of a latent vector: exp where ``log_positive_mask``, the latent value elsewhere."""
    params_array = jnp.asarray(params, dtype=jnp.float64)
    return jnp.where(log_positive_mask, jnp.exp(jnp.where(log_positive_mask, params_array, 0.0)), params_array)


def clip_value_to_safe_bounds(value: float, spec: ParameterSpec, boundary_frac: float = 0.0) -> float:
    """``value`` (latent) inside the support of the latent prior, at least ``boundary_frac`` of the width (uniform) or of
    the bound's magnitude (bounded below) away from a bound."""
    latent = spec.latent_prior
    lower, upper = priors.bounds(latent)
    kind = priors.family(latent)
    clipped = float(np.clip(float(value), lower, upper))
    if kind == "Uniform":
        width = float(upper - lower)
        margin = min(max(float(boundary_frac) * width, 0.0), 0.5 * width - 1.0e-12) if width > 0.0 else 0.0
        if margin > 0.0:
            clipped = float(np.clip(clipped, lower + margin, upper - margin))
    elif kind == "TruncatedNormal" and np.isfinite(lower):
        margin = max(float(boundary_frac) * max(abs(lower), 1.0e-12), 0.0)
        if spec.field == "beta_radius":
            margin = max(margin, float(boundary_frac) * priors.parameters(latent)["scale"])
        if np.isfinite(upper):
            width = float(upper - lower)
            margin = min(margin, 0.5 * width - 1.0e-12) if width > 0.0 else 0.0
        if margin > 0.0:
            clipped = max(clipped, lower + margin)
    return float(np.clip(clipped, lower, upper))


def clip_theta_to_support(theta: np.ndarray, parameter_specs: list[ParameterSpec], boundary_frac: float = 0.0) -> np.ndarray:
    clipped = np.asarray(theta, dtype=float).copy()
    for idx, spec in enumerate(parameter_specs):
        clipped[idx] = clip_value_to_safe_bounds(clipped[idx], spec, boundary_frac=boundary_frac)
    return clipped


def default_theta(parameter_specs: list[ParameterSpec], boundary_frac: float = 0.02) -> np.ndarray:
    """Latent vector at prior centres: the configured values for lens parameters, zero offsets."""
    values = np.asarray([priors.center(spec.latent_prior) for spec in parameter_specs], dtype=float)
    return clip_theta_to_support(values, parameter_specs, boundary_frac=boundary_frac)


def member_log_deviations(spec: PackedLensSpec, physical) -> tuple[Any, Any]:
    """Per-component (d ln sigma, d ln radius) implied by the sampled deviations in ``physical``.

    Centered members hold the deviation itself; non-centered members hold a unit offset times the population
    scatter.  In "mass" mode one deviation moves dispersion and radii together along the relation
    (d ln sigma : d ln r = alpha : beta); in "sigma_rcut" mode the radius has its own offset and scatter.
    Works with NumPy or JAX arrays.
    """
    import jax.numpy as jnp
    physical = jnp.asarray(physical)
    centered = jnp.asarray(spec.deviation_centered, dtype=bool)

    def pick(offset_index, scatter_index):
        offset_index, scatter_index = jnp.asarray(offset_index), jnp.asarray(scatter_index)
        present = offset_index >= 0
        offset = jnp.where(present, physical[jnp.maximum(offset_index, 0)], 0.0)
        scatter = jnp.where(scatter_index >= 0, physical[jnp.maximum(scatter_index, 0)], 1.0)
        return jnp.where(present, jnp.where(centered, offset, offset * scatter), 0.0)

    delta_v = pick(spec.velocity_offset_param_index, spec.velocity_scatter_param_index)
    active = np.asarray(spec.radius_deviation_active, dtype=bool)
    radius_mask = jnp.asarray(active, dtype=jnp.float64) if active.size else 1.0

    def pick_radius():
        return radius_mask * pick(spec.radius_offset_param_index, spec.radius_scatter_param_index)

    if spec.deviation_mode == "sigma":
        return delta_v, jnp.zeros_like(delta_v)
    if spec.deviation_mode in ("mass", "mass_rcut"):
        alpha = jnp.where(jnp.asarray(spec.alpha_sigma_param_index) >= 0,
                          physical[jnp.maximum(jnp.asarray(spec.alpha_sigma_param_index), 0)], jnp.asarray(spec.alpha_sigma_base))
        beta = jnp.where(jnp.asarray(spec.beta_radius_param_index) >= 0,
                         physical[jnp.maximum(jnp.asarray(spec.beta_radius_param_index), 0)], jnp.asarray(spec.beta_radius_base))
        total = 2.0 * alpha + beta
        extra = pick_radius() if spec.deviation_mode == "mass_rcut" else 0.0
        return delta_v * alpha / total, delta_v * beta / total + extra
    if spec.deviation_mode == "sigma_rcut":
        return delta_v, pick_radius()
    raise ValueError(f"unknown member deviation mode {spec.deviation_mode!r}")
