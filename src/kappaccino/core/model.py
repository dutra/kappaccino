"""Data model for the single-stage smoothed-local-image-plane solver."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from .. import priors

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
    """One sampled parameter of the model.

    ``prior`` is the NumPyro distribution of the parameter's physical value.  The parameter is sampled in that
    distribution's base coordinate, the "latent" value: :math:`\\ln` of the physical value for an exp-transformed prior
    (``dist.LogNormal``, ``dist.LogUniform``, ``priors.truncated_lognormal``, ``priors.lognormal_from_moments``), the
    physical value itself otherwise (see :mod:`kappaccino.priors`).  The ordered list of specs in
    ``ModelState.parameter_specs`` defines the parameter axis of every latent vector ``theta``, of
    ``PosteriorResults.samples`` and of ``PosteriorResults.physical``.

    Physical units by ``field``: ``x_centre``, ``y_centre`` arcsec offsets from the reference
    (:math:`x = (\\alpha_0 - \\alpha)\\cos\\delta_0 \\cdot 3600`, positive west; :math:`y = (\\delta - \\delta_0) \\cdot 3600`);
    ``e1``, ``e2``, ``gamma1``, ``gamma2``, ``kappa`` dimensionless; ``core_radius_kpc``, ``cut_excess_kpc``,
    ``scale_radius_kpc``, ``cutkpc``, ``corekpc`` and ``softening_length_kpc`` kpc; ``v_disp`` and ``sigma`` km s^-1;
    ``log10_m200`` :math:`\\log_{10}(M_{200}/M_\\odot)`; ``alpha_sigma``, ``beta_radius`` dimensionless slopes;
    ``velocity_log_scatter``, ``radius_log_scatter`` dimensionless scatters in :math:`\\ln\\sigma` and :math:`\\ln r`;
    ``velocity_offset``, ``radius_offset``, ``shape_e1_offset``, ``shape_e2_offset`` and the source ``beta_x``,
    ``beta_y`` dimensionless unit offsets; the error-model scales (``sigma_def``, ``sigma_meas``, ``sigma_sel``,
    ``lambda_sel``, ``sigma_gp``, ``ell_gp``, ``sigma_grf``, ``lambda_b``, ``sigma_grf2``,
    ``lambda_b2``) arcsec, except ``lambda_sel``, ``gp_knee_gamma`` and ``gp_knee_beta_low``, which are dimensionless.

    Attributes
    ----------
    name : str
        Unique dotted name, ``"<potential_id>.<field>"`` (for example ``"halo1.v_disp"``, ``"images.sigma_def"``,
        ``"source.<family_id>.beta_x"`` or ``"<population>.<catalog_id>.velocity_offset"``).  Used as the parameter
        label in tables and figures.
    sample_name : str
        The NumPyro sample-site name (``name`` with every character outside ``[0-9A-Za-z_]`` replaced by ``_``) for a
        parameter with its own site.
    potential_id : str
        The lens component or member population the parameter belongs to; ``"images"`` for error-model scales, the
        family id for source offsets, ``"softening_length"`` for the softening length.
    profile_type : int
        Lenstool profile code of the component: 81 (dPIE), 12 (NFW), 14 (external shear), 101 (mass sheet); 0 for
        parameters that are not lens parameters (error scales, sources, softening).
    field : str
        The component field the parameter sets (for example ``"v_disp"``, ``"e1"``, ``"cut_excess_kpc"``,
        ``"sigma"``, ``"velocity_offset"``, ``"beta_x"``, ``"sigma_def"``).
    prior : numpyro.distributions.Distribution
        Prior of the physical value.
    component_family : str, optional
        Parameter group: ``"large"`` (halo, shear, sheet parameters; default), ``"scaling"`` (member scaling
        relation), ``"member_velocity_scatter"`` (population scatters of the member deviations, velocity and radius),
        ``"member_scatter_offset"`` (per-member deviation offsets), ``"member_shape_offset"``, ``"error_scale"``,
        ``"source_position"`` or ``"softening_length"``.
    sample_site_name : str or None, optional
        Name of the vector NumPyro site that holds this parameter together with others of its kind (for example
        ``"source_position_beta"``, ``"member_velocity_offsets"``); None (default) for a parameter with its own site.
    sample_site_index : int or None, optional
        Position of the parameter in the vector site ``sample_site_name``; None (default) otherwise.
    centered : bool, optional
        True when a member deviation is sampled directly with prior :math:`N(\\mathrm{loc}, s)` instead of as a unit
        offset multiplied by the population scatter :math:`s`.  Default False.
    scale_site_name : str or None, optional
        For member deviation offsets, the ``sample_name`` of the population scatter parameter whose
        :math:`\\exp(\\mathrm{latent})` is the scatter :math:`s`; None (default) otherwise.
    """
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
        """The distribution of the latent (sampled) value.

        Returns
        -------
        numpyro.distributions.Distribution
            The base distribution of ``prior``: ``prior.base_dist`` for an exp-transformed prior (a distribution of
            :math:`\\ln` of the physical value), ``prior`` itself otherwise.

        Raises
        ------
        ValueError
            If ``prior`` is not a supported distribution (see :func:`kappaccino.priors.split`).
        """
        return priors.split(self.prior)[0]

    @cached_property
    def log(self) -> bool:
        """Whether the parameter is sampled in log space.

        Returns
        -------
        bool
            True when the physical value is :math:`\\exp(\\mathrm{latent})` (an exp-transformed prior), False when the
            latent and physical values are equal.

        Raises
        ------
        ValueError
            If ``prior`` is not a supported distribution (see :func:`kappaccino.priors.split`).
        """
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
    """The observed images of one multiple-image family (one source), in catalog order.

    Positions are arcsec offsets from the reference, :math:`x = (\\alpha_0 - \\alpha)\\cos\\delta_0 \\cdot 3600`
    (positive west) and :math:`y = (\\delta - \\delta_0) \\cdot 3600`.  ``ModelState.family_data`` holds one per family,
    sorted by family id.

    Attributes
    ----------
    family_id : str
        Family identifier from the image catalog.
    z_source : float
        Source redshift of the family (one value for all its images).
    image_labels : list of str
        Image labels, shape (image,).
    x_obs : numpy.ndarray
        Observed image x positions in arcsec (positive west), shape (image,).
    y_obs : numpy.ndarray
        Observed image y positions in arcsec (positive north), shape (image,).
    reliability : numpy.ndarray
        Catalog family reliability per image in [0, 1] (1 where the catalog gives none), shape (image,).
    """
    family_id: str
    z_source: float
    image_labels: list[str]
    x_obs: np.ndarray
    y_obs: np.ndarray
    reliability: np.ndarray

    @property
    def effective_z_source(self) -> float:
        """The source redshift used for lensing (equal to ``z_source``)."""
        return self.z_source

    @property
    def n_images(self) -> int:
        """Number of images in the family."""
        return len(self.image_labels)

    @property
    def x_center(self) -> float:
        """Midpoint of the image x range in arcsec (NaN for an empty family)."""
        if self.n_images == 0:
            return float("nan")
        return float(0.5 * (np.min(self.x_obs) + np.max(self.x_obs)))

    @property
    def y_center(self) -> float:
        """Midpoint of the image y range in arcsec (NaN for an empty family)."""
        if self.n_images == 0:
            return float("nan")
        return float(0.5 * (np.min(self.y_obs) + np.max(self.y_obs)))

    @property
    def search_window(self) -> float:
        """Side of the image-search window in arcsec: the larger image span plus 8 arcsec of padding."""
        span_x = np.ptp(self.x_obs) if self.n_images > 1 else 0.0
        span_y = np.ptp(self.y_obs) if self.n_images > 1 else 0.0
        return float(max(span_x, span_y) + DEFAULT_SEARCH_PADDING)


@dataclass(frozen=True)
class ImageData:
    """All constrained images of every family, flattened, with per-family bookkeeping.

    Images are ordered by source redshift, then family (families sorted by id within a redshift), then catalog order.
    The family axis of ``family_ids``, ``source_x_param_index``, ``source_y_param_index`` and of the per-family
    outputs of :class:`~kappaccino.likelihood.evaluator.Evaluator` (for example ``source_positions``) follows this order, which
    can differ from the order of ``ModelState.family_data``.  Positions are arcsec offsets from the reference
    (:math:`x` positive west, :math:`y` positive north).

    Attributes
    ----------
    family_ids : tuple of str
        Family identifiers in image-data order, length (family,).
    n_families : int
        Number of families.
    x_obs : jax.numpy.ndarray
        Observed image x positions in arcsec (positive west), shape (image,).
    y_obs : jax.numpy.ndarray
        Observed image y positions in arcsec (positive north), shape (image,).
    parity : jax.numpy.ndarray
        Catalog image parity, -1, +1, or 0 when unknown, int32 of shape (image,).
    z_index_per_image : jax.numpy.ndarray
        Index of each image's source redshift in ``ModelState.z_source_values``, int32 of shape (image,).
    family_index_per_image : jax.numpy.ndarray
        Index of each image's family in ``family_ids``, int32 of shape (image,).
    source_x_param_index : jax.numpy.ndarray
        Parameter index of each family's source x coordinate (``beta_x``), int32 of shape (family,).  The latent
        value there is a whitened unit offset; the reported physical value is the source position in source-plane
        arcsec.
    source_y_param_index : jax.numpy.ndarray
        The same for the source y coordinate (``beta_y``).
    image_labels : tuple of str
        Image labels, length (image,).
    system_ids : tuple of str, optional
        Source systems: a system holds the knots (families) of one source galaxy (the ``system_id`` catalog column;
        each family is its own system without it).  Default empty.
    system_index_per_image : jax.numpy.ndarray or None, optional
        Index of each image's system in ``system_ids``, int32 of shape (image,); None by default.
    """
    family_ids: tuple[str, ...]
    n_families: int
    x_obs: jnp.ndarray
    y_obs: jnp.ndarray
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
    """Population members: identities, component rows, parameter indices, and classes.

    Members are the galaxies of every member population placed on the scaling relations, in population then catalog
    order; "position" below means the index along this member axis.  Exact members are evaluated with the full dPIE
    kernel; far-field members with a series expansion that holds while every image stays beyond a guard distance.

    Attributes
    ----------
    member_ids : tuple of str
        Member identifiers ``"<population_id>:<catalog_id>"``, length (member,).
    component_indices : numpy.ndarray
        Row of each member in the packed lens components, int32 of shape (member,).
    scatter_param_index : numpy.ndarray
        Parameter index of each member's population velocity (:math:`\\ln\\sigma`) scatter, shape (member,).
    offset_param_index : numpy.ndarray
        Parameter index of each member's sampled velocity deviation offset, shape (member,).
    exact_positions : numpy.ndarray
        Member positions evaluated exactly, int32 of shape (exact member,).
    far_positions : numpy.ndarray
        Member positions evaluated by the far-field series, int32 of shape (far member,).
    source_prior_mean : tuple of float
        Mean (x, y) of the Gaussian source-position prior, source-plane arcsec.
    source_prior_sigma : float
        Width of the source-position prior per coordinate, source-plane arcsec.
    far_field_guard_cut_radii : float
        Guard distance of the far-field approximation in units of each member's cut radius.
    classification : dict, optional
        The exact/far-field classification record (written to ``classification.json``), with ``centered_count`` and
        ``centered_members``.  Default empty.
    far_field_barrier_width : float, optional
        Width of the soft guard barrier in :math:`\\ln(R / (g\\, r_\\mathrm{cut}))` (dimensionless).  Default 0.05.
    far_field_barrier_strength : float, optional
        Log-likelihood penalty scale of the barrier.  Default 25.
    radius_scatter_param_index : numpy.ndarray, optional
        Parameter index of each member's population radius (:math:`\\ln r`) scatter, shape (member,); -1 where absent
        (only the ``sigma_rcut`` and ``mass_rcut`` deviation modes have one).
    radius_offset_param_index : numpy.ndarray, optional
        Parameter index of each member's radius deviation offset, shape (member,); -1 where absent.
    centered : numpy.ndarray, optional
        True for members whose deviations are sampled directly (centered) rather than as unit offsets, bool of shape
        (member,).
    membership_mode : str, optional
        ``"fixed"`` (every member counts fully) or ``"marginalize"`` (``MemberConfig.membership``).  Default ``"fixed"``.
    membership_prior : numpy.ndarray, optional
        Prior membership probability :math:`p_i` of each member in (0, 1], shape (member,); empty means all 1.
    membership_max_enumerated : int, optional
        Most uncertain exact members whose membership indicators are summed exactly.  Default 4.
    membership_calibration_param_index : int, optional
        Parameter index of the log-odds shift :math:`\\delta` of the membership probabilities; -1 when absent.
    """
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
    membership_mode: str = "fixed"
    membership_prior: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    membership_max_enumerated: int = 4
    membership_calibration_param_index: int = -1


@dataclass
class ModelState:
    """Everything fixed about one lens model: parameters, lens components, images, members and error model.

    Built by :func:`kappaccino.core.state.build_state` from a configuration and passed to
    :class:`~kappaccino.likelihood.evaluator.Evaluator`.  Sky positions throughout are arcsec offsets from the reference
    (``reference``): :math:`x = (\\alpha_0 - \\alpha)\\cos\\delta_0 \\cdot 3600` (positive west) and
    :math:`y = (\\delta - \\delta_0) \\cdot 3600`.

    Attributes
    ----------
    run_name : str
        Run name (``paths.run_name``, ``"cluster"`` when unset).
    cosmo_config : dict
        Cosmology: ``"class"`` (astropy class name), ``"H0"`` (km s^-1 Mpc^-1), ``"Om0"``, ``"Ode0"``, ``"w0"``.
    z_lens : float
        Lens (cluster) redshift.
    kpc_per_arcsec : float
        Proper kpc per arcsec at ``z_lens`` for ``cosmo_config``.
    reference : tuple of (int, float, float)
        ``(reference mode, ra0, dec0)``: the Lenstool reference mode and the reference position in degrees that all
        arcsec offsets are measured from.
    parameter_specs : list of ParameterSpec
        The sampled parameters, in the order of the parameter axis of ``initial_theta``, of every latent vector and
        of ``PosteriorResults.samples`` / ``PosteriorResults.physical``.  Order: halo parameters, population scaling
        parameters, member scatters and offsets, member shape offsets, error-model scales, source offsets
        (``beta_x``, ``beta_y`` per family), softening length.
    base_components : list of dict
        One dict per lens component (halos first, then members) with its fixed Lenstool-style fields (``id``,
        ``profil``, ``x_centre``, ``y_centre`` in arcsec, ``ellipticite``, ``angle_pos`` in degrees,
        ``core_radius_kpc``, ``cut_radius_kpc`` in kpc, ``v_disp`` in km s^-1, ...).
    packed_lens_spec : PackedLensSpec
        Per-component fixed values and the parameter index that overrides each field (-1 where the field is fixed).
    lens_model_list : list of str
        Profile name of each lens component.
    family_data : list of FamilyData
        The observed families, sorted by family id.
    images : ImageData
        All constrained images, flattened (see :class:`ImageData` for the image and family order).
    potfiles : list of dict
        The member populations (Lenstool potfiles): ``id``, ``type``, ``catalog_df`` (the member catalog as a
        DataFrame), ``mag0`` (pivot magnitude), ``zlens``, nominal ``sigma_nominal`` (km s^-1), ``cutkpc_nominal`` and
        ``corekpc`` (kpc), and the priors of the reference parameters.
    scaling_component_records : list of dict
        One record per member: ``potfile_id``, ``catalog_id``, ``component_index``, ``catalog_mag``,
        ``luminosity_ratio`` :math:`L/L_*`, ``x_centre``, ``y_centre`` (arcsec), ``ellipticite``, ``angle_pos``
        (degrees), and catalog flags.
    members : MemberModel
        Member identities, parameter indices and exact/far-field classes.
    z_source_values : list of float
        The distinct source redshifts, sorted ascending (``ImageData.z_index_per_image`` indexes this list).
    dpie_sigma0_factor_by_z : dict
        Lensing-efficiency factor of the dPIE normalization for each source redshift (filled on demand for new
        redshifts by ``Evaluator.packed_state_for_z``).
    softening_length_param_index : int, optional
        Index into ``parameter_specs`` of the softening length (kpc).  Like every ``*_param_index`` field below:
        -1 when the model does not sample the parameter.
    sigma_def_param_index : int, optional
        ``smoothed`` likelihood: the source-plane deflection scatter :math:`\\sigma_\\mathrm{def}` (arcsec).
    sigma_meas_param_index : int, optional
        Every mode: the sampled independent measurement scatter :math:`\\sigma_\\mathrm{meas}` of every image, the
        only measurement error (arcsec).
    sigma_sel_param_index : int, optional
        ``smoothed``: the feature-selection scatter :math:`\\sigma_\\mathrm{sel}` (arcsec).
    lambda_sel_param_index : int, optional
        ``smoothed``: the stretching scale :math:`\\lambda_\\mathrm{sel}` of the selection weight :math:`R = (I
        + A A^T / \\lambda_\\mathrm{sel}^2)^{-1}` (dimensionless).
    sigma_gp_param_index : int, optional
        ``gp_image_plane``: the correlated image-plane scale :math:`\\sigma_\\mathrm{shared}` (arcsec).
    ell_gp_param_index : int, optional
        ``gp_image_plane``: the correlation length :math:`\\ell` (arcsec); -1 when it is fixed.
    sigma_grf_param_index : int, optional
        Knee kernel: the mass-field deflection amplitude :math:`\\sigma_\\alpha` (arcsec); -1 without the mass
        field.
    lambda_b_param_index : int, optional
        Knee kernel: the break scale :math:`\\lambda_b` (arcsec).
    gp_knee_gamma_param_index : int, optional
        Knee kernel: the sampled spectral slope :math:`\\gamma` (dimensionless); -1 when fixed.
    sigma_grf2_param_index : int, optional
        Knee kernel, ``two_knee`` / ``knee_foreground`` / ``knee_background`` spectra: the second field's
        amplitude (arcsec).
    lambda_b2_param_index : int, optional
        Knee kernel: the second field's knee (arcsec).
    gp_knee_beta_low_param_index : int, optional
        Knee kernel, ``broken_power_law`` spectrum: the low-k slope (dimensionless).
    gp_correlation_length_arcsec : float, optional
        ``gp_image_plane``: the correlation length :math:`\\ell` in arcsec (fixed value, or the start when sampled).
        Default 7.
    gp_floor_arcsec : float, optional
        ``gp_image_plane``: the source-plane floor :math:`\\sigma_\\mathrm{fl}` in arcsec.  Default 0.0015.
    gp_kernel : str, optional
        ``"image_plane"`` (default) or ``"knee"`` (mass field plus image-plane GP; see ``LikelihoodConfig``).
    gp_reference_redshift : float, optional
        Knee kernel: the reference redshift :math:`z_\\mathrm{ref}` of the redshift weights.  Default 2.
    gp_blocks : str, optional
        Knee kernel covariance blocks: ``"all"`` (default), ``"system"`` or ``"family"``.
    gp_knee : dict, optional
        Knee kernel settings: slope ``gamma``, ``spectrum``, ``scales`` and ``lambda_grid`` of the tables, ``r_step``
        (arcsec), fixed ``knee_fixed`` / ``ell_fixed`` values, projected modes, and the slope grids.
    freeze_determinant_gradient : bool, optional
        Hold :math:`\\nabla \\det A` fixed in the backward pass.  Default False.
    dropped_member_ids : tuple of str, optional
        Catalog members removed from the lens model (``members.min_deflection_arcsec``).  Default empty.
    member_kinematics : dict, optional
        Measured member dispersions in the likelihood (``member_ids``, ``sources``, ``component_index``, ``sigma`` and
        ``err`` in km s^-1, ``log_err``, ``aperture_factor``); empty when no kinematics source is on.
    initial_theta : numpy.ndarray or None, optional
        The latent start vector, shape (parameter,): prior centers, with the configured starts of the error-model
        scales.  Also the known-valid reference at which ``Evaluator.log_likelihood`` evaluates rejected proposals.
    slots : partition.slots.SlotModel or None, optional
        Uncertain images and the source slots they may come from (``partition.slots.add_slots``); their parameters are
        the last ones.  None (default) without uncertain images.
    """
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
    sigma_sel_param_index: int = -1              # feature-selection scatter; -1 when the term is disabled
    lambda_sel_param_index: int = -1             # stretching scale of the selection weight R = (I + A A^T / lambda_sel^2)^-1; -1 with the term
    sigma_gp_param_index: int = -1           # gp_image_plane: the correlated image-plane scale; -1 under the other likelihoods
    sigma_meas_param_index: int = -1            # the sampled independent measurement scatter of every image (all modes)
    ell_gp_param_index: int = -1                 # gp_image_plane: the sampled correlation length; -1 when ell is fixed
    gp_correlation_length_arcsec: float = 7.0    # gp_image_plane: ell (fixed value or start) and sigma_fl (LikelihoodConfig)
    gp_floor_arcsec: float = 0.0015
    gp_kernel: str = "image_plane"               # gp_image_plane: "image_plane" (B S B^T) or "knee" (mass field + B S B^T, see LikelihoodConfig)
    gp_reference_redshift: float = 2.0           # knee kernel: z_ref of the redshift weights
    gp_blocks: str = "all"                       # knee kernel: "all" | "system" | "family" covariance blocks
    gp_knee: dict[str, Any] = field(default_factory=dict)   # knee kernel: gamma, scales, lambda grid, r step
    sigma_grf_param_index: int = -1         # knee kernel: sigma_grf; -1 when the mass field is dropped
    lambda_b_param_index: int = -1                # knee kernel: the break scale lambda_b
    gp_knee_gamma_param_index: int = -1          # knee kernel: the sampled slope gamma; -1 when fixed
    sigma_grf2_param_index: int = -1        # knee kernel, two_knee / knee_foreground spectra: the second field's sigma_grf
    lambda_b2_param_index: int = -1               # ... and its knee
    gp_knee_beta_low_param_index: int = -1       # knee kernel, broken_power_law spectrum: the low-k slope
    freeze_determinant_gradient: bool = False    # hold grad det A fixed in the backward pass (LikelihoodConfig)
    dropped_member_ids: tuple[str, ...] = ()  # catalog members removed from the lens model (members.min_deflection_arcsec)
    # Measured member dispersions in the likelihood (state.build_member_kinematics); empty when no source is on.
    member_kinematics: dict[str, Any] = field(default_factory=dict)
    initial_theta: np.ndarray | None = None
    # Uncertain images assigned to source slots (partition.slots.SlotModel); its parameters follow every other parameter.
    slots: Any = None

    @property
    def n_parameters(self) -> int:
        """Number of sampled parameters (``len(parameter_specs)``)."""
        return len(self.parameter_specs)


# Plotting code written against the previous solver uses this name.
BuildState = ModelState


@dataclass
class PosteriorResults:
    """NUTS posterior draws and sampler diagnostics of one run (saved as ``final/artifacts/posterior.h5``).

    The draw axis concatenates the chains chain by chain: ``draw = num_chains * draws_per_chain``, where
    ``draws_per_chain`` is ``sample_steps`` divided by the thinning.  The parameter axis follows
    ``ModelState.parameter_specs``.  Non-finite draws are kept (their count is
    ``diagnostics["n_nonfinite_draws"]``).

    Attributes
    ----------
    samples : numpy.ndarray
        Latent draws, shape (draw, parameter): :math:`\\ln` of the value for log-sampled parameters, the value
        otherwise; source coordinates are whitened unit offsets.  Pass rows to ``Evaluator`` methods.
    physical : numpy.ndarray
        Reported physical draws, shape (draw, parameter), in the units of each parameter (arcsec, kpc, km s^-1,
        :math:`\\log_{10} M_\\odot` for ``log10_m200``, dimensionless; see :class:`ParameterSpec`).  Source
        coordinates are the physical source positions in source-plane arcsec
        (:meth:`Evaluator.reported_physical <kappaccino.likelihood.evaluator.Evaluator.reported_physical>`).
    log_prob : numpy.ndarray
        Unnormalized log posterior density of each draw (minus the NumPyro potential energy, in NumPyro's
        unconstrained coordinates), shape (draw,).
    accept_prob : numpy.ndarray
        NUTS acceptance probability of each transition, shape (draw,).
    diverging : numpy.ndarray
        True for divergent transitions, bool of shape (draw,).
    num_steps : numpy.ndarray
        Leapfrog steps of each transition, shape (draw,).
    warmup_steps : int
        Warmup iterations per chain.
    sample_steps : int
        Sampling iterations per chain (before thinning).
    num_chains : int
        Number of chains.
    energy : numpy.ndarray
        Hamiltonian energy of each transition (used for E-BFMI), shape (draw,).
    step_size : numpy.ndarray
        Adapted step size of each chain at the end of warmup, shape (chain,).
    grouped_samples : numpy.ndarray or None, optional
        Latent draws grouped by chain, shape (chain, draw per chain, parameter).  After
        :func:`~kappaccino.io.artifacts.load_posterior` a missing value is a 0-d object array holding None.
    diagnostics : dict, optional
        Sampler diagnostics (``posterior_diagnostics.json``): split R-hat ``r_hat`` and effective sample size
        ``n_eff`` per parameter (lists of length parameter), ``max_r_hat``, ``min_n_eff``, ``adaptation`` (per-chain step
        size, E-BFMI, acceptance, divergences, tree-depth saturation), ``mass_seed``, ``elapsed_seconds``,
        ``chain_method``, ``n_nonfinite_draws``, ``stiffness`` and ``geometry``.
    median_fit : numpy.ndarray or None, optional
        Per-parameter median of ``physical`` over the finite draws, shape (parameter,).
    maximum_likelihood_fit : numpy.ndarray or None, optional
        ``physical`` at the draw with the highest ``log_prob``, shape (parameter,).  Despite the name this is the
        maximum-posterior draw, not a likelihood maximum.
    """
    samples: np.ndarray                     # (draws, n_params) latent
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
    """Latent (sampled) value of one physical parameter value.

    Parameters
    ----------
    value : float
        Physical value, in the units of the parameter (see :class:`ParameterSpec`).
    spec : ParameterSpec
        The parameter.

    Returns
    -------
    float
        :math:`\\ln(\\max(\\mathrm{value}, 10^{-12}))` when ``spec.log``, ``value`` otherwise.
    """
    return float(np.log(max(float(value), 1.0e-12))) if spec.log else float(value)


def latent_to_physical(value: float, spec: ParameterSpec) -> float:
    """Physical value of one latent (sampled) parameter value.

    Parameters
    ----------
    value : float
        Latent value.
    spec : ParameterSpec
        The parameter.

    Returns
    -------
    float
        :math:`\\exp(\\mathrm{value})` when ``spec.log``, ``value`` otherwise, in the units of the parameter.
    """
    return float(np.exp(float(value))) if spec.log else float(value)


def latent_array_to_physical(values: np.ndarray, spec: ParameterSpec) -> np.ndarray:
    """Physical values of an array of latent values of one parameter (for example its column of draws).

    Parameters
    ----------
    values : array_like
        Latent values of the parameter, any shape.
    spec : ParameterSpec
        The parameter.

    Returns
    -------
    numpy.ndarray
        Physical values, same shape as ``values``: ``exp(values)`` when ``spec.log``, ``values`` as float otherwise.
    """
    array = np.asarray(values, dtype=float)
    return np.exp(array) if spec.log else array


def convert_theta_to_physical(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    """Physical values of one latent parameter vector, parameter by parameter.

    Source coordinates stay whitened unit offsets; use
    :meth:`Evaluator.reported_physical <kappaccino.likelihood.evaluator.Evaluator.reported_physical>` for physical source
    positions.

    Parameters
    ----------
    theta : array_like
        Latent vector, shape (parameter,).
    parameter_specs : list of ParameterSpec
        ``ModelState.parameter_specs``, the same order as ``theta``.

    Returns
    -------
    numpy.ndarray
        Physical vector, shape (parameter,).
    """
    theta_array = np.asarray(theta, dtype=float)
    return np.asarray([latent_to_physical(v, spec) for v, spec in zip(theta_array, parameter_specs)], dtype=float).reshape(theta_array.shape)


def convert_theta_to_latent(theta: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    """Latent values of one physical parameter vector, parameter by parameter (the inverse of
    :func:`convert_theta_to_physical`).

    Parameters
    ----------
    theta : array_like
        Physical vector, shape (parameter,).  Source entries are taken as unit offsets, unchanged; use
        :meth:`Evaluator.latent_from_reported_physical
        <kappaccino.likelihood.evaluator.Evaluator.latent_from_reported_physical>` for a vector holding source positions.
    parameter_specs : list of ParameterSpec
        ``ModelState.parameter_specs``, the same order as ``theta``.

    Returns
    -------
    numpy.ndarray
        Latent vector, shape (parameter,).  Values of log-sampled parameters are floored at :math:`10^{-12}` before
        the logarithm.
    """
    theta_array = np.asarray(theta, dtype=float)
    return np.asarray([physical_to_latent(v, spec) for v, spec in zip(theta_array, parameter_specs)], dtype=float).reshape(theta_array.shape)


def convert_sample_matrix_to_physical(samples: np.ndarray, parameter_specs: list[ParameterSpec]) -> np.ndarray:
    """Physical values of an array of latent draws.

    Unlike ``PosteriorResults.physical``, source coordinates stay whitened unit offsets.

    Parameters
    ----------
    samples : array_like
        Latent draws, shape (..., parameter), for example (draw, parameter) or (chain, draw, parameter).
    parameter_specs : list of ParameterSpec
        ``ModelState.parameter_specs``, matching the last axis of ``samples``.

    Returns
    -------
    numpy.ndarray
        Physical draws, same shape as ``samples`` (a copy).
    """
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
    """Latent vector at the prior centers: the configured values for lens parameters, zero offsets.

    Each value is the center of the latent prior (:func:`kappaccino.priors.center`), clipped into the prior support
    and at least ``boundary_frac`` of the prior width away from a bound.

    Parameters
    ----------
    parameter_specs : list of ParameterSpec
        ``ModelState.parameter_specs``.
    boundary_frac : float, optional
        Minimum distance from a prior bound, as a fraction of the width of a uniform prior or of the bound's magnitude
        for a prior bounded below.  Default 0.02.

    Returns
    -------
    numpy.ndarray
        Latent vector, shape (parameter,).
    """
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
