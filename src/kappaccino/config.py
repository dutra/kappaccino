"""Typed configuration of a kappaccino run.

A run is described by one :class:`LensClusterSolverConfig`, which groups the lens model
(:class:`LensModelConfig`: reference frame, cosmology, cluster-scale halos, member populations and image
constraints) with the settings of each stage: output paths (:class:`RunPathsConfig`), runtime
(:class:`RuntimeConfig`), the initial search (:class:`SearchConfig`), NUTS sampling (:class:`SamplingConfig`), member
classification (:class:`MemberConfig`), the member scaling relation (:class:`ScalingModelConfig`), the likelihood and
error model (:class:`LikelihoodConfig`) and the output diagnostics (:class:`ImageDiagnosticsConfig`,
:class:`TruthRecoveryConfig`, :class:`ImageCatalogCutoutConfig`) and the posterior draws behind them
(:class:`ReportingConfig`).  Every class is a frozen dataclass; change a field
with :func:`dataclasses.replace` or :meth:`LensClusterSolverConfig.with_updates`, and check a finished configuration
with :meth:`LensClusterSolverConfig.validate`.

Image-plane coordinates are offsets in arcsec from the reference point of :class:`ReferenceFrameConfig`, with
:math:`x = 3600\\,(\\alpha_0 - \\alpha)\\cos\\delta_0` (positive toward the west) and
:math:`y = 3600\\,(\\delta - \\delta_0)` (positive toward the north).  Position angles are in degrees, measured from
the +x axis toward +y.

Priors are ``numpyro.distributions`` objects of the physical value of each sampled quantity (see
:mod:`kappaccino.priors`).  A prior transformed by ``exp`` (``dist.LogNormal``, ``dist.LogUniform``,
``priors.truncated_lognormal``, ``priors.lognormal_from_moments``) is sampled as the natural log of the value; any
other supported prior (``dist.Uniform``, ``dist.Normal``, ``dist.TruncatedNormal``, ``dist.StudentT``) is sampled as
the value itself.  Creating one of the exp-transformed priors starts JAX, so call
``cpu_runtime.configure_cpu_runtime`` before building a configuration.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field, fields, is_dataclass, replace
from numbers import Integral
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import numpyro.distributions as dist

from .priors import (cut_excess_from_moments, ellipticity_box, is_distribution, lognormal_from_moments, physical_bounds, shear_box, split,
                     truncated_lognormal)

# Lower truncation (arcsec) of the default error-scale priors of LikelihoodConfig.
ERROR_SCALE_FLOOR_ARCSEC = 1.0e-4


def _error_scale_prior(median: float, log_sigma: float, upper: float):
    """Log-normal error-scale prior truncated to [ERROR_SCALE_FLOOR_ARCSEC, upper]."""
    return truncated_lognormal(median, log_sigma, low=ERROR_SCALE_FLOOR_ARCSEC, high=upper)


@dataclass(frozen=True)
class RunPathsConfig:
    """Where a run writes its products and how its figures are named.

    Parameters
    ----------
    output_dir : str or pathlib.Path, optional
        Parent directory of all runs.  A run writes into ``output_dir / run_name``.  Default ``"results"``.
    run_name : str or None, optional
        Name of the run directory: a nonempty single directory name without path separators or surrounding
        whitespace.  Required in practice: :meth:`LensClusterSolverConfig.validate` raises ``ValueError`` for None.
        Default None.
    figure_prefix : str or None, optional
        Prefix of figure file names, joined to the name with an underscore (``"<prefix>_corner_globals.pdf"``).  None
        writes most figures without a prefix (a few use ``"cluster"``).  Default None.
    """
    output_dir: str | Path = "results"
    run_name: str | None = None
    figure_prefix: str | None = None


@dataclass(frozen=True)
class ReferenceFrameConfig:
    """Reference point of the image-plane coordinates.

    Every sky position (images, members) is converted to arcsec offsets from (``ra0_deg``, ``dec0_deg``):
    :math:`x = 3600\\,(\\alpha_0 - \\alpha)\\cos\\delta_0` (positive toward the west) and
    :math:`y = 3600\\,(\\delta - \\delta_0)` (positive toward the north).

    Parameters
    ----------
    reference : int, optional
        Lenstool ``runmode reference`` code, carried with the frame (``lenstool_par`` reads it from a Lenstool
        parameter file).  The coordinate conversion above does not depend on it.  Default ``3``.
    ra0_deg : float, optional
        Right ascension of the reference point, in degrees.  Default ``0.0``.
    dec0_deg : float, optional
        Declination of the reference point, in degrees.  Default ``0.0``.
    """
    reference: int = 3
    ra0_deg: float = 0.0
    dec0_deg: float = 0.0


@dataclass(frozen=True)
class CosmologyConfig:
    """Background cosmology of the lens model.

    The lensing distances (JAX, ``jax_cosmology``) are those of a flat universe with matter density ``Om0`` and a
    dark-energy equation of state ``w0``; ``class_name`` selects the astropy cosmology of the exact-image solver and
    of the output maps.

    Parameters
    ----------
    H0 : float, optional
        Hubble constant, in km s^-1 Mpc^-1.  Must be positive.  Default ``70.0``.
    Om0 : float, optional
        Matter density parameter at z = 0, dimensionless.  Must be nonnegative.  Default ``0.3``.
    Ode0 : float or None, optional
        Dark-energy density parameter at z = 0, dimensionless.  Used only in :math:`H(z)` for the critical density
        that defines an NFW halo's :math:`M_{200}`; the distances assume a flat universe.  None means ``1 - Om0``.
        Default None.
    w0 : float, optional
        Dark-energy equation-of-state parameter, dimensionless.  Used by the JAX distances and by the astropy
        ``"FlatwCDM"`` cosmology.  Default ``-1.0``.
    class_name : str, optional
        astropy cosmology used by the exact-image solver, the convergence maps and the truth comparison:
        ``"FlatLambdaCDM"`` (ignores ``w0``) or ``"FlatwCDM"`` (uses ``w0``).  Any other value behaves as
        ``"FlatLambdaCDM"``.  Keep it consistent with ``w0``.  Default ``"FlatLambdaCDM"``.
    """
    H0: float = 70.0
    Om0: float = 0.3
    Ode0: float | None = None
    w0: float = -1.0
    class_name: str = "FlatLambdaCDM"


@dataclass(frozen=True)
class DPIEHaloConfig:
    """Elliptical dPIE (Lenstool profile 81) halo, usually a cluster-scale component.

    Each field below is the fixed value of the halo's parameter; a parameter with an entry in ``priors`` is sampled
    instead.  :meth:`cluster_scale` builds a halo with the standard priors.

    Parameters
    ----------
    id : str
        Name of the component; the sampled parameters are named ``"<id>.<field>"``.
    x_centre : float
        Center x offset from the reference point, in arcsec.
    y_centre : float
        Center y offset from the reference point, in arcsec.
    ellipticite : float
        Lenstool ellipticity :math:`(a^2 - b^2)/(a^2 + b^2)` of the mass distribution, dimensionless, in [0, 1).
    angle_pos : float
        Position angle of the major axis, in degrees from the +x axis toward +y.
    core_radius_kpc : float
        Core radius, in kpc.  Must be positive.
    cut_radius_kpc : float
        Cut (truncation) radius, in kpc.  Must be positive.  When the cut radius is sampled (``priors["cut_excess_kpc"]``)
        the model uses the effective core plus the sampled excess instead.
    v_disp : float
        Velocity dispersion of the dPIE in km s^-1 (Lenstool's fiducial :math:`\\sigma_{LT}`).  Must be positive.
    z_lens : float
        Redshift of the halo.  Must be positive.  The first halo's redshift sets the kpc-per-arcsec scale of the model.
    profile_type : int, optional
        Lenstool profile code; must be ``81`` (dPIE).  Default ``81``.
    priors : dict of str to numpyro distribution or None, optional
        Priors of the sampled parameters, keyed by: ``"x_centre"`` and ``"y_centre"`` (arcsec); ``"e1"`` and ``"e2"``
        (the shape as the Cartesian pair :math:`|e|(\\cos 2\\phi, \\sin 2\\phi)` with :math:`|e| = (1 - q)/(1 + q)`,
        dimensionless; give both or neither, see ``priors.ellipticity_box``); ``"core_radius_kpc"`` (kpc);
        ``"cut_excess_kpc"`` (the cut radius above the effective core, :math:`r_{cut} - \\sqrt{r_{core}^2 +
        s^2}` with :math:`s` the softening of ``ScalingModelConfig.softening_length_prior``, in kpc; see
        ``priors.cut_excess_from_moments``); ``"v_disp"`` (km s^-1).  A parameter without an entry, or with None, is
        fixed at its field value.  Any other key raises ``ValueError`` when the model is built.  Default: empty
        (every parameter fixed).
    ellipticity_soft_cap : float or None, optional
        Smooth upper limit on the ellipticity, in Lenstool ``ellipticite`` (converted internally to the modulus
        :math:`|e|`).  Below the cap the shape prior is the (``e1``, ``e2``) box of ``priors``; above it the log prior
        falls as :math:`-\\mathrm{softplus}((|e| - \\mathrm{cap}) / \\mathrm{width})`, a smooth wall in place of the
        hard edge of the box, which is stiff for NUTS.  Needs uniform ``e1`` and ``e2`` priors whose box reaches beyond
        the cap; must lie in (0, 1).  None keeps the hard box only.  Default None.
    ellipticity_soft_width : float, optional
        Width of the soft wall in the modulus :math:`|e|`, dimensionless.  Used only with ``ellipticity_soft_cap``;
        must be positive.  Default ``0.02``.
    """
    id: str
    x_centre: float
    y_centre: float
    ellipticite: float
    angle_pos: float
    core_radius_kpc: float
    cut_radius_kpc: float
    v_disp: float
    z_lens: float
    profile_type: int = 81
    priors: dict[str, Any] = field(default_factory=dict)
    # Soft cap history: a hard bound is stiff for NUTS: halo 1 of M0416 at |e| 0.61 against 0.627 set the step size
    # (2026-11-03).
    ellipticity_soft_cap: float | None = None
    ellipticity_soft_width: float = 0.02

    @classmethod
    def cluster_scale(cls, id: str, x_centre: float, y_centre: float, z_lens: float, *, v_disp: float = 800.0,
                      position_half_width_arcsec: float = 5.0, ellipticity_upper: float = 0.8, softening_kpc: float = 0.0) -> "DPIEHaloConfig":
        """Build a cluster-scale dPIE halo with the standard priors.

        The center is uniform in a box around (``x_centre``, ``y_centre``); the shape is uniform in the (e1, e2) box of
        Lenstool ellipticities up to ``ellipticity_upper`` at any angle; the core radius is log-uniform on 0.5-100 kpc;
        the cut radius is about 1500 +- 300 kpc on 500-3000 kpc (the arcs do not constrain it), as a log-normal prior
        of its excess above the effective core; the dispersion is a log-normal of mean ``v_disp`` and standard
        deviation ``0.35 * v_disp`` truncated to 100-2200 km s^-1.  The core, cut excess and dispersion are sampled in
        log.  The fixed fields (used only if a prior is later removed) are ``ellipticite`` 0.3, ``angle_pos`` 0,
        a 10 kpc core and a 1500 kpc cut radius.

        Parameters
        ----------
        id : str
            Name of the component.
        x_centre, y_centre : float
            Center of the position prior, in arcsec from the reference point.
        z_lens : float
            Redshift of the halo.
        v_disp : float, optional
            Mean of the velocity-dispersion prior and the fixed value, in km s^-1.  Default ``800.0``.
        position_half_width_arcsec : float, optional
            Half-width of the uniform position box, in arcsec.  Default ``5.0``.
        ellipticity_upper : float, optional
            Largest Lenstool ``ellipticite`` the shape box covers, dimensionless.  Default ``0.8``.
        softening_kpc : float, optional
            Resolution softening added in quadrature to the core (``ScalingModelConfig.softening_length_prior``), in
            kpc; it sets where the cut radius's excess starts.  Default ``0.0``.

        Returns
        -------
        DPIEHaloConfig
            The halo, with ``priors`` on the center, shape, core, cut excess and dispersion.
        """
        half, core = float(position_half_width_arcsec), 10.0
        return cls(id=str(id), x_centre=float(x_centre), y_centre=float(y_centre), ellipticite=0.3, angle_pos=0.0, core_radius_kpc=core,
                   cut_radius_kpc=1500.0, v_disp=float(v_disp), z_lens=float(z_lens), priors={
                       "x_centre": dist.Uniform(x_centre - half, x_centre + half),
                       "y_centre": dist.Uniform(y_centre - half, y_centre + half),
                       **ellipticity_box(float(ellipticity_upper)),
                       "core_radius_kpc": dist.LogUniform(0.5, 100.0),
                       "cut_excess_kpc": cut_excess_from_moments(1500.0, 300.0, low=500.0, high=3000.0,
                                                                 core=math.hypot(core, float(softening_kpc))),
                       "v_disp": lognormal_from_moments(float(v_disp), 0.35 * float(v_disp), low=100.0, high=2200.0)})


@dataclass(frozen=True)
class NFWHaloConfig:
    """Elliptical NFW (Lenstool profile 12) cluster-scale halo.

    The halo is set by its mass :math:`M_{200}` (the mass within the radius enclosing 200 times the critical density
    at ``z_lens``, in Msun with no factor of h) and its scale radius; the ellipticity is that of the convergence, in
    Lenstool's ``ellipticite`` convention.

    Parameters
    ----------
    id : str
        Name of the component; the sampled parameters are named ``"<id>.<field>"``.
    x_centre : float
        Center x offset from the reference point, in arcsec.
    y_centre : float
        Center y offset from the reference point, in arcsec.
    ellipticite : float
        Lenstool ellipticity :math:`(a^2 - b^2)/(a^2 + b^2)` of the convergence, dimensionless, in [0, 1).
    angle_pos : float
        Position angle of the major axis, in degrees from the +x axis toward +y.
    log10_m200 : float
        :math:`\\log_{10}(M_{200} / M_\\odot)`.  Must be finite.
    scale_radius_kpc : float
        NFW scale radius :math:`r_s`, in kpc.  Must be positive.
    z_lens : float
        Redshift of the halo.  Must be positive.
    profile_type : int, optional
        Lenstool profile code; must be ``12`` (NFW).  Default ``12``.
    priors : dict of str to numpyro distribution or None, optional
        Priors of the sampled parameters, keyed by ``"x_centre"``, ``"y_centre"`` (arcsec), ``"e1"``, ``"e2"`` (the
        Cartesian shape, dimensionless; both or neither, see ``priors.ellipticity_box``), ``"log10_m200"``
        (:math:`\\log_{10}` Msun) and ``"scale_radius_kpc"`` (kpc).  A parameter without an entry, or with None, is
        fixed at its field value.  Default: empty (every parameter fixed).
    """
    id: str
    x_centre: float
    y_centre: float
    ellipticite: float
    angle_pos: float
    log10_m200: float
    scale_radius_kpc: float
    z_lens: float
    profile_type: int = 12
    priors: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MassSheetConfig:
    """Uniform external convergence (profile code 101) at the lens redshift.

    ``kappa`` is the convergence for a source at infinite redshift; a family at redshift :math:`z_s` sees
    :math:`\\kappa\\,D_{ls}/D_s`.  Families at several source redshifts constrain it; a single source redshift cannot
    (mass-sheet degeneracy).

    Parameters
    ----------
    id : str
        Name of the component; the sampled parameter is named ``"<id>.kappa"``.
    kappa : float
        Convergence for a source at infinite redshift, dimensionless.  Must be finite.
    z_lens : float
        Redshift of the sheet.  Must be positive.
    profile_type : int, optional
        Profile code; must be ``101``.  Default ``101``.
    priors : dict of str to numpyro distribution or None, optional
        Prior of the sampled convergence under the key ``"kappa"`` (dimensionless).  Without it, or with None, the
        convergence is fixed at ``kappa``.  Default: empty (fixed).
    """
    id: str
    kappa: float
    z_lens: float
    profile_type: int = 101
    priors: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ShearHaloConfig:
    """External shear (Lenstool profile 14).

    Parameters
    ----------
    id : str
        Name of the component; the sampled parameters are named ``"<id>.gamma1"`` and ``"<id>.gamma2"``.
    gamma : float
        Shear amplitude, dimensionless.  Must be finite and nonnegative.
    angle_pos : float
        Shear angle, in degrees from the +x axis toward +y; the Cartesian components are
        :math:`(\\gamma_1, \\gamma_2) = \\gamma(\\cos 2\\phi, \\sin 2\\phi)`.
    z_lens : float
        Redshift of the shear component.  Must be positive.
    profile_type : int, optional
        Lenstool profile code; must be ``14``.  Default ``14``.
    priors : dict of str to numpyro distribution or None, optional
        Priors of the sampled Cartesian components under the keys ``"gamma1"`` and ``"gamma2"`` (dimensionless; both
        or neither, see ``priors.shear_box``).  Without them the shear is fixed at (``gamma``, ``angle_pos``).
        Default: empty (fixed).
    """
    id: str
    gamma: float
    angle_pos: float
    z_lens: float
    profile_type: int = 14
    priors: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def free(cls, id: str, z_lens: float, *, gamma_upper: float = 0.3) -> "ShearHaloConfig":
        """Build an external shear with a free amplitude and angle.

        The prior is uniform in the smallest (gamma1, gamma2) box that holds every amplitude up to ``gamma_upper`` at
        every angle (``priors.shear_box``).

        Parameters
        ----------
        id : str
            Name of the component.
        z_lens : float
            Redshift of the shear component.
        gamma_upper : float, optional
            Largest shear amplitude the box covers, dimensionless.  Default ``0.3``.

        Returns
        -------
        ShearHaloConfig
            The shear, with fixed values ``gamma`` 0 and ``angle_pos`` 0 and priors on ``gamma1`` and ``gamma2``.
        """
        return cls(id=str(id), gamma=0.0, angle_pos=0.0, z_lens=float(z_lens), priors=shear_box(float(gamma_upper)))


@dataclass(frozen=True)
class IndependentMemberHaloConfig:
    """A cluster member taken off the scaling relation and modeled as its own dPIE halo.

    The halo starts at the member's catalog position and light shape, with the population's ``corekpc``, ``cutkpc``
    and ``sigma`` as its fixed core, cut radius and dispersion, and is removed from the population's
    scaling relation.  Members whose catalog ``treatment`` is ``independent`` get such a halo with the default priors
    even when they are not listed here.

    Parameters
    ----------
    population_id : str
        ``id`` of the :class:`MemberPopulationConfig` whose catalog holds the member.
    catalog_id : str
        The member's id in that catalog.
    id : str or None, optional
        Name of the component.  None means ``"independent_member_<population_id>_<catalog_id>"``.  Default None.
    priors : dict of str to numpyro distribution or None, optional
        Priors keyed as in :attr:`DPIEHaloConfig.priors`; each entry replaces the default prior of that parameter, and
        None fixes the parameter at its starting value.  The defaults are: position truncated normal of width 0.3
        arcsec within +-2 arcsec of the catalog position; shape uniform in the (e1, e2) box of ``ellipticite`` up to
        0.8; core radius log-normal around the population's ``corekpc`` (0.5 dex width) on 0.01-50 kpc; cut radius
        log-normal around ``cutkpc`` (0.5 dex width) on 1-1000 kpc, as an excess above the core; dispersion log-normal
        of mean ``sigma`` and standard deviation ``0.5 * sigma`` on 30-1000 km s^-1 (all log-normals sampled in log).
        Default: empty (all defaults).
    """
    population_id: str
    catalog_id: str
    id: str | None = None
    priors: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemberPopulationConfig:
    """A population of cluster members whose dPIE halos follow a luminosity scaling relation.

    Member :math:`i` with luminosity ratio :math:`L_i/L_0 = 10^{-0.4 (m_i - m_0)}` gets dispersion
    :math:`\\sigma_0 (L_i/L_0)^{\\alpha}` and core and cut radii :math:`r_0 (L_i/L_0)^{\\beta}`, times its sampled
    deviation from the relation (see :class:`ScalingModelConfig`, which also holds the slope priors).
    :meth:`bergamini2019` and :meth:`neutral` build populations with published or scale-free priors.

    Parameters
    ----------
    id : str
        Name of the population; must be unique.
    catalog_path : str or pathlib.Path
        Member catalog: a YAML catalog (``.yaml``/``.yml``, see ``kappaccino.io.member_catalog``) or a Lenstool potfile
        (``.cat``).  Must exist.  Members with catalog ``treatment`` ``exclude`` are left out of the model.
    mag0 : float or str
        Pivot magnitude :math:`m_0` (mag, in the relation's band) at which ``sigma``, ``cutkpc``, ``corekpc`` and their
        priors apply; or ``"auto"``: the image-leverage-weighted mean member magnitude (weights
        :math:`\\sum_j 1/\\max(R_{ij}, 1'')^2` over images :math:`j`), rounded to the nearest magnitude.  With
        ``"auto"`` the values and priors are given at ``prior_mag0`` and moved to the pivot with the centers of the
        slope priors.
    corekpc : float
        Core radius at the pivot, in kpc.  Fixed unless ``ScalingModelConfig.fit_corekpc``; must be positive.
    sigma : float
        Velocity dispersion at the pivot, in km s^-1 (Lenstool :math:`\\sigma_{LT}`).  Used when it is not sampled
        (``ScalingModelConfig.fixed_relation_fields``); must be positive.
    cutkpc : float
        Cut radius at the pivot, in kpc.  Used when it is not sampled; must be positive.
    z_lens : float
        Redshift of the members.  Must be positive.
    profile_type : int, optional
        Lenstool profile code of the members; only ``81`` (dPIE) is supported.  Default ``81``.
    sigma_prior : numpyro distribution, optional
        Prior of the pivot dispersion, in km s^-1, given at ``mag0`` (or ``prior_mag0`` when ``mag0`` is ``"auto"``).
        Default: log-normal of mean 100 and standard deviation 40 km s^-1 truncated to 20-500 km s^-1, sampled in log.
    cutkpc_prior : numpyro distribution, optional
        Prior of the pivot cut radius, in kpc, at the same magnitude.  Default: log-normal of mean 300 and standard
        deviation 120 kpc truncated to 20-1000 kpc, sampled in log.
    exclude_catalog_ids : tuple of str, optional
        Catalog ids of members to leave out of this population.  Each must exist in the catalog, without duplicates.
        Default ``()``.
    prior_mag0 : float or None, optional
        Magnitude (mag) at which ``sigma``, ``cutkpc``, ``corekpc`` and their priors are given when ``mag0`` is
        ``"auto"``; required then, ignored otherwise.  Default None.
    band : str or None, optional
        YAML catalogs only: the magnitude band of the scaling relation.  None uses the catalog's ``band``.  Setting it
        for a ``.cat`` catalog raises an error.  Default None.
    kinematics_sources : tuple of str, optional
        YAML catalogs only: names of the catalog's kinematics sources whose measured stellar velocity dispersions
        enter the likelihood, one log-normal term per measured member on the relation:
        :math:`\\ln\\sigma_{meas} \\sim N(\\ln(c\\,\\sigma_i), e_i)`, with :math:`\\sigma_i` the member's model
        dispersion (relation times deviation), :math:`c` = ``kinematics_aperture_factor`` and
        :math:`e_i^2 = (\\mathrm{err}_i/\\sigma_{meas})^2 + \\mathrm{kinematics\\_systematic}^2`.  At most one listed
        source per galaxy, and each must appear on some member on the relation.  Empty: no kinematics term.
        Default ``()``.
    kinematics_systematic : float, optional
        Fractional systematic error of the measured dispersions (template and fitting systematics), added in quadrature
        to the measurement error in ln sigma, dimensionless.  The conversion from stellar to dPIE dispersion is left to
        the member scatter.  Must be nonnegative.  Default ``0.05``.
    kinematics_aperture_factor : float, optional
        Ratio :math:`c = \\sigma_{ap}/\\sigma_{LT}` of the measured aperture dispersion to the dPIE dispersion,
        dimensionless.  Must be positive.  Default ``295.2 / 248.0`` (Bergamini et al. 2021: an aperture dispersion
        of 295.2 km s^-1 at the BCG magnitude deprojected to a :math:`\\sigma_{LT}` prior of 248 km s^-1).
    """
    id: str
    catalog_path: str | Path
    mag0: float | str
    corekpc: float
    sigma: float
    cutkpc: float
    z_lens: float
    profile_type: int = 81
    sigma_prior: Any = field(default_factory=lambda: lognormal_from_moments(100.0, 40.0, low=20.0, high=500.0))
    cutkpc_prior: Any = field(default_factory=lambda: lognormal_from_moments(300.0, 120.0, low=20.0, high=1000.0))
    exclude_catalog_ids: tuple[str, ...] = ()
    prior_mag0: float | None = None
    band: str | None = None
    kinematics_sources: tuple[str, ...] = ()
    kinematics_systematic: float = 0.05
    kinematics_aperture_factor: float = 295.2 / 248.0

    @classmethod
    def bergamini2019(cls, id: str, catalog_path: str | Path, z_lens: float, *, prior_mag0: float, sigma: float = 200.0,
                      cutkpc: float = 12.0, corekpc: float = 0.15) -> "MemberPopulationConfig":
        """Build a member population with priors from the MUSE-calibrated relation of Bergamini et al. 2019.

        Their three clusters give a central dispersion of about 220 km s^-1 and a cut radius of about 10 (+13, -7) kpc at
        L* (``prior_mag0``).  The priors are widened to log-normals (sampled in log): dispersion of mean ``sigma`` and
        standard deviation ``0.35 * sigma`` on 40-500 km s^-1 (200 +- 70 km s^-1 by default), and cut radius of mean
        ``cutkpc`` and standard deviation 10 kpc on 2-60 kpc.  The pivot is chosen automatically from image leverage
        (``mag0="auto"``).  Pair with :meth:`ScalingModelConfig.bergamini2019` for the slopes.

        Parameters
        ----------
        id : str
            Name of the population.
        catalog_path : str or pathlib.Path
            Member catalog (YAML or Lenstool ``.cat``).
        z_lens : float
            Redshift of the members.
        prior_mag0 : float
            Magnitude of L*, at which the values and priors are given, in mag.
        sigma : float, optional
            Dispersion at L* and mean of its prior, in km s^-1.  Default ``200.0``.
        cutkpc : float, optional
            Cut radius at L* and mean of its prior, in kpc.  Default ``12.0``.
        corekpc : float, optional
            Core radius at L*, in kpc.  Default ``0.15``.

        Returns
        -------
        MemberPopulationConfig
            The population.
        """
        return cls(id=str(id), catalog_path=catalog_path, mag0="auto", prior_mag0=float(prior_mag0), corekpc=float(corekpc),
                   sigma=float(sigma), cutkpc=float(cutkpc), z_lens=float(z_lens),
                   sigma_prior=lognormal_from_moments(float(sigma), 0.35 * float(sigma), low=40.0, high=500.0),
                   cutkpc_prior=lognormal_from_moments(float(cutkpc), 10.0, low=2.0, high=60.0))

    @classmethod
    def neutral(cls, id: str, catalog_path: str | Path, z_lens: float, *, prior_mag0: float, corekpc: float = 0.15) -> "MemberPopulationConfig":
        """Build a member population with priors that come from no cluster's lens model.

        At L* (``prior_mag0``) the dispersion prior is a log-normal of mean 200 and standard deviation 100 km s^-1 on
        50-500 km s^-1 (an L* early type in any Faber-Jackson calibration; sampled in log), and the cut radius is
        log-uniform between 2 and 100 kpc (lensing barely constrains it, so the prior is scale-free).  The fixed values
        are 200 km s^-1 and 15 kpc.  The pivot is chosen automatically from image leverage (``mag0="auto"``).  Pair
        with :meth:`ScalingModelConfig.neutral` for the slopes.

        Parameters
        ----------
        id : str
            Name of the population.
        catalog_path : str or pathlib.Path
            Member catalog (YAML or Lenstool ``.cat``).
        z_lens : float
            Redshift of the members.
        prior_mag0 : float
            Magnitude of L*, at which the values and priors are given, in mag.
        corekpc : float, optional
            Core radius at L*, in kpc.  Default ``0.15``.

        Returns
        -------
        MemberPopulationConfig
            The population.
        """
        return cls(id=str(id), catalog_path=catalog_path, mag0="auto", prior_mag0=float(prior_mag0), corekpc=float(corekpc),
                   sigma=200.0, cutkpc=15.0, z_lens=float(z_lens),
                   sigma_prior=lognormal_from_moments(200.0, 100.0, low=50.0, high=500.0), cutkpc_prior=dist.LogUniform(2.0, 100.0))


@dataclass(frozen=True)
class ImageConstraintsConfig:
    """The multiple-image catalog that constrains the lens.

    Families without a positive redshift and families with a single image are dropped when the model is built.  The
    catalog carries no positional errors: the independent measurement scatter of every image is the sampled
    ``images.sigma_meas`` (``LikelihoodConfig.sigma_meas_prior``).

    Parameters
    ----------
    catalog_path : str or pathlib.Path
        YAML image catalog (see ``kappaccino.io.image_catalog``); convert Lenstool files with
        ``scripts/convert_lenstool_catalog.py``.  Must exist.
    trusted_prior_min : float, optional
        Images whose catalog ``prior`` (the prior probability that the image belongs to its system) is below this are
        not trusted: they leave their families, and with a :class:`PartitionConfig` without ``catalog_path`` they are
        the uncertain images, each weighted toward its catalog system by its prior.  In [0, 1].  Default ``0.0`` (every
        image trusted).
    """
    catalog_path: str | Path
    trusted_prior_min: float = 0.0


@dataclass(frozen=True)
class LensModelConfig:
    """The lens model: reference frame, cosmology, mass components and image constraints.

    Parameters
    ----------
    reference : ReferenceFrameConfig, optional
        Reference point of the arcsec coordinates.  Default: ``ReferenceFrameConfig()``.
    cosmology : CosmologyConfig, optional
        Background cosmology.  Default: ``CosmologyConfig()`` (flat, H0 = 70 km s^-1 Mpc^-1, Om0 = 0.3).
    large_halos : tuple of DPIEHaloConfig, NFWHaloConfig, ShearHaloConfig or MassSheetConfig, optional
        Cluster-scale components.  At least one is required by :meth:`LensClusterSolverConfig.validate`; the first
        component's ``z_lens`` sets the lens redshift.  Default ``()``.
    member_populations : tuple of MemberPopulationConfig, optional
        Member populations on scaling relations.  Default ``()``.
    independent_member_halos : tuple of IndependentMemberHaloConfig, optional
        Members modeled as their own halos.  Each must name an existing population, without duplicates.  Default
        ``()``.
    image_constraints : ImageConstraintsConfig or None, optional
        The multiple-image catalog.  Required by :meth:`LensClusterSolverConfig.validate`.  Default None.
    """
    reference: ReferenceFrameConfig = field(default_factory=ReferenceFrameConfig)
    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    large_halos: tuple[DPIEHaloConfig | NFWHaloConfig | ShearHaloConfig | MassSheetConfig, ...] = ()
    member_populations: tuple[MemberPopulationConfig, ...] = ()
    independent_member_halos: tuple[IndependentMemberHaloConfig, ...] = ()
    image_constraints: ImageConstraintsConfig | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Random seed, chain layout, resumption and logging of a run.

    Parameters
    ----------
    seed : int, optional
        Random seed (nonnegative): seeds NumPy and Python ``random``, the prior draws of the search starts, the NUTS
        PRNG key and the posterior draws of the truth comparison.  Default ``12345``.
    chains : int, optional
        Number of NUTS chains (positive).  Each chain starts from one of the kept search modes, the best first,
        cycling through them.  Default ``4``.
    resume : bool, optional
        Skip a stage (init or final) whose saved artifacts are complete and reuse them.  Default True.
    skip_plots : bool, optional
        Skip the output figures and tables written after each stage.  Default False.
    quiet : bool, optional
        Write the log to the run's log files only, not to the terminal.  Default False.
    nuts_chain_method : str, optional
        NumPyro chain method: ``"parallel"`` (chains run concurrently, one JAX device per chain; needs ``chains``
        visible JAX devices, e.g. via ``JAX_NUM_CPU_DEVICES``), ``"vectorized"`` (chains run concurrently on one
        device) or ``"sequential"`` (one chain at a time).  Default ``"parallel"``.
    cpu_threads_per_chain : int or None, optional
        Logical CPUs per concurrently running chain, positive.  Read by ``cpu_runtime.resolve_cpu_plan`` and
        ``cpu_runtime.configure_cpu_runtime``, which a launcher calls before JAX starts, to pin the process to that
        many CPUs per chain (Linux only); the pipeline itself does not apply it.  None pins nothing.  Default None.
    numpyro_print_summary : bool, optional
        Print a summary table of the global parameters after sampling.  Default False.
    """
    seed: int = 12345
    chains: int = 4
    resume: bool = True
    skip_plots: bool = False
    quiet: bool = False
    nuts_chain_method: str = "parallel"
    cpu_threads_per_chain: int | None = None
    numpyro_print_summary: bool = False


@dataclass(frozen=True)
class SearchConfig:
    """The initial optimization that finds the chain starts.

    The search runs in two stages and a polish.  Stage A fits the large-scale model (halos, shear, independent halos,
    sources and the member scaling relation, every member on its relation, error scales held at their starts) from
    ``n_starts`` prior draws with ``halo_method``.  Stage B starts from each carried stage-A mode with
    ``member_starts`` prior draws of the relation and optimizes the members, sources and error scales with the
    large-scale parameters frozen, with ``member_method``.  The modes that can seed chains are then polished with
    L-BFGS in full dimension.  Optionally (``nuts_start_max_iter``) the kept modes are optimized again on the exact NUTS
    target before sampling.  Throughout, end points are grouped into modes, degenerate modes are rejected, and modes
    within ``mode_margin`` of the best are kept.

    Parameters
    ----------
    n_starts : int, optional
        Number of stage-A starts (positive); the first is the configured model, the others are prior draws.
        Default ``32``.
    halo_method : str, optional
        Optimizer of stage A: ``"lbfgs"`` (scipy L-BFGS-B, one start at a time, with screening), ``"optax-lbfgs"``
        (vmapped optax L-BFGS over all starts, for GPUs), ``"adam"`` (vmapped Adam) or ``"svi"`` (vmapped mean-field
        Gaussian guide; the guide mean is the estimate).  Default ``"lbfgs"``.
    member_method : str, optional
        Optimizer of stage B, with the same choices as ``halo_method``.  Default ``"lbfgs"``.
    member_starts : int, optional
        Stage-B starts per carried mode (positive): the first keeps the stage-A relation, the others draw the relation
        from its prior.  Default ``8``.
    lbfgs_max_iter : int, optional
        Iteration cap per start of ``"lbfgs"`` (screening included) and ``"optax-lbfgs"``.  Positive.  Default ``400``.
    screen_iterations : int or None, optional
        ``"lbfgs"`` only: every start first runs this many iterations, and only the ``screen_keep`` best continue, from
        where they stopped, up to ``lbfgs_max_iter`` iterations in total.  None runs every start to the end.  Default
        ``50``.
    screen_keep : int, optional
        Number of screened starts that continue.  Default ``16``.
    nuts_start_max_iter : int, optional
        Iteration cap of the chain-start step: after the final member classification, every kept mode is optimized by
        scipy L-BFGS-B on the NUTS target (all parameters free except the pinned member scatters, source-transport
        volume included); the modes within ``mode_margin`` of the best on that target, without copies of the same
        large-scale model, seed the chains.  ``0`` skips the step; the start's conditioning on the target is then only
        checked and a warning logged.  Default ``4000``.
    nuts_start_tolerance : float, optional
        Stopping rule of the chain-start step: a mode stops once it gains less than this in log posterior over 50
        iterations.  Default ``0.1``.
    nuts_start_pin_scatter : bool, optional
        Pin the member scatters at ``polish_scatter`` and ``polish_radius_scatter`` in the chain-start step; NUTS frees
        them.  Left free, an optimizer drives them toward zero (with centered deviations the density grows as
        :math:`-\\ln s` per member as :math:`s \\to 0`).  Default True.
    warm_start : str or None, optional
        A previous run directory, or its ``init/artifacts`` directory.  Its saved start, matched by parameter name,
        replaces stages A and B; one L-BFGS run and the polish start from it.  Parameters it lacks start at the
        configured model.  None runs the full search.  Default None.
    stage_b_free_amplitudes : bool, optional
        Keep the large-scale parameters other than centers, shapes and shear components (dPIE dispersion, core and cut
        radius; NFW mass and scale radius; sheet convergence) free in stage B next to the members.  False freezes the
        whole large-scale family.  Default False.
    diverse_carry : int, optional
        Besides the stage-A modes within ``mode_margin``, up to this many more non-degenerate stage-A modes within
        ``diverse_carry_margin`` of the best go to stage B, picked by farthest-point sampling in the large-scale
        parameters.  Each gets its own stage B and polish; the chain-start step keeps it only if it lands within
        ``mode_margin`` of the best on the NUTS target.  ``0`` turns this off.  Default ``8``.
    diverse_carry_margin : float, optional
        Largest log-posterior gap to the best stage-A mode for a diverse-carry mode.  Default ``500.0``.
    adam_steps : int, optional
        Steps per start of ``"adam"`` (positive).  Default ``2000``.
    adam_learning_rate : float, optional
        Learning rate of ``"adam"`` (positive), in the unconstrained coordinates.  Default ``0.05``.
    svi_steps : int, optional
        Steps of ``"svi"`` (positive).  Default ``2000``.
    svi_learning_rate : float, optional
        Learning rate of ``"svi"`` (positive).  Default ``0.05``.
    polish_max_iter : int, optional
        Iteration cap of the full-dimension scipy L-BFGS-B polish of each mode that can seed chains (positive); scipy's
        own convergence tests can stop it earlier.  Default ``6000``.
    polish_scatter : float or None, optional
        Member velocity scatter (in ln sigma, dimensionless) at which the member scatters are pinned while optimizing
        (search, polish, chain-start step); NUTS frees them.  Also the scatter at which ``MemberConfig.centering="auto"``
        compares curvatures.  None pins neither scatter.  Must be positive when set.  Default ``0.2``.
    polish_radius_scatter : float or None, optional
        Member radius scatter (in ln r_cut, dimensionless) pinned with ``polish_scatter``.  None pins it at
        ``polish_scatter``.  Default ``0.3`` (the prior median).
    polish_free_scatter : bool, optional
        Free the member scatters in the polish (the search keeps them pinned).  Default False.
    mode_margin : float, optional
        Log-posterior margin: modes within it of the best are carried to stage B, polished, and can seed chains.
        Positive.  Default ``10.0``.
    mode_parameter_tolerance : float, optional
        End points whose active coordinates all differ by less than this, in units of the spread (standard deviation)
        of the starts in each coordinate, are grouped into one mode.  Positive.  Default ``0.05``.
    max_chain_modes : int, optional
        Largest number of modes carried from stage A, polished per stage, and used as chain starts.  Positive.
        Default ``4``.
    max_median_abs_determinant : float, optional
        A mode whose median :math:`|\\det A|` over the image positions exceeds this, dimensionless, is degenerate (a
        lens so massive that the images are demagnified) and cannot seed chains; cluster lenses sit near 0.2.
        Positive.  Default ``10.0``.
    """
    n_starts: int = 32
    halo_method: str = "lbfgs"
    member_method: str = "lbfgs"
    member_starts: int = 8
    lbfgs_max_iter: int = 400
    # Screening benchmark: on CANUCS (256 starts) 50/16 reaches the same best stage-A mode as running all starts (log
    # posterior equal to 1e-4) in 54 s instead of 201 s.
    screen_iterations: int | None = 50
    screen_keep: int = 16
    # Chain-start step since 2026-11-04 (before, only the search's best by one evaluation was optimized).  Even with the
    # search and the target on the same blocks, the search's optimum (no transport volume, scatters pinned, provisional
    # classes) was 47-162 in Newton decrement off the target's mode, and a start there cost a short NUTS check 11
    # divergences in 60 iterations; converged (1700 iterations, 145 s) it ran clean (2026-11-03).
    nuts_start_max_iter: int = 4000
    nuts_start_tolerance: float = 0.1
    # Free, the step on the nov3i shared-field run drove the scatters to 7e-4 and 5e-6 (z -16 and -22) (2026-11-03).
    nuts_start_pin_scatter: bool = True
    warm_start: str | None = None
    # stage_b_free_amplitudes made no difference on AS1063 or M0416 (2026-11-05).
    stage_b_free_amplitudes: bool = False
    # Diverse carry: a mode that is poor with the members on their relation can be the best once the members deviate
    # (M0416's southern halo, 2026-11-05).
    diverse_carry: int = 8
    diverse_carry_margin: float = 500.0
    adam_steps: int = 2000
    adam_learning_rate: float = 0.05
    svi_steps: int = 2000
    svi_learning_rate: float = 0.05
    polish_max_iter: int = 6000             # the only full-dimension optimization: AS1063 needs 400 iterations to reach the joint
                                            # optimum, M0416 passes the one-stage multistart after 2000 (400 left it 27 short) and
                                            # was still 10-13 short of its basin's optimum at 2000 (nov5_reltest, 2026-11-04);
                                            # scipy L-BFGS-B stops a converged polish early (its default ftol, gtol)
    # Pinned scatters: 0.2 for the velocity: the MUSE dispersions of Granata et al. 2026 scatter by 0.22 in ln sigma
    # about the relation, and 25-31 km/s is typical of MUSE-calibrated clusters.  0.3 for the radius: its prior median
    # (2026-11-03).
    polish_scatter: float | None = 0.2
    polish_radius_scatter: float | None = 0.3
    # polish_free_scatter off: on the search objective (no source-transport term) a free scatter collapsed to 0.001 on
    # CANUCS (deviation rms 0.06, fragile verdict, 2026-11-03).
    polish_free_scatter: bool = False
    mode_margin: float = 10.0
    mode_parameter_tolerance: float = 0.05
    max_chain_modes: int = 4
    max_median_abs_determinant: float = 10.0


@dataclass(frozen=True)
class SamplingConfig:
    """NUTS settings of the final stage.

    Parameters
    ----------
    warmup : int, optional
        Warmup (adaptation) iterations per chain.  Positive.  Default ``250``.
    samples : int, optional
        Post-warmup iterations per chain, before thinning.  Positive.  Default ``200``.
    max_tree_depth : int, optional
        Largest NUTS tree depth (at most :math:`2^{depth} - 1` leapfrog steps per transition).  Positive.  Default
        ``8``.
    target_accept : float, optional
        Target acceptance probability of the step-size adaptation, in (0, 1).  Default ``0.85``.
    initial_step_size : float, optional
        Step size at the start of warmup, in the unconstrained coordinates.  Positive.  Default ``1e-3``.
    thin : int, optional
        Keep every ``thin``-th post-warmup draw.  Positive.  Default ``1``.
    mass_matrix : str, optional
        Mass matrix: ``"hessian"`` (dense block of the global parameters, seeded by the Hessian at the start and
        adapted), ``"dense"`` (the same dense block, adapted from the identity) or ``"diagonal"`` (NumPyro's diagonal
        adaptation; any other value behaves the same).  Default ``"hessian"``.
    dense_member_offsets : bool, optional
        Add the member deviation offsets of ``dense_offset_members`` to the dense block (global parameters plus two per
        member), seeded and adapted with it; the source coordinates stay diagonal.  Default True.
    dense_offset_members : str, optional
        Which members' offsets join the dense block when ``dense_member_offsets`` is set: ``"all"``; ``"centered"``
        (members whose deviation is sampled directly, see ``MemberConfig.centering``); ``"exact"`` (not the far-field
        members); ``"none"``.  The other members' offsets get a diagonal mass.  Default ``"centered"``.
    far_field_offsets : str, optional
        ``"sample"``: far-field members keep free offsets; ``"fixed"``: their (non-centered) offsets are pinned at the
        relation by a normal prior of width ``sampling.FIXED_OFFSET_STD`` (1e-3) started at zero.  Far-field members
        barely move the images, so their offsets are close to prior-only.  Default ``"sample"``.
    stiffness_diagnostic : bool, optional
        After sampling, find the stiffest direction at each chain's last state with its adapted mass (power iteration on
        Hessian-vector products, about 30 per chain), the step size it allows and the parameters in it
        (``sampling.stiffness_diagnostic``).  Default True.
    wbic : bool, optional
        After the main NUTS run, run NUTS again on the tempered posterior (the data likelihood raised to
        :math:`\\beta = 1/\\ln n`) and report WBIC, an estimate of :math:`-2 \\ln Z` for model comparison, in
        ``final/artifacts/model_comparison.json`` (see :mod:`kappaccino.inference.model_comparison` and the "Model
        comparison" page of the documentation).  Roughly doubles the sampling time.  Turning ``wbic`` on for a finished run
        under the same run name adds the WBIC run without sampling the posterior again.  Default False.
    wbic_warmup : int or None, optional
        Warmup iterations per chain of the WBIC run.  None uses ``warmup``.  Default None.
    wbic_samples : int or None, optional
        Post-warmup iterations per chain of the WBIC run.  None uses ``samples``.  Default None.
    """
    warmup: int = 250
    samples: int = 200
    max_tree_depth: int = 8
    target_accept: float = 0.85
    initial_step_size: float = 1.0e-3
    thin: int = 1
    mass_matrix: str = "hessian"
    # Dense member offsets: the bright members' offsets trade off against sigma*, the core and the slopes (correlations
    # to 0.74 on M0416), which the diagonal mass cannot follow (2026-11-03).
    dense_member_offsets: bool = True
    # A 600-dimensional dense block cannot be estimated from a warmup whose draws have an ESS of tens; "centered" (the
    # default since 2026-11-04, quick tests on AS1063) cut the chain disagreement by 3, made the far-field offsets mix
    # (ESS 4 -> 785 of 400 draws) and the softest direction 4x stiffer, at 2/3 of the time per draw.
    dense_offset_members: str = "centered"
    far_field_offsets: str = "sample"
    stiffness_diagnostic: bool = True
    wbic: bool = False
    wbic_warmup: int | None = None
    wbic_samples: int | None = None


@dataclass(frozen=True)
class MemberConfig:
    """Member classification (exact kernels near images, far-field series elsewhere) and far-field validity.

    Members whose images lie within ``far_field_cut_radii`` cut radii, or whose circular far-field series would leave a
    whitened ellipticity error above ``template_tolerance`` at an image, are computed with the exact dPIE kernel
    ("exact" members); the others use the far-field series and are kept away from the images by a smooth barrier during
    sampling.  The classification uses a conservative (inflated) cut radius so that the sampled cut radius can grow
    before a far-field member reaches the guard.

    Parameters
    ----------
    classification_prior_quantile : float or None, optional
        Classify with cut radii at least this upper quantile of the ``cutkpc`` prior, with the radius slope spanning
        the current value and the ``1 - q`` and ``q`` quantiles of the ``beta_radius`` prior (and never below the
        current model inflated by ``cut_radius_inflation``).  Must lie in (0.5, 1).  None uses the current model only.
        Default ``0.975``.
    centering : str, optional
        Which members sample their deviation directly (centered) rather than as a unit offset times the scatter:
        ``"none"`` (every member a unit offset: one coordinate system throughout and no optimizer trap), ``"auto"``
        (members whose data curvature of the deviation exceeds ``centering_ratio`` times the prior's
        :math:`1/s^2` at the start, with :math:`s` = ``SearchConfig.polish_scatter``) or ``"exact"`` (every exact
        member).  Default ``"none"``.
    centering_ratio : float, optional
        ``centering="auto"`` threshold on the ratio of data to prior curvature, dimensionless.  Positive.  Default
        ``1.0``.
    far_field_cut_radii : float, optional
        Guard distance :math:`k`: a member with any image within :math:`k` of its (classification) cut radii is exact,
        and a far-field member is kept at least :math:`k` of its sampled cut radii from every image.  Positive.
        Default ``3.0``.
    cut_radius_inflation : float, optional
        Members are classified with this multiple of the current pivot cut radius (or the prior quantile, whichever is
        larger), so the sampled cut radius can grow by that factor before any far-field member reaches the guard.  At
        least 1.  Default ``3.0``.
    template_tolerance : float, optional
        Tolerance on the whitened ellipticity error of the circular far-field series at an image, in units of the
        source-plane error (dimensionless), per member and summed over the far-field members.  In (0, 1).  Default
        ``0.1``.
    top_exact : int or None, optional
        Make at least the ``top_exact`` highest-scoring members exact.  Must be at least the number of members the rule
        above makes exact (``ValueError`` otherwise).  None applies the rule only.  Default None.
    far_field_barrier_width : float, optional
        Width :math:`w` of the far-field guard's log-barrier, in :math:`x = \\ln(R / (k\\,r_{cut}))` (dimensionless):
        each far-field member adds :math:`-\\mathrm{strength}\\cdot\\mathrm{softplus}(-x/w)` to the log density, so NUTS
        turns back at the boundary instead of diverging on a hard wall (about :math:`-0.7\\,\\mathrm{strength}` at the
        guard, :math:`-0.01\\,\\mathrm{strength}` one width outside, linear in :math:`-x` inside).  Positive.  Default
        ``0.05``.
    far_field_barrier_strength : float, optional
        Strength of that barrier, in log-density units.  Positive.  Default ``25.0``.
    min_deflection_arcsec : float or None, optional
        Drop members whose deflection at every image, at the initial model, is below this, in arcsec.  Members with the
        catalog ``keep`` flag or with kinematics are kept.  None (or 0) drops none.  Default None.
    membership : str, optional
        ``"fixed"`` (every catalog member is a member; the catalog ``p_member`` is ignored) or ``"marginalize"``: each
        member :math:`i` has a membership indicator :math:`z_i \\sim \\mathrm{Bernoulli}(p_i)` that the likelihood sums
        over.  Uncertain exact members (at most ``membership_max_enumerated``, highest classification score first) are enumerated
        exactly, so the images update their membership; every other uncertain member enters with its expected mass
        :math:`p_i` and the deflection covariance :math:`p_i(1-p_i)\\,\\alpha_i\\alpha_i^T` (docs: Likelihoods,
        Uncertain membership).  Default ``"fixed"``.
    membership_default_probability : float, optional
        Prior :math:`p_i` of a member without a catalog ``p_member`` that is not spectroscopically confirmed (a
        ``z`` of kind ``spec``, kinematics, or role ``bcg`` all give :math:`p_i = 1`).  In (0, 1].  Default ``1.0``.
    membership_max_enumerated : int, optional
        Most uncertain exact members whose indicators are summed exactly; the likelihood costs :math:`2^K` evaluations
        for :math:`K` of them.  Nonnegative.  Default ``4``.
    membership_calibration_prior : numpyro distribution or None, optional
        Prior of a global shift :math:`\\delta` of the classifier's log-odds, :math:`\\mathrm{logit}\\,p_i' =
        \\mathrm{logit}\\,p_i + \\delta` for every uncertain member (for example ``dist.Normal(0.0, 1.0)``), which
        corrects an over- or under-confident classifier.  None (default) keeps the catalog probabilities.
    """
    classification_prior_quantile: float | None = 0.975
    centering: str = "none"                 # "none" since 2026-11-03
    centering_ratio: float = 1.0
    """Member classification and far-field validity."""
    far_field_cut_radii: float = 3.0
    cut_radius_inflation: float = 3.0
    template_tolerance: float = 0.1
    top_exact: int | None = None
    far_field_barrier_width: float = 0.05
    far_field_barrier_strength: float = 25.0
    min_deflection_arcsec: float | None = None
    membership: str = "fixed"
    membership_default_probability: float = 1.0
    membership_max_enumerated: int = 4
    membership_calibration_prior: Any = None


@dataclass(frozen=True)
class ScalingModelConfig:
    """Priors of the member scaling relation and of the members' deviations from it.

    For a member of luminosity ratio :math:`L/L_0` the relation gives :math:`\\sigma = \\sigma_0 (L/L_0)^{\\alpha}` and
    :math:`r_{core}, r_{cut} \\propto (L/L_0)^{\\beta}` (pivot values in :class:`MemberPopulationConfig`).  Members
    leave the relation in dispersion (a tight scatter: Faber-Jackson) and, for the exact members, in cut radius (a
    loose scatter: tidal truncation depends on the orbit).  :meth:`bergamini2019` and :meth:`neutral` set the slope
    priors.

    Parameters
    ----------
    member_velocity_scatter_prior : numpyro distribution, optional
        Prior of the scatter of the members' dispersion deviations about the relation, in ln sigma (dimensionless),
        one per population; in the ``"mass"`` modes the scatter of the joint deviation along the relation.  Default
        ``dist.LogNormal(ln 0.10, 0.3)`` (median 0.10), sampled in log.
    member_deviation : str, optional
        How a member leaves the relation: ``"sigma"`` (dispersion only); ``"mass"`` (dispersion and radii together
        along the relation, :math:`d\\ln\\sigma : d\\ln r = \\alpha : \\beta`, one scatter); ``"sigma_rcut"``
        (independent dispersion and radius deviations, two scatters); ``"mass_rcut"`` (a mass deviation along the
        relation plus an independent radius deviation).  Default ``"sigma_rcut"``.
    member_radius_scatter_prior : numpyro distribution, optional
        Prior of the scatter of the radius deviations, in ln r (dimensionless), one per population; used in the
        ``"sigma_rcut"`` and ``"mass_rcut"`` modes.  Default ``dist.LogNormal(ln 0.3, 0.5)`` (median 0.3), sampled in
        log.
    member_deviation_tail : str, optional
        Distribution of the members' unit offsets (deviation = offset times scatter): ``"normal"``, or
        ``"student_t"`` with ``member_deviation_dof`` degrees of freedom, so a few members can sit far from the relation
        without widening the scatter for all.  With the Student-t the scatter is the scale; the standard deviation is
        :math:`s\\sqrt{\\nu/(\\nu - 2)}`.  Default ``"student_t"``.
    member_deviation_dof : float, optional
        Degrees of freedom :math:`\\nu` of the Student-t offsets; must exceed 2 (finite variance).  Default ``4.0``.
    corekpc_prior : numpyro distribution, optional
        Prior of the member core radius at the pivot, in kpc, when ``fit_corekpc`` is set.  Default: log-normal of
        median 0.15 kpc and ln-width 0.7 truncated to 0.01-10 kpc, sampled in log.
    radius_deviation_members : str, optional
        Which members' radius deviations act on the model: ``"exact"`` (far-field members keep the relation radius, so
        they cannot cross the far-field guard) or ``"all"``.  Default ``"exact"``.
    fit_corekpc : bool, optional
        Sample the member core radius at the pivot.  False fixes it at the population's ``corekpc`` moved to the pivot
        (a 0.1 kpc member core is not a lensing observable).  Default True.
    fixed_relation_fields : tuple of str, optional
        Relation parameters held at nominal values instead of sampled, any of: ``"sigma"`` and ``"cutkpc"`` (the
        population's values moved to the pivot), ``"alpha_sigma"`` and ``"beta_radius"`` (the centers of their
        priors).  For tests against a published relation.  Default ``()``.
    member_shape_scatter : float or None, optional
        Scatter :math:`s` of the sampled member shapes, dimensionless, in (0, 1).  For every exact member
        :math:`e_1 = e_{1,light} + s u_1` and :math:`e_2 = e_{2,light} + s u_2` with :math:`u \\sim N(0, 1)`, where
        :math:`e = (1 - q)/(1 + q)(\\cos 2\\phi, \\sin 2\\phi)`; far-field members keep the light shape and their
        offsets are prior-only.  Round catalog members are centered on :math:`e = 0`.  None fixes every member at its
        catalog (light) shape.  Default None.
    alpha_sigma_prior : numpyro distribution, optional
        Prior of the dispersion slope :math:`\\alpha` (:math:`\\sigma \\propto L^{\\alpha}`), dimensionless.  Default
        ``dist.TruncatedNormal(0.27, 0.05, low=0.05, high=0.50)`` (Faber-Jackson slope 0.27-0.28 in three
        MUSE-calibrated clusters, Bergamini et al. 2019), sampled linearly.
    beta_radius_prior : numpyro distribution, optional
        Prior of the radius slope :math:`\\beta` (:math:`r_{cut}, r_{core} \\propto L^{\\beta}`), dimensionless; its
        lower bound must be at least 0.  Default ``dist.TruncatedNormal(0.65, 0.15, low=0.0, high=1.5)`` (0.64-0.66 in
        Bergamini et al. 2019), sampled linearly.
    softening_length_prior : numpyro distribution or None, optional
        Prior of a resolution softening length, in kpc, added in quadrature to every dPIE core (for members, capped
        smoothly just below the cut radius), e.g. ``dist.LogNormal(ln(2.3 / h), 0.15)`` for an N-body simulation's
        softening.  None for no softening.  Default None.
    """
    # Dispersion deviation alone: the sampler turned the biggest subhalos into point masses to patch mass the cluster
    # halos lacked.
    member_velocity_scatter_prior: Any = field(default_factory=lambda: dist.LogNormal(math.log(0.10), 0.3))
    member_deviation: str = "sigma_rcut"
    member_radius_scatter_prior: Any = field(default_factory=lambda: dist.LogNormal(math.log(0.3), 0.5))
    # Student-t tail: the radius scatter grew to 1.6-1.8 against a 0.3 prior for five members on M0416 (2026-11-03).
    # Student-t is the default since 2026-11-04 (a scale-mixture form was tried and dropped: per-member funnels).
    member_deviation_tail: str = "student_t"
    member_deviation_dof: float = 4.0
    corekpc_prior: Any = field(default_factory=lambda: truncated_lognormal(0.15, 0.7, low=0.01, high=10.0))
    radius_deviation_members: str = "exact"
    fit_corekpc: bool = True
    fixed_relation_fields: tuple[str, ...] = ()
    member_shape_scatter: float | None = None
    alpha_sigma_prior: Any = field(default_factory=lambda: dist.TruncatedNormal(0.27, 0.05, low=0.05, high=0.50))
    beta_radius_prior: Any = field(default_factory=lambda: dist.TruncatedNormal(0.65, 0.15, low=0.0, high=1.5))
    softening_length_prior: Any | None = None

    @classmethod
    def bergamini2019(cls, **overrides) -> "ScalingModelConfig":
        """Build the scaling model with the default slope priors of Bergamini et al. 2019.

        The slopes are :math:`\\alpha` = 0.27 +- 0.05 and :math:`\\beta` = 0.65 +- 0.15 (the class defaults).

        Parameters
        ----------
        **overrides
            Other fields of :class:`ScalingModelConfig` to set.

        Returns
        -------
        ScalingModelConfig
            The scaling model.
        """
        return cls(**overrides)

    @classmethod
    def neutral(cls, **overrides) -> "ScalingModelConfig":
        """Build the scaling model with slope priors from scaling arguments only.

        :math:`\\alpha` = 0.25 +- 0.10 on 0-0.5 (:math:`L \\propto \\sigma^4`) and :math:`\\beta` = 0.5 +- 0.25 on 0-1.5
        (constant mass-to-light ratio, :math:`2\\alpha + \\beta = 1`), both truncated normals.

        Parameters
        ----------
        **overrides
            Other fields of :class:`ScalingModelConfig` to set; ``alpha_sigma_prior`` and ``beta_radius_prior`` cannot
            be overridden here.

        Returns
        -------
        ScalingModelConfig
            The scaling model.

        Raises
        ------
        TypeError
            If ``overrides`` contains ``alpha_sigma_prior`` or ``beta_radius_prior``.
        """
        return cls(alpha_sigma_prior=dist.TruncatedNormal(0.25, 0.10, low=0.0, high=0.5),
                   beta_radius_prior=dist.TruncatedNormal(0.5, 0.25, low=0.0, high=1.5), **overrides)


LIKELIHOOD_MODES = frozenset({"smoothed", "gp_image_plane", "source-anisotropic", "source-isotropic"})


@dataclass(frozen=True)
class LikelihoodConfig:
    """Likelihood of the image positions and its error model.

    In the ``"smoothed"`` mode (the model of the paper) each image contributes a local image-plane term with the
    source-plane positional covariance :math:`C = \\sigma_{meas}^2 A A^T + \\sigma_{def}^2 I + \\sigma_{sel}^2 R`
    (the paper's Eq. position_covariance), where :math:`A` is the lens Jacobian at the image,
    :math:`\\sigma_{meas}` is the sampled independent measurement scatter of every image and
    :math:`R = (I + A A^T/\\lambda_{sel}^2)^{-1}` (Eq. selection_weight).  The ``"gp_image_plane"`` mode replaces the
    floors by correlated errors within blocks of images, with an image-plane kernel (``gp_kernel="image_plane"``) or a
    mass field with a knee spectrum plus an image-plane GP (``gp_kernel="knee"``).  Fields prefixed ``gp_`` apply to the
    ``"gp_image_plane"`` mode only, and ``gp_knee*`` fields to its knee kernel.  The default error-scale priors are
    log-normals truncated below at 1e-4 arcsec and above at the stated value, sampled in log.

    Parameters
    ----------
    likelihood_mode : str, optional
        ``"smoothed"`` (local image-plane term with per-image source-plane floors); ``"gp_image_plane"`` (correlated
        image-plane displacements, no source-plane floors); and, for diagnostics only, ``"source-anisotropic"`` (the
        Gaussian source-plane term with covariance :math:`C`, without the determinant factor) and
        ``"source-isotropic"`` (a source-plane Gaussian of variance :math:`\\sigma_{def}^2 + \\sigma_{meas}^2` per image,
        without the Jacobian).  Default ``"smoothed"``.
    sigma_gp_prior : numpyro distribution, optional
        ``gp_kernel="image_plane"``: prior of the shared (correlated) image-plane displacement amplitude
        :math:`\\sigma_{shared}`, in arcsec.  The family covariance is
        :math:`S = \\sigma_{shared}^2 (K \\otimes I) + \\sigma_{meas}^2 I` with the Gaussian kernel
        :math:`K_{ij} = \\exp(-|\\theta_i - \\theta_j|^2 / 2\\ell^2)`, and the source-plane covariance
        :math:`C = B S B^T + \\sigma_{fl}^2 I` with :math:`B` = blockdiag(:math:`A_i`); :math:`\\sigma_{shared}` and
        :math:`\\sigma_{meas}` replace :math:`\\sigma_{def}` and :math:`\\sigma_{sel}`.
        Default: median 0.15 arcsec, ln-width 0.5, truncated at 1 arcsec.
    gp_sigma_meas_prior : numpyro distribution, optional
        ``gp_kernel="image_plane"``: prior of the independent measurement scatter :math:`\\sigma_{meas}` of every image,
        in arcsec, the only measurement error (no fixed instrumental error).  Log-normal so that it is learned and kept
        away from zero; ``dist.TruncatedNormal(0.0, 0.05, low=0.0)`` is the half-normal alternative.  Default: median 0.015
        arcsec, ln-width 0.5, truncated at 0.1 arcsec.
    sigma_gp_start_arcsec : float, optional
        ``gp_kernel="image_plane"``: start of :math:`\\sigma_{shared}` in the search, in arcsec.  Positive.  Default
        ``0.168``.
    gp_sigma_meas_start_arcsec : float, optional
        ``gp_kernel="image_plane"``: start of :math:`\\sigma_{meas}`, in arcsec.  Positive.  Default ``0.02``.
    gp_correlation_length_arcsec : float, optional
        ``gp_kernel="image_plane"``: the correlation length :math:`\\ell`, in arcsec: its fixed value when
        ``gp_correlation_length_prior`` is None, else its start.  Positive.  Default ``7.0``.
    gp_correlation_length_prior : numpyro distribution or None, optional
        ``gp_kernel="image_plane"``: prior of :math:`\\ell`, in arcsec.  None fixes :math:`\\ell` at
        ``gp_correlation_length_arcsec``.  Only same-family pairs separated by a few to about 15 arcsec constrain it,
        and above about 30 arcsec every family's kernel is a block of ones.  Default: median 7 arcsec, ln-width 0.5
        (68% between 4.2 and 11.5 arcsec), truncated at 30 arcsec.
    gp_floor_arcsec : float, optional
        Fixed source-plane floor :math:`\\sigma_{fl}` of the ``"gp_image_plane"`` covariance, in arcsec; a smoothness
        floor only.  Nonnegative.  Default ``0.0015`` (1.5 mas).
    gp_kernel : str, optional
        Structure of the GP term: ``"image_plane"`` (an image-plane displacement times the Jacobian,
        :math:`C = B S B^T`) or ``"knee"`` (a mass field plus the image-plane GP; see ``gp_knee``).  Default
        ``"image_plane"``.
    gp_reference_redshift : float, optional
        Knee kernel: reference source redshift :math:`z_{ref}` of the redshift weights
        :math:`w = (D_{ds}/D_s)(z) / (D_{ds}/D_s)(z_{ref})` that scale each family's deflection error.  Positive.
        Default ``2.0``.
    gp_blocks : str, optional
        Knee kernel: blocks of images that share one covariance: ``"all"`` (every image), ``"system"`` (all knots of a
        galaxy) or ``"family"``.  The init search always uses ``"system"`` blocks.  Default ``"all"``.
    gp_knee_gamma : float, optional
        Knee kernel: slope :math:`\\gamma` of the convergence spectrum
        :math:`P_\\kappa = P_0 / (1 + (k/k_t)^{\\gamma})`, dimensionless: its fixed value, or its start when sampled.
        Positive, and inside ``gp_knee_gamma_prior`` when that is set.  Default ``4.0`` (the slope of an NFW cusp).
    gp_knee_gamma_prior : numpyro distribution or None, optional
        Knee kernel: prior of the sampled slope, a ``dist.Uniform`` or a two-sided ``dist.TruncatedNormal`` whose bounds
        are also those of the slope tables.  Bounds 2-6 span the slopes of :math:`P_\\kappa` below the truncation for
        halos with :math:`\\rho \\propto r^{-n}`, n = 2 (isothermal) to 0 (cored).  None fixes the slope at
        ``gp_knee_gamma``.  Default ``dist.Uniform(2.0, 6.0)``.
    gp_knee_gamma_step : float, optional
        Spacing of the tabulated slopes (cubic spline between), dimensionless; must divide the prior's range into at
        least 2 intervals.  Default ``0.25``.
    gp_knee_scales_arcsec : tuple of float, optional
        Knee kernel: (shortest, longest) wavelength of the mass field, in arcsec: from the smallest image separation
        resolved to the scale of the cluster halos.  ``0 < min < max``; the upper bound of ``lambda_b_prior`` must not
        exceed the longest.  Default ``(0.2, 30.0)``.
    gp_knee_project_modes : tuple of str, optional
        Knee kernel: low-order modes removed from the mass field over the images, any of ``"translation"``,
        ``"convergence"`` and ``"shear"`` (``gp_knee.project_low_order``).  Under a flat shear prior it leaves the
        likelihood unchanged and makes the external shear the total shear over the images.  Needs
        ``gp_blocks="all"``; not with the ``"knee_foreground"`` or ``"knee_background"`` spectra.  Empty: none.  Default
        ``()``.
    gp_knee_lambda_grid : tuple of (float, float, int), optional
        Knee kernel: tabulated knee wavelengths (min, max in arcsec, count), log-spaced; the sampled
        :math:`\\ln\\lambda_b` is clamped to the grid.  ``0 < min < max`` and count at least 8.  Default
        ``(0.5, 80.0, 64)``.
    gp_knee_r_step_arcsec : float, optional
        Knee kernel: separation step of the kernel tables, in arcsec.  Positive.  Default ``0.02``.
    sigma_grf_prior : numpyro distribution or None, optional
        Knee kernel: prior of the mass field's rms deflection error per component at :math:`z_{ref}`, in arcsec.  The
        upper truncation keeps the error model from replacing the lens: a deflection error above 0.3 arcsec is a
        missing cluster component, not substructure.  None drops the mass field (only the image-plane GP remains).
        Default: median 0.08 arcsec, ln-width 0.7, truncated at 0.3 arcsec.
    sigma_grf_start_arcsec : float, optional
        Start of the rms deflection error, in arcsec.  Positive.  Default ``0.08``.
    lambda_b_prior : numpyro distribution, optional
        Knee kernel: prior of the break scale :math:`\\lambda_b = 2\\pi/k_t`, in arcsec.  Its upper bound must not
        exceed ``gp_knee_scales_arcsec[1]`` (a knee above it is the scale of the cluster halos and the external shear)
        nor the largest tabulated wavelength.  Default: median 6 arcsec, ln-width 0.5, truncated at 30 arcsec.
    lambda_b_start_arcsec : float, optional
        Start of the knee wavelength, in arcsec, strictly inside the bounds of ``lambda_b_prior``.  Default ``6.0``.
    gp_knee_sigma_gp_prior : numpyro distribution, optional
        Knee kernel: prior of the image-plane GP amplitude :math:`\\sigma_{shared}`, in arcsec; an image-plane
        displacement above 0.3 arcsec is lensing, not astrometry or feature selection.  Default: median 0.10 arcsec,
        ln-width 0.7, truncated at 0.3 arcsec.
    gp_knee_sigma_gp_start_arcsec : float, optional
        Start of that amplitude, in arcsec.  Positive.  Default ``0.10``.
    gp_knee_ell_gp_prior : numpyro distribution, optional
        Knee kernel: prior of the image-plane GP correlation length, in arcsec; coherence over more than 5 arcsec is
        lensing.  Default: median 2 arcsec, ln-width 0.5, truncated at 5 arcsec.
    gp_knee_ell_gp_start_arcsec : float, optional
        Start of that correlation length, in arcsec.  Positive.  Default ``2.0``.
    gp_knee_spectrum : str, optional
        Knee kernel: shape of the mass-field spectrum (``scripts/compare_grf_spectra.py`` compares them):
        ``"knee"`` (:math:`P = 1/(1 + (k/k_t)^{\\gamma})`, the one-halo term of one population of truncated halos);
        ``"two_knee"`` (a second, independent knee field with the same slope, for a second population; the upper bound
        of ``lambda_b2_prior`` must not exceed the lower bound of ``lambda_b_prior``, so the two cannot swap);
        ``"broken_power_law"`` (:math:`P = x^{-\\beta_{low}} (1 + x^{\\gamma})^{-(\\gamma - \\beta_{low})/\\gamma}`,
        :math:`x = k/k_b`, with :math:`\\beta_{low}` sampled on ``gp_knee_beta_low_prior`` and :math:`\\gamma` fixed;
        :math:`\\beta_{low} = 0` is the knee); ``"power_law"`` (:math:`P = k^{-\\gamma}` between the cutoffs, no knee,
        :math:`\\gamma` sampled on ``gp_knee_gamma_prior``); ``"knee_background"`` (a second knee field on a thin mass
        plane at ``gp_knee2_plane_redshift`` behind the cluster, weighted by :math:`D_{ps}/D_s`, zero for sources in
        front of it; to first order, without the plane's coupling to the cluster deflection); ``"knee_foreground"``
        (a second knee field with redshift weight 1 for every source, for mass well in front of the cluster).  The
        spectra other than ``"knee"`` need the mass field (``sigma_grf_prior``).  The image solver's
        mean-deflection correction is implemented for ``"knee"`` only.  Default ``"knee"``.
    gp_knee2_sigma_grf_prior : numpyro distribution, optional
        Second-field spectra: prior of the second field's rms deflection error, in arcsec.  Default: median 0.05
        arcsec, ln-width 0.7, truncated at 0.3 arcsec.
    gp_knee2_sigma_grf_start_arcsec : float, optional
        Start of that rms deflection error, in arcsec.  Positive.  Default ``0.05``.
    lambda_b2_prior : numpyro distribution, optional
        Second-field spectra: prior of the second knee wavelength, in arcsec; bounded above by at most the largest
        tabulated wavelength.  Default: median 3 arcsec, ln-width 0.5, truncated at 10 arcsec.
    lambda_b2_start_arcsec : float, optional
        Start of the second knee wavelength, in arcsec, strictly inside the bounds of ``lambda_b2_prior``.  Default
        ``3.0``.
    gp_knee2_plane_redshift : float, optional
        ``"knee_background"``: redshift of the background mass plane; must exceed the cluster redshift.  Default
        ``0.7``.
    gp_knee_beta_low_prior : numpyro distribution, optional
        ``"broken_power_law"``: prior of the low-k slope :math:`\\beta_{low}`, dimensionless; its bounds are those of
        the tables and must lie below ``gp_knee_gamma``.  Default ``dist.Uniform(-1.0, 3.5)``.
    gp_knee_beta_low_step : float, optional
        Spacing of the tabulated low-k slopes, dimensionless; must divide the prior's range into at least 2 intervals.
        Default ``0.25``.
    gp_knee_beta_low_start : float, optional
        Start of :math:`\\beta_{low}`, inside its prior's bounds.  Default ``0.0``.
    lambda_b_fixed_arcsec : float or None, optional
        Fixed knee wavelength of the mass field, in arcsec (no sampled knee); must lie inside the knee tables and
        ``gp_knee_scales_arcsec``; not with ``"power_law"``.  None samples it.  Default None.
    gp_knee_ell_gp_fixed_arcsec : float or None, optional
        Fixed correlation length of the knee kernel's image-plane GP, in arcsec (no sampled length).  Positive.  None
        samples it.  Default None.
    gp_knee_sigma_meas_prior : numpyro distribution, optional
        Knee kernel: prior of the independent measurement scatter :math:`\\sigma_{meas}` of every image, in arcsec;
        log-normal rather than half-normal (``dist.TruncatedNormal(0.0, 0.03, low=0.0)``), so that it is inferred and
        kept away from zero.  Default: median 0.015 arcsec, ln-width 0.5, truncated at 0.1
        arcsec.
    gp_knee_sigma_meas_start_arcsec : float, optional
        Start of that measurement scatter, in arcsec.  Positive.  Default ``0.012``.
    sigma_meas_prior : numpyro distribution, optional
        ``"smoothed"`` and source-plane modes: prior of the independent measurement scatter :math:`\\sigma_{meas}` of
        every image, in arcsec (astrometry, image registration, feature centering, manual marking), the only
        measurement error.  Always sampled; cannot be None.  Default: median 0.05 arcsec, ln-width 1.0, truncated at
        1 arcsec.
    sigma_def_prior : numpyro distribution, optional
        ``"smoothed"`` and source-plane modes: prior of the residual deflection error :math:`\\sigma_{def}`, isotropic
        in the source plane, in arcsec.  Always sampled; cannot be None.  Default: median 0.1 arcsec, ln-width 1.0,
        truncated at 1 arcsec.
    sigma_sel_prior : numpyro distribution or None, optional
        ``"smoothed"`` and source-plane modes: prior of the feature-selection scatter :math:`\\sigma_{sel}` in the
        source plane, in arcsec, weighted by the stretching through :math:`R`: the selection variance is halved where
        the absolute Jacobian eigenvalue equals :math:`\\lambda_{sel}`.  None disables the term.  Default: median 0.04
        arcsec, ln-width 1.0, truncated at 1 arcsec.
    lambda_sel_prior : numpyro distribution, optional
        Prior of the stretching scale :math:`\\lambda_{sel}` of the selection term, dimensionless; sampled with
        :math:`\\sigma_{sel}` (not used when ``sigma_sel_prior`` is None).  Default ``dist.LogNormal(ln 0.7, 1.0)``,
        sampled in log.
    source_position_prior_mean_arcsec : tuple of float, optional
        Mean (x, y) of the Gaussian prior on each family's source position, in arcsec.  Default ``(0.0, 0.0)``.
    source_position_prior_sigma_arcsec : float, optional
        Standard deviation of that prior per axis, in arcsec.  Positive.  Default ``100.0``.
    freeze_determinant_gradient : bool, optional
        Hold :math:`\\nabla\\det A` (the third derivatives of the potential) fixed in the backward pass.  The
        log-likelihood is unchanged; its gradient differs by about 1e-4 relative (AS1063, M0416) and costs 15-35% less.
        NUTS stays exact (the energy is exact and the leapfrog map stays reversible and volume preserving); only the
        integrator's accuracy, hence the acceptance rate, can suffer.  Default False.
    """
    likelihood_mode: str = "smoothed"
    # sigma_gp prior truncated at 1": the untruncated prior puts 7e-5 of its mass there, and a 32" value carried a
    # spurious init mode.
    sigma_gp_prior: Any = field(default_factory=lambda: _error_scale_prior(0.15, 0.5, 1.0))
    gp_sigma_meas_prior: Any = field(default_factory=lambda: _error_scale_prior(0.015, 0.5, 0.1))
    sigma_gp_start_arcsec: float = 0.168      # the sigma_img posterior median of the oct31a M0416 run
    gp_sigma_meas_start_arcsec: float = 0.02
    gp_correlation_length_arcsec: float = 7.0
    # ell prior: only the ~40 same-family pairs separated by 3" to 15" constrain ell (M0416).
    gp_correlation_length_prior: Any | None = field(default_factory=lambda: _error_scale_prior(7.0, 0.5, 30.0))
    gp_floor_arcsec: float = 0.0015
    gp_kernel: str = "image_plane"
    gp_reference_redshift: float = 2.0
    # Knee kernel: gp_knee.py, .agents/gp-knee-plan.md.  gp_blocks "system" is run.xsh --fast; "family" is for
    # regression.  Sites: images.sigma_grf, images.lambda_b (lambda_t), images.gp_knee_gamma, images.sigma_gp,
    # images.ell_gp, images.sigma_meas.
    gp_blocks: str = "all"
    gp_knee_gamma: float = 4.0
    # run.xsh uses dist.TruncatedNormal(4, 0.5, low=2, high=6): the slope of an NFW-like cusp, within a quarter of n.
    gp_knee_gamma_prior: Any | None = field(default_factory=lambda: dist.Uniform(2.0, 6.0))
    gp_knee_gamma_step: float = 0.25
    # With the long end at 100" and the halos free (all-image blocks), the field took the halo-scale misfit: sigma_grf
    # and the knee at their caps (0.28", 25"), the external shear down to 0.02 (lens-free test, 2026-11-02).  Truncated at
    # 30" the halos and the shear carry those scales and the field is left with the structure below them.
    gp_knee_scales_arcsec: tuple[float, float] = (0.2, 30.0)
    gp_knee_project_modes: tuple[str, ...] = ()
    gp_knee_lambda_grid: tuple[float, float, int] = (0.5, 80.0, 64)
    gp_knee_r_step_arcsec: float = 0.02
    sigma_grf_prior: Any | None = field(default_factory=lambda: _error_scale_prior(0.08, 0.7, 0.3))
    sigma_grf_start_arcsec: float = 0.08
    lambda_b_prior: Any = field(default_factory=lambda: _error_scale_prior(6.0, 0.5, 30.0))
    lambda_b_start_arcsec: float = 6.0
    gp_knee_sigma_gp_prior: Any = field(default_factory=lambda: _error_scale_prior(0.10, 0.7, 0.3))
    gp_knee_sigma_gp_start_arcsec: float = 0.10
    gp_knee_ell_gp_prior: Any = field(default_factory=lambda: _error_scale_prior(2.0, 0.5, 5.0))
    gp_knee_ell_gp_start_arcsec: float = 2.0
    gp_knee_spectrum: str = "knee"
    gp_knee2_sigma_grf_prior: Any = field(default_factory=lambda: _error_scale_prior(0.05, 0.7, 0.3))
    gp_knee2_sigma_grf_start_arcsec: float = 0.05
    lambda_b2_prior: Any = field(default_factory=lambda: _error_scale_prior(3.0, 0.5, 10.0))
    lambda_b2_start_arcsec: float = 3.0
    gp_knee2_plane_redshift: float = 0.7
    gp_knee_beta_low_prior: Any = field(default_factory=lambda: dist.Uniform(-1.0, 3.5))
    gp_knee_beta_low_step: float = 0.25
    gp_knee_beta_low_start: float = 0.0
    lambda_b_fixed_arcsec: float | None = None
    gp_knee_ell_gp_fixed_arcsec: float | None = None
    # gp_knee_sigma_meas log-normal: the fits put it at 0.006-0.02" (2026-11-05).
    gp_knee_sigma_meas_prior: Any = field(default_factory=lambda: _error_scale_prior(0.015, 0.5, 0.1))
    gp_knee_sigma_meas_start_arcsec: float = 0.012
    # smoothed and source-plane modes: the independent measurement scatter of every image, always sampled.
    sigma_meas_prior: Any = field(default_factory=lambda: _error_scale_prior(0.05, 1.0, 1.0))
    sigma_def_prior: Any = field(default_factory=lambda: _error_scale_prior(0.1, 1.0, 1.0))
    sigma_sel_prior: Any | None = field(default_factory=lambda: _error_scale_prior(0.04, 1.0, 1.0))
    lambda_sel_prior: Any = field(default_factory=lambda: dist.LogNormal(math.log(0.7), 1.0))
    source_position_prior_mean_arcsec: tuple[float, float] = (0.0, 0.0)
    source_position_prior_sigma_arcsec: float = 100.0
    freeze_determinant_gradient: bool = False


@dataclass(frozen=True)
class ImageDiagnosticsConfig:
    """Exact-image solver and posterior image diagnostics written after sampling.

    Every reporting draw's model images are found with ``exact_image_finder``; the lens-only images (without the GRF
    conditional mean) use one batched local Levenberg-Marquardt (LM) solve started at the observed positions, without
    fallback.  Predicted images are matched to the observed ones within ``match_tolerance_arcsec``.

    Parameters
    ----------
    fold_pair_max_separation_arcsec : float, optional
        Largest separation of a candidate fold image pair (same family, model-verified), in arcsec.  Positive.
        Default ``5.0``.
    exact_image_min_distance_arcsec : float, optional
        ``"lenstronomy"``: its ``min_distance``, in arcsec.  ``"local-lm-adaptive"``: starting half-size of the 3 x 3
        box of fallback starts (at most ``exact_image_lm_trust_radius_arcsec``).  Default ``0.5``.
    exact_image_precision_limit : float, optional
        ``"lenstronomy"``: its ``precision_limit``.  LM solvers: lower limit of the source-plane tolerance of
        convergence, in arcsec.  Default ``1e-2``.
    exact_image_num_iter_max : int, optional
        ``"lenstronomy"``: its ``num_iter_max``.  Default ``100``.
    exact_image_finder : str, optional
        Finder of each reporting draw's images: ``"global-jax"`` (seeds from a coarse grid over each family window, one
        batched LM solve; also reports extra model images), ``"local-lm-adaptive"`` (LM from the observed positions,
        with a shrinking-box fallback for images not found), ``"local-lm"`` (LM from the observed positions) or
        ``"lenstronomy"`` (lenstronomy's image-position solver).  Default ``"global-jax"``.
    exact_image_grid_pitch_arcsec : float, optional
        ``"global-jax"``: pitch of the coarse grid over each family window, in arcsec.  Positive.  Default ``0.5``.
    exact_image_fine_pitch_arcsec : float, optional
        ``"global-jax"``: pitch of the sub-grid seeded inside near-source cells close to a critical curve, in arcsec;
        at most the coarse pitch.  Default ``0.1``.
    exact_image_critical_det_threshold : float, optional
        ``"global-jax"``: :math:`|\\det A|` (dimensionless) below which a cell counts as close to a critical curve.
        Default ``0.25``.
    exact_image_seed_tolerance_pitches : float, optional
        ``"global-jax"``: a cell is near the source when :math:`|\\beta - \\beta_{source}|` is within this many coarse
        pitches (scaled by the local stretching).  Positive.  Default ``3.0``.
    exact_image_window_margin_arcsec : float, optional
        ``"global-jax"``: margin added around each family's image span to form its search window, in arcsec.  Default
        ``3.0``.
    exact_image_min_separation_arcsec : float, optional
        Local solver: two observed images that converge onto roots closer than this, in arcsec, are one model image.
        Default ``0.05``.
    exact_image_root_residual_arcsec : float, optional
        ``"global-jax"``: a candidate root is polished with Newton steps and accepted only when its source-plane residual
        is below this, in arcsec.  Points on a critical curve with the source just outside (or inside) the caustic stall at
        a residual equal to the caustic offset, so a loose tolerance would accept them; a true image converges to round-
        off.  A fold pair 0.001" apart comes from a source about 1e-8" inside its caustic, so the tolerance sits well
        below that.  Default ``1e-11``.
    exact_image_root_separation_arcsec : float, optional
        ``"global-jax"``: two accepted roots closer than this, in arcsec, are one image; below the separation of the
        tightest fold pairs to be resolved (0.001").  Default ``1e-4``.
    exact_image_polish_iterations : int, optional
        ``"global-jax"``: Newton iterations polishing each candidate root.  Default ``40``.
    exact_image_displacement_tol_arcsec : float, optional
        LM convergence tolerance on the step and the source-plane residual, in arcsec; a ``"global-jax"`` root must
        meet it in both planes.  Default ``1e-4``.
    exact_image_identification_tol_arcsec : float, optional
        A solve identifies an image when the Newton step at its end point is below this, in arcsec (a looser test than
        convergence, met by near-roots).  Default ``1e-3``.
    exact_image_lm_max_iter : int, optional
        LM iteration budget per solve.  Default ``100``.
    exact_image_lm_trust_radius_arcsec : float, optional
        LM steps are softly limited to this length (tanh scaling), in arcsec; ``0`` or less disables the limit.
        Default ``1.0``.
    exact_image_adaptive_max_levels : int, optional
        ``"local-lm-adaptive"``: number of times the fallback box is halved.  Default ``8``.
    match_tolerance_arcsec : float, optional
        Largest distance between a predicted and an observed image for a match, in arcsec; unmatched predictions are
        extra images.  Default ``5.0``.
    """
    fold_pair_max_separation_arcsec: float = 5.0
    exact_image_min_distance_arcsec: float = 0.5
    exact_image_precision_limit: float = 1.0e-2
    exact_image_num_iter_max: int = 100
    exact_image_finder: str = "global-jax"
    exact_image_grid_pitch_arcsec: float = 0.5
    exact_image_fine_pitch_arcsec: float = 0.1
    exact_image_critical_det_threshold: float = 0.25
    exact_image_seed_tolerance_pitches: float = 3.0
    exact_image_window_margin_arcsec: float = 3.0
    exact_image_min_separation_arcsec: float = 0.05
    exact_image_root_residual_arcsec: float = 1.0e-11
    exact_image_root_separation_arcsec: float = 1.0e-4
    exact_image_polish_iterations: int = 40
    exact_image_displacement_tol_arcsec: float = 1.0e-4
    exact_image_identification_tol_arcsec: float = 1.0e-3
    exact_image_lm_max_iter: int = 100
    exact_image_lm_trust_radius_arcsec: float = 1.0
    exact_image_adaptive_max_levels: int = 8
    match_tolerance_arcsec: float = 5.0


@dataclass(frozen=True)
class TruthRecoveryConfig:
    """Comparison with true lens maps (simulated clusters) and the source redshift of the critical-curve outputs.

    Parameters
    ----------
    kappa_true_fits : str or pathlib.Path or None, optional
        FITS map (with WCS) of the true convergence.  Setting it turns on the truth comparison: model maps of the
        posterior median and draws on the map's grid, residuals and the projected-mass aperture profile.  None skips
        it.  Default None.
    gammax_true_fits : str or pathlib.Path or None, optional
        FITS map of the true shear component :math:`\\gamma_1`.  With ``gammay_true_fits`` it adds the true
        :math:`\\det A` and magnification.  Default None.
    gammay_true_fits : str or pathlib.Path or None, optional
        FITS map of the true shear component :math:`\\gamma_2`.  Default None.
    truth_grid_size : int, optional
        Largest number of pixels per side of the comparison grid (the true map's footprint, resampled).  Nonnegative;
        ``0`` keeps the native grid.  Default ``256``.
    caustic_source_redshift : float, optional
        Source redshift of the critical curves, convergence maps and truth comparison.  The true maps are compared at
        this redshift; the magnification comparison reads the maps' ``ZSOURCE`` (or ``REDSHIFT``) header and falls back
        to it.  When it is not behind the lens, the critical-curve figure uses the highest family redshift and the
        truth comparison is skipped.  Default ``9.0``.
    """
    kappa_true_fits: str | Path | None = None
    gammax_true_fits: str | Path | None = None
    gammay_true_fits: str | Path | None = None
    truth_grid_size: int = 256
    caustic_source_redshift: float = 9.0


@dataclass(frozen=True)
class RGBDisplayConfig:
    """Overrides of the calibrated Lupton (asinh) RGB display of the imaging cutouts.

    Each None keeps the display recipe's default for the bands (HST Frontier Fields, JWST NIRCam or a three-band
    composite).

    Parameters
    ----------
    q : float or None, optional
        Lupton asinh softening parameter, dimensionless.  Default None.
    stretch : float or None, optional
        Lupton linear stretch, in the reference band's flux units.  Default None.
    minimum : float or None, optional
        Black level, in the reference band's flux units.  Default None.
    red_gain : float or None, optional
        Gain of the red channel, dimensionless.  The three gains apply only when all three are set.  Default None.
    green_gain : float or None, optional
        Gain of the green channel, dimensionless.  Default None.
    blue_gain : float or None, optional
        Gain of the blue channel, dimensionless.  Default None.
    """
    q: float | None = None
    stretch: float | None = None
    minimum: float | None = None
    red_gain: float | None = None
    green_gain: float | None = None
    blue_gain: float | None = None


@dataclass(frozen=True)
class ImageCatalogCutoutConfig:
    """Imaging behind the cluster mass figure, the RGB cutouts and the fold-diagnostic panels.

    Parameters
    ----------
    image_dir : str or pathlib.Path or None, optional
        Directory of the cluster's FITS imaging.  None skips the figures that need imaging.  Default None.
    candidate_catalog_path : str or pathlib.Path or None, optional
        Optional catalog of proposed image associations, drawn on the RGB critical-curve figure for display only.
        Default None.
    paper_small_lambda_cutouts : tuple of str, optional
        Panels of the standalone small-eigenvalue posterior figure, one per selector, in order: ``image:<label>`` (one
        image), ``family:<id>`` (a knot's images) or ``system:<id>`` (every knot of a saved fold cutout).  Empty
        (default) skips the figure.
    paper_fold_cutouts : tuple of str, optional
        Panels of the standalone fold-cutout figure, same selectors.  Empty (default) skips the figure.
    image_scale : str, optional
        Pixel scale of the imaging to use when several exist: ``"auto"``, ``"20mas"``, ``"30mas"``, ``"40mas"`` or
        ``"60mas"``.  Default ``"60mas"``.
    bands : tuple of str or None, optional
        Bands of the RGB composite.  None uses the HST Frontier Fields bands (F435W, F606W, F814W, F105W, F125W,
        F140W, F160W).  Default None.
    figure_suffix : str, optional
        Suffix appended to the cluster mass figure name and related products, e.g. ``"_jwst"``.  Default ``""``.
    rgb : RGBDisplayConfig, optional
        Display overrides.  Default: ``RGBDisplayConfig()`` (recipe defaults).
    mode : str, optional
        Not read by the current pipeline.  Default ``"full"``.
    dpi : int or None, optional
        Not read by the current pipeline.  Default None.
    max_side_pixels : int or None, optional
        Not read by the current pipeline.  Default None.
    critical_lines : str, optional
        Not read by the current pipeline.  Default ``"auto"``.
    cutouts : bool, optional
        Not read by the current pipeline.  Default True.
    """
    image_dir: str | Path | None = None
    candidate_catalog_path: str | Path | None = None
    paper_small_lambda_cutouts: tuple[str, ...] = ()  # image:/family:/system: selectors, one per panel
    paper_fold_cutouts: tuple[str, ...] = ()          # same syntax; empty disables the figure
    image_scale: str = "60mas"
    bands: tuple[str, ...] | None = None
    figure_suffix: str = ""
    rgb: RGBDisplayConfig = field(default_factory=RGBDisplayConfig)
    mode: str = "full"
    dpi: int | None = None
    max_side_pixels: int | None = None
    critical_lines: str = "auto"
    cutouts: bool = True


@dataclass(frozen=True)
class PartitionConfig:
    """Uncertain images whose sources are inferred jointly with the lens (``kappaccino.partition.groups``).

    Each uncertain image may be a new image of a trusted family, one of a group of uncertain images that share a new
    source, or unrelated.  The likelihood sums exactly over the partitions of the uncertain images into the groups of a
    library, with every source integrated out, so NUTS samples the lens only.  The partition is drawn after sampling.
    The trusted families are kept as given; knots are not supported for uncertain images.

    Parameters
    ----------
    catalog_path : str or pathlib.Path or None, optional
        CSV table of the uncertain images: ``image_label``, ``ra``, ``dec`` (degrees); optional ``parity`` (-1, 0,
        +1), redshifts (``zspec_best``, ``catalog_z``, ``zphot_best`` with ``catalog_z_sigma`` or ``pagul_zpdf_low``/``pagul_zpdf_high``), ``mag_<band>`` and ``mag_err_<band>``, and the
        claimed association: ``claimed_family`` with ``association_prior``.  Must exist.  None (default) takes the
        uncertain images from the image catalog: every image the run does not trust (``ImageConstraintsConfig.
        trusted_prior_min``, a family without a positive redshift, a family left with one image), claimed by its own
        family with its catalog ``prior``.
    association_weight : bool, optional
        Weight each image with a claimed family by its ``association_prior``: that family (trusted, or the new source of
        the uncertain images claimed by the same family) gets the prior, the other options share the rest as they would
        without a claim.  The images a family claims always form a group of the library.  Default True.
    population_catalog_path : str or pathlib.Path or None, optional
        CSV photometric catalog of the field (a redshift column and the same ``mag_<band>`` columns) for the prior of a
        source's redshift and colors.  None fits it on the trusted and uncertain images themselves.  Default None.
    bands : tuple of str or None, optional
        Bands whose adjacent differences are the colors.  None: bands measured for at least half the uncertain images.
    redshift_grid : tuple of (float, float, int) or None, optional
        (z_min, z_max, points) of the redshift quadrature of a source, spaced uniformly in the lensing efficiency
        D_ls / D_s.  A family of several images fixes the efficiency to a few percent, so the grid needs many points:
        with too few, the evidence of a true family is underestimated and the family is split.  None spans the catalog
        redshifts +- 2 sigma with 96 points, above z_lens + 0.05.  Default None.
    source_prior_sigma_arcsec : float or None, optional
        Width per axis of the Gaussian prior on a new or unrelated source's position: the evidence cost of a source.
        None uses ``likelihood.source_position_prior_sigma_arcsec``.  Default None.
    color_scatter : float or None, optional
        Same-source color scatter in mag; None fits it on the trusted families.  Default None.
    outlier_fraction, outlier_scatter : float, optional
        Fraction of images whose colors scatter by ``outlier_scatter`` mag instead.  Defaults 0.05 and 1.0.
    photoz_outlier_fraction : float, optional
        Fraction of catalog redshifts that are catastrophic outliers, spread uniformly over redshifts 0.5 to 6, so that
        one wrong photo-z does not split a family.  Default 0.05.
    population_components : int, optional
        Components of the population mixture.  Default 6.
    unrelated_fraction : float, optional
        Prior probability that an uncertain image is unrelated to every other image; the rest is shared equally by the
        trusted families and the new sources.  Default 0.5.
    library_pair_margin : float, optional
        Two images are linked when, at a library lens, the prior-weighted evidence of the pair as one source comes
        within this (log) of the two images alone.  Default 3.0.
    library_group_margin : float, optional
        A clique of linked images enters the library when its prior-weighted evidence as one source comes within this
        (log) of its images alone at a library lens.  Default 5.0.
    library_max_group : int, optional
        Most images in one library group.  Default 8.
    library_refits : int, optional
        MAP refits of the lens with the library summed, each followed by a rebuild of the library at every lens so far.
        The chains start at the last refit.  Default 2.
    posterior_draws : int, optional
        Posterior draws used for the assignment probabilities and joint draws.  Default 500.
    """
    catalog_path: str | Path | None = None
    association_weight: bool = True
    population_catalog_path: str | Path | None = None
    bands: tuple[str, ...] | None = None
    redshift_grid: tuple[float, float, int] | None = None
    source_prior_sigma_arcsec: float | None = None
    color_scatter: float | None = None
    outlier_fraction: float = 0.05
    outlier_scatter: float = 1.0
    photoz_outlier_fraction: float = 0.05
    population_components: int = 6
    unrelated_fraction: float = 0.5
    library_pair_margin: float = 3.0
    library_group_margin: float = 5.0
    library_max_group: int = 8
    library_refits: int = 2
    posterior_draws: int = 500


@dataclass(frozen=True)
class ReportingConfig:
    """The posterior draws behind every reported product.

    One set of random posterior draws (over all chains, chosen with ``runtime.seed``) is evaluated once, each draw with
    its own GRF conditional mean, and every figure, table and results macro is a median over those draws with its 16th
    to 84th percentile range.  No product uses the single best sample.

    The image positions, residuals, RMS and recovered counts are cheap (a local solve from the observed positions, about
    0.1 s per draw) and need many draws: the median RMS of 64 draws scatters by about 0.01" between draw sets on Hera.
    They use ``image_draws`` draws; the maps, curves and other products use the first ``draws`` of them (one seeded
    shuffle of the posterior, so the product draws are a subset of the image draws).

    Parameters
    ----------
    draws : int, optional
        Number of draws behind the maps, curves, masses and the other products (all of them when the posterior has
        fewer).  Positive.  Default ``64``.
    image_draws : int, optional
        Number of draws behind the image positions, residuals, RMS and recovered counts, solved with the local solver.
        At least ``draws``.  Default ``256``.
    extra_images : bool, optional
        Run the global image search on the ``draws`` product draws to find extra (unobserved) model images, at about
        15 s (Hera) to 140 s (Ares) per draw.  Default True.
    """
    draws: int = 64
    image_draws: int = 256
    extra_images: bool = True


@dataclass(frozen=True)
class LensClusterSolverConfig:
    """Complete configuration of a kappaccino run.

    Parameters
    ----------
    model : LensModelConfig or None, optional
        The lens model.  Required by :meth:`validate`.  Default None.
    paths : RunPathsConfig, optional
        Output directory and naming.  Default: ``RunPathsConfig()``.
    runtime : RuntimeConfig, optional
        Seed, chains and logging.  Default: ``RuntimeConfig()``.
    search : SearchConfig, optional
        The initial search.  Default: ``SearchConfig()``.
    sampling : SamplingConfig, optional
        NUTS settings.  Default: ``SamplingConfig()``.
    members : MemberConfig, optional
        Member classification.  Default: ``MemberConfig()``.
    scaling : ScalingModelConfig, optional
        Member scaling relation and deviations.  Default: ``ScalingModelConfig()``.
    likelihood : LikelihoodConfig, optional
        Likelihood and error model.  Default: ``LikelihoodConfig()``.
    image_diagnostics : ImageDiagnosticsConfig, optional
        Exact-image solver and posterior image diagnostics.  Default: ``ImageDiagnosticsConfig()``.
    truth : TruthRecoveryConfig, optional
        Truth comparison and critical-curve redshift.  Default: ``TruthRecoveryConfig()``.
    reporting : ReportingConfig, optional
        The posterior draws behind the reported products.  Default: ``ReportingConfig()``.
    image_catalog : ImageCatalogCutoutConfig, optional
        Imaging for the figures.  Default: ``ImageCatalogCutoutConfig()``.
    partition : PartitionConfig or None, optional
        Uncertain images assigned to families jointly with the lens.  Default None (none).
    """
    model: LensModelConfig | None = None
    paths: RunPathsConfig = field(default_factory=RunPathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    members: MemberConfig = field(default_factory=MemberConfig)
    scaling: ScalingModelConfig = field(default_factory=ScalingModelConfig)
    likelihood: LikelihoodConfig = field(default_factory=LikelihoodConfig)
    image_diagnostics: ImageDiagnosticsConfig = field(default_factory=ImageDiagnosticsConfig)
    truth: TruthRecoveryConfig = field(default_factory=TruthRecoveryConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    image_catalog: ImageCatalogCutoutConfig = field(default_factory=ImageCatalogCutoutConfig)
    partition: PartitionConfig | None = None

    def with_updates(self, **updates: Any) -> "LensClusterSolverConfig":
        """Return a copy with some top-level sections replaced.

        Parameters
        ----------
        **updates
            Fields of :class:`LensClusterSolverConfig` and their new values, e.g. ``sampling=SamplingConfig(warmup=500)``.

        Returns
        -------
        LensClusterSolverConfig
            The new configuration (this one is unchanged).

        Raises
        ------
        TypeError
            If a key is not a field of :class:`LensClusterSolverConfig`.
        """
        return replace(self, **updates)

    def validate(self) -> "LensClusterSolverConfig":
        """Check the configuration (see :func:`validate_config`).

        Returns
        -------
        LensClusterSolverConfig
            This configuration, unchanged, so the call can be chained.

        Raises
        ------
        ValueError
            If a field is missing, out of range or inconsistent with another, or a catalog file does not exist.
        """
        validate_config(self)
        return self

    def to_json_dict(self) -> dict[str, Any]:
        """Return the configuration as JSON-serializable nested dictionaries.

        Nested configurations become dictionaries keyed by field name, with a ``"_class"`` entry (``module:name``) where
        the field's type does not determine the class (the halos of ``LensModelConfig.large_halos``).  Priors become their
        ``priors.to_dict`` descriptions, paths become strings, infinite floats become ``"inf"`` or ``"-inf"`` and tuples
        become lists.  :meth:`from_json_dict` is the inverse.

        Returns
        -------
        dict
            Nested dictionaries keyed by field name.
        """
        return _jsonable(self, LensClusterSolverConfig)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "LensClusterSolverConfig":
        """Rebuild a configuration from :meth:`to_json_dict` output (``config.json`` of a run).

        Parameters
        ----------
        payload : dict
            The JSON dictionaries.

        Returns
        -------
        LensClusterSolverConfig
            The configuration; ``to_json_dict`` of the result equals ``payload``.
        """
        return _from_jsonable(payload, cls)


CLASS_KEY = "_class"


def _hint_class(hint: Any) -> type | None:
    """The dataclass a type hint names unambiguously (``X`` or ``X | None``), else None."""
    if is_dataclass(hint):
        return hint
    options = [a for a in get_args(hint) if a is not type(None)] if get_origin(hint) in (Union, UnionType) else []
    return options[0] if len(options) == 1 and is_dataclass(options[0]) else None


def _element_hint(hint: Any) -> Any:
    """The element type of ``tuple[X, ...]`` or ``list[X]`` (through ``| None``), else Any."""
    if get_origin(hint) in (Union, UnionType):
        options = [a for a in get_args(hint) if a is not type(None)]
        hint = options[0] if len(options) == 1 else Any
    args = get_args(hint) if get_origin(hint) in (tuple, list) else ()
    return args[0] if args and (len(args) == 1 or args[1] is Ellipsis) else Any


def _jsonable(value: Any, hint: Any = Any) -> Any:
    if is_distribution(value):
        from .priors import to_dict
        return to_dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        hints = get_type_hints(type(value))
        payload = {f.name: _jsonable(getattr(value, f.name), hints.get(f.name, Any)) for f in fields(value)}
        if _hint_class(hint) is not type(value):
            payload = {CLASS_KEY: f"{type(value).__module__}:{type(value).__qualname__}", **payload}
        return payload
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        element = _element_hint(hint)
        return [_jsonable(item, element) for item in value]
    return value


def _from_jsonable(value: Any, hint: Any = Any) -> Any:
    if isinstance(value, dict):
        if CLASS_KEY in value or _hint_class(hint) is not None:
            if CLASS_KEY in value:
                module, name = value[CLASS_KEY].split(":")
                cls = getattr(importlib.import_module(module), name)
            else:
                cls = _hint_class(hint)
            hints = get_type_hints(cls)
            return cls(**{key: _from_jsonable(item, hints.get(key, Any)) for key, item in value.items() if key != CLASS_KEY})
        if {"family", "log"} <= value.keys():
            from .priors import from_dict
            return from_dict(value)
        return {key: _from_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        element = _element_hint(hint)
        items = [_from_jsonable(item, element) for item in value]
        return items if get_origin(hint) is list else tuple(items)
    if value in ("inf", "-inf"):
        return float(value)
    return value


def validate_run_label(label: str) -> str:
    """Check that a run name is usable as a single directory name.

    Parameters
    ----------
    label : str
        The run name.

    Returns
    -------
    str
        ``label``, unchanged.

    Raises
    ------
    ValueError
        If ``label`` is empty, has leading or trailing whitespace, is ``"."`` or ``".."``, or contains a path
        separator, a null character or a line break.
    """
    if not label or label.strip() != label or label in {".", ".."} or any(c in label for c in "/\\\0\n\r"):
        raise ValueError("run label must be a nonempty single directory name")
    return label


def _positive(name: str, value: float, *, allow_zero: bool = False) -> None:
    if not math.isfinite(float(value)) or value < 0.0 or (not allow_zero and value <= 0.0):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}.")


def _positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def validate_config(config: LensClusterSolverConfig) -> None:
    """Check a configuration before a run.

    Checks the run name, the lens model (at least one large halo, the image constraints, positive halo and population
    parameters, existing catalog files, unique ids), the allowed string options, the numeric ranges of every section,
    the priors (supported ``numpyro.distributions``, positive quantities where required) and the consistency of the
    GP knee settings with their tables.

    Parameters
    ----------
    config : LensClusterSolverConfig
        The configuration to check.

    Raises
    ------
    ValueError
        At the first problem found, with a message naming the field.
    """
    validate_run_label(config.paths.run_name or "")
    _validate_model_config(config.model)
    runtime = config.runtime
    if isinstance(runtime.seed, bool) or not isinstance(runtime.seed, Integral) or int(runtime.seed) < 0:
        raise ValueError("seed must be a nonnegative integer.")
    _positive_int("chains", runtime.chains)
    if runtime.nuts_chain_method not in {"parallel", "sequential", "vectorized"}:
        raise ValueError("nuts_chain_method must be 'parallel', 'sequential', or 'vectorized'.")
    if runtime.cpu_threads_per_chain is not None:
        _positive_int("cpu_threads_per_chain", runtime.cpu_threads_per_chain)
    search = config.search
    _positive_int("search.n_starts", search.n_starts)
    _positive_int("search.lbfgs_max_iter", search.lbfgs_max_iter)
    for label in ("halo_method", "member_method"):
        if getattr(search, label) not in {"lbfgs", "optax-lbfgs", "adam", "svi"}:
            raise ValueError(f"search.{label} must be 'lbfgs', 'optax-lbfgs', 'adam' or 'svi'.")
    _positive_int("search.member_starts", search.member_starts)
    _positive_int("search.adam_steps", search.adam_steps)
    _positive_int("search.svi_steps", search.svi_steps)
    _positive("search.adam_learning_rate", search.adam_learning_rate)
    _positive("search.svi_learning_rate", search.svi_learning_rate)
    _positive_int("search.polish_max_iter", search.polish_max_iter)
    if search.polish_scatter is not None:
        _positive("search.polish_scatter", search.polish_scatter)
    if search.polish_radius_scatter is not None:
        _positive("search.polish_radius_scatter", search.polish_radius_scatter)
    _positive("search.mode_margin", search.mode_margin)
    _positive("search.mode_parameter_tolerance", search.mode_parameter_tolerance)
    _positive_int("search.max_chain_modes", search.max_chain_modes)
    _positive("search.max_median_abs_determinant", search.max_median_abs_determinant)
    sampling = config.sampling
    for name in ("warmup", "samples", "max_tree_depth", "thin"):
        _positive_int(f"sampling.{name}", getattr(sampling, name))
    for name in ("wbic_warmup", "wbic_samples"):
        if getattr(sampling, name) is not None:
            _positive_int(f"sampling.{name}", getattr(sampling, name))
    if not 0.0 < sampling.target_accept < 1.0:
        raise ValueError("sampling.target_accept must lie between 0 and 1.")
    _positive("sampling.initial_step_size", sampling.initial_step_size)
    if sampling.dense_offset_members not in {"all", "centered", "exact", "none"}:
        raise ValueError("sampling.dense_offset_members must be 'all', 'centered', 'exact' or 'none'.")
    if sampling.far_field_offsets not in {"sample", "fixed"}:
        raise ValueError("sampling.far_field_offsets must be 'sample' or 'fixed'.")
    bad = set(getattr(config.scaling, "fixed_relation_fields", ())) - {"sigma", "cutkpc", "alpha_sigma", "beta_radius"}
    if bad:
        raise ValueError(f"scaling.fixed_relation_fields: unknown fields {sorted(bad)}; allowed: sigma, cutkpc, alpha_sigma, beta_radius.")
    members = config.members
    _positive("members.far_field_cut_radii", members.far_field_cut_radii)
    if members.cut_radius_inflation < 1.0 or not math.isfinite(members.cut_radius_inflation):
        raise ValueError("members.cut_radius_inflation must be at least one.")
    if not 0.0 < members.template_tolerance < 1.0:
        raise ValueError("members.template_tolerance must lie between zero and one.")
    _positive("members.far_field_barrier_width", members.far_field_barrier_width)
    _positive("members.far_field_barrier_strength", members.far_field_barrier_strength)
    if members.top_exact is not None and (isinstance(members.top_exact, bool) or not isinstance(members.top_exact, Integral) or members.top_exact < 0):
        raise ValueError("members.top_exact must be a nonnegative integer or None.")
    if members.min_deflection_arcsec is not None:
        _positive("members.min_deflection_arcsec", members.min_deflection_arcsec)
    if members.membership not in {"fixed", "marginalize"}:
        raise ValueError("members.membership must be 'fixed' or 'marginalize'.")
    if not 0.0 < float(members.membership_default_probability) <= 1.0:
        raise ValueError("members.membership_default_probability must lie in (0, 1].")
    if (isinstance(members.membership_max_enumerated, bool) or not isinstance(members.membership_max_enumerated, Integral)
            or members.membership_max_enumerated < 0):
        raise ValueError("members.membership_max_enumerated must be a nonnegative integer.")
    if members.membership_calibration_prior is not None and not is_distribution(members.membership_calibration_prior):
        raise ValueError("members.membership_calibration_prior must be a numpyro distribution or None.")
    scaling = config.scaling
    if config.members.centering not in {"auto", "exact", "none"}:
        raise ValueError("members.centering must be 'auto', 'exact' or 'none'.")
    _positive("members.centering_ratio", config.members.centering_ratio)
    if config.members.classification_prior_quantile is not None and not (0.5 < float(config.members.classification_prior_quantile) < 1.0):
        raise ValueError("members.classification_prior_quantile must lie in (0.5, 1).")
    if config.likelihood.likelihood_mode not in LIKELIHOOD_MODES:
        raise ValueError(f"likelihood.likelihood_mode must be one of {sorted(LIKELIHOOD_MODES)}.")
    if config.likelihood.likelihood_mode != "gp_image_plane" and config.likelihood.sigma_meas_prior is None:
        raise ValueError("likelihood.sigma_meas_prior is required: sigma_meas is the only measurement error of the images.")
    _positive("likelihood.gp_correlation_length_arcsec", config.likelihood.gp_correlation_length_arcsec)
    _positive("likelihood.gp_floor_arcsec", config.likelihood.gp_floor_arcsec, allow_zero=True)
    _positive("likelihood.sigma_gp_start_arcsec", config.likelihood.sigma_gp_start_arcsec)
    _positive("likelihood.gp_sigma_meas_start_arcsec", config.likelihood.gp_sigma_meas_start_arcsec)
    if config.likelihood.gp_kernel not in {"image_plane", "knee"}:
        raise ValueError("likelihood.gp_kernel must be 'image_plane' or 'knee'.")
    if config.likelihood.gp_blocks not in {"all", "system", "family"}:
        raise ValueError("likelihood.gp_blocks must be 'all', 'system' or 'family'.")
    _positive("likelihood.gp_knee_gamma", config.likelihood.gp_knee_gamma)
    if config.likelihood.gp_knee_gamma_prior is not None:
        g_lo, g_hi = _bounded_prior("likelihood.gp_knee_gamma_prior", config.likelihood.gp_knee_gamma_prior)
        step = float(config.likelihood.gp_knee_gamma_step)
        if not (0.0 < g_lo < g_hi) or not (g_lo <= float(config.likelihood.gp_knee_gamma) <= g_hi):
            raise ValueError("likelihood.gp_knee_gamma_prior must have bounds 0 < low < high holding gp_knee_gamma (the start).")
        if step <= 0.0 or abs((g_hi - g_lo) / step - round((g_hi - g_lo) / step)) > 1e-9 or round((g_hi - g_lo) / step) < 2:
            raise ValueError("likelihood.gp_knee_gamma_step must divide the gp_knee_gamma_prior range into at least 2 intervals.")
    modes = tuple(config.likelihood.gp_knee_project_modes)
    if any(m not in {"translation", "convergence", "shear"} for m in modes) or len(set(modes)) != len(modes):
        raise ValueError("likelihood.gp_knee_project_modes takes distinct names among 'translation', 'convergence', 'shear'.")
    if modes and config.likelihood.gp_blocks != "all":
        raise ValueError("likelihood.gp_knee_project_modes needs gp_blocks = 'all' (field-wide modes over one covariance).")
    lo, hi = config.likelihood.gp_knee_scales_arcsec
    if not (0.0 < float(lo) < float(hi)):
        raise ValueError("likelihood.gp_knee_scales_arcsec must be (min, max) with 0 < min < max.")
    lam_lo, lam_hi, lam_n = config.likelihood.gp_knee_lambda_grid
    if not (0.0 < float(lam_lo) < float(lam_hi)) or int(lam_n) < 8:
        raise ValueError("likelihood.gp_knee_lambda_grid must be (min, max, count) with 0 < min < max and count >= 8.")
    spectrum = config.likelihood.gp_knee_spectrum
    if spectrum not in {"knee", "two_knee", "broken_power_law", "power_law", "knee_foreground", "knee_background"}:
        raise ValueError("likelihood.gp_knee_spectrum must be 'knee', 'two_knee', 'broken_power_law', 'power_law', 'knee_foreground' "
                         "or 'knee_background'.")
    if config.likelihood.gp_kernel == "knee" and spectrum != "knee":
        if config.likelihood.sigma_grf_prior is None:
            raise ValueError(f"likelihood.gp_knee_spectrum {spectrum!r} needs the mass field (sigma_grf_prior).")
        if spectrum == "power_law" and config.likelihood.gp_knee_gamma_prior is None:
            raise ValueError("likelihood.gp_knee_spectrum 'power_law' samples the slope: set gp_knee_gamma_prior.")
        if spectrum == "broken_power_law":
            b_lo, b_hi = _bounded_prior("likelihood.gp_knee_beta_low_prior", config.likelihood.gp_knee_beta_low_prior)
            b_step = float(config.likelihood.gp_knee_beta_low_step)
            if not (b_lo < b_hi < float(config.likelihood.gp_knee_gamma)) or not (b_lo <= float(config.likelihood.gp_knee_beta_low_start) <= b_hi):
                raise ValueError("likelihood.gp_knee_beta_low_prior must have bounds below gp_knee_gamma that hold gp_knee_beta_low_start.")
            if b_step <= 0.0 or abs((b_hi - b_lo) / b_step - round((b_hi - b_lo) / b_step)) > 1e-9 or round((b_hi - b_lo) / b_step) < 2:
                raise ValueError("likelihood.gp_knee_beta_low_step must divide gp_knee_beta_low_prior into at least 2 intervals.")
            if config.likelihood.gp_knee_gamma_prior is not None:
                raise ValueError("likelihood.gp_knee_spectrum 'broken_power_law' fixes the high-k slope: set gp_knee_gamma_prior to None.")
        if spectrum in {"two_knee", "knee_foreground", "knee_background"}:
            upper2 = physical_bounds(config.likelihood.lambda_b2_prior)[1]
            if float(upper2) > float(config.likelihood.gp_knee_lambda_grid[1]) * (1.0 + 1.0e-12):
                raise ValueError("likelihood.lambda_b2_prior must be bounded above by at most the largest tabulated knee wavelength.")
            if spectrum == "two_knee" and float(upper2) > physical_bounds(config.likelihood.lambda_b_prior)[0] * (1.0 + 1.0e-12):
                raise ValueError("likelihood.gp_knee_spectrum 'two_knee': the upper bound of lambda_b2_prior must not exceed the lower "
                                 "bound of lambda_b_prior, so the small and the large knee cannot swap.")
            if spectrum in {"knee_foreground", "knee_background"} and config.likelihood.gp_knee_project_modes:
                raise ValueError(f"likelihood.gp_knee_project_modes assume one redshift weighting; not with {spectrum!r}.")
            if spectrum == "knee_background" and not float(config.likelihood.gp_knee2_plane_redshift) > float(config.model.large_halos[0].z_lens):
                raise ValueError("likelihood.gp_knee2_plane_redshift must lie behind the cluster.")
            for name in ("gp_knee2_sigma_grf_start_arcsec", "lambda_b2_start_arcsec"):
                _positive(f"likelihood.{name}", getattr(config.likelihood, name))
    for start, prior, label in ((config.likelihood.lambda_b_start_arcsec, config.likelihood.lambda_b_prior, "lambda_b"),
                                (config.likelihood.lambda_b2_start_arcsec, config.likelihood.lambda_b2_prior, "lambda_b2")):
        low, high = physical_bounds(prior)
        if not low < float(start) < high:
            raise ValueError(f"likelihood.{label}_start_arcsec ({float(start):g}\") must lie inside the bounds of {label}_prior.")
    if config.likelihood.lambda_b_fixed_arcsec is not None:
        fixed = float(config.likelihood.lambda_b_fixed_arcsec)
        if spectrum == "power_law" or not (float(lam_lo) <= fixed <= min(float(lam_hi), float(hi))):
            raise ValueError("likelihood.lambda_b_fixed_arcsec must lie inside the knee tables and the band, and a power law has no knee.")
    if config.likelihood.gp_knee_ell_gp_fixed_arcsec is not None:
        _positive("likelihood.gp_knee_ell_gp_fixed_arcsec", config.likelihood.gp_knee_ell_gp_fixed_arcsec)
    knee_upper = physical_bounds(config.likelihood.lambda_b_prior)[1]
    # Bounds come back from the log coordinate (exp(ln 30) = 30.000000000000004): compare with a relative tolerance.
    exceeds = lambda value, limit: float(value) > float(limit) * (1.0 + 1.0e-12)
    if config.likelihood.gp_kernel == "knee" and math.isfinite(knee_upper) and exceeds(knee_upper, hi):
        raise ValueError(f"the upper bound of likelihood.lambda_b_prior ({float(knee_upper):g}\") must not exceed the longest GRF "
                         f"wavelength gp_knee_scales_arcsec[1] ({float(hi):g}\"): beyond it the spectrum is flat and the knee has no meaning.")
    if config.likelihood.gp_kernel == "knee" and exceeds(knee_upper, lam_hi):
        raise ValueError("likelihood.lambda_b_prior must be bounded above by at most the largest tabulated knee wavelength "
                         f"({float(lam_hi):g}\"): beyond the tables the knee is clamped and only the prior moves it.")
    for name in ("gp_knee_r_step_arcsec", "sigma_grf_start_arcsec", "lambda_b_start_arcsec", "gp_knee_sigma_gp_start_arcsec",
                 "gp_knee_ell_gp_start_arcsec", "gp_knee_sigma_meas_start_arcsec"):
        _positive(f"likelihood.{name}", getattr(config.likelihood, name))
    _positive("likelihood.gp_reference_redshift", config.likelihood.gp_reference_redshift)
    for name in ("sigma_def_prior", "sigma_grf_prior", "lambda_b_prior", "gp_knee2_sigma_grf_prior", "lambda_b2_prior",
                 "gp_knee_sigma_gp_prior", "gp_knee_ell_gp_prior", "gp_knee_sigma_meas_prior", "gp_sigma_meas_prior", "sigma_meas_prior",
                 "sigma_sel_prior", "lambda_sel_prior", "sigma_gp_prior", "gp_correlation_length_prior"):
        _positive_prior(f"likelihood.{name}", getattr(config.likelihood, name))
    for name in ("gp_sigma_meas_prior", "gp_knee_sigma_meas_prior", "sigma_gp_prior", "gp_knee_sigma_gp_prior", "sigma_def_prior",
                 "lambda_sel_prior", "lambda_b_prior", "gp_knee_ell_gp_prior"):
        if getattr(config.likelihood, name) is None:
            raise ValueError(f"likelihood.{name} is always sampled and needs a prior (e.g. dist.TruncatedNormal(0.0, 0.05, low=0.0) for a "
                             "half-normal).")
    if scaling.member_deviation not in {"sigma", "mass", "sigma_rcut", "mass_rcut"}:
        raise ValueError("scaling.member_deviation must be 'sigma', 'mass', 'sigma_rcut' or 'mass_rcut'.")
    if scaling.radius_deviation_members not in {"exact", "all"}:
        raise ValueError("scaling.radius_deviation_members must be 'exact' or 'all'.")
    if scaling.member_shape_scatter is not None and not (0.0 < float(scaling.member_shape_scatter) < 1.0):
        raise ValueError("scaling.member_shape_scatter must be None or in (0, 1).")
    if scaling.member_deviation_tail not in {"normal", "student_t"}:
        raise ValueError("scaling.member_deviation_tail must be 'normal' or 'student_t'.")
    if not float(scaling.member_deviation_dof) > 2.0:
        raise ValueError("scaling.member_deviation_dof must exceed 2 (a finite variance).")
    for halo in config.model.large_halos:
        cap = getattr(halo, "ellipticity_soft_cap", None)
        if cap is None:
            continue
        if not (0.0 < float(cap) < 1.0) or not float(halo.ellipticity_soft_width) > 0.0:
            raise ValueError(f"halo {halo.id}: ellipticity_soft_cap must lie in (0, 1) and ellipticity_soft_width be positive.")
        box = [(halo.priors or {}).get(name) for name in ("e1", "e2")]
        reach = ellipticity_box(float(cap))["e1"].high
        if any(prior is None or type(prior).__name__ != "Uniform" for prior in box) or max(float(prior.high) for prior in box) <= float(reach):
            raise ValueError(f"halo {halo.id}: ellipticity_soft_cap needs uniform e1, e2 priors whose box reaches beyond it.")
    for name in ("member_velocity_scatter_prior", "member_radius_scatter_prior", "corekpc_prior"):
        _positive_prior(f"scaling.{name}", getattr(scaling, name))
    for name in ("alpha_sigma_prior", "beta_radius_prior"):
        _supported_prior(f"scaling.{name}", getattr(scaling, name))
    if physical_bounds(scaling.beta_radius_prior)[0] < 0.0:
        raise ValueError("scaling.beta_radius_prior must be bounded below by zero or more.")
    if scaling.softening_length_prior is not None:
        _positive_prior("scaling.softening_length_prior", scaling.softening_length_prior)
    for population in config.model.member_populations:
        for name in ("sigma_prior", "cutkpc_prior"):
            _positive_prior(f"member population {population.id!r} {name}", getattr(population, name))
    for halo in tuple(config.model.large_halos) + tuple(config.model.independent_member_halos or ()):
        for name, prior in (halo.priors or {}).items():
            if prior is None:
                continue
            _supported_prior(f"halo {getattr(halo, 'id', None) or getattr(halo, 'catalog_id', '')} prior {name!r}", prior)
    likelihood = config.likelihood
    mean = likelihood.source_position_prior_mean_arcsec
    if len(mean) != 2 or not all(math.isfinite(float(v)) for v in mean):
        raise ValueError("source_position_prior_mean_arcsec requires two finite coordinates.")
    _positive("source_position_prior_sigma_arcsec", likelihood.source_position_prior_sigma_arcsec)
    diagnostics = config.image_diagnostics
    draws = config.reporting.draws
    if isinstance(draws, bool) or not isinstance(draws, Integral) or draws < 1:
        raise ValueError("reporting.draws must be a positive integer.")
    image_draws = config.reporting.image_draws
    if isinstance(image_draws, bool) or not isinstance(image_draws, Integral) or image_draws < draws:
        raise ValueError("reporting.image_draws must be an integer of at least reporting.draws.")
    if not math.isfinite(diagnostics.fold_pair_max_separation_arcsec) or diagnostics.fold_pair_max_separation_arcsec <= 0:
        raise ValueError("image_diagnostics.fold_pair_max_separation_arcsec must be finite and positive.")
    if diagnostics.exact_image_finder not in {"global-jax", "lenstronomy", "local-lm", "local-lm-adaptive"}:
        raise ValueError("image_diagnostics.exact_image_finder must be 'global-jax', 'lenstronomy', 'local-lm', or 'local-lm-adaptive'.")
    _positive("image_diagnostics.exact_image_grid_pitch_arcsec", diagnostics.exact_image_grid_pitch_arcsec)
    _positive("image_diagnostics.exact_image_fine_pitch_arcsec", diagnostics.exact_image_fine_pitch_arcsec)
    if diagnostics.exact_image_fine_pitch_arcsec > diagnostics.exact_image_grid_pitch_arcsec:
        raise ValueError("image_diagnostics.exact_image_fine_pitch_arcsec must not exceed the coarse grid pitch.")
    _positive("image_diagnostics.exact_image_root_residual_arcsec", diagnostics.exact_image_root_residual_arcsec)
    _positive("image_diagnostics.exact_image_root_separation_arcsec", diagnostics.exact_image_root_separation_arcsec)
    if int(diagnostics.exact_image_polish_iterations) < 1:
        raise ValueError("image_diagnostics.exact_image_polish_iterations must be positive.")
    _positive("image_diagnostics.exact_image_seed_tolerance_pitches", diagnostics.exact_image_seed_tolerance_pitches)
    if int(config.truth.truth_grid_size) < 0:
        raise ValueError("truth.truth_grid_size must be nonnegative.")
    if config.partition is not None:
        _validate_partition_config(config.partition, config)


def _validate_partition_config(partition: PartitionConfig, config: LensClusterSolverConfig) -> None:
    for label in ("catalog_path", "population_catalog_path"):
        path = getattr(partition, label)
        if path is not None and not Path(path).exists():
            raise ValueError(f"partition.{label} does not exist: {path}")
    if partition.redshift_grid is not None:
        z_min, z_max, points = partition.redshift_grid
        z_lens = float(config.model.large_halos[0].z_lens)
        if not z_lens < float(z_min) < float(z_max):
            raise ValueError(f"partition.redshift_grid needs z_lens ({z_lens}) < z_min < z_max.")
        _positive_int("partition.redshift_grid points", points)
    for label in ("source_prior_sigma_arcsec", "color_scatter"):
        if getattr(partition, label) is not None:
            _positive(f"partition.{label}", getattr(partition, label))
    _positive("partition.outlier_scatter", partition.outlier_scatter)
    if not 0.0 <= float(partition.outlier_fraction) < 1.0:
        raise ValueError("partition.outlier_fraction must be in [0, 1).")
    if not 0.0 < float(partition.unrelated_fraction) < 1.0:
        raise ValueError("partition.unrelated_fraction must be in (0, 1).")
    if not 0.0 <= float(partition.photoz_outlier_fraction) < 1.0:
        raise ValueError("partition.photoz_outlier_fraction must be in [0, 1).")
    for label in ("library_pair_margin", "library_group_margin"):
        if not math.isfinite(float(getattr(partition, label))) or float(getattr(partition, label)) < 0.0:
            raise ValueError(f"partition.{label} must be finite and >= 0.")
    if int(partition.library_max_group) < 2:
        raise ValueError("partition.library_max_group must be at least 2.")
    if int(partition.library_refits) < 0:
        raise ValueError("partition.library_refits must be >= 0.")
    _positive_int("partition.population_components", partition.population_components)
    _positive_int("partition.posterior_draws", partition.posterior_draws)


def _supported_prior(label: str, prior: Any) -> None:
    if not is_distribution(prior):
        raise ValueError(f"{label} must be a numpyro.distributions object, got {type(prior).__name__}.")
    try:
        split(prior)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from None


def _positive_prior(label: str, prior: Any) -> None:
    """A prior of a positive quantity (``None`` passes: the quantity is fixed or the term is off)."""
    if prior is None:
        return
    _supported_prior(label, prior)
    if not physical_bounds(prior)[0] >= 0.0:
        raise ValueError(f"{label} must be a prior of a positive quantity (e.g. dist.LogNormal or kappaccino.priors.truncated_lognormal).")


def _bounded_prior(label: str, prior: Any) -> tuple[float, float]:
    """The finite bounds of a prior that needs them (tabulated or truncated quantities)."""
    _supported_prior(label, prior)
    low, high = physical_bounds(prior)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(f"{label} must have finite bounds low < high (a Uniform or a two-sided TruncatedNormal).")
    return low, high


def _validate_model_config(model: LensModelConfig | None) -> None:
    if model is None:
        raise ValueError("LensClusterSolverConfig.model is required.")
    if not model.large_halos:
        raise ValueError("model.large_halos must contain at least one halo.")
    if model.image_constraints is None:
        raise ValueError("model.image_constraints is required.")
    if model.cosmology.H0 <= 0.0 or model.cosmology.Om0 < 0.0:
        raise ValueError("model.cosmology.H0 must be positive and Om0 nonnegative.")
    for halo in model.large_halos:
        if isinstance(halo, ShearHaloConfig):
            if int(halo.profile_type) != 14 or not math.isfinite(float(halo.gamma)) or halo.gamma < 0.0 or not math.isfinite(float(halo.angle_pos)):
                raise ValueError(f"shear halo {halo.id!r} must use profile_type=14 with finite non-negative gamma and finite angle_pos.")
        elif isinstance(halo, MassSheetConfig):
            if int(halo.profile_type) != 101 or not math.isfinite(float(halo.kappa)):
                raise ValueError(f"mass sheet {halo.id!r} must use profile_type=101 with finite kappa.")
        elif isinstance(halo, NFWHaloConfig):
            if int(halo.profile_type) != 12 or halo.scale_radius_kpc <= 0.0 or not math.isfinite(float(halo.log10_m200)):
                raise ValueError(f"NFW halo {halo.id!r} must use profile_type=12 with a positive scale radius and finite log10_m200.")
            if not (0.0 <= float(halo.ellipticite) < 1.0):
                raise ValueError(f"NFW halo {halo.id!r} must have ellipticite in [0, 1).")
        elif halo.core_radius_kpc <= 0.0 or halo.cut_radius_kpc <= 0.0 or halo.v_disp <= 0.0:
            raise ValueError(f"large halo {halo.id!r} must have positive core, cut, and velocity dispersion.")
        if halo.z_lens <= 0.0:
            raise ValueError(f"large halo {halo.id!r} must have positive z_lens.")
    populations: dict[str, MemberPopulationConfig] = {}
    for population in model.member_populations:
        if isinstance(population.mag0, str):
            if population.mag0 != "auto":
                raise ValueError(f"member population {population.id!r}: mag0 must be a number or 'auto'.")
            if population.prior_mag0 is None:
                raise ValueError(f"member population {population.id!r}: mag0='auto' needs prior_mag0, the magnitude the sigma and cutkpc priors refer to.")
        if population.id in populations:
            raise ValueError(f"duplicate member population id {population.id!r}.")
        populations[population.id] = population
        if not Path(population.catalog_path).is_file():
            raise ValueError(f"member population catalog does not exist: {population.catalog_path}")
        if population.corekpc <= 0.0 or population.sigma <= 0.0 or population.cutkpc <= 0.0 or population.z_lens <= 0.0:
            raise ValueError(f"member population {population.id!r} must have positive corekpc, sigma, cutkpc, and z_lens.")
        exclude_ids = tuple(str(value).strip() for value in population.exclude_catalog_ids)
        if any(not value for value in exclude_ids) or len(set(exclude_ids)) != len(exclude_ids):
            raise ValueError(f"member population {population.id!r} has empty or duplicate exclude_catalog_ids entries.")
        if population.kinematics_sources:
            if Path(population.catalog_path).suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(f"member population {population.id!r}: kinematics_sources need a YAML member catalog.")
            if len(set(population.kinematics_sources)) != len(population.kinematics_sources):
                raise ValueError(f"member population {population.id!r} has duplicate kinematics_sources.")
        if not (population.kinematics_systematic >= 0.0 and population.kinematics_aperture_factor > 0.0):
            raise ValueError(f"member population {population.id!r}: kinematics_systematic must be >= 0 and kinematics_aperture_factor > 0.")
    keys: set[tuple[str, str]] = set()
    for independent in model.independent_member_halos:
        key = (str(independent.population_id).strip(), str(independent.catalog_id).strip())
        if not key[0] or not key[1] or key[0] not in populations:
            raise ValueError(f"independent member halo {independent!r} references an unknown or empty population/catalog id.")
        if key in keys:
            raise ValueError(f"duplicate independent member halo {key!r}.")
        keys.add(key)
    constraints = model.image_constraints
    if not Path(constraints.catalog_path).is_file():
        raise ValueError(f"image constraint catalog does not exist: {constraints.catalog_path}")
    if not 0.0 <= float(constraints.trusted_prior_min) <= 1.0:
        raise ValueError("image_constraints.trusted_prior_min must be in [0, 1].")
