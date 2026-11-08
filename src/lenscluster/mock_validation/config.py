"""Configuration of a mock-validation campaign and the solver model built from each realized truth.

A campaign is one mock generator configuration, one solver template and a number of realizations.  The mock is an
analytic cluster: a cluster-scale dPIE halo, a BCG dPIE, and a population of member dPIE subhalos on a luminosity
scaling relation; its sources are lensed exactly and the images carry astrometric noise.  The solver model of each
realization is built from that realization's truth (never from the nominal configuration) in one of two modes:

* ``production``: the priors an observer would use, i.e. the same factories as the real clusters, with the halo centred
  on the BCG light and the member population on the Bergamini et al. 2019 relation at the mock's pivot magnitude;
* ``idealized``: priors centred on the truth with moderate widths, which isolates the sampler and the likelihood.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpyro.distributions as dist

from ..priors import cut_excess_from_moments, ellipticity_box, lognormal_from_moments, physical_bounds
from ..config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                      MemberPopulationConfig, ReferenceFrameConfig, RunPathsConfig, ShearHaloConfig)

VALID_MODES = ("production", "idealized")


@dataclass(frozen=True)
class MockConfig:
    """How the analytic mock clusters are drawn, lensed and observed.

    Each realization draws its own truth from the ranges below (``draw_truth``), lenses its sources exactly through that
    truth (``generate_images``), and adds observational noise.  Unless stated otherwise a ``(min, max)`` range is sampled
    uniformly once per realization.  Angular positions are offsets in arcsec from the reference point
    (``reference_ra_deg``, ``reference_dec_deg``); ellipticities follow the Lenstool convention
    :math:`(a^2 - b^2) / (a^2 + b^2)` (dimensionless) and position angles are drawn uniformly in (-90, 90) degrees.
    Masses follow the dPIE convention :math:`M = 1.5 \\pi \\sigma^2 r_\\mathrm{cut} / G`, from which each component's
    velocity dispersion is set given its mass and cut radius.

    Parameters
    ----------
    seed : int, optional
        Default random seed of ``draw_truth`` when it is called without one.  A campaign ignores it: realization ``i``
        uses ``runtime.seed + i`` (``MockValidationRuntimeConfig``), and that seed is what ``truth.json`` records.
        Default ``12345``.
    reference_ra_deg : float, optional
        Right ascension of the reference point in degrees.  Image offsets are converted to RA/Dec about it in the image
        catalog, and the solver's reference frame uses it.  Default ``64.0381417``.
    reference_dec_deg : float, optional
        Declination of the reference point in degrees.  Default ``-24.0674722``.
    cosmology : CosmologyConfig, optional
        Cosmology of the truth and of the solver model.  The truth uses a flat LambdaCDM with its ``H0``
        (km s^-1 Mpc^-1), ``Om0`` and ``Ode0`` (dimensionless) and :math:`w_0 = -1` for the kpc-per-arcsec scale and the
        lensing efficiencies.  Default ``CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7)``.
    z_lens_range : tuple of float, optional
        (min, max) cluster redshift (dimensionless), drawn uniformly.  Default ``(0.30, 0.50)``.
    total_mass_range_msun : tuple of float, optional
        (min, max) total dPIE mass of halo plus BCG in Msun, drawn log-uniformly and then split by the BCG mass fraction.
        Default ``(5.0e14, 2.0e15)``.
    mass_pivot_msun : float, optional
        Halo mass in Msun at which the halo radii equal ``halo_core_pivot_kpc`` and ``halo_cut_pivot_kpc``.
        Default ``7.6e14``.
    radius_mass_slope : float, optional
        Slope (dimensionless) of the mass-size relations of both halo and BCG:
        :math:`r = r_\\mathrm{pivot} (M / M_\\mathrm{pivot})^\\mathrm{slope}` for core and cut radii.
        Default ``1/3``.
    halo_core_pivot_kpc : float, optional
        Halo core radius in kpc at ``mass_pivot_msun``.  Default ``27.0``.
    halo_cut_pivot_kpc : float, optional
        Halo cut radius in kpc at ``mass_pivot_msun``.  Default ``1175.0``.
    halo_radius_scatter_dex : float, optional
        Log-normal scatter in dex of each halo radius (core and cut drawn independently) about the mass-size relation;
        ``0`` for none.  Default ``0.08``.
    halo_ellipticity_range : tuple of float, optional
        (min, max) halo ellipticity (dimensionless, Lenstool convention).  Default ``(0.10, 0.40)``.
    halo_position_half_width_arcsec : float, optional
        Half width in arcsec of the box about the reference point in which the halo centre is drawn uniformly (each axis).
        Default ``10.0``.
    bcg_mass_fraction_median : float, optional
        Median fraction (dimensionless) of the total mass in the BCG.  Default ``0.02``.
    bcg_mass_fraction_scatter_dex : float, optional
        Log-normal scatter in dex of the BCG mass fraction about its median.  Default ``0.25``.
    bcg_mass_fraction_range : tuple of float, optional
        (min, max) dimensionless bounds the drawn BCG mass fraction is clipped to.  Default ``(0.005, 0.08)``.
    bcg_mass_pivot_msun : float, optional
        BCG mass in Msun at which the BCG radii equal ``bcg_core_pivot_kpc`` and ``bcg_cut_pivot_kpc``.
        Default ``1.7e13``.
    bcg_core_pivot_kpc : float, optional
        BCG core radius in kpc at ``bcg_mass_pivot_msun``.  Default ``0.8``.
    bcg_cut_pivot_kpc : float, optional
        BCG cut radius in kpc at ``bcg_mass_pivot_msun``.  Default ``187.0``.
    bcg_radius_scatter_dex : float, optional
        Log-normal scatter in dex of each BCG radius about its mass-size relation.  Default ``0.10``.
    bcg_ellipticity_range : tuple of float, optional
        (min, max) BCG ellipticity (dimensionless, Lenstool convention).  Default ``(0.0, 0.30)``.
    bcg_offset_arcsec : float, optional
        Mean in arcsec of the BCG's offset from the halo centre.  The offset distance is drawn from a normal with this
        mean and width ``bcg_offset_width_arcsec``, truncated to 0 - ``bcg_offset_max_arcsec``, in a uniform direction
        (1 arcsec is 4.5-6 kpc at z 0.3-0.5).  Must be nonnegative and at most ``bcg_offset_max_arcsec``.
        Default ``1.0``.
    bcg_offset_width_arcsec : float, optional
        Width (standard deviation) in arcsec of the BCG offset distance before truncation.  Default ``0.5``.
    bcg_offset_max_arcsec : float, optional
        Largest BCG offset from the halo centre in arcsec.  Default ``2.0``.
    n_members : int, optional
        Number of member subhalos in the member catalog the solver reads (the modelled members).  Members are drawn from
        a Schechter luminosity function and placed on a luminosity scaling relation.  Default ``50``.
    schechter_alpha : float, optional
        Faint-end slope (dimensionless) of the Schechter luminosity function of the members; must exceed -1.
        Default ``-0.7``.
    pivot_magnitude : float, optional
        Magnitude m* (mag, member band) of an L* member: member magnitudes are
        :math:`m^* - 2.5 \\log_{10}(L / L^*)`, the scaling relation is quoted here, and the solver's member population
        uses it as its prior pivot magnitude.  Default ``19.82``.
    faint_magnitude_limit : float, optional
        Faintest member magnitude in mag: drawn members fainter than this are discarded and redrawn.  Default ``24.0``.
    luminosity_range : tuple of float, optional
        (min, max) :math:`L / L^*` (dimensionless) the Schechter draw is truncated to.  Default ``(0.02, 20.0)``.
    sigma_star_range_kms : tuple of float, optional
        (min, max) velocity dispersion of an L* member in km s^-1.  A member's dispersion is
        :math:`\\sigma^* (L / L^*)^{\\alpha}` times its scatter.  Default ``(160.0, 240.0)``.
    cut_star_range_kpc : tuple of float, optional
        (min, max) cut radius of an L* member in kpc; a member's cut radius is :math:`r^*_\\mathrm{cut} (L / L^*)^{\\beta}`
        times its scatter.  Default ``(8.0, 20.0)``.
    core_star_kpc : float, optional
        Core radius of an L* member in kpc (fixed, not drawn); a member's core radius is
        :math:`r^*_\\mathrm{core} (L / L^*)^{\\beta}` without scatter.  Default ``0.15``.
    alpha_sigma_range : tuple of float, optional
        (min, max) dimensionless slope :math:`\\alpha` of the dispersion-luminosity relation.  Default ``(0.22, 0.32)``.
    beta_radius_range : tuple of float, optional
        (min, max) dimensionless slope :math:`\\beta` of the radius-luminosity relation (cut and core radii).
        Default ``(0.50, 0.80)``.
    sigma_scatter_ln : float, optional
        Intrinsic scatter (standard deviation of :math:`\\ln \\sigma`, dimensionless) of member dispersions about the
        relation; also the truth of the solver's ``potfile.velocity_log_scatter``.  Default ``0.10``.
    cut_scatter_ln : float, optional
        Intrinsic scatter (standard deviation of :math:`\\ln r_\\mathrm{cut}`, dimensionless) of member cut radii about
        the relation; also the truth of ``potfile.radius_log_scatter``.  Default ``0.0``.
    member_ellipticity_range : tuple of float, optional
        (min, max) member ellipticity (dimensionless, Lenstool convention), drawn per member.  Default ``(0.0, 0.5)``.
    member_spatial_core_arcsec : float, optional
        Core radius in arcsec of the members' radial distribution about the halo centre: radii follow
        :math:`r \\Sigma(r)` of a projected dPIE profile with this core and ``member_spatial_cut_arcsec``, in uniform
        directions.  Default ``25.0``.
    member_spatial_cut_arcsec : float, optional
        Cut radius in arcsec of the members' radial distribution.  Default ``180.0``.
    member_field_radius_arcsec : float, optional
        Largest member distance from the halo centre in arcsec (the radial distribution is truncated there).
        Default ``250.0``.
    unmodelled_members : int, optional
        Model misspecification: number of additional subhalos present in the truth (drawn like the others, chosen at
        random among all members) but missing from the member catalog.  ``0`` for none.  Default ``0``.
    n_primary_families : int, optional
        Number of planned source families on the primary caustic (that of the largest tangential critical curve), at
        source redshifts evenly spaced over ``source_redshift_range``; at least 1.  A family is dropped (and recorded in
        ``truth.json``) if no source with enough images is found.  Default ``15``.
    n_member_families : int, optional
        Number of planned families on secondary caustics (tangential caustics other than the primary one), sources
        drawn uniformly inside a caustic chosen with probability proportional to its area, at redshifts evenly spaced
        over ``source_redshift_range``.  Requires ``n_members > 0``.  Default ``5``.
    n_central_families : int, optional
        Number of planned families whose source lies behind the BCG: the source is the back-projection of a point drawn
        uniformly within ``central_radius_arcsec`` of the BCG, and it is accepted only if one of its images falls within
        that radius (a radial arc or central image).  Redshifts are evenly spaced strictly inside
        ``source_redshift_range``.  Default ``1``.
    central_radius_arcsec : float, optional
        Radius in arcsec about the BCG used by the central families.  Default ``3.0``.
    source_placement : str, optional
        How primary-family sources are placed.  ``"image_plane"``: the source is the back-projection of a point drawn
        uniformly inside the tangential critical curve, so the images spread over all radii.  ``"caustic"``: the source
        is drawn uniformly inside the primary caustic.  Secondary families always use the caustic.
        Default ``"image_plane"``.
    source_redshift_range : tuple of float, optional
        (min, max) source redshift (dimensionless).  Family redshifts are evenly spaced over it, not drawn at random.
        Default ``(1.5, 9.0)``.
    n_knotted_families : int, optional
        Number of primary families (spread evenly over them; at most ``n_primary_families``) that are galaxies with
        knots.  Each knot is its own family of one system in the catalog (system 3, knots A, B -> families 3A, 3B), so
        every image becomes a tight group of images whose separations scale with the magnification.  Each knot adds as
        many images as its family.  Default ``0``.
    knots_per_family : tuple of int, optional
        (min, max) number of knots per knotted galaxy, the main source included, drawn as an integer uniformly in this
        inclusive range; within 2-26.  A knot whose image count differs from the main source's (it crossed a caustic) or
        whose images do not converge is redrawn, and dropped after 50 attempts.  Default ``(2, 3)``.
    knot_offset_range_arcsec : tuple of float, optional
        (min, max) source-plane distance in arcsec of a knot from the main source, drawn log-uniformly, in a uniform
        direction.  Default ``(0.05, 0.4)``.
    knot_magnitude_offset_range : tuple of float, optional
        (min, max) in mag by which a knot is fainter than the main source, drawn uniformly per knot.
        Default ``(0.5, 1.5)``.
    min_images_per_family : int, optional
        Smallest number of images a source must have to be accepted; at least 2.  Default ``3``.
    max_source_attempts : int, optional
        Number of source draws tried per family before the family is dropped.  Default ``400``.
    sigma_inst_arcsec : float, optional
        Standard deviation in arcsec, per axis, of the Gaussian astrometric noise added to every image position in the
        image plane; the truth of the solver's white scatter ``sigma_white`` (``sigma_extra`` under the smoothed
        likelihood).  Written as the images' ``sigma_inst_arcsec``.  Ignored when ``sigma_inst_range_arcsec`` is set.
        Default ``0.10``.
    solver_sigma_inst_arcsec : float, optional
        The solver's instrumental astrometric floor in arcsec (image constraints and likelihood), 1 mas by default so
        the white scatter is inferred, not assumed.  Default ``0.001``.
    sigma_inst_range_arcsec : tuple of float or None, optional
        (min, max) in arcsec from which each realization draws its injected astrometric noise log-uniformly (from its
        own random stream, leaving the other draws unchanged); the drawn value replaces ``sigma_inst_arcsec`` in that
        realization's ``truth.json`` and is the truth of ``sigma_white``.  None keeps ``sigma_inst_arcsec``.
        Default ``None``.
    sigma_def_arcsec : float, optional
        Injected source-plane scatter in arcsec, per axis: each image is re-solved as the exact image of its own source
        position displaced by a Gaussian of this width (an image that no longer exists keeps its true position), before
        the astrometric noise.  The truth of ``images.sigma_def``.  ``0`` for none.  Default ``0.0``.
    field_sigma_alpha_range_arcsec : tuple of float or None, optional
        Correlated error from unmodelled mass: (min, max) in arcsec of the rms deflection per component of an injected
        mass field for a source at ``field_reference_redshift``, drawn log-uniformly per realization (from its own random
        stream).  The field is a sum of ``field_n_modes`` plane waves with the solver's knee convergence spectrum
        :math:`P_\\kappa(k) = 1 / (1 + (k / k_t)^\\gamma)`; other source redshifts see it scaled by
        :math:`D_{ls} / D_s`.  None injects no field (and the ``field_*`` options below are unused).  Default ``None``.
    field_knee_range_arcsec : tuple of float, optional
        (min, max) knee wavelength :math:`2 \\pi / k_t` of the mass field in arcsec, drawn log-uniformly.
        Default ``(3.0, 12.0)``.
    field_gamma_range : tuple of float, optional
        (min, max) dimensionless slope :math:`\\gamma` of the mass-field spectrum beyond the knee, drawn uniformly.
        Default ``(3.0, 5.0)``.
    field_scales_arcsec : tuple of float, optional
        (min, max) wavelength in arcsec of the mass field's modes (the band of the solver's field), i.e. wavenumbers
        between :math:`2 \\pi / \\max` and :math:`2 \\pi / \\min`.  Default ``(0.2, 100.0)``.
    field_n_modes : int, optional
        Number of plane waves in the mass field; at least 100.  Default ``4000``.
    field_reference_redshift : float, optional
        Source redshift (dimensionless) at which the mass field's rms deflection is ``sigma_alpha``.  Default ``2.0``.
    image_plane_sigma_range_arcsec : tuple of float or None, optional
        Correlated image-plane error: (min, max) in arcsec of the amplitude :math:`\\sigma` of a displacement drawn
        jointly over all images, per axis, with covariance :math:`\\sigma^2 \\exp(-r^2 / 2 \\ell^2)` (the solver's
        Gaussian image-plane kernel) at the true positions and added to the observed ones; drawn log-uniformly per
        realization.  The truth of ``images.sigma_shared``.  None injects no displacement.  Default ``None``.
    image_plane_ell_range_arcsec : tuple of float, optional
        (min, max) correlation length :math:`\\ell` in arcsec of the image-plane displacement, drawn log-uniformly.
        Default ``(0.5, 2.0)``.
    source_magnitude_range : tuple of float, optional
        (min, max) unlensed source magnitude in mag, drawn uniformly per family.  An image's observed magnitude is the
        source magnitude minus :math:`2.5 \\log_{10}(|\\mu| + 10^{-3})` at its observed position plus noise.
        Default ``(24.0, 27.0)``.
    image_magnitude_sigma : float, optional
        Gaussian noise in mag added to every image magnitude, and the magnitude error written to the catalog.
        Default ``0.10``.
    caustic_window_arcsec : float, optional
        Full width in arcsec of the square grid, centred on the halo, on which the convergence and shear of the truth
        are evaluated to trace the tangential critical curves; curves touching its edge are discarded.
        Default ``160.0``.
    caustic_grid_arcsec : float, optional
        Pixel size in arcsec of the caustic grid.  Default ``0.2``.
    caustic_min_area_arcsec2 : float, optional
        Smallest source-plane area in arcsec^2 of a caustic that is kept.  Default ``1.0e-5``.
    seed_grid_pixel_arcsec : float, optional
        Pixel size in arcsec of the image-plane grid, ray-shot once per source redshift, whose local minima of the
        source-plane distance seed the image search.  Default ``1.0``.
    search_window_arcsec : float, optional
        Smallest full width in arcsec of that seed grid (centred on the halo); it is enlarged to
        2 (1.6 r + 10) arcsec, where r is the largest critical-curve distance from the halo centre, when that is wider.
        Default ``80.0``.
    image_refine_iterations : int, optional
        Number of Newton iterations of the lens equation used to refine each image.  Default ``100``.
    image_precision_arcsec : float, optional
        Source-plane residual in arcsec below which a refined image counts as converged.  Default ``1.0e-8``.
    mode : str, optional
        How the solver model of each realization is built from its truth.  ``"production"``: the priors an observer
        would use (the real-cluster factories), with the cluster halo centred on the BCG light and the member population
        on the Bergamini et al. 2019 relation at ``pivot_magnitude``.  ``"idealized"``: priors centred on the truth with
        moderate widths (including the scaling-relation slopes), which isolates the sampler and the likelihood.
        Default ``"production"``.
    """
    seed: int = 12345
    reference_ra_deg: float = 64.0381417
    reference_dec_deg: float = -24.0674722
    cosmology: CosmologyConfig = field(default_factory=lambda: CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7))
    z_lens_range: tuple[float, float] = (0.30, 0.50)
    # Cluster-scale halo and BCG.
    total_mass_range_msun: tuple[float, float] = (5.0e14, 2.0e15)
    mass_pivot_msun: float = 7.6e14
    radius_mass_slope: float = 1.0 / 3.0
    halo_core_pivot_kpc: float = 27.0
    halo_cut_pivot_kpc: float = 1175.0
    halo_radius_scatter_dex: float = 0.08
    halo_ellipticity_range: tuple[float, float] = (0.10, 0.40)
    halo_position_half_width_arcsec: float = 10.0
    bcg_mass_fraction_median: float = 0.02
    bcg_mass_fraction_scatter_dex: float = 0.25
    bcg_mass_fraction_range: tuple[float, float] = (0.005, 0.08)
    bcg_mass_pivot_msun: float = 1.7e13
    bcg_core_pivot_kpc: float = 0.8
    bcg_cut_pivot_kpc: float = 187.0
    bcg_radius_scatter_dex: float = 0.10
    bcg_ellipticity_range: tuple[float, float] = (0.0, 0.30)
    bcg_offset_arcsec: float = 1.0
    bcg_offset_width_arcsec: float = 0.5
    bcg_offset_max_arcsec: float = 2.0
    # Member population.
    n_members: int = 50
    schechter_alpha: float = -0.7
    pivot_magnitude: float = 19.82
    faint_magnitude_limit: float = 24.0
    luminosity_range: tuple[float, float] = (0.02, 20.0)
    sigma_star_range_kms: tuple[float, float] = (160.0, 240.0)
    cut_star_range_kpc: tuple[float, float] = (8.0, 20.0)
    core_star_kpc: float = 0.15
    alpha_sigma_range: tuple[float, float] = (0.22, 0.32)
    beta_radius_range: tuple[float, float] = (0.50, 0.80)
    sigma_scatter_ln: float = 0.10
    cut_scatter_ln: float = 0.0
    member_ellipticity_range: tuple[float, float] = (0.0, 0.5)
    member_spatial_core_arcsec: float = 25.0
    member_spatial_cut_arcsec: float = 180.0
    member_field_radius_arcsec: float = 250.0
    # Model misspecification.
    unmodelled_members: int = 0
    # Sources and images.
    n_primary_families: int = 15
    n_member_families: int = 5
    n_central_families: int = 1
    central_radius_arcsec: float = 3.0
    source_placement: str = "image_plane"
    source_redshift_range: tuple[float, float] = (1.5, 9.0)
    # Sources with knots.  The knot separations set the scales of the image-plane GP and the mass field's knee.
    n_knotted_families: int = 0
    knots_per_family: tuple[int, int] = (2, 3)
    knot_offset_range_arcsec: tuple[float, float] = (0.05, 0.4)
    knot_magnitude_offset_range: tuple[float, float] = (0.5, 1.5)
    min_images_per_family: int = 3
    max_source_attempts: int = 400
    # Noise.  solver_sigma_inst_arcsec is 1 mas as in run.xsh.
    sigma_inst_arcsec: float = 0.10
    solver_sigma_inst_arcsec: float = 0.001
    sigma_inst_range_arcsec: tuple[float, float] | None = None
    sigma_def_arcsec: float = 0.0
    # Correlated errors (mock_validation.field), drawn per realization; None leaves each out.
    field_sigma_alpha_range_arcsec: tuple[float, float] | None = None
    field_knee_range_arcsec: tuple[float, float] = (3.0, 12.0)
    field_gamma_range: tuple[float, float] = (3.0, 5.0)
    field_scales_arcsec: tuple[float, float] = (0.2, 100.0)
    field_n_modes: int = 4000
    field_reference_redshift: float = 2.0
    image_plane_sigma_range_arcsec: tuple[float, float] | None = None
    image_plane_ell_range_arcsec: tuple[float, float] = (0.5, 2.0)
    source_magnitude_range: tuple[float, float] = (24.0, 27.0)
    image_magnitude_sigma: float = 0.10
    # Image finding on the exact truth model.
    caustic_window_arcsec: float = 160.0
    caustic_grid_arcsec: float = 0.2
    caustic_min_area_arcsec2: float = 1.0e-5
    seed_grid_pixel_arcsec: float = 1.0
    search_window_arcsec: float = 80.0
    image_refine_iterations: int = 100
    image_precision_arcsec: float = 1.0e-8
    # Solver model built from the truth.
    mode: str = "production"

    def with_updates(self, **updates: Any) -> "MockConfig":
        """Return a copy with the given fields replaced.

        Parameters
        ----------
        **updates
            Field names and their new values.

        Returns
        -------
        MockConfig
            The updated copy (not validated).
        """
        return replace(self, **updates)

    def validate(self) -> "MockConfig":
        """Check the configuration and return it unchanged.

        Checks that ``mode`` and ``source_placement`` are allowed values, that the ``(min, max)`` ranges are finite and
        ordered (and positive where the quantity is drawn log-uniformly or is optional), that counts are consistent
        (at least one primary family, ``n_knotted_families <= n_primary_families``, ``knots_per_family`` within 2-26,
        members present when ``n_member_families > 0``, ``min_images_per_family >= 2``, ``field_n_modes >= 100``), that
        noise levels and grid sizes are positive, and that the BCG offset settings are consistent.

        Returns
        -------
        MockConfig
            ``self``.

        Raises
        ------
        ValueError
            If any check fails.
        """
        if self.mode not in VALID_MODES:
            raise ValueError(f"mock.mode must be one of {VALID_MODES}.")
        for name in ("z_lens_range", "total_mass_range_msun", "halo_ellipticity_range", "bcg_mass_fraction_range", "bcg_ellipticity_range",
                     "luminosity_range", "sigma_star_range_kms", "cut_star_range_kpc", "alpha_sigma_range", "beta_radius_range",
                     "member_ellipticity_range", "source_redshift_range", "source_magnitude_range", "knot_magnitude_offset_range"):
            lo, hi = getattr(self, name)
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError(f"mock.{name} must be an ordered finite range.")
        if self.n_members < 0 or self.unmodelled_members < 0 or self.n_primary_families < 1 or self.n_member_families < 0:
            raise ValueError("mock member and family counts must be nonnegative (at least one primary family).")
        if self.source_placement not in {"image_plane", "caustic"}:
            raise ValueError("source_placement must be 'image_plane' or 'caustic'")
        if self.n_central_families < 0 or self.central_radius_arcsec <= 0:
            raise ValueError("n_central_families must be >= 0 and central_radius_arcsec > 0")
        if self.sigma_inst_range_arcsec is not None:
            lo, hi = self.sigma_inst_range_arcsec
            if not (0.0 < lo <= hi and math.isfinite(hi)):
                raise ValueError("mock.sigma_inst_range_arcsec must be an ordered positive range.")
        if not 0 <= self.n_knotted_families <= self.n_primary_families:
            raise ValueError("mock.n_knotted_families must lie between 0 and n_primary_families.")
        lo, hi = self.knots_per_family
        if not (2 <= int(lo) <= int(hi) <= 26):
            raise ValueError("mock.knots_per_family must be an ordered range within 2-26.")
        lo, hi = self.knot_offset_range_arcsec
        if not (0.0 < lo <= hi and math.isfinite(hi)):
            raise ValueError("mock.knot_offset_range_arcsec must be an ordered positive range.")
        if self.n_member_families > 0 and self.n_members == 0:
            raise ValueError("mock.n_member_families needs members.")
        for name in ("sigma_inst_arcsec", "solver_sigma_inst_arcsec", "caustic_window_arcsec", "caustic_grid_arcsec", "seed_grid_pixel_arcsec", "search_window_arcsec", "image_precision_arcsec"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0.0):
                raise ValueError(f"mock.{name} must be positive.")
        for name in ("sigma_def_arcsec", "sigma_scatter_ln", "cut_scatter_ln", "bcg_offset_arcsec"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) >= 0.0):
                raise ValueError(f"mock.{name} must be nonnegative.")
        if not (self.bcg_offset_width_arcsec > 0.0 and 0.0 < self.bcg_offset_max_arcsec and self.bcg_offset_arcsec <= self.bcg_offset_max_arcsec):
            raise ValueError("mock.bcg_offset_width_arcsec must be positive and bcg_offset_arcsec at most bcg_offset_max_arcsec.")
        for name in ("field_sigma_alpha_range_arcsec", "field_knee_range_arcsec", "field_gamma_range", "field_scales_arcsec",
                     "image_plane_sigma_range_arcsec", "image_plane_ell_range_arcsec"):
            bounds = getattr(self, name)
            if bounds is not None and not (len(bounds) == 2 and 0.0 < float(bounds[0]) <= float(bounds[1]) and math.isfinite(float(bounds[1]))):
                raise ValueError(f"mock.{name} must be None or a positive ordered (min, max).")
        if self.field_n_modes < 100 or not self.field_reference_redshift > 0.0:
            raise ValueError("mock.field_n_modes must be at least 100 and field_reference_redshift positive.")
        if self.min_images_per_family < 2:
            raise ValueError("mock.min_images_per_family must be at least 2.")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serializable dict (nested configs as dicts, tuples as lists, paths as
        strings, infinities as ``"inf"``/``"-inf"``).  This is the ``config`` entry of each realization's
        ``truth.json``.

        Returns
        -------
        dict
            The field values keyed by field name.
        """
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class MockValidationPathsConfig:
    """Where a campaign writes its files.

    The campaign root is ``<output_dir>/<campaign_name>/<run_name>``; every realization lives in
    ``seed_<seed>/`` under it (the seed zero-padded to eight digits).

    Parameters
    ----------
    output_dir : str or Path, optional
        Top-level output directory.  Default ``"validation_runs"``.
    campaign_name : str, optional
        Name of the campaign directory under ``output_dir``.  Default ``"mock_validation"``.
    run_name : str, optional
        Name of the run directory under the campaign directory.  Default ``"analytic"``.
    """
    output_dir: str | Path = "validation_runs"
    campaign_name: str = "mock_validation"
    run_name: str = "analytic"


@dataclass(frozen=True)
class MockValidationRuntimeConfig:
    """How many realizations to run and how.

    Parameters
    ----------
    realizations : int, optional
        Number of realizations; must be positive.  Realization ``i`` (0-based) uses seed ``seed + i``.  Default ``1``.
    seed : int, optional
        Seed of the first realization; must be nonnegative.  It seeds the truth, the image draws and the solver of that
        realization.  Default ``12345``.
    resume : bool, optional
        If True, skip realizations that already have ``recovery/recovery.json``, reuse a complete mock instead of
        regenerating it, and let the solver resume from its saved state.  If False, regenerate the mocks and refit.
        Default ``True``.
    quiet : bool, optional
        Passed to the solver's runtime ``quiet`` setting of every realization.  Default ``False``.
    """
    realizations: int = 1
    seed: int = 12345
    resume: bool = True
    quiet: bool = False


@dataclass(frozen=True)
class MockValidationConfig:
    """A mock-validation campaign: the mock generator, the solver template, where to write and how many realizations.

    Parameters
    ----------
    mock : MockConfig, optional
        How each mock cluster is drawn, lensed and observed, and which solver model (``mock.mode``) is built from its
        truth.  Default ``MockConfig()``.
    paths : MockValidationPathsConfig, optional
        Output location of the campaign.  Default ``MockValidationPathsConfig()``.
    runtime : MockValidationRuntimeConfig, optional
        Number of realizations, seeds, resuming and verbosity.  Default ``MockValidationRuntimeConfig()``.
    solver : LensClusterSolverConfig, optional
        Template of every realization's solver configuration (likelihood, sampler, scaling relation, ...).  Per
        realization its lens model is replaced by the one built from the truth, its likelihood's ``sigma_inst_arcsec``
        by ``mock.solver_sigma_inst_arcsec``, its paths by ``seed_<seed>/solver/``, and its runtime ``seed``,
        ``resume`` and ``quiet`` by the realization's; in ``"idealized"`` mode its scaling-relation slope priors are
        recentred on the truth.  Default ``LensClusterSolverConfig()``.
    """
    mock: MockConfig = field(default_factory=MockConfig)
    paths: MockValidationPathsConfig = field(default_factory=MockValidationPathsConfig)
    runtime: MockValidationRuntimeConfig = field(default_factory=MockValidationRuntimeConfig)
    solver: LensClusterSolverConfig = field(default_factory=LensClusterSolverConfig)

    def with_updates(self, **updates: Any) -> "MockValidationConfig":
        """Return a copy with the given fields (``mock``, ``paths``, ``runtime``, ``solver``) replaced.

        Parameters
        ----------
        **updates
            Field names and their new values.

        Returns
        -------
        MockValidationConfig
            The updated copy (not validated).
        """
        return replace(self, **updates)

    def validate(self) -> "MockValidationConfig":
        """Check the runtime settings and the mock configuration (``MockConfig.validate``); return ``self``.

        Returns
        -------
        MockValidationConfig
            ``self``.

        Raises
        ------
        ValueError
            If ``runtime.realizations`` is not positive, ``runtime.seed`` is negative, or the mock configuration is
            invalid.
        """
        if int(self.runtime.realizations) <= 0:
            raise ValueError("runtime.realizations must be positive.")
        if int(self.runtime.seed) < 0:
            raise ValueError("runtime.seed must be nonnegative.")
        self.mock.validate()
        return self

    def to_json_dict(self) -> dict[str, Any]:
        """Return the campaign configuration as a JSON-serializable dict with keys ``mock``, ``paths``, ``runtime`` and
        ``solver``; written to ``campaign_config.json`` in the campaign root.

        Returns
        -------
        dict
            The nested configuration.
        """
        return _jsonable({"mock": asdict(self.mock), "paths": asdict(self.paths), "runtime": asdict(self.runtime),
                          "solver": self.solver.to_json_dict()})


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ----------------------------------------------------------------------------- solver model from the realized truth
def _bcg_halo(id: str, x: float, y: float, z_lens: float, *, v_disp: float, cut_kpc: float, half_width: float, truth: dict[str, float] | None) -> DPIEHaloConfig:
    """A BCG-scale dPIE: production priors around the light (``truth`` None) or idealized priors around the truth."""
    if truth is None:
        core = 1.0
        priors = {"x_centre": dist.Uniform(x - half_width, x + half_width), "y_centre": dist.Uniform(y - half_width, y + half_width),
                  **ellipticity_box(0.6), "core_radius_kpc": dist.LogUniform(0.1, 5.0),
                  "cut_excess_kpc": cut_excess_from_moments(cut_kpc, 0.5 * cut_kpc, low=20.0, high=600.0, core=core),
                  "v_disp": lognormal_from_moments(v_disp, 0.35 * v_disp, low=100.0, high=800.0)}
        return DPIEHaloConfig(id=id, x_centre=x, y_centre=y, ellipticite=0.2, angle_pos=0.0, core_radius_kpc=core, cut_radius_kpc=cut_kpc,
                              v_disp=v_disp, z_lens=z_lens, priors=priors)
    return _idealized_halo(id, truth, z_lens, half_width=half_width, core_bounds=(0.1, 5.0), cut_bounds=(20.0, 600.0), v_bounds=(100.0, 800.0))


def _idealized_halo(id: str, t: dict[str, float], z_lens: float, *, half_width: float, core_bounds, cut_bounds, v_bounds) -> DPIEHaloConfig:
    """A dPIE with priors centred on its truth: position box, free shape, 30% radii, 15% dispersion."""
    core, cut, v = float(t["core_radius_kpc"]), float(t["cut_radius_kpc"]), float(t["v_disp"])
    priors = {"x_centre": dist.Uniform(t["x_centre"] - half_width, t["x_centre"] + half_width),
              "y_centre": dist.Uniform(t["y_centre"] - half_width, t["y_centre"] + half_width),
              **ellipticity_box(0.8),
              "core_radius_kpc": lognormal_from_moments(core, 0.3 * core, low=core_bounds[0], high=core_bounds[1]),
              "cut_excess_kpc": cut_excess_from_moments(cut, 0.3 * cut, low=cut_bounds[0], high=cut_bounds[1], core=core),
              "v_disp": lognormal_from_moments(v, 0.15 * v, low=v_bounds[0], high=v_bounds[1])}
    return DPIEHaloConfig(id=id, x_centre=t["x_centre"], y_centre=t["y_centre"], ellipticite=t["ellipticite"], angle_pos=t["angle_pos"],
                          core_radius_kpc=t["core_radius_kpc"], cut_radius_kpc=t["cut_radius_kpc"], v_disp=t["v_disp"], z_lens=z_lens, priors=priors)


def mock_lens_model_config(config: MockConfig, truth: dict[str, Any], *, image_catalog: Path, member_catalog: Path | None) -> LensModelConfig:
    """The solver's lens model for one realization, built from its truth according to ``config.mode``."""
    halo, bcg, relation = truth["halo"], truth["bcg"], truth["relation"]
    z_lens = float(truth["z_lens"])
    has_members = member_catalog is not None and int(truth.get("n_modelled_members", 0)) > 0
    if config.mode == "idealized":
        halos = (_idealized_halo("halo", halo, z_lens, half_width=2.0, core_bounds=(1.0, 200.0), cut_bounds=(300.0, 4000.0), v_bounds=(100.0, 2200.0)),
                 _idealized_halo("bcg", bcg, z_lens, half_width=1.0, core_bounds=(0.1, 5.0), cut_bounds=(20.0, 600.0), v_bounds=(100.0, 800.0)),
                 ShearHaloConfig.free("shear", z_lens, gamma_upper=0.1))
        members = ()
        if has_members:
            sigma, cut = float(relation["sigma_star_kms"]), float(relation["cut_star_kpc"])
            members = (MemberPopulationConfig(
                id="potfile", catalog_path=member_catalog, mag0="auto", prior_mag0=float(relation["pivot_magnitude"]),
                corekpc=float(relation["core_star_kpc"]), sigma=sigma, cutkpc=cut, z_lens=z_lens,
                sigma_prior=lognormal_from_moments(sigma, 0.15 * sigma, low=40.0, high=500.0),
                cutkpc_prior=lognormal_from_moments(cut, 0.3 * cut, low=2.0, high=60.0)),)
    else:
        # An observer centres the cluster halo on the BCG light and knows nothing else about the truth.
        halos = (DPIEHaloConfig.cluster_scale("halo", bcg["x_centre"], bcg["y_centre"], z_lens, v_disp=800.0, position_half_width_arcsec=5.0),
                 _bcg_halo("bcg", bcg["x_centre"], bcg["y_centre"], z_lens, v_disp=300.0, cut_kpc=150.0, half_width=1.0, truth=None),
                 ShearHaloConfig.free("shear", z_lens, gamma_upper=0.1))
        members = (MemberPopulationConfig.bergamini2019("potfile", member_catalog, z_lens, prior_mag0=float(relation["pivot_magnitude"])),) if has_members else ()
    return LensModelConfig(
        reference=ReferenceFrameConfig(reference=3, ra0_deg=float(config.reference_ra_deg), dec0_deg=float(config.reference_dec_deg)),
        cosmology=config.cosmology, large_halos=halos, member_populations=members,
        image_constraints=ImageConstraintsConfig(catalog_path=image_catalog, sigma_inst_arcsec=float(config.solver_sigma_inst_arcsec)))


def solver_config_for_mock(config: MockValidationConfig, truth: dict[str, Any], *, image_catalog: Path, member_catalog: Path | None,
                           seed: int, output_dir: str | Path) -> LensClusterSolverConfig:
    """The solver configuration of one realization: the template's settings, the realization's lens model."""
    template = config.solver
    model = mock_lens_model_config(config.mock, truth, image_catalog=image_catalog, member_catalog=member_catalog)
    scaling = template.scaling
    if config.mock.mode == "idealized":
        relation = truth["relation"]
        # Centred on the truth, narrower, on the template's bounds.
        recentred = lambda prior, loc, scale: dist.TruncatedNormal(loc, scale, low=physical_bounds(prior)[0], high=physical_bounds(prior)[1])
        scaling = replace(scaling, alpha_sigma_prior=recentred(scaling.alpha_sigma_prior, float(relation["alpha_sigma"]), 0.03),
                          beta_radius_prior=recentred(scaling.beta_radius_prior, float(relation["beta_radius"]), 0.08))
    likelihood = replace(template.likelihood, sigma_inst_arcsec=float(config.mock.solver_sigma_inst_arcsec))
    return replace(template, model=model, scaling=scaling, likelihood=likelihood,
                   paths=RunPathsConfig(output_dir=output_dir, run_name="solver"),
                   runtime=replace(template.runtime, seed=int(seed), resume=bool(config.runtime.resume), quiet=bool(config.runtime.quiet)))
