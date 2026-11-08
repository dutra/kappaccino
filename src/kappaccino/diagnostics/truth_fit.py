"""Fit the lens model directly to a simulated cluster's true deflection field: how well the model family can represent
the truth, with no image data in the way.

The model (the same halos, members, scaling relation and priors as the image fit) is fitted by least squares to the
true deflection over the strong-lensing region:

    log L(theta) = -1/2 sum_p |alpha_model(x_p; theta) - alpha_true(x_p) - c|^2 / sigma^2

on a square grid of points x_p inside a circle around the multiple images, with the constant deflection c (which the
images cannot see: the sources absorb it) profiled out, and the model's priors kept as a weak regularizer.  There is no
noise in the truth: sigma only sets the weight of the data against the priors.  Two variants:

* ``relation``: the members on their scaling relation (their offsets held at 0); tests the relation;
* ``free``: as the image fit, the offsets of the exact members free and the far-field members on the relation.  The
  members are classified at the ``relation`` fit by the pipeline's own rule (``pipeline.classify``: exact when an image
  lies within the far-field guard of the member or its ellipticity matters there), and the far-field offsets pinned
  as ``SamplingConfig(far_field_offsets="fixed")`` pins them; it starts from the ``relation`` fit.

What is reported for each (``truth_fit/<variant>/`` of the run directory):

* the deflection residual: rms over the region and at the images, and its map; the convergence residual at z = 9;
* the image-plane floor: the observed images pushed to the source plane through the fitted model, the scatter of each
  family's sources mapped back through the model's inverse magnification (linearized); the same through the true
  deflection gives the catalog's own scatter, the baseline;
* the residual deflection's correlation functions (longitudinal, transverse) against the knee prior of the GRF at its
  prior median scale, and its rms against the prior of sigma_alpha;
* with ``compare_theta`` (an image fit of the same model), the same metrics for it and the split
  image - truth = (oracle - truth) + (image - oracle).

The truth maps of FF-SIMS (``deflAngles<Cluster>.fits``): two 2048 x 2048 images, the deflection in arcsec toward
increasing column (east) and increasing row (north) for a source at the header's ``ZS``, on the pixel grid of the
convergence map (``kappa_z9_0.fits``, whose WCS places the pixels).  Checked on Hera and Ares (2026-11-08): with
alpha_west = -extension 0 and alpha_north = extension 1 the catalog images of each family meet in the source plane to
0.15" (median), and half the divergence of the deflection reproduces the convergence map to 0.6%.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

from ..core.model import ModelState
from ..io.sky import radec_to_offsets_arcsec

VARIANTS = ("relation", "free")
LENS_FAMILIES = ("large", "scaling", "softening_length")
MEMBER_FAMILIES = ("member_scatter_offset", "member_velocity_scatter")


# ----------------------------------------------------------------------------- the truth
@dataclass(frozen=True)
class TruthDeflection:
    """A true deflection map in the model frame (arcsec, x positive west, y north), for a source at ``z_source``."""
    alpha_west: np.ndarray
    alpha_north: np.ndarray
    z_source: float
    z_lens: float
    pixel_arcsec: float
    column0: float          # 0-based pixel position of x = 0
    row0: float             # 0-based pixel position of y = 0
    kappa9: np.ndarray | None = None   # the true convergence for a source at z = 9 on the same grid

    def pixels(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        return self.row0 + np.asarray(y) / self.pixel_arcsec, self.column0 - np.asarray(x) / self.pixel_arcsec

    def sample(self, x, y) -> np.ndarray:
        """The true deflection at points (arcsec), shape (point, 2), bilinear."""
        rows, cols = self.pixels(x, y)
        return np.stack([map_coordinates(self.alpha_west, [rows, cols], order=1),
                         map_coordinates(self.alpha_north, [rows, cols], order=1)], axis=-1)

    def sample_jacobian(self, x, y) -> np.ndarray:
        """d alpha / d theta of the truth at points, shape (point, 2, 2), from finite differences of the maps."""
        step = self.pixel_arcsec
        out = np.empty((np.size(x), 2, 2))
        for j, (dx, dy) in enumerate(((step, 0.0), (0.0, step))):
            plus, minus = self.sample(np.asarray(x) + dx, np.asarray(y) + dy), self.sample(np.asarray(x) - dx, np.asarray(y) - dy)
            out[:, :, j] = (plus - minus) / (2.0 * step)
        return out

    def sample_kappa9(self, x, y) -> np.ndarray:
        rows, cols = self.pixels(x, y)
        return map_coordinates(self.kappa9, [rows, cols], order=1)


def load_truth_deflection(path: str | Path, kappa_path: str | Path, reference: tuple[float, float] = (0.0, 0.0)) -> TruthDeflection:
    """Read an FF-SIMS deflection map; the convergence map's WCS places its pixels (see the module docstring)."""
    with fits.open(path) as hdus:
        east, north = np.asarray(hdus[0].data, dtype=float), np.asarray(hdus[1].data, dtype=float)
        header = hdus[0].header
        z_source, z_lens = float(header["ZS"]), float(header["ZL"])
        extent = float(header["XMAX"]) - float(header["XMIN"])
    with fits.open(kappa_path) as hdus:
        kappa9, wcs = np.asarray(hdus[0].data, dtype=float), WCS(hdus[0].header)
    if kappa9.shape != east.shape:
        raise ValueError(f"{path} and {kappa_path} are not on the same pixel grid: {east.shape} and {kappa9.shape}.")
    pixel = extent / east.shape[1]
    column0, row0 = (float(v) for v in wcs.world_to_pixel_values(reference[0], reference[1]))
    return TruthDeflection(alpha_west=-east, alpha_north=north, z_source=z_source, z_lens=z_lens, pixel_arcsec=pixel,
                           column0=column0, row0=row0, kappa9=kappa9)


def region_grid(state: ModelState, *, step_arcsec: float = 1.0, radius_factor: float = 1.5) -> dict[str, Any]:
    """Grid points (arcsec) inside the circle around the image centroid of radius ``radius_factor`` times the farthest
    image's distance."""
    x, y = np.asarray(state.images.x_obs, dtype=float), np.asarray(state.images.y_obs, dtype=float)
    centre = (float(x.mean()), float(y.mean()))
    radius = float(radius_factor * np.hypot(x - centre[0], y - centre[1]).max())
    axis = np.arange(-radius, radius + step_arcsec / 2, step_arcsec)
    gx, gy = np.meshgrid(centre[0] + axis, centre[1] + axis)
    inside = np.hypot(gx - centre[0], gy - centre[1]) <= radius
    return {"x": gx[inside], "y": gy[inside], "centre": centre, "radius": radius, "step": float(step_arcsec),
            "shape": gx.shape, "mask": inside, "axis": axis}


# ----------------------------------------------------------------------------- the model
def efficiency(state: ModelState, z_source: float) -> float:
    """The lensing efficiency of the model at ``z_source`` (the dPIE sigma0 factor, proportional to D_ls / D_s)."""
    from ..lens.cosmology import dpie_sigma0_factor_from_config
    table = state.dpie_sigma0_factor_by_z
    if float(z_source) not in table:
        table[float(z_source)] = float(dpie_sigma0_factor_from_config(state.z_lens, float(z_source), state.cosmo_config))
    return float(table[float(z_source)])


def model_deflection(evaluator, theta, z_source: float, x, y, *, components=None) -> tuple[np.ndarray, np.ndarray]:
    """The model's deflection (point, 2) and its Jacobian (point, 2, 2) at points for a source at ``z_source``."""
    efficiency(evaluator.state, z_source)
    physical = evaluator.physical(jnp.asarray(theta, dtype=jnp.float64))
    packed = evaluator.packed_state_for_z(physical, z_source)
    return evaluator.total_fields_at(np.column_stack((x, y)), packed, components)


def truth_log_likelihood(evaluator, truth: TruthDeflection, x, y, sigma_arcsec: float):
    """log L(theta) of the module docstring, jittable and differentiable in the latent vector."""
    efficiency(evaluator.state, truth.z_source)
    points = jnp.asarray(np.column_stack((x, y)), dtype=jnp.float64)
    target = jnp.asarray(truth.sample(x, y), dtype=jnp.float64)
    components = np.arange(evaluator.n_components, dtype=np.int32)
    scale = 1.0 / float(sigma_arcsec) ** 2

    def log_likelihood(params):
        packed = evaluator.packed_state_for_z(evaluator.physical(params), truth.z_source)
        alpha, _ = evaluator.component_rows(points, packed, components)
        residual = jnp.sum(alpha, axis=0) - target
        residual = residual - jnp.mean(residual, axis=0)
        return -0.5 * scale * jnp.sum(residual ** 2)

    return log_likelihood


def active_sites(state: ModelState, variant: str) -> set[str]:
    """The sample sites the variant fits: the lens (halos, shear, relation, softening), plus the member offsets for "free"."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}; got {variant!r}.")
    from ..inference.sampling import FIXED_SITE_SUFFIX
    families = LENS_FAMILIES + (MEMBER_FAMILIES if variant == "free" else ())
    sites = {spec.sample_site_name or spec.sample_name for spec in state.parameter_specs if spec.component_family in families}
    return {site for site in sites if not site.endswith(FIXED_SITE_SUFFIX)}   # far-field offsets pinned at the relation


def free_member_state(state: ModelState, evaluator, theta: np.ndarray, config, *, log=print) -> tuple[ModelState, dict[str, Any]]:
    """The state classified at ``theta`` (pipeline.classify) with the far-field members' offsets moved to pinned sites,
    as the "free" variant fits it, and the classification counts."""
    from dataclasses import replace
    from ..inference.sampling import split_offset_sites
    from ..pipeline import _scaling_value, classify
    physical = np.asarray(evaluator.physical(jnp.asarray(theta, dtype=jnp.float64)))
    classified = classify(state, evaluator, theta, config, reference_cut_kpc=_scaling_value(state, physical, "cutkpc"),
                          beta_radius=_scaling_value(state, physical, "beta_radius"), log=log)
    specs, _ = split_offset_sites(list(classified.parameter_specs), np.asarray(theta, dtype=float)[None], classified,
                                  dense_offset_members="all", far_field_offsets="fixed", log=log)
    members = classified.members
    counts = {"members": len(members.member_ids), "exact": len(members.exact_positions), "far_field": len(members.far_positions)}
    return replace(classified, parameter_specs=specs), counts


def fit_variant(state: ModelState, evaluator, truth: TruthDeflection, grid: dict, variant: str, *, start_theta: np.ndarray,
                sigma_arcsec: float = 0.1, max_iter: int = 2000, extra_starts: int = 0, seed: int = 0, log=print) -> dict[str, Any]:
    """The best fit of the model to the true deflection (the variant's sites free, the others held at ``start_theta``)."""
    from ..inference.search import Objective, prior_starts, run_scipy_lbfgs
    log_likelihood = truth_log_likelihood(evaluator, truth, grid["x"], grid["y"], sigma_arcsec)
    objective = Objective(state, evaluator, start_theta, log_likelihood=log_likelihood, active_sites=active_sites(state, variant))
    starts = [np.asarray(start_theta, dtype=float)]
    if extra_starts:
        draws = np.asarray(prior_starts(state, int(extra_starts), seed))
        fitted = set(active_sites(state, variant))
        for draw in draws:   # prior draws of the fitted sites only; the rest as in the start
            theta = np.asarray(start_theta, dtype=float).copy()
            for i, spec in enumerate(state.parameter_specs):
                if (spec.sample_site_name or spec.sample_name) in fitted:
                    theta[i] = draw[i]
            starts.append(theta)
    starts_g = np.stack([objective.flatten(theta)[np.asarray(objective.global_index)] for theta in starts])
    began = time.perf_counter()
    result = run_scipy_lbfgs(objective, starts_g, max_iter)
    best = int(np.nanargmin(result["potential"]))
    theta = objective.theta(objective.expand(result["final"][best]))
    log(f"[truth fit] {variant}: {objective.n_global} parameters, {len(starts)} start(s), potentials "
        f"{np.round(result['potential'], 1).tolist()}, {result['iterations'][best]} iterations, {time.perf_counter() - began:.0f}s")
    return {"theta": np.asarray(theta), "potential": float(result["potential"][best]), "iterations": int(result["iterations"][best]),
            "n_parameters": int(objective.n_global), "potentials": result["potential"].tolist()}


# ----------------------------------------------------------------------------- metrics
def deflection_metrics(evaluator, theta, truth: TruthDeflection, grid: dict) -> dict[str, Any]:
    """Residual deflection over the region (constant removed) and at the images; the convergence residual at z = 9."""
    state = evaluator.state
    alpha, _ = model_deflection(evaluator, theta, truth.z_source, grid["x"], grid["y"])
    residual = alpha - truth.sample(grid["x"], grid["y"])
    offset = residual.mean(axis=0)
    residual = residual - offset
    xi, yi = np.asarray(state.images.x_obs, dtype=float), np.asarray(state.images.y_obs, dtype=float)
    at_images = model_deflection(evaluator, theta, truth.z_source, xi, yi)[0] - truth.sample(xi, yi) - offset
    _, u9 = model_deflection(evaluator, theta, 9.0, grid["x"], grid["y"])
    kappa_model = 0.5 * (u9[:, 0, 0] + u9[:, 1, 1])
    kappa_true = truth.sample_kappa9(grid["x"], grid["y"])
    norm = np.hypot(residual[:, 0], residual[:, 1])
    return {"residual": residual, "kappa_residual": kappa_model - kappa_true, "kappa_true": kappa_true,
            "summary": {"constant_offset_arcsec": offset.round(4).tolist(),
                        "rms_region_arcsec": float(np.sqrt(np.mean(norm ** 2))),
                        "rms_per_component_region_arcsec": float(np.sqrt(np.mean(residual ** 2))),
                        "max_region_arcsec": float(norm.max()),
                        "rms_at_images_arcsec": float(np.sqrt(np.mean(np.sum(at_images ** 2, axis=1)))),
                        "kappa_rms_residual_z9": float(np.sqrt(np.mean((kappa_model - kappa_true) ** 2))),
                        "kappa_mean_ratio_z9": float(np.mean(kappa_model) / np.mean(kappa_true))}}


def image_plane_floor(state: ModelState, deflection_at: callable, jacobian_at: callable) -> dict[str, Any]:
    """Image-plane scatter of each family's sources: beta_i = theta_i - alpha(theta_i) at the family's redshift, the
    offset from the family's mean source mapped back by A_i^-1 (A = I - d alpha / d theta), linearized.  The linear
    mapping diverges at critical curves, so the summary gives the median and the 84th percentile, not an rms."""
    images = state.images
    x, y = np.asarray(images.x_obs, dtype=float), np.asarray(images.y_obs, dtype=float)
    z_of_image = np.asarray(state.z_source_values, dtype=float)[np.asarray(images.z_index_per_image)]
    family = np.asarray(images.family_index_per_image)
    beta = np.empty((x.size, 2)); a_inverse = np.empty((x.size, 2, 2))
    for z in np.unique(z_of_image):
        sel = z_of_image == z
        beta[sel] = np.column_stack((x[sel], y[sel])) - deflection_at(x[sel], y[sel], z)
        a = np.eye(2)[None] - jacobian_at(x[sel], y[sel], z)
        a_inverse[sel] = np.linalg.inv(a)
    offsets = np.zeros((x.size, 2)); source_scatter = np.zeros(x.size)
    for f in np.unique(family):
        sel = family == f
        delta = beta[sel] - beta[sel].mean(axis=0)
        source_scatter[sel] = np.hypot(delta[:, 0], delta[:, 1])
        offsets[sel] = np.einsum("nij,nj->ni", a_inverse[sel], delta)
    norm = np.hypot(offsets[:, 0], offsets[:, 1])
    return {"image_offsets_arcsec": norm, "labels": list(images.image_labels),
            "summary": {"image_plane_median_arcsec": float(np.median(norm)), "image_plane_p84_arcsec": float(np.percentile(norm, 84)),
                        "source_plane_rms_arcsec": float(np.sqrt(np.mean(source_scatter ** 2)))}}


def model_floor(evaluator, theta) -> dict[str, Any]:
    def deflection_at(x, y, z):
        return model_deflection(evaluator, theta, z, x, y)[0]

    def jacobian_at(x, y, z):
        return model_deflection(evaluator, theta, z, x, y)[1]
    return image_plane_floor(evaluator.state, deflection_at, jacobian_at)


def truth_floor(state: ModelState, truth: TruthDeflection) -> dict[str, Any]:
    reference = efficiency(state, truth.z_source)

    def deflection_at(x, y, z):
        return truth.sample(x, y) * efficiency(state, z) / reference

    def jacobian_at(x, y, z):
        return truth.sample_jacobian(x, y) * efficiency(state, z) / reference
    return image_plane_floor(state, deflection_at, jacobian_at)


def correlation_functions(x, y, residual, *, max_points: int = 3000, bins: int = 25, r_max: float | None = None, seed: int = 0) -> dict[str, Any]:
    """Empirical longitudinal and transverse correlation functions of a residual deflection field, normalized at r = 0:
    C_L(r) = <r_L(a) r_L(b)> / sigma^2, with r_L the component along the separation, C_T across it."""
    rng = np.random.default_rng(seed)
    index = rng.choice(len(x), size=min(max_points, len(x)), replace=False)
    px, py, rr = np.asarray(x)[index], np.asarray(y)[index], np.asarray(residual)[index]
    i, j = np.triu_indices(len(px), k=1)
    dx, dy = px[j] - px[i], py[j] - py[i]
    r = np.hypot(dx, dy)
    nx, ny = dx / r, dy / r
    long_i, long_j = rr[i, 0] * nx + rr[i, 1] * ny, rr[j, 0] * nx + rr[j, 1] * ny
    tran_i, tran_j = -rr[i, 0] * ny + rr[i, 1] * nx, -rr[j, 0] * ny + rr[j, 1] * nx
    variance = float(np.mean(rr ** 2))
    edges = np.linspace(0.0, r_max or float(r.max()) / 2.0, bins + 1)
    which = np.digitize(r, edges) - 1
    centres, c_l, c_t = [], [], []
    for b in range(bins):
        sel = which == b
        if sel.sum() < 50:
            continue
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        c_l.append(float(np.mean(long_i[sel] * long_j[sel]) / variance))
        c_t.append(float(np.mean(tran_i[sel] * tran_j[sel]) / variance))
    return {"r": np.asarray(centres), "c_l": np.asarray(c_l), "c_t": np.asarray(c_t), "sigma_alpha": float(np.sqrt(variance))}


def knee_prior_curves(config, r: np.ndarray) -> dict[str, Any] | None:
    """The knee GRF's correlation functions at the prior median of its scale (and slope), and the prior median of
    sigma_alpha, when the run uses the knee likelihood."""
    from ..likelihood.gp_knee import knee_tables
    from ..priors import physical_center
    likelihood = config.likelihood
    if getattr(likelihood, "gp_kernel", None) != "knee":
        return None
    lam = float(physical_center(likelihood.gp_knee_prior)) if likelihood.gp_knee_prior is not None else float(likelihood.gp_knee_start_arcsec)
    gamma = float(likelihood.gp_knee_gamma) if likelihood.gp_knee_gamma is not None else 4.0
    scale_min, scale_max = (float(v) for v in likelihood.gp_knee_scales_arcsec)
    step = max(float(r.max()) / 400.0, 0.05)
    tables = knee_tables(r_max=float(r.max()) + step, r_step=step, lam_min=lam / 1.001, lam_max=lam * 1.001, n_lam=2,
                         scale_min=scale_min, scale_max=scale_max, gamma=gamma)
    grid = np.asarray(tables["r_grid"])
    c_l = np.interp(r, grid, np.asarray(tables["c_l"])[:, 0, 0])
    c_t = np.interp(r, grid, np.asarray(tables["c_t"])[:, 0, 0])
    sigma = (float(physical_center(likelihood.gp_sigma_alpha_prior)) if getattr(likelihood, "gp_sigma_alpha_prior", None) is not None
             else None)
    return {"knee_arcsec": lam, "gamma": gamma, "c_l": c_l, "c_t": c_t, "sigma_alpha_prior_median": sigma}


# ----------------------------------------------------------------------------- driver
def run_truth_fit(config, *, deflection_path: str | Path, kappa_path: str | Path, output_dir: str | Path,
                  variants: tuple[str, ...] = VARIANTS, step_arcsec: float = 1.0, radius_factor: float = 1.5,
                  sigma_arcsec: float = 0.1, max_iter: int = 2000, extra_starts: int = 0, compare_theta: np.ndarray | None = None,
                  compare_label: str = "image fit", log=print) -> dict[str, Any]:
    """Fit each variant to the truth and write ``<output_dir>/<variant>/`` (summary.json, parameters.csv, the residual maps
    and correlation figure) and ``<output_dir>/summary.json`` (all variants, the truth baseline, the comparison)."""
    import pandas as pd
    from ..core.state import build_state
    from ..likelihood.evaluator import Evaluator
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = build_state(config)
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)   # the image likelihood itself is never evaluated
    reference = (float(config.model.reference.ra0_deg), float(config.model.reference.dec0_deg))
    truth = load_truth_deflection(deflection_path, kappa_path, reference)
    if abs(truth.z_lens - float(state.z_lens)) > 0.01:
        raise ValueError(f"the truth's lens redshift {truth.z_lens} is not the model's {state.z_lens}.")
    grid = region_grid(state, step_arcsec=step_arcsec, radius_factor=radius_factor)
    log(f"[truth fit] {len(grid['x'])} points, step {step_arcsec}\", radius {grid['radius']:.1f}\" about "
        f"({grid['centre'][0]:.1f}, {grid['centre'][1]:.1f}); truth for z_s = {truth.z_source}")
    baseline = truth_floor(state, truth)
    summary: dict[str, Any] = {"grid": {k: grid[k] for k in ("centre", "radius", "step")}, "n_points": int(len(grid["x"])),
                               "sigma_arcsec": sigma_arcsec, "z_source_truth": truth.z_source,
                               "truth_catalog_floor": baseline["summary"], "variants": {}}
    names = [spec.name for spec in state.parameter_specs]
    families = [spec.component_family for spec in state.parameter_specs]
    reported = lambda theta: np.asarray(evaluator.reported_physical(jnp.asarray(theta, dtype=jnp.float64)))
    table = pd.DataFrame({"parameter": names, "family": families, "start": reported(state.initial_theta)})
    thetas, fits = {}, {}
    if "relation" in variants or "free" in variants:
        fits["relation"] = fit_variant(state, evaluator, truth, grid, "relation", start_theta=np.asarray(state.initial_theta),
                                       sigma_arcsec=sigma_arcsec, max_iter=max_iter, extra_starts=extra_starts, log=log)
    if "free" in variants:
        free_state, counts = free_member_state(state, evaluator, fits["relation"]["theta"], config, log=log)
        summary["member_classification"] = counts
        fits["free"] = fit_variant(free_state, evaluator, truth, grid, "free", start_theta=fits["relation"]["theta"],
                                   sigma_arcsec=sigma_arcsec, max_iter=max_iter, log=log)
        log(f"[truth fit] free: offsets of the {counts['exact']} exact members free, the {counts['far_field']} far-field members on the relation")
    for variant in variants:
        fit = fits[variant]
        thetas[variant] = fit["theta"]
        metrics = deflection_metrics(evaluator, fit["theta"], truth, grid)
        floor = model_floor(evaluator, fit["theta"])
        corr = correlation_functions(grid["x"], grid["y"], metrics["residual"])
        prior = knee_prior_curves(config, corr["r"]) if len(corr["r"]) else None
        variant_summary = {"fit": {k: fit[k] for k in ("potential", "iterations", "n_parameters", "potentials")},
                           "deflection": metrics["summary"], "image_plane": floor["summary"],
                           "residual_sigma_alpha_arcsec": corr["sigma_alpha"],
                           "knee_prior": None if prior is None else {"knee_arcsec": prior["knee_arcsec"], "gamma": prior["gamma"],
                                                                      "sigma_alpha_prior_median": prior["sigma_alpha_prior_median"]}}
        summary["variants"][variant] = variant_summary
        table[variant] = reported(fit["theta"])
        directory = output_dir / variant
        directory.mkdir(exist_ok=True)
        (directory / "summary.json").write_text(json.dumps(variant_summary, indent=2))
        pd.DataFrame({"image_label": floor["labels"], "image_offset_arcsec": floor["image_offsets_arcsec"],
                      "catalog_floor_arcsec": baseline["image_offsets_arcsec"]}).to_csv(directory / "image_plane_floor.csv", index=False)
        np.savez_compressed(directory / "residual_maps.npz", x=grid["x"], y=grid["y"], residual=metrics["residual"],
                            kappa_residual=metrics["kappa_residual"], kappa_true=metrics["kappa_true"])
        plot_truth_fit(directory, grid, metrics, corr, prior, state, title=f"{config.paths.run_name or ''} {variant}")
        log(f"[truth fit] {variant}: deflection rms {metrics['summary']['rms_region_arcsec']:.3f}\" (region), "
            f"{metrics['summary']['rms_at_images_arcsec']:.3f}\" (images); image plane median {floor['summary']['image_plane_median_arcsec']:.3f}\", "
            f"84% {floor['summary']['image_plane_p84_arcsec']:.3f}\" (catalog floor {baseline['summary']['image_plane_median_arcsec']:.3f}\", "
            f"{baseline['summary']['image_plane_p84_arcsec']:.3f}\"); kappa rms {metrics['summary']['kappa_rms_residual_z9']:.3f}")
    if compare_theta is not None:
        compare_theta = np.asarray(compare_theta, dtype=float)
        metrics = deflection_metrics(evaluator, compare_theta, truth, grid)
        floor = model_floor(evaluator, compare_theta)
        split = {}
        for variant, theta in thetas.items():
            a_image = model_deflection(evaluator, compare_theta, truth.z_source, grid["x"], grid["y"])[0]
            a_oracle = model_deflection(evaluator, theta, truth.z_source, grid["x"], grid["y"])[0]
            difference = a_image - a_oracle
            difference = difference - difference.mean(axis=0)
            split[variant] = {"oracle_minus_truth_rms_arcsec": summary["variants"][variant]["deflection"]["rms_region_arcsec"],
                              "image_minus_oracle_rms_arcsec": float(np.sqrt(np.mean(np.sum(difference ** 2, axis=1))))}
        summary["comparison"] = {"label": compare_label, "deflection": metrics["summary"], "image_plane": floor["summary"],
                                 "image_minus_truth_rms_arcsec": metrics["summary"]["rms_region_arcsec"], "split": split}
        table[compare_label] = reported(compare_theta)
    table.to_csv(output_dir / "parameters.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_truth_fit(directory: Path, grid: dict, metrics: dict, corr: dict, prior: dict | None, state: ModelState, *, title: str = "") -> None:
    """residual_maps.pdf (deflection residual and convergence residual, with the images) and correlation.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shape, mask, axis, centre = grid["shape"], grid["mask"], grid["axis"], grid["centre"]
    extent = [centre[0] + axis[0], centre[0] + axis[-1], centre[1] + axis[0], centre[1] + axis[-1]]

    def image(values):
        out = np.full(shape, np.nan)
        out[mask] = values
        return out

    residual = metrics["residual"]
    norm = np.hypot(residual[:, 0], residual[:, 1])
    xi, yi = np.asarray(state.images.x_obs), np.asarray(state.images.y_obs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    shown = axes[0].imshow(image(norm), origin="lower", extent=extent, cmap="viridis")
    every = max(1, int(len(grid["x"]) / 400))
    axes[0].quiver(grid["x"][::every], grid["y"][::every], residual[::every, 0], residual[::every, 1], color="white", scale_units="xy",
                   angles="xy", scale=norm.max() / (6 * grid["step"]) if norm.max() > 0 else 1.0, width=0.003)
    fig.colorbar(shown, ax=axes[0], label="|model - true deflection| (arcsec)")
    limit = float(np.nanpercentile(np.abs(metrics["kappa_residual"]), 99))
    shown = axes[1].imshow(image(metrics["kappa_residual"]), origin="lower", extent=extent, cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(shown, ax=axes[1], label=r"$\kappa_{\rm model} - \kappa_{\rm true}$ ($z_s = 9$)")
    for ax in axes:
        ax.scatter(xi, yi, s=8, facecolor="none", edgecolor="k", lw=0.6)
        ax.set_xlim(extent[1], extent[0])    # x positive west: east to the left
        ax.set_xlabel("x (arcsec, west)")
        ax.set_ylabel("y (arcsec, north)")
    fig.savefig(directory / "residual_maps.pdf")
    plt.close(fig)
    if not len(corr["r"]):
        return
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.plot(corr["r"], corr["c_l"], "o-", color="C0", label=r"residual $C_L$")
    ax.plot(corr["r"], corr["c_t"], "s-", color="C1", label=r"residual $C_T$")
    if prior is not None:
        ax.plot(corr["r"], prior["c_l"], "--", color="C0", label=f"knee prior $c_L$ ($\\lambda$ = {prior['knee_arcsec']:.1f}\")")
        ax.plot(corr["r"], prior["c_t"], "--", color="C1", label="knee prior $c_T$")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("separation (arcsec)")
    ax.set_ylabel("correlation of the residual deflection")
    text = f"residual $\\sigma_\\alpha$ = {corr['sigma_alpha']:.3f}\""
    if prior is not None and prior["sigma_alpha_prior_median"] is not None:
        text += f"\nprior median {prior['sigma_alpha_prior_median']:.3f}\""
    ax.text(0.97, 0.95, text, transform=ax.transAxes, ha="right", va="top")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(directory / "correlation.pdf")
    plt.close(fig)
