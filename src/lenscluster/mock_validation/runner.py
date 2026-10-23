"""Mock-validation campaign: generate a mock, run the single inference stage, compare with the truth."""
from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .. import pipeline
from ..config import DPIEHaloConfig
from ..model import ModelState, PosteriorResults, member_log_deviations
from ..utils import log10_dpie_mass_msun, progress
from .aperture import APERTURE_RADII_KPC, COMPONENT_GROUPS, aperture_profiles, truth_groups
from .config import MockValidationConfig, solver_config_for_mock
from .generation import MockClusterPaths, generate_single_bcg_mock

RECOVERY_FILE = "recovery.json"
M2D_POSTERIOR_DRAWS = 24


def campaign_root(config: MockValidationConfig) -> Path:
    return Path(str(config.paths.output_dir)) / str(config.paths.campaign_name) / str(config.paths.run_name)


def seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{int(seed):08d}"


def mock_paths(mock_dir: Path) -> MockClusterPaths:
    return MockClusterPaths(root=mock_dir, par_path=mock_dir / "single_bcg_mock.par", image_catalog_path=mock_dir / "obs_arcs.cat",
                            truth_path=mock_dir / "truth.json", mock_images_path=mock_dir / "mock_images.json")


def mock_complete(paths: MockClusterPaths) -> bool:
    return all(p.is_file() for p in (paths.image_catalog_path, paths.truth_path, paths.mock_images_path))


# ----------------------------------------------------------------------------- truth values
def parameter_truth(truth: dict[str, Any]) -> dict[str, float]:
    """Truth value per model parameter name, including derived masses, scatter and source positions."""
    values = {str(k): float(v) for k, v in truth["parameter_truth"].items() if _finite(v)}
    for prefix in ("halo", "bcg"):
        if f"{prefix}.v_disp" in values and f"{prefix}.cut_radius_kpc" in values:
            values[f"{prefix}.log10_mass_msun"] = float(log10_dpie_mass_msun(values[f"{prefix}.v_disp"], values[f"{prefix}.cut_radius_kpc"]))
    if "potfile.sigma" in values and "potfile.cutkpc" in values:
        values["potfile.log10_mass_msun"] = float(log10_dpie_mass_msun(values["potfile.sigma"], values["potfile.cutkpc"]))
    scatter = values.get("potfile.sigma_log_scatter", truth.get("config", {}).get("subhalo_sigma_log_scatter"))
    if _finite(scatter):
        values["potfile.velocity_log_scatter"] = float(scatter)
    sigma_int = truth.get("config", {}).get("source_sigma_int_arcsec")
    if _finite(sigma_int):
        values["source.sigma_src"] = float(sigma_int)
    for source in truth.get("sources", []):
        for key in ("beta_x", "beta_y"):
            if _finite(source.get(key)):
                values[f"source.{source['family_id']}.{key}"] = float(source[key])
    return values


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------------- recovery tables
def posterior_columns(state: ModelState, evaluator, posterior: PosteriorResults) -> dict[str, np.ndarray]:
    """Posterior draws per parameter name in physical units, with derived masses, gamma_ml and source positions."""
    physical = np.asarray(posterior.physical, dtype=float)
    columns = {spec.name: physical[:, i] for i, spec in enumerate(state.parameter_specs) if spec.component_family != "source_position"}
    for prefix in {name.rsplit(".", 1)[0] for name in columns}:
        v, cut = columns.get(f"{prefix}.v_disp"), columns.get(f"{prefix}.cut_radius_kpc")
        if v is not None and cut is not None:
            columns[f"{prefix}.log10_mass_msun"] = log10_dpie_mass_msun(v, cut)
        sigma, cutkpc = columns.get(f"{prefix}.sigma"), columns.get(f"{prefix}.cutkpc")
        if sigma is not None and cutkpc is not None:
            columns[f"{prefix}.log10_mass_msun"] = log10_dpie_mass_msun(sigma, cutkpc)
        alpha, beta = columns.get(f"{prefix}.alpha_sigma"), columns.get(f"{prefix}.beta_radius")
        if alpha is not None and beta is not None:
            columns[f"{prefix}.gamma_ml"] = 2.0 * alpha + beta - 1.0
    sources = np.asarray(jax.vmap(jax.jit(evaluator.source_positions))(jnp.asarray(posterior.samples)))
    for k, family_id in enumerate(state.images.family_ids):
        columns[f"source.{family_id}.beta_x"] = sources[:, k, 0]
        columns[f"source.{family_id}.beta_y"] = sources[:, k, 1]
    return columns


def parameter_recovery(columns: dict[str, np.ndarray], truth: dict[str, float], best_index: int) -> pd.DataFrame:
    rows = []
    for name, values in columns.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        q16, median, q84 = np.quantile(finite, [0.16, 0.5, 0.84])
        truth_value = truth.get(name, np.nan)
        rows.append({"parameter": name, "truth": truth_value, "best_fit": float(values[best_index]), "q16": float(q16), "median": float(median),
                     "q84": float(q84), "bias": float(median - truth_value), "truth_percentile": float(np.mean(finite <= truth_value)),
                     "covered_68": bool(q16 <= truth_value <= q84)})
    return pd.DataFrame(rows)


def magnification_recovery(truth: dict[str, Any], magnification: pd.DataFrame) -> pd.DataFrame:
    truth_images = pd.DataFrame(truth["images"])[["image_label", "family_id", "magnification_true", "magnification_true_at_obs"]]
    merged = truth_images.merge(magnification.drop(columns=[c for c in ("family_id",) if c in magnification]), on="image_label", how="left")
    mu_true = merged["magnification_true_at_obs"].to_numpy(float)
    for suffix in ("", "_q16", "_q50", "_q84"):
        values = merged[f"magnification_model{suffix}"].to_numpy(float)
        merged[f"abs_fractional_error{suffix}"] = np.abs(np.abs(values) - np.abs(mu_true)) / np.maximum(np.abs(mu_true), 1e-8)
    merged["parity_match"] = np.sign(mu_true) == np.sign(merged["magnification_model"].to_numpy(float))
    return merged


def subhalo_masses(truth: dict[str, Any], member_table: pd.DataFrame) -> dict[str, Any]:
    truth_masses = np.asarray([float(s["subhalo_dpie_mass_msun"]) for s in truth.get("subhalos", []) if _finite(s.get("subhalo_dpie_mass_msun"))])
    recovered = member_table["log10_mass_msun_median"].to_numpy(float) if not member_table.empty else np.asarray([])
    return {"truth_log10_mass": np.log10(truth_masses[truth_masses > 0]).tolist(), "recovered_log10_mass": recovered[np.isfinite(recovered)].tolist(),
            "selection": {k: v for k, v in truth.get("subhalo_selection", {}).items() if k != "candidates"}}


def model_groups(state: ModelState, config_model, physical_row: np.ndarray, name_index: dict[str, int]) -> dict[str, tuple[np.ndarray, ...]]:
    """Physical dPIE components implied by one posterior draw, grouped as the truth is."""
    kpc = float(state.kpc_per_arcsec)
    halos = {h.id: h for h in config_model.large_halos if isinstance(h, DPIEHaloConfig)}

    def value(name: str, default: float) -> float:
        index = name_index.get(name)
        return float(physical_row[index]) if index is not None else float(default)

    def large(prefix: str) -> tuple[np.ndarray, ...]:
        h = halos[prefix]
        return tuple(np.asarray([value(f"{prefix}.{key}", getattr(h, key)) * scale]) for key, scale in
                     (("v_disp", 1.0), ("core_radius_kpc", 1.0), ("cut_radius_kpc", 1.0), ("x_centre", kpc), ("y_centre", kpc)))

    spec = state.packed_lens_spec
    dev_v, dev_r = (np.asarray(a) for a in member_log_deviations(spec, physical_row))
    v, core, cut, xs, ys = [], [], [], [], []
    for record in state.scaling_component_records:
        c = int(record["component_index"])
        lum = float(spec.luminosity_ratio[c])
        sigma_ref, cut_ref, core_ref = (physical_row[int(idx[c])] for idx in (spec.sigma_ref_param_index, spec.cut_ref_param_index, spec.core_ref_param_index))
        alpha, beta = physical_row[int(spec.alpha_sigma_param_index[c])], physical_row[int(spec.beta_radius_param_index[c])]
        v.append(sigma_ref * lum ** alpha * math.exp(float(dev_v[c])))
        core.append(core_ref * lum ** beta * math.exp(float(dev_r[c])))
        cut.append(cut_ref * lum ** beta * math.exp(float(dev_r[c])))
        xs.append(float(record["x_centre"]) * kpc)
        ys.append(float(record["y_centre"]) * kpc)
    return {"halo": large("halo"), "bcg": large("bcg"), "members": tuple(np.asarray(a, dtype=float) for a in (v, core, cut, xs, ys))}


def m2d_recovery(state: ModelState, config_model, posterior: PosteriorResults, truth: dict[str, Any], *, max_draws: int = M2D_POSTERIOR_DRAWS) -> pd.DataFrame:
    radii = np.asarray(APERTURE_RADII_KPC)
    truth_profiles = aperture_profiles(radii, truth_groups(truth))
    name_index = {spec.name: i for i, spec in enumerate(state.parameter_specs)}
    physical = np.asarray(posterior.physical, dtype=float)
    draws = physical[np.unique(np.linspace(0, len(physical) - 1, min(int(max_draws), len(physical))).astype(int))]
    per_group = {g: [] for g in (*COMPONENT_GROUPS, "total")}
    for row in progress(draws, desc="aperture draws"):
        for group, profile in aperture_profiles(radii, model_groups(state, config_model, row, name_index)).items():
            per_group[group].append(profile)
    rows = []
    for group, profiles in per_group.items():
        stacked = np.stack(profiles)
        q16, q50, q84 = np.percentile(stacked, [16, 50, 84], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(truth_profiles[group] > 0, stacked / truth_profiles[group], np.nan)
        r16, r50, r84 = np.nanpercentile(ratio, [16, 50, 84], axis=0)
        for k, radius in enumerate(radii):
            rows.append({"component": group, "radius_kpc": float(radius), "truth_m2d_msun": float(truth_profiles[group][k]),
                         "model_q16": float(q16[k]), "model_q50": float(q50[k]), "model_q84": float(q84[k]),
                         "ratio_q16": float(r16[k]), "ratio_q50": float(r50[k]), "ratio_q84": float(r84[k])})
    return pd.DataFrame(rows)


def write_recovery(recovery_dir: Path, result: pipeline.RunResult, solver_config, truth: dict[str, Any], seed: int, *, log=print) -> dict[str, Any]:
    """Compare the posterior with the mock truth and write recovery.json plus CSV tables."""
    from ..plots import member_scaling_table
    recovery_dir.mkdir(parents=True, exist_ok=True)
    state, evaluator, posterior = result.state, result.evaluator, result.posterior
    tables = result.paths.tables
    best_index = int(np.nanargmax(np.asarray(posterior.log_prob)))
    columns = posterior_columns(state, evaluator, posterior)
    parameters = parameter_recovery(columns, parameter_truth(truth), best_index)
    magnification = magnification_recovery(truth, pd.read_csv(tables / "model_magnification.csv"))
    image_fit = pd.read_csv(tables / "image_fit_quality.csv")
    residual_draws = pd.read_csv(tables / "posterior_image_residual_draws.csv")
    member_table = member_scaling_table(state, posterior)
    m2d = m2d_recovery(state, solver_config.model, posterior, truth)
    for name, frame in (("parameter_recovery", parameters), ("magnification_recovery", magnification), ("m2d_aperture_recovery", m2d)):
        frame.to_csv(recovery_dir / f"{name}.csv", index=False)
    fit = json.loads((tables / "run_summary.json").read_text()).get("fit", {})
    diagnostics = posterior.diagnostics or {}
    payload = {
        "seed": int(seed), "n_parameters": int(state.n_parameters), "images": int(state.images.n_images), "families": int(state.images.n_families),
        "members": int(len(state.members.member_ids)), "exact_members": int(len(state.members.exact_positions)),
        "fit": fit, "max_r_hat": diagnostics.get("max_r_hat"), "min_n_eff": diagnostics.get("min_n_eff"),
        "divergences": int(np.asarray(posterior.diverging).sum()), "audit": result.audit.get("approximation_validation"),
        "parameters": parameters.to_dict("records"), "magnification": magnification.to_dict("records"),
        "image_residuals": image_fit[["family_id", "image_label", "image_recovery_status", "image_residual_arcsec"]].to_dict("records"),
        "image_residual_draws": residual_draws.to_dict("records"), "subhalos": subhalo_masses(truth, member_table),
        "m2d": m2d.to_dict("records"),
    }
    (recovery_dir / RECOVERY_FILE).write_text(json.dumps(payload, indent=1, default=_json_default))
    covered = parameters["covered_68"].mean() if not parameters.empty else float("nan")
    log(f"[recovery] seed={seed} images recovered {fit.get('recovered_image_count')}/{fit.get('observed_image_count')} "
        f"rms={fit.get('image_rms_arcsec')} covered_68={covered:.2f} max_r_hat={diagnostics.get('max_r_hat')}")
    return payload


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


# ----------------------------------------------------------------------------- campaign
def run_mock_validation(config: MockValidationConfig, *, log=print) -> list[Path]:
    """Run every realization of the campaign; returns the recovery directories."""
    config = config.validate()
    root = campaign_root(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign_config.json").write_text(json.dumps(config.to_json_dict(), indent=2, default=_json_default))
    outputs = []
    for realization in range(int(config.runtime.realizations)):
        seed = int(config.runtime.seed) + realization
        directory = seed_dir(root, seed)
        recovery_dir = directory / "recovery"
        if config.runtime.resume and (recovery_dir / RECOVERY_FILE).is_file():
            log(f"[campaign] seed={seed} already complete; skipping")
            outputs.append(recovery_dir)
            continue
        started = time.perf_counter()
        paths = mock_paths(directory / "mock")
        if config.runtime.resume and mock_complete(paths):
            log(f"[campaign] seed={seed} reusing mock under {paths.root}")
        else:
            log(f"[campaign] seed={seed} generating mock under {paths.root}")
            paths, _, _ = generate_single_bcg_mock(paths.root, replace(config.mock, seed=seed))
        truth = json.loads(paths.truth_path.read_text())
        solver_config = solver_config_for_mock(config, paths=paths, seed=seed, output_dir=directory)
        result = pipeline.run(solver_config)
        write_recovery(recovery_dir, result, solver_config, truth, seed, log=log)
        log(f"[campaign] seed={seed} done in {time.perf_counter() - started:.0f}s")
        outputs.append(recovery_dir)
    return outputs
