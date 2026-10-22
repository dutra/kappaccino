"""Saved run artifacts: one directory with the config, classification, search, posterior and audit."""
from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import LensClusterSolverConfig
from .model import ModelState, PosteriorResults

STATE_FILE = "state.json"
CONFIG_FILE = "config.json"
CONFIG_PICKLE = "config.pkl"
CLASSIFICATION_FILE = "classification.json"
SEARCH_FILE = "search.npz"
POSTERIOR_FILE = "posterior.npz"
AUDIT_FILE = "audit.json"
SUMMARY_FILE = "summary.json"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=True), encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_config(directory: Path, config: LensClusterSolverConfig) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / CONFIG_FILE, config.to_json_dict())
    (directory / CONFIG_PICKLE).write_bytes(pickle.dumps(config))


def load_config(directory: Path) -> LensClusterSolverConfig:
    return pickle.loads((directory / CONFIG_PICKLE).read_bytes())


def save_state(directory: Path, state: ModelState) -> None:
    members = state.members
    write_json(directory / STATE_FILE, {
        "run_name": state.run_name, "z_lens": state.z_lens, "kpc_per_arcsec": state.kpc_per_arcsec,
        "n_parameters": state.n_parameters, "parameter_specs": [asdict(spec) for spec in state.parameter_specs],
        "families": [{"family_id": f.family_id, "z_source": f.z_source, "image_labels": f.image_labels} for f in state.family_data],
        "member_ids": list(members.member_ids), "exact_positions": members.exact_positions.tolist(),
        "far_positions": members.far_positions.tolist(), "initial_theta": None if state.initial_theta is None else state.initial_theta.tolist(),
    })
    write_json(directory / CLASSIFICATION_FILE, members.classification)


def save_search(directory: Path, search: dict[str, Any]) -> None:
    np.savez_compressed(directory / SEARCH_FILE, theta=np.asarray(search["theta"]), potential=np.asarray(search["potential"]),
                        starts=np.asarray(search["starts"]), assignment=np.asarray(search["assignment"]),
                        iterations=np.asarray(search["iterations"]), gradient_norm=np.asarray(search["gradient_norm"]))
    write_json(directory / "modes.json", {"modes": [{k: v for k, v in m.items() if k != "members"} | {"count": m["count"]} for m in search["modes"]],
                                           "elapsed_seconds": search["elapsed_seconds"], "n_finite": search["n_finite"]})


def save_posterior(directory: Path, posterior: PosteriorResults) -> None:
    np.savez_compressed(directory / POSTERIOR_FILE, samples=posterior.samples, physical=posterior.physical,
                        log_prob=posterior.log_prob, accept_prob=posterior.accept_prob, diverging=posterior.diverging,
                        num_steps=posterior.num_steps, grouped_samples=posterior.grouped_samples,
                        median_fit=posterior.median_fit, maximum_likelihood_fit=posterior.maximum_likelihood_fit,
                        warmup_steps=posterior.warmup_steps, sample_steps=posterior.sample_steps, num_chains=posterior.num_chains)
    write_json(directory / "posterior_diagnostics.json", posterior.diagnostics)


def load_posterior(directory: Path) -> PosteriorResults:
    with np.load(directory / POSTERIOR_FILE, allow_pickle=True) as data:
        diagnostics = json.loads((directory / "posterior_diagnostics.json").read_text()) if (directory / "posterior_diagnostics.json").exists() else {}
        return PosteriorResults(samples=data["samples"], physical=data["physical"], log_prob=data["log_prob"],
                                accept_prob=data["accept_prob"], diverging=data["diverging"], num_steps=data["num_steps"],
                                warmup_steps=int(data["warmup_steps"]), sample_steps=int(data["sample_steps"]),
                                num_chains=int(data["num_chains"]), grouped_samples=data["grouped_samples"],
                                diagnostics=diagnostics, median_fit=data["median_fit"], maximum_likelihood_fit=data["maximum_likelihood_fit"])


def load_exact_positions(directory: Path) -> np.ndarray:
    payload = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    return np.asarray(payload["exact_positions"], dtype=np.int32)


def is_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in (CONFIG_PICKLE, STATE_FILE, POSTERIOR_FILE, AUDIT_FILE))
