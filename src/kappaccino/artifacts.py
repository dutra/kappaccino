"""Saved run artifacts: the init stage (config, state, classification, search, start) and the final stage (posterior, audit).

The pipeline writes these files to ``<run root>/init/artifacts/`` and ``<run root>/final/artifacts/``; the functions
here take that directory.

``config.json``, ``config.pkl``
    The run configuration as JSON (for reading) and as a pickle (:func:`load_config` restores it exactly).
``init_signature.json``
    The part of the configuration the init stage depends on (:func:`init_signature`); resuming with a configuration
    that does not match it raises an error.
``state.json``
    A summary of the model state (:func:`save_state`): parameter specs with their priors, families, member ids, the
    exact/far-field member positions and the initial latent vector.  The state itself is rebuilt from the
    configuration: :func:`kappaccino.state.build_state` (with the ``dropped_members`` of ``classification.json``),
    then :func:`kappaccino.state.with_member_classes` with :func:`load_exact_positions`.
``classification.json``
    The exact/far-field member classification record.
``search.npz``, ``modes.json``
    The initial global search: arrays of starts, optimized vectors and traces, and the modes found
    (:func:`save_search`, :func:`load_search`).
``start.json``
    The latent start vector of the chains and how it was chosen (:func:`save_start`, :func:`load_start`).
``summary.json``
    Stage summary.
``posterior.npz``, ``posterior_diagnostics.json``
    The posterior draws and sampler diagnostics (:func:`save_posterior`, :func:`load_posterior`).
``audit.json``
    Whether the far-field approximation held over the posterior.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from . import priors
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
START_FILE = "start.json"
MODES_FILE = "modes.json"
INIT_SIGNATURE_FILE = "init_signature.json"


def write_json(path: Path, payload: Any) -> None:
    """Write a payload as indented JSON, converting NumPy arrays and scalars and paths.

    Parameters
    ----------
    path : pathlib.Path
        Output file; parent directories are created.
    payload : Any
        JSON-like value: dicts (keys converted to str), lists, tuples, NumPy arrays and scalars, ``Path`` objects,
        and plain values.  NaN and infinity are written as ``NaN`` / ``Infinity``.
    """
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
    """Write ``config.json`` and ``config.pkl`` of a run configuration.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory (created if missing).
    config : LensClusterSolverConfig
        The run configuration.
    """
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / CONFIG_FILE, config.to_json_dict())
    (directory / CONFIG_PICKLE).write_bytes(pickle.dumps(config))


def load_config(directory: Path) -> LensClusterSolverConfig:
    """Load the run configuration saved by :func:`save_config`.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory holding ``config.pkl`` (``init/artifacts`` or ``final/artifacts``).

    Returns
    -------
    LensClusterSolverConfig
        The configuration, unpickled.  Only load pickles you trust.
    """
    return pickle.loads((directory / CONFIG_PICKLE).read_bytes())


def spec_to_json(spec) -> dict[str, Any]:
    """A parameter spec as JSON: its fields, with the prior as priors.to_dict.

    Parameters
    ----------
    spec : ParameterSpec
        The parameter.

    Returns
    -------
    dict
        Every ``ParameterSpec`` field by name (``name``, ``sample_name``, ``potential_id``, ``profile_type``,
        ``field``, ``component_family``, ``sample_site_name``, ``sample_site_index``, ``centered``,
        ``scale_site_name``), and ``"prior"``: the prior as :func:`kappaccino.priors.to_dict`.
    """
    return {**{f.name: getattr(spec, f.name) for f in fields(spec) if f.name != "prior"}, "prior": priors.to_dict(spec.prior)}


def save_state(directory: Path, state: ModelState) -> None:
    """Write ``state.json`` and ``classification.json`` of a model state.

    ``state.json`` holds ``run_name``, ``z_lens``, ``kpc_per_arcsec``, ``n_parameters``, ``parameter_specs`` (as
    :func:`spec_to_json`), ``families`` (``family_id``, ``z_source``, ``image_labels``), ``member_ids``,
    ``exact_positions``, ``far_positions`` and ``initial_theta`` (latent, or null).  ``classification.json`` is
    ``state.members.classification``.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory.
    state : ModelState
        The model state.
    """
    members = state.members
    write_json(directory / STATE_FILE, {
        "run_name": state.run_name, "z_lens": state.z_lens, "kpc_per_arcsec": state.kpc_per_arcsec,
        "n_parameters": state.n_parameters, "parameter_specs": [spec_to_json(spec) for spec in state.parameter_specs],
        "families": [{"family_id": f.family_id, "z_source": f.z_source, "image_labels": f.image_labels} for f in state.family_data],
        "member_ids": list(members.member_ids), "exact_positions": members.exact_positions.tolist(),
        "far_positions": members.far_positions.tolist(), "initial_theta": None if state.initial_theta is None else state.initial_theta.tolist(),
    })
    write_json(directory / CLASSIFICATION_FILE, members.classification)


def save_search(directory: Path, search: dict[str, Any]) -> None:
    """Write ``search.npz`` and ``modes.json`` of the initial global search.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory.
    search : dict
        The search result.  Arrays ``theta`` (best latent vector), ``potential``, ``starts``, ``assignment``,
        ``iterations``, ``evaluations``, ``traces`` and ``final_theta`` go to ``search.npz``; ``modes`` (list of
        dicts, each with ``count``; ``members`` is dropped), ``method``, ``n_global``, ``search_seconds``,
        ``polish_seconds``, ``n_finite`` and the optional ``stages``, ``stage_traces``, ``starts_in_basin``,
        ``configured_in_basin`` and ``warm_start*`` entries go to ``modes.json``.
    """
    np.savez_compressed(directory / SEARCH_FILE, theta=np.asarray(search["theta"]), potential=np.asarray(search["potential"]),
                        starts=np.asarray(search["starts"]), assignment=np.asarray(search["assignment"]),
                        iterations=np.asarray(search["iterations"]), evaluations=np.asarray(search["evaluations"]),
                        traces=np.asarray(search["traces"]), final_theta=np.asarray(search["final_theta"]))
    write_json(directory / MODES_FILE, {"modes": [{k: v for k, v in m.items() if k != "members"} | {"count": m["count"]} for m in search["modes"]],
                                           "method": search["method"], "n_global": search["n_global"], "search_seconds": search["search_seconds"],
                                           "polish_seconds": search["polish_seconds"], "n_finite": search["n_finite"],
                                           "stages": search.get("stages"), "stage_traces": search.get("stage_traces"),
                                           "starts_in_basin": search.get("starts_in_basin"), "configured_in_basin": search.get("configured_in_basin"),
                                           **{k: search[k] for k in ("warm_start", "warm_start_shared", "warm_start_fragile") if k in search}})


def save_posterior(directory: Path, posterior: PosteriorResults) -> None:
    """Write ``posterior.npz`` and ``posterior_diagnostics.json``.

    ``posterior.npz`` holds every :class:`~kappaccino.model.PosteriorResults` field except ``diagnostics`` under its
    own name (``samples`` and ``physical`` of shape (draw, parameter), the per-draw sampler arrays, ``step_size``,
    ``grouped_samples``, ``median_fit``, ``maximum_likelihood_fit`` and the scalar ``warmup_steps``,
    ``sample_steps``, ``num_chains``); ``diagnostics`` goes to ``posterior_diagnostics.json``.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory (``final/artifacts``).
    posterior : PosteriorResults
        The posterior.
    """
    np.savez_compressed(directory / POSTERIOR_FILE, samples=posterior.samples, physical=posterior.physical,
                        log_prob=posterior.log_prob, accept_prob=posterior.accept_prob, diverging=posterior.diverging,
                        num_steps=posterior.num_steps, energy=posterior.energy, step_size=posterior.step_size, grouped_samples=posterior.grouped_samples,
                        median_fit=posterior.median_fit, maximum_likelihood_fit=posterior.maximum_likelihood_fit,
                        warmup_steps=posterior.warmup_steps, sample_steps=posterior.sample_steps, num_chains=posterior.num_chains)
    write_json(directory / "posterior_diagnostics.json", posterior.diagnostics)


def load_posterior(directory: Path) -> PosteriorResults:
    """Load the posterior saved by :func:`save_posterior`.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory holding ``posterior.npz`` (``final/artifacts``).

    Returns
    -------
    PosteriorResults
        The posterior; ``diagnostics`` is empty when ``posterior_diagnostics.json`` is missing.  Optional fields
        saved as None load as 0-d object arrays holding None.
    """
    with np.load(directory / POSTERIOR_FILE, allow_pickle=True) as data:
        diagnostics = json.loads((directory / "posterior_diagnostics.json").read_text()) if (directory / "posterior_diagnostics.json").exists() else {}
        return PosteriorResults(samples=data["samples"], physical=data["physical"], log_prob=data["log_prob"],
                                accept_prob=data["accept_prob"], diverging=data["diverging"], num_steps=data["num_steps"],
                                energy=data["energy"], step_size=data["step_size"],
                                warmup_steps=int(data["warmup_steps"]), sample_steps=int(data["sample_steps"]),
                                num_chains=int(data["num_chains"]), grouped_samples=data["grouped_samples"],
                                diagnostics=diagnostics, median_fit=data["median_fit"], maximum_likelihood_fit=data["maximum_likelihood_fit"])


def load_exact_positions(directory: Path) -> np.ndarray:
    """Load the exact-member positions saved in ``state.json``.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory holding ``state.json`` (``init/artifacts``).

    Returns
    -------
    numpy.ndarray
        Positions into the member arrays of the members evaluated exactly, int32 of shape (exact member,).
    """
    payload = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    return np.asarray(payload["exact_positions"], dtype=np.int32)


def load_search(directory: Path) -> dict[str, Any]:
    """The saved search in the shape chain_starts expects (modes with theta and within_margin, best theta).

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory holding ``search.npz`` and ``modes.json`` (``init/artifacts``).

    Returns
    -------
    dict
        Every array of ``search.npz`` by name (``theta`` as float: the best latent vector, shape (parameter,)), and
        from ``modes.json`` ``modes``, ``method``, ``n_finite``, ``search_seconds``, ``polish_seconds`` and, when
        saved, ``stages``, ``stage_traces``, ``starts_in_basin`` and the ``warm_start*`` entries.
    """
    payload = json.loads((directory / MODES_FILE).read_text(encoding="utf-8"))
    with np.load(directory / SEARCH_FILE) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    return {**arrays, "theta": np.asarray(arrays["theta"], dtype=float), "modes": payload["modes"], "method": payload.get("method"),
            "n_finite": payload.get("n_finite"), "search_seconds": payload.get("search_seconds"), "polish_seconds": payload.get("polish_seconds"),
            **({"stages": payload["stages"], "stage_traces": payload["stage_traces"]} if payload.get("stages") else {}),
            **({"starts_in_basin": payload["starts_in_basin"]} if payload.get("starts_in_basin") is not None else {}),
            **{k: payload[k] for k in ("warm_start", "warm_start_shared", "warm_start_fragile") if k in payload}}


def save_start(directory: Path, theta: np.ndarray, info: dict[str, Any]) -> None:
    """Write ``start.json``: the chains' latent start vector and how it was chosen.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory.
    theta : array_like
        Latent start vector, shape (parameter,).
    info : dict
        Extra JSON-serializable entries stored next to ``"theta"``.
    """
    write_json(directory / START_FILE, {"theta": np.asarray(theta, dtype=float).tolist(), **info})


def load_start(directory: Path) -> np.ndarray:
    """Load the latent start vector saved by :func:`save_start`.

    Parameters
    ----------
    directory : pathlib.Path
        Artifact directory holding ``start.json`` (``init/artifacts``).

    Returns
    -------
    numpy.ndarray
        Latent vector, shape (parameter,).
    """
    return np.asarray(json.loads((directory / START_FILE).read_text(encoding="utf-8"))["theta"], dtype=float)


def init_signature(config: LensClusterSolverConfig) -> dict[str, Any]:
    """The part of the configuration the init stage depends on: model, members, scaling, likelihood, search, seed.

    Parameters
    ----------
    config : LensClusterSolverConfig
        The run configuration.

    Returns
    -------
    dict
        ``config.to_json_dict()`` without the ``sampling``, ``image_diagnostics``, ``truth``, ``image_catalog`` and
        ``paths`` sections, and with ``runtime`` reduced to ``{"seed": ...}``.
    """
    payload = config.to_json_dict()
    for key in ("sampling", "image_diagnostics", "truth", "image_catalog", "paths"):
        payload.pop(key, None)
    payload["runtime"] = {"seed": config.runtime.seed}
    if payload.get("members", {}).get("membership") == "fixed":
        # Inactive membership settings stay out, so init stages saved before they existed still resume.
        for key in ("membership", "membership_default_probability", "membership_max_enumerated", "membership_calibration_prior"):
            payload["members"].pop(key, None)
    return payload


def is_init_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in (INIT_SIGNATURE_FILE, STATE_FILE, CLASSIFICATION_FILE, SEARCH_FILE, MODES_FILE, START_FILE))


def is_final_complete(directory: Path) -> bool:
    return all((directory / name).exists() for name in (CONFIG_PICKLE, POSTERIOR_FILE, AUDIT_FILE))
