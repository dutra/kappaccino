from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from lenscluster import cluster_solver as solver




def _sequential_args(tmp_path: Path, **updates: Any) -> argparse.Namespace:
    payload: dict[str, Any] = {
        "output_dir": str(tmp_path), "run_name": "root",
        "fit_mode": solver.FIT_MODE_SEQUENTIAL,
    }
    payload.update(updates)
    return argparse.Namespace(**payload)




def test_fresh_stage_process_error_is_propagated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeQueue:
        def get(self, timeout: float) -> dict[str, str]:
            del timeout
            return {"status": "error", "traceback": "child boom"}

        def close(self) -> None:
            pass

        def join_thread(self) -> None:
            pass

    class FakeProcess:
        exitcode = 0

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FakeContext:
        def Queue(self) -> FakeQueue:
            return FakeQueue()

        def Process(self, **_kwargs: Any) -> FakeProcess:
            return FakeProcess()

    monkeypatch.setattr(solver.multiprocessing, "get_context", lambda _method: FakeContext())
    with pytest.raises(RuntimeError, match="child boom"):
        solver._run_single_stage_in_fresh_process(
            _sequential_args(tmp_path),
            "joint",
            "root/stage2_free_source_forward_fit",
        )


def test_prepare_direct_evaluator_clears_after_initial_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class FakeEvaluator:
        surrogate_enabled = True
        active_scaling_component_indices: list[int] = []
        inactive_scaling_component_indices: list[int] = []
        traced_bin_data: list[Any] = []
        sample_likelihood_mode = solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE
        sampling_engine = solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT
        surrogate_cache_by_z: dict[float, Any] = {}
        scaling_scatter_cache_by_z: dict[float, Any] = {}
        source_metric_cache_by_z: dict[float, Any] = {}
        surrogate_reference_params = None
        flat_surrogate_cache = object()
        scaling_scatter_reference_params = None
        source_metric_reference_params = None
        state = argparse.Namespace(family_data=[], bin_data=[], parameter_specs=[])
        timing_totals = {"initial_jit_compile": 0.0}

        def refresh_surrogate(self, _params: Any, reason: str) -> None:
            events.append(f"surrogate:{reason}")

        def refresh_scaling_scatter_cache(self, _params: Any, reason: str) -> None:
            events.append(f"scaling:{reason}")

        def refresh_source_metric_cache(self, _params: Any, reason: str) -> None:
            events.append(f"source_metric:{reason}")

        def source_loglike(self, _params: Any) -> float:
            events.append("compile")
            return -1.0

    monkeypatch.setattr(solver, "_build_cluster_evaluator_from_args", lambda *_args, **_kwargs: FakeEvaluator())
    monkeypatch.setattr(solver, "_log_evaluator_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(solver.jax, "clear_caches", lambda: events.append("clear"))
    args = argparse.Namespace(jax_clear_caches_after_svi_refresh=True)
    state = argparse.Namespace(parameter_specs=[])

    solver._prepare_direct_evaluator(args, state)

    assert events == [
        "surrogate:svi_nuts_initial",
        "scaling:svi_nuts_initial",
        "source_metric:svi_nuts_initial",
        "clear",
        "compile",
    ]


def test_evaluator_shape_diagnostics_report_one_dimensional_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    bin_data = solver.TracedBinData(
        effective_z_source=1.0,
        family_ids=("1",),
        n_families=1,
        family_index_per_image=jnp.array([0, 0, 0]),
        x_obs=jnp.array([1.0, 2.0, 3.0]),
        y_obs=jnp.array([4.0, 5.0, 6.0]),
        sigma_per_image=jnp.ones(3),
        reliability_per_image=jnp.ones(3),
        image_has_constraint=jnp.array([True, True, True]),
    )
    evaluator = argparse.Namespace(
        traced_bin_data=[bin_data],
        sampling_engine=solver.SAMPLING_ENGINE_REFRESHING_SURROGATE_FLAT,
        sample_likelihood_mode=solver.SAMPLE_LIKELIHOOD_CRITICAL_ARC_MIXTURE_IMAGE_PLANE,
        surrogate_enabled=True,
        active_scaling_component_indices=[1, 2],
        inactive_scaling_component_indices=[3],
        surrogate_cache_by_z={},
        scaling_scatter_cache_by_z={},
        source_metric_cache_by_z={},
        surrogate_reference_params=None,
        scaling_scatter_reference_params=None,
        source_metric_reference_params=None,
    )
    monkeypatch.setattr(solver, "_log", lambda _args, message: messages.append(message))

    solver._log_evaluator_memory_shape_diagnostics(argparse.Namespace(), evaluator, reason="test")

    assert messages
    assert "images=3" in messages[0]
    assert "x_obs_ndim=1" in messages[0]
    assert "y_obs_ndim=1" in messages[0]
    assert "jacobian_surrogate=True" in messages[0]
