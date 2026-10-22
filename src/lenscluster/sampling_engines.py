"""Engine names and compatibility aliases, independent of the JAX runtime."""
from __future__ import annotations

FULL_FLAT = "full_flat"
REFRESHING_SURROGATE_FLAT = "refreshing_surrogate_flat"
TIERED_MEMBERS = "tiered_members"
BAD_LOG_LIKE = -1.0e30
SMOOTHED_LIKELIHOOD = "smoothed-local-image-plane"
SAMPLING_ENGINES = (FULL_FLAT, REFRESHING_SURROGATE_FLAT, TIERED_MEMBERS)


def canonical_sampling_engine(engine: str, likelihood: str) -> str:
    """Resolve the historical smoothed name without changing legacy engines."""
    engine = str(engine)
    if engine == TIERED_MEMBERS and likelihood != SMOOTHED_LIKELIHOOD:
        raise ValueError("tiered_members requires smoothed-local-image-plane.")
    if engine == FULL_FLAT and likelihood == SMOOTHED_LIKELIHOOD:
        return TIERED_MEMBERS
    return engine


def canonical_workflow_engines(workflow: dict) -> dict:
    result = dict(workflow)
    for engine_key, mode_key in (
        ("sampling_engine", "stage0_likelihood"),
        ("stage1_sampling_engine", "stage1_likelihood"),
        ("stage2_sampling_engine", "stage2_forward_mode"),
    ):
        if engine_key in result:
            result[engine_key] = canonical_sampling_engine(result[engine_key], result.get(mode_key, "source"))
    return result


def canonical_runtime_engines(runtime: dict) -> dict:
    result = canonical_workflow_engines({key: value for key, value in runtime.items() if key != "sampling_engine"})
    if "sampling_engine" in runtime:
        mode = runtime.get("sample_likelihood_mode", runtime.get("stage0_likelihood", "source"))
        result["sampling_engine"] = canonical_sampling_engine(runtime["sampling_engine"], mode)
    return result
