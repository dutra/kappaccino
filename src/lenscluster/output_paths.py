"""Versioned output locations shared by writers and saved-run readers.

Stage directories always own inputs and checkpoints. In results-v1 only the
configured final stage writes scientific products at the run root.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

RESULTS_LAYOUT = "results-v1"
STAGES = {
    "stage0": "stage0_fast_initializer",
    "stage1": "stage1_backprojected_centroid_fit",
    "stage2": "stage2_free_source_forward_fit",
}


def validate_run_label(label: str) -> str:
    if not label or label.strip() != label or label in {".", ".."} or any(c in label for c in "/\\\0\n\r"):
        raise ValueError("run label must be a nonempty single directory name")
    return label


def _manifest(root: Path) -> dict[str, Any] | None:
    path = root / "config.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("layout_version") == RESULTS_LAYOUT else None


def is_results_root(root: Path) -> bool:
    return _manifest(root) is not None


def stage_name(path: str | Path) -> str:
    path = Path(path)
    if path.parent.name == "stages":
        manifest = _manifest(path.parent.parent)
        if manifest:
            for semantic, relative in manifest["stages"].items():
                if Path(relative).name == path.name:
                    return semantic
        # Relative run names are also passed by the solver before state creation.
        return STAGES.get(path.name, path.name)
    return path.name


def stage_directory(root: Path, semantic: str, *, layout: str | None = None) -> Path:
    manifest = _manifest(root)
    if manifest:
        relative = manifest["stages"].get(semantic)
        if relative is not None:
            return root / relative
        layout = RESULTS_LAYOUT
    if layout == RESULTS_LAYOUT:
        key = next(key for key, name in STAGES.items() if name == semantic)
        return root / "stages" / key
    return root / semantic


@dataclass(frozen=True)
class OutputPaths:
    stage: Path
    root: Path
    products: Path
    modern: bool

    @property
    def figures(self) -> Path:
        return self.products / "figures" if self.modern else self.stage

    @property
    def figure_prefix(self) -> str | None:
        manifest = _manifest(self.root) if self.modern else None
        return manifest.get("figure_prefix") if manifest else None

    @property
    def tables(self) -> Path:
        return self.products / "tables"

    @property
    def fits(self) -> Path:
        return self.products / "fits"

    @property
    def artifacts(self) -> Path:
        return self.stage / "artifacts"


def output_paths(stage: str | Path) -> OutputPaths:
    stage = Path(stage)
    if stage.parent.name == "stages":
        root = stage.parent.parent
        manifest = _manifest(root)
        if manifest:
            semantic = stage_name(stage)
            products = root if semantic == manifest["final_stage"] else stage
            return OutputPaths(stage, root, products, True)
    return OutputPaths(stage, stage, stage, False)


def debug_log_path(args: Any, run_name: str, run_dir: Path | None) -> Path:
    stage = run_dir if run_dir is not None else Path(args.output_dir) / run_name
    paths = output_paths(stage)
    if paths.modern:
        return paths.root / "logs" / f"{stage.name}.log"
    if _manifest(stage):
        return stage / "logs" / "workflow.log"
    return (Path(args.output_dir) / f"{run_name}.debug.log") if run_dir is None else stage / "run_debug.log"


def _fingerprints(config: dict[str, Any]) -> dict[str, Any]:
    fingerprints: dict[str, Any] = {}

    def visit(value: Any, key: str = "", location: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, child_key, f"{location}.{child_key}".strip("."))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, key, f"{location}[{index}]")
        elif value and (key == "catalog_path" or key.endswith("_fits") or key == "corner_overlay_bayes_dat"):
            path = Path(value)
            if path.is_file():
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                fingerprints[location] = {"sha256": digest}
            else:
                fingerprints[location] = {"missing": True}

    visit(config)
    return fingerprints


def _identity(config: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    result["figure_prefix"] = result.pop("paths", {}).get("figure_prefix")
    # Rendering and execution policy may change without invalidating samples.
    for section in ("image_catalog", "truth"):
        result.pop(section, None)
    for key in ("posterior_image_diagnostic_draws", "posterior_image_diagnostic_mode"):
        result["image_diagnostics"].pop(key, None)
    result["runtime"] = {key: value for key, value in result["runtime"].items()
                         if key in {"seed", "chains", "dense_mass"}}
    result["workflow"].pop("stage2_fresh_process", None)
    result["workflow"].pop("exact_image_diagnostics_stage2", None)
    result["workflow"].pop("exact_image_diagnostics_stage3", None)
    return result


def _differences(old: Any, new: Any, prefix: str = "") -> list[str]:
    if isinstance(old, dict) and isinstance(new, dict):
        return [item for key in sorted(old.keys() | new.keys())
                for item in _differences(old.get(key), new.get(key), f"{prefix}.{key}".strip("."))]
    return [] if old == new else [prefix]


def prepare_output(plan: Any) -> None:
    """Validate identity before any log, checkpoint, or inference output is written."""
    if plan.config.paths.layout != RESULTS_LAYOUT:
        return
    root = plan.output.output_dir / plan.output.run_name
    config = plan.config.to_json_dict()
    payload = {
        "layout_version": RESULTS_LAYOUT,
        "config": config,
        "figure_prefix": plan.config.paths.figure_prefix,
        "identity": _identity(config),
        "input_fingerprints": _fingerprints(config),
        "stages": {stage.name: f"stages/{key}" for stage in plan.stages
                   for key, semantic in STAGES.items() if semantic == stage.name},
        "final_stage": plan.stages[-1].name,
        "resolved_stages": [{key: value for key, value in asdict(stage).items() if key != "output_plan"}
                            for stage in plan.stages],
    }
    path = root / "config.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        changes = _differences(
            {key: saved.get(key) for key in ("layout_version", "identity", "input_fingerprints", "stages", "final_stage")},
            {key: payload[key] for key in ("layout_version", "identity", "input_fingerprints", "stages", "final_stage")},
        )
        if changes:
            raise ValueError("Run label already contains incompatible results. Use a different --run-label. Changed: " + ", ".join(changes))
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to reuse nonempty run directory without config.json: {root}. Use a different --run-label.")
    root.mkdir(parents=True, exist_ok=True)
    # Publish a complete manifest atomically, without replacing a competing run.
    with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=".config-", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()
    for directory in ("figures", "tables", "fits", "logs", *payload["stages"].values()):
        (root / directory).mkdir(parents=True, exist_ok=True)


def resolve_artifacts_dir(run_dir: str | Path) -> Path:
    """Accept a checkpoint directory, a stage directory, or a results run root."""
    path = Path(run_dir)
    if path.name == "artifacts":
        return path
    manifest = _manifest(path)
    if manifest:
        candidates = [path / relative / "artifacts" for relative in manifest["stages"].values()]
        for candidate in reversed(candidates):
            if (candidate / "plot_bundle.h5").exists() or (candidate / "posterior_arrays.npz").exists():
                return candidate
        return path / manifest["stages"][manifest["final_stage"]] / "artifacts"
    return path / "artifacts"
