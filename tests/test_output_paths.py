from __future__ import annotations

from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from lenscluster.config import RunPathsConfig, TruthRecoveryConfig
from lenscluster.output_paths import (
    RESULTS_LAYOUT, STAGES, debug_log_path, output_paths, prepare_output,
    stage_directory, stage_name, validate_run_label,
)
from lenscluster.planning import compile_run_plan


@pytest.fixture(scope="module")
def launcher():
    source = Path("run.xsh").read_text()
    source = "\n".join(line for line in source.splitlines() if not line.startswith("$"))
    namespace = {"__file__": str(Path("run.xsh").resolve()), "__name__": "launcher_test"}
    exec(compile(source, "run.xsh", "exec"), namespace)
    return namespace


@pytest.fixture
def config(tmp_path, launcher):
    config = launcher["build_config"](
        "HERA", cores=1, campaign_name="test",
        use_magnitude_likelihood=False,
    )
    # Keep these general output and resume tests on a legacy model as well as
    # the dedicated smoothed-likelihood tests in test_smoothed_config.py.
    return replace(
        config, paths=replace(config.paths, output_dir=tmp_path), truth=TruthRecoveryConfig(),
        workflow=replace(config.workflow, stage0_likelihood="source", stage1_likelihood="source",
                         sampling_engine="full_flat", stage1_sampling_engine="full_flat", stage2_sampling_engine="full_flat",
                         stage2_forward_mode="isotropic-latent-displacement",
                         source_position_parameterization="prior-whitened"),
        runtime=replace(config.runtime, dense_mass="full"),
        likelihood=replace(config.likelihood, latent_displacement_sigma_beta_arcsec=0.02,
                           source_plane_covariance_mode="unit"),
    )


def test_launcher_labels(launcher):
    parsed = launcher["_parse_args"](["run.xsh", "hera", "--campaign", "example", "--magnitude-likelihood", "off"])
    assert parsed == dict(cluster="HERA", campaign_name="example", run_label=None, dry_run=False, image_metadata=None)
    config = launcher["build_config"]("HERA", cores=1, campaign_name="example")
    assert Path(config.paths.output_dir) / config.paths.run_name == Path("results/example/hera/smoothed-local-image-plane")
    assert config.paths.layout == RESULTS_LAYOUT


def test_explicit_label(launcher):
    assert launcher["_parse_args"](["run.xsh", "hera", "--run-label", "comparison-2"])["run_label"] == "comparison-2"
    config = launcher["build_config"]("HERA", cores=1, campaign_name="example",
                                      use_magnitude_likelihood=False, run_label="comparison-2")
    assert config.paths.run_name == "comparison-2"


@pytest.mark.parametrize("label", ["", ".", "..", "../escape", "a/b", "a\\b", "/absolute", " spaced ", "a\n"])
def test_invalid_label(launcher, label):
    with pytest.raises(ValueError):
        validate_run_label(label)
    with pytest.raises(SystemExit) as exc:
        launcher["_parse_args"](["run.xsh", "hera", "--run-label", label])
    assert exc.value.code == 2


def test_figure_paths_stay_with_each_stage(config):
    plan = compile_run_plan(config)
    prepare_output(plan)
    root = plan.output.output_dir / plan.output.run_name
    for number in range(3):
        stage = root / f"stages/stage{number}"
        figures = output_paths(stage).figures
        assert figures == stage / "figures"
        figures.mkdir(parents=True)
        (figures / "hera_corner.pdf").write_bytes(str(number).encode())
    for number in range(3):
        assert (root / f"stages/stage{number}/figures/hera_corner.pdf").read_bytes() == str(number).encode()


@pytest.mark.parametrize("stage2", [False, True])
def test_output_tree_and_completion(config, stage2):
    from lenscluster import cluster_solver as solver, plotting
    config = replace(config, workflow=replace(config.workflow, stage2_forward_mode="isotropic-latent-displacement" if stage2 else "none"))
    if not stage2:
        schedule = config.schedule
        config = replace(config, schedule=replace(
            schedule, svi_steps=schedule.svi_steps[:2], refresh_every=schedule.refresh_every[:2],
            **{name: getattr(schedule, name)[:1] for name in ("fit_method", "warmup", "samples", "sampling_refresh_runs", "max_tree_depth")},
        ))
    plan = compile_run_plan(config)
    prepare_output(plan)
    root = plan.output.output_dir / plan.output.run_name
    manifest = json.loads((root / "config.json").read_text())
    assert manifest["final_stage"] == STAGES["stage2" if stage2 else "stage1"]
    assert manifest["resolved_stages"][-1]["sample_likelihood_mode"] == plan.stages[-1].sample_likelihood_mode
    final = stage_directory(root, manifest["final_stage"])
    for stage in plan.stages:
        directory = stage_directory(root, stage.name)
        paths = output_paths(directory)
        assert stage_name(directory) == stage.name
        assert paths.products == (root if directory == final else directory)
        assert paths.figures == directory / "figures"
        assert plotting.plot_path(directory, "corner.pdf") == paths.figures / "hera_corner.pdf"
        assert paths.artifacts == directory / "artifacts"
        paths.artifacts.mkdir()
        (paths.artifacts / "plot_bundle.h5").touch()
        assert solver._stage_run_checkpointed(directory)
        paths.tables.mkdir(exist_ok=True)
        (paths.tables / "run_summary.json").write_text('{}')
        assert solver._stage_run_complete(directory)
        assert not solver._stage_run_checkpointed(directory)
        assert debug_log_path(plan.runtime_args, str(directory.relative_to(plan.output.output_dir)), None) == root / "logs" / f"{directory.name}.log"
    assert not (final / "tables").exists()
    assert debug_log_path(plan.runtime_args, plan.output.run_name, None) == root / "logs/workflow.log"
    assert solver._previous_stage_artifacts_dir_for_run_dir(final) == final.parent / ("stage1" if stage2 else "stage0") / "artifacts"
    assert solver._plots_only_exact_diagnostics_stage(final) == manifest["final_stage"]


def test_resume_identity(config):
    plan = compile_run_plan(config)
    prepare_output(plan)
    prepare_output(plan)
    operational = replace(config, runtime=replace(config.runtime, quiet=True, skip_plots=True, plots_only=True,
                                                    resume="fast", jax_default_device="cpu"),
                          image_diagnostics=replace(config.image_diagnostics, posterior_image_diagnostic_draws=5))
    prepare_output(compile_run_plan(operational))
    for changed, field in [
        (replace(config, likelihood=replace(config.likelihood, pos_sigma_arcsec=0.9)), "pos_sigma_arcsec"),
        (replace(config, schedule=replace(config.schedule, samples=(20, 20))), "samples"),
        (replace(config, runtime=replace(config.runtime, seed=17)), "seed"),
        (replace(config, image_diagnostics=replace(config.image_diagnostics, match_tolerance_arcsec=3.0)), "match_tolerance_arcsec"),
    ]:
        with pytest.raises(ValueError, match=field):
            prepare_output(compile_run_plan(changed))


def test_changed_input_rejected_before_logging(config, tmp_path):
    from lenscluster.runner import LensClusterRunner
    source = Path(config.model.image_constraints.catalog_path)
    catalog = tmp_path / "images.cat"
    catalog.write_bytes(source.read_bytes())
    config = replace(config, model=replace(config.model, image_constraints=replace(config.model.image_constraints, catalog_path=catalog)))
    plan = compile_run_plan(config)
    prepare_output(plan)
    catalog.write_text(catalog.read_text() + "\n")
    with pytest.raises(ValueError, match="input_fingerprints"):
        LensClusterRunner().run(plan)
    assert not list((tmp_path / config.paths.run_name / "logs").iterdir())


def test_unknown_directory_is_not_adopted(config):
    root = Path(config.paths.output_dir) / config.paths.run_name
    root.mkdir()
    (root / "unrelated.txt").write_text("keep")
    with pytest.raises(ValueError, match="nonempty"):
        prepare_output(compile_run_plan(config))
    assert (root / "unrelated.txt").read_text() == "keep"


def _spawn_resolve(queue, runtime):
    root = Path(runtime.output_dir) / runtime.run_name
    directory = stage_directory(root, STAGES["stage2"])
    queue.put((str(output_paths(directory).tables), stage_name(directory)))


def test_fresh_process_reads_manifest(config):
    plan = compile_run_plan(config)
    prepare_output(plan)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_spawn_resolve, args=(queue, plan.runtime_args))
    process.start()
    try:
        assert queue.get(timeout=45) == (str(Path(config.paths.output_dir) / config.paths.run_name / "tables"), STAGES["stage2"])
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()


@pytest.mark.parametrize("modern", [False, True])
def test_warm_start_and_plots_only(config, monkeypatch, tmp_path, modern):
    import h5py
    from lenscluster import cluster_solver as solver, multi_cluster_solver as multi
    if not modern:
        config = replace(config, paths=replace(config.paths, layout="legacy"))
    plan = compile_run_plan(config)
    prepare_output(plan)
    root = Path(config.paths.output_dir) / config.paths.run_name
    for name in (STAGES["stage1"], STAGES["stage2"]):
        artifacts = stage_directory(root, name) / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        with h5py.File(artifacts / "plot_bundle.h5", "w") as handle:
            solver._write_h5_json(handle, "runtime_args_json", {
                key: value for key, value in plan.runtime_args.values.items()
                if key == "scaling_relation_mode" or key.startswith("potfile_")
            })
    par = tmp_path / "input.par"
    par.touch()
    warm = multi._resolve_warm_stage(multi.ClusterInput("hera", par, root), "auto")
    assert warm.stage_artifacts_dir == stage_directory(root, STAGES["stage2"]) / "artifacts"
    seen = []
    monkeypatch.setattr(solver, "_rerender_plots", lambda args, path, **kwargs: seen.append(path))
    plotting = compile_run_plan(replace(config, runtime=replace(config.runtime, plots_only=True)))
    solver._run_typed_plan_dispatch(plotting.runtime_args, {})
    assert seen == [stage_directory(root, STAGES[key]) for key in ("stage1", "stage2")]
    if not modern:
        assert output_paths(seen[-1]).tables == seen[-1] / "tables"
        assert not (root / "config.json").exists()


def test_logs_do_not_duplicate_workflow(config):
    from lenscluster.utils import configure_debug_log, debug_log_line, close_debug_log
    plan = compile_run_plan(config)
    prepare_output(plan)
    configure_debug_log(plan.runtime_args, plan.output.run_name)
    debug_log_line("workflow-only-marker")
    stage = stage_directory(Path(config.paths.output_dir) / config.paths.run_name, STAGES["stage1"])
    path = configure_debug_log(plan.runtime_args, str(stage.relative_to(plan.output.output_dir)))
    debug_log_line("stage-only-marker")
    close_debug_log()
    assert "stage-only-marker" in path.read_text()
    assert "workflow-only-marker" not in path.read_text()


def test_synthetic_sequential_outputs_and_resume(config, monkeypatch):
    """Exercise real stage orchestration and writers with inference replaced."""
    import numpy as np
    from astropy.wcs import WCS
    from lenscluster import cluster_solver as solver, plotting
    from lenscluster.runner import _stage_fit_controls_from_plan
    from lenscluster.utils import close_debug_log

    config = replace(config, workflow=replace(config.workflow, stage2_fresh_process=False))
    plan = compile_run_plan(config)
    prepare_output(plan)
    controls = _stage_fit_controls_from_plan(plan, solver)
    inferred, rendered = [], []

    def build(args, **kwargs):
        return SimpleNamespace(run_name=args.run_name, par_path="", potfiles=[], parsed={})

    def products(args, directory):
        paths = output_paths(directory)
        paths.tables.mkdir(parents=True, exist_ok=True)
        (paths.tables / "run_summary.json").write_text(json.dumps({"run_name": args.run_name, "sampler": "synthetic"}))
        plotting._write_truth_grid_summary(directory, [{"quantity": "kappa", "value": 1.0}])
        plotting._write_truth_grid_quantile_fits(directory, WCS(naxis=2), {"kappa": {"median": np.ones((2, 2))}}, suffixes=("median",))
        fig, ax = plotting.plt.subplots()
        ax.plot([0, 1], [0, 1])
        plotting._finish_figure(fig, plotting.plot_path(directory, "smoke.pdf"))

    def infer(args, state, directory):
        inferred.append(directory)
        output_paths(directory).artifacts.mkdir()
        (output_paths(directory).artifacts / "plot_bundle.h5").touch()
        products(args, directory)

    def rerender(args, directory, **kwargs):
        rendered.append(directory)
        products(args, directory)

    monkeypatch.setattr(solver, "_build_state_from_inputs", build)
    monkeypatch.setattr(solver, "_apply_previous_stage_frozen_active_scaling", lambda args, state, *a, **kw: state)
    monkeypatch.setattr(solver, "_log_state_summary", lambda *a: None)
    monkeypatch.setattr(solver, "_run_inference", infer)
    monkeypatch.setattr(solver, "_rerender_plots", rerender)
    monkeypatch.setattr(solver, "_stage_run_cosmology_compatible", lambda *a: (True, ""))
    for name in ("_physical_best_fit_values_from_artifacts", "_source_position_prior_values_from_artifacts", "_member_shape_values_from_artifacts"):
        monkeypatch.setattr(solver, name, lambda *a: {})
    monkeypatch.setattr(solver, "_selected_independent_by_potfile_from_artifacts", lambda *a: [])
    monkeypatch.setattr(solver, "_audit_frozen_independent_selection", lambda *a: None)
    try:
        solver._run_sequential_v2(plan.runtime_args, controls)
        root = plan.output.output_dir / plan.output.run_name
        assert [path.name for path in inferred] == ["stage0", "stage1", "stage2"]
        assert (root / "summary.txt").is_file()
        assert (root / "tables/sequential_summary.json").is_file()
        for stage in ("stage0", "stage1", "stage2"):
            assert (root / "stages" / stage / "figures/hera_smoke.pdf").is_file()
        assert not (root / "figures").exists()
        assert (root / "fits/truth_recovery_kappa_model_median.fits").is_file()
        assert (root / "stages/stage2/inputs/input_manifest.json").is_file()
        assert not (root / "stages/stage2/tables").exists()
        assert (root / "stages/stage2/figures").is_dir()
        assert not (root / "stages/stage2/fits").exists()
        # Simulate interruption after the final checkpoint, then resume.
        (root / "tables/run_summary.json").unlink()
        solver._run_sequential_v2(plan.runtime_args, controls)
        assert len(inferred) == 3
        assert [path.name for path in rendered] == ["stage0", "stage1", "stage2"]
        assert (root / "tables/run_summary.json").is_file()
        rendered.clear()
        fast = compile_run_plan(replace(config, runtime=replace(config.runtime, resume="fast")))
        solver._run_sequential_v2(fast.runtime_args, controls)
        assert [path.name for path in rendered] == ["stage2"]
        assert len(inferred) == 3
    finally:
        close_debug_log()


def test_saved_run_readers(config):
    from lenscluster.artifacts import artifacts_dir, _resolve_parameter_artifact_path
    from lenscluster.mock_validation.runner import _validation_stage_dirs, _validation_stage_manifest
    plan = compile_run_plan(config)
    prepare_output(plan)
    root = plan.output.output_dir / plan.output.run_name
    for key in ("stage0", "stage1", "stage2"):
        directory = stage_directory(root, STAGES[key])
        paths = output_paths(directory)
        paths.artifacts.mkdir()
        (paths.artifacts / "plot_bundle.h5").touch()
        paths.tables.mkdir(exist_ok=True)
        (paths.tables / "run_summary.json").write_text('{}')
    final = root / "stages/stage2"
    assert artifacts_dir(root) == final / "artifacts"
    assert _resolve_parameter_artifact_path(root) == final / "artifacts/plot_bundle.h5"
    assert _validation_stage_dirs(root) == [(STAGES[key], stage_directory(root, STAGES[key]))
                                            for key in ("stage0", "stage1", "stage2")]
    assert _validation_stage_dirs(final) == _validation_stage_dirs(root)
    manifest = _validation_stage_manifest(root)
    assert manifest[-1]["tables_dir"] == root / "tables"
    assert manifest[-1]["stage"] == STAGES["stage2"]
