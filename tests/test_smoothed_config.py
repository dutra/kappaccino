"""Configuration, measured-image metadata, and nonexecuting launcher checks."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lenscluster.config import LogNormalPriorConfig
from lenscluster.lenstool_parser import _load_multiple_images_catalog, _merge_image_metadata
from lenscluster.output_paths import _fingerprints, _identity, prepare_output
from lenscluster.planning import compile_run_plan


def launcher_namespace():
    launcher = Path("run.xsh").resolve()
    namespace = {"__file__": str(launcher), "__name__": "launcher_test"}
    text = "\n".join(line for line in launcher.read_text().splitlines() if not line.startswith("$"))
    exec(compile(text, str(launcher), "exec"), namespace)
    return namespace


def configured(cluster="HERA", **kwargs):
    return launcher_namespace()["build_config"](cluster, cores=1, campaign_name="test-smoothed", **kwargs)


@pytest.mark.parametrize('value', [None, 0, 3, 'all', 'ALL', -1, True, 1.5])
@pytest.mark.parametrize('axis', ['top_sampled', 'top_exact'])
def test_top_member_config(value, axis):
    config = configured()
    config = replace(config, member_tiers=replace(config.member_tiers, **{axis:value}))
    if value != 'all' and value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        with pytest.raises(ValueError, match=axis):
            compile_run_plan(config)
    else:
        assert getattr(compile_run_plan(config).runtime_args, f'member_tier_{axis}') == value


def test_legacy_none_top_limits_have_same_manifest_identity():
    base = configured()
    config = replace(base, member_tiers=replace(base.member_tiers, top_sampled=None, top_exact=None)).to_json_dict()
    legacy = json.loads(json.dumps(config))
    for key in ('top_sampled', 'top_exact'):
        legacy['member_tiers'].pop(key)
    assert _identity(config) == _identity(legacy)
    config['member_tiers']['top_exact'] = 0
    assert _identity(config) != _identity(legacy)


@pytest.mark.parametrize("cluster", ["HERA", "ARES", "AS1063"])
def test_launcher_resolves_full_likelihood_and_fixed_priors(cluster):
    config = configured(cluster)
    plan = compile_run_plan(config)
    assert [stage.sample_likelihood_mode for stage in plan.stages] == ["smoothed-local-image-plane"] * 3
    assert [stage.member_tier_phase for stage in plan.stages] == ["initializer", "fixed", "fixed"]
    assert all(stage.source_position_policy == "sampled" for stage in plan.stages)
    assert all(stage.sampling_engine == "tiered_members" for stage in plan.stages)
    assert config.workflow.source_position_parameterization == "likelihood-whitened"
    assert config.runtime.dense_mass == "lens-and-hyperparameters"
    assert config.likelihood.source_scatter_prior == LogNormalPriorConfig(0.02, 0.7)
    assert config.scaling.member_velocity_scatter_prior == LogNormalPriorConfig(0.02, 1.0)
    assert plan.runtime_args.source_position_prior_sigma_arcsec == 100.0
    assert plan.runtime_args.source_position_prior_mean_x_arcsec == 0.0
    assert plan.runtime_args.source_position_prior_mean_y_arcsec == 0.0
    assert plan.runtime_args.member_tier_scatter_deflection_fraction == pytest.approx(np.sqrt(0.1))
    assert plan.runtime_args.member_tier_absorbed_variance_fraction == 0.25
    assert config.model.image_constraints.sigma_floor_arcsec == 0.5
    assert plan.runtime_args.source_plane_covariance_floor == 0.0
    assert config.schedule.refresh_every == (None, None, None)


@pytest.mark.parametrize("change,message", [
    ("magnitude", "positions only"),
    ("surrogate", "tiered_members"),
    ("whitening", "likelihood-whitened"),
    ("prior", "positive"),
    ("covariance_floor", "covariance_floor"),
    ("revision", "revision"),
])
def test_smoothed_configuration_rejects_incompatible_settings(change, message):
    config = configured()
    if change == "magnitude":
        config = replace(config, likelihood=replace(config.likelihood, use_magnitude_likelihood=True))
    elif change == "surrogate":
        config = replace(config, workflow=replace(config.workflow, stage2_sampling_engine="refreshing_surrogate_flat"))
    elif change == "whitening":
        config = replace(config, workflow=replace(config.workflow, source_position_parameterization="prior-whitened"))
    elif change == "prior":
        config = replace(config, likelihood=replace(config.likelihood, source_scatter_prior=LogNormalPriorConfig(0.0, 0.7)))
    elif change == "covariance_floor":
        config = replace(config, likelihood=replace(config.likelihood, source_plane_covariance_floor=1e-6))
    else:
        config = replace(config, likelihood=replace(config.likelihood, smoothed_likelihood_revision="old"))
    with pytest.raises(ValueError, match=message):
        compile_run_plan(config)


def test_direct_coordinates_remain_available_for_reference_checks():
    config = configured()
    compile_run_plan(replace(config, workflow=replace(config.workflow, source_position_parameterization="direct")))


@pytest.mark.parametrize("field,value", [
    ("stage0_likelihood", "source"),
    ("stage1_likelihood", "source"),
    ("stage2_forward_mode", "none"),
    ("stage2_forward_mode", "isotropic-latent-displacement"),
    ("fit_mode", "joint"),
])
def test_smoothed_mode_requires_all_three_stages(field, value):
    config = configured()
    # A legacy layout permits joint mode generally; the smoothed contract must
    # reject it before a plan could silently choose the source-plane likelihood.
    config = replace(config, paths=replace(config.paths, layout="legacy"),
                     workflow=replace(config.workflow, **{field: value}))
    with pytest.raises(ValueError, match="all three stages"):
        compile_run_plan(config)


def test_metadata_uses_image_labels_and_measured_errors(tmp_path):
    path = tmp_path / "metadata.csv"
    path.write_text("image_label,sigma_obs_arcsec,parity\n2.a,0.8,-1\n1.a,0.1,1\n1.b,,\n")
    images = pd.DataFrame({"image_label": ["1.a", "1.b", "2.a", "2.b"]})
    result = _merge_image_metadata(images, metadata_path=path, sigma_arcsec=0.6, sigma_floor_arcsec=0.5)
    np.testing.assert_array_equal(result["image_label"], images["image_label"])
    np.testing.assert_allclose(result["sigma_obs_arcsec"], [0.5, 0.6, 0.8, 0.6])
    np.testing.assert_array_equal(result["parity"], [1, 0, -1, 0])
    assert "parity" not in images


@pytest.mark.parametrize("contents,message", [
    ("image_label,parity\n1.a,1\n1.a,-1\n", "unique"),
    ("image_label,parity\n9.z,1\n", "unknown"),
    ("image_label,parity\n1.a,2\n", "Invalid parity"),
    ("image_label,parity\n1.a,NaN\n", "Invalid parity"),
    ("image_label,sigma_obs_arcsec\n1.a,0\n", "Invalid sigma_obs"),
    ("image_label,sigma_obs_arcsec\n1.a,inf\n", "Invalid sigma_obs"),
    ("image_label,sigma_obs_arcsec\n1.a,bad\n", "Invalid sigma_obs"),
])
def test_invalid_metadata_fails_before_execution(tmp_path, contents, message):
    path = tmp_path / "metadata.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        _merge_image_metadata(pd.DataFrame({"image_label": ["1.a"]}), metadata_path=path,
                              sigma_arcsec=0.5, sigma_floor_arcsec=0.5)


def test_catalog_parser_propagates_metadata_without_interpreting_morphology(tmp_path):
    config = configured()
    constraints = config.model.image_constraints
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("image_label,sigma_obs_arcsec,parity\n1.a,0.7,-1\n")
    images = _load_multiple_images_catalog(constraints.catalog_path, (3, 0.0, 0.0),
                                          metadata_path=metadata, sigma_arcsec=0.5, sigma_floor_arcsec=0.5)
    assert images.loc[images.image_label == "1.a", "sigma_obs_arcsec"].item() == 0.7
    assert images.loc[images.image_label == "1.a", "parity"].item() == -1
    assert set(images.loc[images.image_label != "1.a", "sigma_obs_arcsec"]) == {0.5}
    assert set(images.loc[images.image_label != "1.a", "parity"]) == {0}


def test_metadata_and_algorithm_revision_are_resume_identity(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("image_label,parity\n1.a,1\n")
    config = configured(image_metadata=str(metadata))
    config = replace(config, paths=replace(config.paths, output_dir=str(tmp_path), run_name="run"))
    plan = compile_run_plan(config)
    prepare_output(plan)
    manifest = tmp_path / "run/config.json"
    original = manifest.read_bytes()
    metadata.write_text("image_label,parity\n1.a,-1\n")
    with pytest.raises(ValueError, match="incompatible"):
        prepare_output(plan)
    assert manifest.read_bytes() == original
    original_config = config.to_json_dict()
    changed_config = json.loads(json.dumps(original_config))
    changed_config["likelihood"]["smoothed_likelihood_revision"] = "future"
    assert _identity(original_config) != _identity(changed_config)


def test_legacy_identity_does_not_change_from_unused_new_defaults():
    config = configured().to_json_dict()
    config["workflow"].update(stage0_likelihood="source", stage1_likelihood="source", stage2_forward_mode="none",
                              sampling_engine="full_flat", stage1_sampling_engine="full_flat", stage2_sampling_engine="full_flat")
    config["model"]["image_constraints"]["sigma_floor_arcsec"] = 0.0
    before = json.loads(json.dumps(config))
    before.pop("member_tiers")
    before["scaling"].pop("member_velocity_scatter_prior")
    for key in ("source_scatter_prior", "source_position_prior_mean_arcsec", "source_position_prior_sigma_arcsec", "smoothed_likelihood_revision"):
        before["likelihood"].pop(key)
    before["model"]["image_constraints"].pop("metadata_path")
    before["model"]["image_constraints"].pop("sigma_floor_arcsec")
    assert _identity(before) == _identity(config)


def test_metadata_and_optional_sidecar_are_fingerprinted(tmp_path):
    config = configured().to_json_dict()
    catalog = tmp_path / "images.cat"
    catalog.write_text("#REFERENCE 3\n")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("image_label,parity\n")
    config["model"]["image_constraints"].update(catalog_path=str(catalog), metadata_path=str(metadata))
    before = _fingerprints(config)
    assert "sha256" in before["model.image_constraints.metadata_path"]
    assert before["model.image_constraints.band_magnitudes"] == {"missing": True}
    catalog.with_name("images_band_magnitudes.csv").write_text("image_label,mag_F160W\n")
    after = _fingerprints(config)
    assert "sha256" in after["model.image_constraints.band_magnitudes"]
    assert before != after


def test_dry_run_never_prepares_or_runs_output(tmp_path, monkeypatch, capsys):
    namespace = launcher_namespace()
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run attempted inference or output creation")
    monkeypatch.setattr(namespace["LensClusterRunner"], "run", forbidden)
    import lenscluster.output_paths as output_paths
    monkeypatch.setattr(output_paths, "prepare_output", forbidden)
    namespace["configure_cpu_runtime"] = forbidden
    campaign = "smoothed-dry-run-must-not-exist"
    assert not Path("results", campaign).exists()
    assert namespace["main"](["run.xsh", "HERA", "--campaign", campaign, "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["cpu_plan"]["chains"] == report["config"]["runtime"]["chains"]
    assert report["cpu_plan"]["jax_cpu_devices"] == report["config"]["runtime"]["chains"]
    assert report["images"] == 65
    assert report["known_parities"] == 0
    assert report["sigma_obs_arcsec_range"] == [0.5, 0.5]
    assert not Path("results", campaign).exists()


def test_launcher_rejects_flux_before_building_configuration(capsys):
    namespace = launcher_namespace()
    with pytest.raises(SystemExit) as error:
        namespace["main"](["run.xsh", "HERA", "--magnitude-likelihood", "on", "--dry-run"])
    assert error.value.code == 2
    assert "positions only" in capsys.readouterr().out
