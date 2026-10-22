"""Construct complete launcher catalogs without sampling or evaluating likelihoods."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from lenscluster import cluster_solver as cs
from lenscluster.planning import compile_run_plan
from test_smoothed_config import configured


@pytest.mark.parametrize(
    "cluster,image_count,family_count,member_count,has_shared_bin",
    [
        ("ARES", 242, 85, 332, True),
        ("HERA", 65, 19, 332, True),
        ("AS1063", 55, 20, 222, False),
    ],
)
def test_full_catalog_stage0_metadata_alignment(
    tmp_path, cluster, image_count, family_count, member_count, has_shared_bin,
):
    config = configured(cluster)
    _, images, _, _ = cs._declared_model_inputs(config.model)
    labels = images["image_label"].astype(str).to_numpy()
    # Unique measured errors and varied parity expose incorrect label lookups;
    # reverse the sidecar order so correctness also requires joining by label.
    metadata = pd.DataFrame({
        "image_label": labels,
        "sigma_obs_arcsec": 0.51 + 0.001 * np.arange(len(labels)),
        "parity": np.resize(np.array([-1, 0, 1]), len(labels)),
    })
    metadata_path = tmp_path / "metadata.csv"
    metadata.iloc[::-1].to_csv(metadata_path, index=False)
    expected_metadata = metadata.set_index("image_label")
    output_dir = tmp_path / "unused_output"
    config = replace(
        config,
        model=replace(config.model, image_constraints=replace(
            config.model.image_constraints, metadata_path=str(metadata_path))),
        paths=replace(config.paths, output_dir=str(output_dir)),
        runtime=replace(config.runtime, quiet=True, skip_plots=True,
                        skip_validation=True, debug_sampler_diagnostics=False),
    )
    args = cs._clone_args(
        compile_run_plan(config).runtime_args,
        sample_likelihood_mode="smoothed-local-image-plane",
        sampling_engine="tiered_members",
        member_tier_phase="initializer",
        perturbation_discovery_stage0=False,
        refresh_every=None,
    )
    state = cs._build_state_from_inputs(args, fit_mode_override="joint")
    evaluator = cs._build_cluster_evaluator_from_args(args, state)
    data = evaluator.flat_critical_arc_data

    assert len(data.x_obs) == image_count
    assert data.n_families == family_count
    assert len(evaluator.smoothed_member_components) == member_count
    assert (len(state.bin_data) < family_count) == has_shared_bin
    assert np.asarray(data.image_has_constraint).all()
    assert len(evaluator.smoothed_far_positions) == 0
    np.testing.assert_array_equal(
        evaluator.smoothed_absorbed_positions, np.arange(member_count))

    families = {str(family.family_id): family for family in state.family_data}
    recovered_labels = []
    for family_index, family_id in enumerate(data.family_ids):
        rows = np.flatnonzero(
            np.asarray(data.global_family_index_per_image) == family_index)
        family = families[family_id]
        recovered_labels.extend(family.image_labels)
        expected = expected_metadata.loc[list(family.image_labels)]
        assert len(rows) == len(family.image_labels)
        np.testing.assert_array_equal(np.asarray(data.x_obs)[rows], family.x_obs)
        np.testing.assert_array_equal(np.asarray(data.y_obs)[rows], family.y_obs)
        np.testing.assert_allclose(np.asarray(data.sigma_per_image)[rows],
                                   expected["sigma_obs_arcsec"])
        np.testing.assert_array_equal(np.asarray(evaluator.smoothed_parity)[rows],
                                      expected["parity"])
        for axis, indices in (
            ("x", data.global_family_source_x_param_index),
            ("y", data.global_family_source_y_param_index),
        ):
            spec = state.parameter_specs[int(indices[family_index])]
            assert spec.component_family == "source_position"
            assert spec.potential_id == family_id
            assert spec.field == f"beta_{axis}"

    assert len(set(recovered_labels)) == image_count
    assert set(recovered_labels) == set(labels)
    assert not output_dir.exists()
