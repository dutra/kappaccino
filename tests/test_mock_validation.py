"""Mock generation, truth bookkeeping and aperture profiles without running inference."""
from __future__ import annotations

import json

import numpy as np

from lenscluster.config import LensClusterSolverConfig, LikelihoodConfig
from lenscluster.mock_validation import MockValidationConfig, SingleBCGMockConfig, generate_single_bcg_mock, solver_config_for_mock
from lenscluster.mock_validation.aperture import APERTURE_RADII_KPC, aperture_profiles, truth_groups
from lenscluster.mock_validation.runner import mock_paths, parameter_truth
from lenscluster.state import build_state


def test_small_mock_truth_and_solver_state(tmp_path):
    mock = SingleBCGMockConfig(seed=11, n_primary_families=2, n_subhalo_families=0, n_subhalos=3, pos_sigma_arcsec=0.05, min_images_per_family=3)
    paths, images, truth = generate_single_bcg_mock(tmp_path / "mock", mock)
    assert paths.truth_path.is_file() and paths.image_catalog_path.is_file()
    assert images["family_id"].nunique() == 2
    truth = json.loads(paths.truth_path.read_text())
    values = parameter_truth(truth)
    for key in ("halo.v_disp", "bcg.v_disp", "potfile.sigma", "potfile.alpha_sigma", "potfile.beta_radius", "halo.log10_mass_msun",
                "potfile.log10_mass_msun", "potfile.velocity_log_scatter", "source.sigma_src", "source.1.beta_x"):
        assert key in values and np.isfinite(values[key])
    profiles = aperture_profiles(np.asarray(APERTURE_RADII_KPC), truth_groups(truth))
    assert set(profiles) == {"halo", "bcg", "members", "total"}
    assert np.all(np.diff(profiles["total"]) >= 0) and profiles["total"][-1] > 0
    np.testing.assert_allclose(profiles["total"], profiles["halo"] + profiles["bcg"] + profiles["members"], rtol=1e-10)

    config = MockValidationConfig(mock=mock, solver=LensClusterSolverConfig(likelihood=LikelihoodConfig(pos_sigma_arcsec=0.05))).validate()
    solver = solver_config_for_mock(config, paths=mock_paths(paths.root), seed=11, output_dir=tmp_path / "run").validate()
    state = build_state(solver)
    assert state.images.n_images == len(images) and len(state.members.member_ids) == 3
    names = {spec.name for spec in state.parameter_specs}
    assert {"halo.v_disp", "bcg.v_disp", "potfile.sigma", "potfile.cutkpc", "potfile.alpha_sigma", "potfile.beta_radius"} <= names
