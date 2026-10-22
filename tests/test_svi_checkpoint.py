"""SVI samples survive final-posterior replacement in reloadable physical units."""
from dataclasses import replace

import h5py
import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from lenscluster.planning import compile_run_plan
from test_latent_displacement import configured, MODE


@pytest.mark.parametrize('stage', ['stage0', 'stage1', 'stage2'])
def test_svi_checkpoint_roundtrip_and_final_independence(tmp_path, stage):
    config = configured()
    config = replace(config, likelihood=replace(
        config.likelihood, sample_latent_displacement_sigma_beta=True))
    args = cs._clone_args(compile_run_plan(config).runtime_args,
                         sample_likelihood_mode=MODE, sampling_engine='full_flat')
    initial = cs._build_state_from_inputs(
        cs._clone_args(args, sample_likelihood_mode='source'), fit_mode_override='joint')
    centers = {str(f.family_id): (0., 0.) for f in initial.family_data}
    state = cs._build_state_from_inputs(args, fit_mode_override='joint',
                                       source_position_prior_values=centers)
    evaluator = cs.ClusterJAXEvaluator(state, .1, sampling_engine='full_flat',
                                     sample_likelihood_mode=MODE,
                                     latent_displacement_sigma_beta_arcsec=.03)
    center = cs._default_theta(state.parameter_specs)
    draws = np.stack([center.copy(), center.copy()])
    idx = next(i for i, s in enumerate(state.parameter_specs)
               if s.sample_name == 'source_sigma_beta')
    draws[1, idx] += np.log(2)
    posterior = cs.PosteriorResults(
        samples=draws, log_prob=np.array([-2., -3.]),
        accept_prob=np.empty(0), diverging=np.empty(0, dtype=bool), num_steps=np.empty(0),
        warmup_steps=0, sample_steps=2, num_chains=0, sampler='svi',
        init_diagnostics={'svi_steps': 20})
    run_dir = tmp_path / 'stages' / stage
    cs._save_svi_checkpoint(args, state, evaluator, run_dir, center, posterior)
    saved_dir = run_dir / 'artifacts' / 'svi'
    path = saved_dir / 'plot_bundle.h5'
    original_bytes = path.read_bytes()
    rebuilt, _, arrays, diagnostics = cs._load_artifacts(saved_dir)
    np.testing.assert_allclose(arrays['samples'][:, idx], [.03, .06])
    np.testing.assert_allclose(arrays['best_fit'][idx], .03)
    np.testing.assert_allclose(arrays['log_prob'], [-2., -3.])
    assert [s.sample_name for s in rebuilt.parameter_specs] == [s.sample_name for s in state.parameter_specs]
    assert diagnostics['best_value_selected'] == 'svi_guide_center'
    assert posterior.init_diagnostics == {'svi_steps': 20}
    np.testing.assert_array_equal(posterior.samples, draws)
    with h5py.File(path) as f:
        assert f['posterior'].attrs['sampler'] == 'svi'
    # Saving the later NUTS result must not replace the retained guide.
    final = replace(posterior, samples=arrays['samples'], sampler='numpyro_nuts')
    cs._save_artifacts(run_dir / 'artifacts', state, args, arrays['best_fit'], final)
    assert path.read_bytes() == original_bytes
    with h5py.File(run_dir / 'artifacts' / 'plot_bundle.h5') as f:
        assert f['posterior'].attrs['sampler'] == 'numpyro_nuts'
