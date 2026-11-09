"""Saved-run adapter for covariance figure regeneration."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def with_catalogs(config, catalogs):
    """``config`` reading the image and member catalogs from ``catalogs`` = (images.yaml, members.yaml): for a saved run
    whose catalog files have since moved (the FF-SIMS catalogs went from data/ff_sims to catalogs/<cluster>, 2026-11-08).
    The catalog contents must be the ones the run used; the rebuilt parameter order is checked against the saved state."""
    if not catalogs:
        return config
    from dataclasses import replace
    images, members = catalogs
    model = replace(config.model, image_constraints=replace(config.model.image_constraints, catalog_path=images),
                    member_populations=tuple(replace(p, catalog_path=members) for p in config.model.member_populations))
    return replace(config, model=model)


def with_ffsims_dir(config, directory):
    """``config`` reading the FF-SIMS truth maps and simulated HST frames from ``directory`` (data/imaging/<cluster>/ffsims)
    instead of the paths saved in the run (data/ff_sims before 2026-11-08)."""
    if not directory:
        return config
    from dataclasses import replace
    directory = str(directory).rstrip('/')
    truth = replace(config.truth, kappa_true_fits=f'{directory}/kappa_z9_0.fits', gammax_true_fits=f'{directory}/gammax_z9_0.fits',
                    gammay_true_fits=f'{directory}/gammay_z9_0.fits')
    return replace(config, truth=truth, image_catalog=replace(config.image_catalog, image_dir=directory))


def add_catalogs_argument(parser):
    parser.add_argument('--catalogs', nargs=2, metavar=('IMAGES_YAML', 'MEMBERS_YAML'),
                        help='read the catalogs from these files instead of the paths saved in the run (moved catalog files)')
    parser.add_argument('--ffsims-dir', metavar='DIR',
                        help='read the FF-SIMS truth maps and simulated frames from DIR (data/imaging/<cluster>/ffsims) instead of the saved paths')


def load_run(run_directory, *, physical_draw_indices=None, catalogs=None, ffsims_dir=None):
    import jax
    jax.config.update('jax_enable_x64', True)
    import numpy as np
    from kappaccino.io import artifacts
    from kappaccino.likelihood.evaluator import Evaluator
    from kappaccino.core.state import build_state, with_member_classes
    run = Path(run_directory).resolve()
    init, final = run / 'init/artifacts', run / 'final'
    config = with_ffsims_dir(with_catalogs(artifacts.load_config(final / 'artifacts'), catalogs), ffsims_dir)
    classes = json.loads((init / artifacts.CLASSIFICATION_FILE).read_text())
    state = build_state(config, drop_member_ids=classes.get('dropped_members', ()))
    centered = set(classes.get('centered_members', []))
    state = with_member_classes(state, artifacts.load_exact_positions(init), classes,
                                centered=np.array([m in centered for m in state.members.member_ids]))
    state.initial_theta = artifacts.load_start(init)
    saved = json.loads((init / artifacts.STATE_FILE).read_text())
    assert [p.name for p in state.parameter_specs] == [p['name'] for p in saved['parameter_specs']]
    assert list(state.members.member_ids) == saved['member_ids']
    posterior = artifacts.load_posterior(final / 'artifacts')
    best = int(np.nanargmax(posterior.log_prob))
    evaluator = Evaluator(state, config.likelihood.likelihood_mode)
    if physical_draw_indices is not None:
        # Plot saved physical draws explicitly, preserving physical source positions.
        # This is not a fallback or a modification of the posterior on disk.
        indices = np.unique(np.r_[physical_draw_indices, best]).astype(int)
        source_indices = np.r_[state.images.source_x_param_index, state.images.source_y_param_index]
        mass_indices = np.setdiff1d(np.arange(posterior.samples.shape[1]), source_indices)
        if evaluator.gp_knee:
            from kappaccino.core.model import convert_theta_to_latent
            evaluator = evaluator._padded_evaluator()
            whitening_fn = jax.jit(lambda theta: evaluator.statistics(theta)['whitening'])
            reported_fn = jax.jit(evaluator.reported_physical)
        for index in indices:
            np.testing.assert_allclose(np.asarray(evaluator.physical(posterior.samples[index]))[mass_indices],
                                       posterior.physical[index, mass_indices], rtol=1e-10, atol=1e-10)
            if evaluator.gp_knee:
                theta = convert_theta_to_latent(posterior.physical[index], state.parameter_specs)
                theta[source_indices] = 0.
                whitening = whitening_fn(theta)
                layout = evaluator.gp
                mask = np.asarray(layout['knot_mask'])
                family = np.asarray(layout['knot_family'])
                source = np.stack((posterior.physical[index, state.images.source_x_param_index],
                                   posterior.physical[index, state.images.source_y_param_index]), -1)
                target = np.where(mask[..., None], source[np.where(mask, family, 0)], 0.).reshape(mask.shape[0], -1)
                offset = np.linalg.solve(np.asarray(whitening['cholesky']),
                                         (target-np.asarray(whitening['centroid']))[..., None])[..., 0].reshape(*mask.shape, 2)
                real = offset[np.asarray(layout['real_block']), np.asarray(layout['real_knot'])]
                real_family = np.asarray(layout['real_family'])
                theta[np.asarray(state.images.source_x_param_index)[real_family]] = real[:, 0]
                theta[np.asarray(state.images.source_y_param_index)[real_family]] = real[:, 1]
                reported = reported_fn(theta)
            else:
                theta = evaluator.latent_from_reported_physical(posterior.physical[index])
                reported = evaluator.reported_physical(theta)
            np.testing.assert_allclose(reported, posterior.physical[index],
                                       rtol=1e-10, atol=1e-10)
            posterior.samples[index] = theta
        print(f'[outputs] verified {len(indices)} saved physical draws including source positions', flush=True)
    np.testing.assert_allclose(evaluator.reported_physical(posterior.samples[best]), posterior.physical[best], rtol=1e-10, atol=1e-10)

    paths = SimpleNamespace(figures=final / 'figures', tables=final / 'tables')
    return config, state, evaluator, posterior, paths
