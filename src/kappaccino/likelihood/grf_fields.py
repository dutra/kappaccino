"""Exact parametric fields plus the draw-conditioned shared GRF for reporting.

This does not alter the likelihood covariance. A family- or system-specific GP cannot
define one cluster-wide mass map, so the maps use one shared field.  For a run sampled
with system blocks it is rebuilt after sampling: an all-block twin of the run's model
(same parameters, hyperparameters and physical sources; the field-wide translation,
convergence and shear projected out as in a full run) gives the conditional mean of a
single field over every image.
"""
import copy

import numpy as np

from ..lens import cosmology


def _sigma0_factor(state, z_source):
    """The dPIE sigma0 factor at ``z_source``, from the state's table; computed (a cosmology quadrature) and stored the
    first time a redshift is asked for, as ``Evaluator.packed_state_for_z`` does."""
    table = getattr(state, "dpie_sigma0_factor_by_z", None)
    factor = None if table is None else table.get(float(z_source))
    if factor is None:
        factor = float(cosmology.dpie_sigma0_factor_from_config(state.z_lens, float(z_source), state.cosmo_config))
        if table is not None:
            table[float(z_source)] = factor
    return float(factor)

SHARED_PROJECT_MODES = ("translation", "convergence", "shear")


def shared_field_evaluator(evaluator):
    """The evaluator of one shared knee mass field over every image, or None when the run has no such field.

    An all-block run returns itself.  Otherwise a cached all-block twin with the same parameter vector; its latent
    source coordinates are whitened differently, so convert draws with ``shared_theta``."""
    if not (getattr(evaluator, 'gp_knee', False) and getattr(evaluator, 'sigma_grf_index', 0) >= 0
            and getattr(evaluator, 'gp_knee_spectrum', 'knee') == 'knee'):
        return None
    if str(evaluator.state.gp_blocks) == 'all':
        return evaluator
    twin = getattr(evaluator, '_shared_field_twin', None)
    if twin is None:
        from .evaluator import Evaluator
        state = copy.copy(evaluator.state)
        state.gp_blocks = 'all'
        state.gp_group_blocks = False
        state.gp_knee = {**dict(getattr(state, 'gp_knee', {})), 'project': SHARED_PROJECT_MODES}
        twin = Evaluator(state, evaluator.likelihood_mode)
        if [p.name for p in twin.state.parameter_specs] != [p.name for p in evaluator.state.parameter_specs]:
            raise ValueError('The shared-field twin has a different parameter vector')
        evaluator._shared_field_twin = twin
    return twin


def latent_from_physical(evaluator, physical):
    """The latent vector of this evaluator whose reported physical parameters, sources included, are ``physical``."""
    import jax
    from ..core.model import convert_theta_to_latent
    physical = np.asarray(physical, dtype=float)
    if not getattr(evaluator, 'gp_knee', False):
        return np.asarray(evaluator.latent_from_reported_physical(physical))
    e = evaluator._padded_evaluator()
    theta = convert_theta_to_latent(physical, e.state.parameter_specs)
    images = e.state.images
    ix, iy = np.asarray(images.source_x_param_index), np.asarray(images.source_y_param_index)
    theta[np.r_[ix, iy]] = 0.
    if not hasattr(e, '_whitening_jit'):
        e._whitening_jit = jax.jit(lambda t: e.statistics(t)['whitening'])
    whitening = e._whitening_jit(theta)
    layout = e.gp
    mask, family = np.asarray(layout['knot_mask']), np.asarray(layout['knot_family'])
    sources = np.stack((physical[ix], physical[iy]), axis=-1)
    target = np.where(mask[..., None], sources[np.where(mask, family, 0)], 0.).reshape(len(mask), -1)
    offset = np.linalg.solve(np.asarray(whitening['cholesky']),
                            (target - np.asarray(whitening['centroid']))[..., None])[..., 0].reshape(*mask.shape, 2)
    real = offset[np.asarray(layout['real_block']), np.asarray(layout['real_knot'])]
    families = np.asarray(layout['real_family'])
    theta[ix[families]], theta[iy[families]] = real[:, 0], real[:, 1]
    if not hasattr(evaluator, '_reported_physical_jit'):
        evaluator._reported_physical_jit = jax.jit(evaluator.reported_physical)
    np.testing.assert_allclose(evaluator._reported_physical_jit(theta), physical, rtol=1e-10, atol=1e-10)
    return np.asarray(theta)


def shared_theta(evaluator, theta):
    """``theta`` of ``evaluator`` re-expressed for its shared-field evaluator, keeping the physical sources."""
    twin = shared_field_evaluator(evaluator)
    if twin is None or twin is evaluator:
        return theta
    import jax
    theta = np.asarray(theta, dtype=float)
    cached = getattr(evaluator, '_shared_theta_cache', None)
    if cached is not None and cached[0] == theta.tobytes():
        return cached[1]
    if not hasattr(evaluator, '_reported_physical_jit'):
        evaluator._reported_physical_jit = jax.jit(evaluator.reported_physical)
    result = latent_from_physical(twin, np.asarray(evaluator._reported_physical_jit(theta)))
    evaluator._shared_theta_cache = (theta.tobytes(), result)
    return result


def shared_posterior(evaluator, posterior):
    """``posterior`` with every draw re-expressed for the shared-field evaluator; the physical values are unchanged."""
    twin = shared_field_evaluator(evaluator)
    if twin is None or twin is evaluator:
        return posterior
    import hashlib
    from dataclasses import replace
    key = hashlib.sha256(np.asarray(posterior.samples).tobytes()).hexdigest()
    cached = getattr(evaluator, '_shared_posterior_cache', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    physical = np.asarray(posterior.physical, dtype=float)
    samples = np.stack([latent_from_physical(twin, p) for p in physical]) if len(physical) else np.asarray(posterior.samples)
    result = replace(posterior, samples=samples)
    evaluator._shared_posterior_cache = (key, result)
    return result


def shared_evaluator_and_posterior(evaluator, posterior):
    """The shared-field evaluator and the posterior in its coordinates: the run itself for an all-block run, the
    all-block twin and the converted draws for a system-block run, or the run unchanged when there is no mass field."""
    twin = shared_field_evaluator(evaluator)
    if twin is None or twin is evaluator:
        return evaluator, posterior
    return twin, shared_posterior(evaluator, posterior)


def shared_grf_label(evaluator):
    """How the maps' GRF was obtained, for provenance."""
    twin = shared_field_evaluator(evaluator)
    if twin is None:
        return 'none: no knee mass field'
    return 'shared field of the run' if twin is evaluator else 'shared field rebuilt from a system-block run (all-block twin, projected)'


def shared_grf_fields(evaluator, theta, z_source, xy, *, chunk=1024):
    """Return deflection and its Jacobian, scaled to the requested redshift.

    The conditional mean already includes family zero's lens efficiency. Remove
    that weight before applying the target efficiency, exactly once.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    alpha = np.zeros_like(xy)
    jacobian = np.zeros((len(xy), 2, 2))
    twin = shared_field_evaluator(evaluator)
    if twin is None:
        return alpha, jacobian
    correction = twin.gp_deflection_correction(shared_theta(evaluator, theta))
    if correction is None:
        return alpha, jacobian
    evaluator = twin
    state = evaluator.state
    reference = _sigma0_factor(state, state.gp_reference_redshift)
    target = _sigma0_factor(state, z_source)
    weight = target / reference / float(evaluator.gp_weights_per_family[0])
    for start in range(0, len(xy), chunk):
        points = xy[start:start + chunk]
        count = len(points)
        points = np.pad(points, ((0, chunk-count), (0, 0)), mode='edge')
        fields = correction(points[:, 0], points[:, 1], np.zeros(chunk, dtype=int))
        alpha[start:start+count] = np.stack(fields[:2], axis=-1)[:count] * weight
        jacobian[start:start+count] = np.stack(fields[2:], axis=-1)[:count].reshape(-1, 2, 2) * weight
    if not (np.isfinite(alpha).all() and np.isfinite(jacobian).all()):
        raise ValueError('Nonfinite conditional GRF reporting fields')
    return alpha, jacobian


def posterior_median_theta(evaluator, posterior):
    """Coordinatewise physical posterior median, including physical sources.

    Whitened source coordinates are model dependent: their median is not the
    coordinate representation of the median physical source positions.
    """
    return latent_from_physical(evaluator, np.median(np.asarray(posterior.physical), axis=0))
