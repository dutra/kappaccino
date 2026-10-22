"""NumPyro block metric for sampled sources and member velocity offsets."""
from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from numpyro.infer.util import potential_energy, unconstrain_fn

_LATENT_FAMILIES = frozenset({"source_position", "member_scatter_offset"})
_EIGENVALUE_FLOOR = 1.0
_FINITE_DIFFERENCE_STEP = 1.0e-4


def prepare_block_mass(args, sample_model, parameter_specs, reference_theta):
    """Seed a dense global block while keeping source/member sites diagonal.

    The global Hessian is evaluated in NumPyro's unconstrained coordinates,
    holding the source coordinates and member offsets fixed at the reference.
    Site tuples are sorted exactly as NumPyro's block metric expects.  Returned
    ``inverse_mass_matrix`` values are inverse Hessians for the dense block and
    ones for the diagonal block, consistent with the existing solver seed.
    Nonfinite curvature raises; eigenvalues below one are floored to one.
    """
    del args  # The caller owns logging and persists the returned diagnostics.
    # Reuse the solver's exact vector-site construction and prior support maps.
    # Import lazily because the solver calls this module during NUTS setup.
    from .cluster_solver import (
        _parameter_sample_sites, _sample_site_model, _site_value_from_theta,
    )

    started = time.perf_counter()
    theta = np.asarray(reference_theta, dtype=float)
    if theta.shape != (len(parameter_specs),):
        raise ValueError("reference_theta must have one entry per parameter specification")
    if not np.all(np.isfinite(theta)):
        raise ValueError("reference_theta must contain finite values")
    sites = _parameter_sample_sites(parameter_specs)
    global_names, latent_names = [], []
    for site in sites:
        latent = [parameter_specs[index].component_family in _LATENT_FAMILIES for index in site.indices]
        if any(latent) and not all(latent):
            raise ValueError(f"Sample site {site.name!r} mixes global and source/member latent parameters")
        (latent_names if all(latent) else global_names).append(site.name)
    global_names, latent_names = tuple(sorted(global_names)), tuple(sorted(latent_names))
    constrained = {site.name: _site_value_from_theta(jnp.asarray(theta), site) for site in sites}
    unconstrained = unconstrain_fn(_sample_site_model(parameter_specs), (), {}, constrained) if sites else {}
    unconstrained = {name: jnp.asarray(value, dtype=jnp.float64) for name, value in unconstrained.items()}
    if any(not np.all(np.isfinite(np.asarray(value))) for value in unconstrained.values()):
        raise ValueError("The reference lies on or outside a prior support boundary")
    global_vector, unravel_globals = ravel_pytree({name: unconstrained[name] for name in global_names})
    global_dim = int(global_vector.size)
    latent_dim = sum(int(unconstrained[name].size) for name in latent_names)
    dense_mass = [global_names] if global_names else []
    inverse_mass = {}
    if latent_names:
        inverse_mass[latent_names] = jnp.ones(latent_dim, dtype=jnp.float64)
    diagnostics = dict(used=False, dim=global_dim, total_dim=global_dim+latent_dim,
                       diagonal_dim=latent_dim, global_sites=list(global_names),
                       diagonal_sites=list(latent_names), eigenvalue_floor=_EIGENVALUE_FLOOR,
                       finite_difference=False, n_negative_eigenvalues=0,
                       n_floored_eigenvalues=0, condition_raw=float("nan"),
                       condition_floored=1.0)
    if not global_names:
        diagnostics.update(reason="no_global_parameters", seconds=time.perf_counter()-started)
        return dense_mass, inverse_mass, diagnostics

    def global_potential(vector):
        current = dict(unconstrained)
        current.update(unravel_globals(vector))
        return potential_energy(sample_model, (), {}, current)

    try:
        hessian = np.asarray(jax.jacfwd(jax.grad(global_potential))(global_vector), dtype=float)
    except Exception as error:
        diagnostics["autodiff_failure"] = repr(error)
        diagnostics["finite_difference"] = True
        gradient = jax.jit(jax.grad(global_potential))
        vector = np.asarray(global_vector)
        columns = []
        try:
            for index in range(global_dim):
                delta = np.zeros(global_dim)
                delta[index] = _FINITE_DIFFERENCE_STEP
                columns.append((np.asarray(gradient(jnp.asarray(vector+delta)))
                                - np.asarray(gradient(jnp.asarray(vector-delta))))
                               / (2.0*_FINITE_DIFFERENCE_STEP))
            hessian = np.stack(columns, axis=1)
        except Exception as fd_error:
            raise RuntimeError("Global-block Hessian seeding failed in autodiff and finite-difference modes") from fd_error
    if not np.all(np.isfinite(hessian)):
        raise RuntimeError("Global-block Hessian seeding produced non-finite curvature at the reference")
    hessian = 0.5*(hessian+hessian.T)
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    floored = np.maximum(eigenvalues, _EIGENVALUE_FLOOR)
    seed = (eigenvectors*(1.0/floored))@eigenvectors.T
    seed = 0.5*(seed+seed.T)
    inverse_mass[global_names] = jnp.asarray(seed, dtype=jnp.float64)
    positive = eigenvalues[eigenvalues > 0]
    diagnostics.update(used=True, seconds=float(time.perf_counter()-started),
                       n_negative_eigenvalues=int(np.sum(eigenvalues < 0)),
                       n_floored_eigenvalues=int(np.sum(eigenvalues < _EIGENVALUE_FLOOR)),
                       condition_raw=float(positive.max()/positive.min()) if positive.size else float("nan"),
                       condition_floored=float(floored.max()/floored.min()))
    return dense_mass, inverse_mass, diagnostics
