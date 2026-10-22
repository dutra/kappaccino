"""Multi-start MAP in NumPyro's unconstrained coordinates, with mode clustering."""
from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from jax.flatten_util import ravel_pytree
from numpyro.infer import Predictive
from numpyro.infer.util import constrain_fn, potential_energy

from .evaluator import Evaluator
from .model import ModelState
from .sampling import (LATENT_FAMILIES, posterior_model, prior_model, sample_sites, theta_from_site_values,
                       unconstrained_from_theta)

HYPERPARAMETER_FAMILIES = frozenset({"member_velocity_scatter", "source_scatter"})


def prior_starts(state: ModelState, n_starts: int, seed: int) -> np.ndarray:
    """Latent starts: lens parameters drawn from the prior; offsets, sources and hyperparameters at their centres."""
    specs = state.parameter_specs
    draws = Predictive(prior_model(specs), num_samples=int(n_starts))(jax.random.PRNGKey(int(seed)))
    theta = theta_from_site_values({k: np.asarray(v) for k, v in draws.items()}, specs)
    reference = np.asarray(state.initial_theta, dtype=float)
    for idx, spec in enumerate(specs):
        if spec.component_family in LATENT_FAMILIES:
            theta[:, idx] = 0.0
        elif spec.component_family in HYPERPARAMETER_FAMILIES:
            theta[:, idx] = reference[idx]
    theta[0] = reference                     # the configured model is always one of the starts
    return theta


def multistart_map(state: ModelState, evaluator: Evaluator, starts: np.ndarray, *, max_iter: int,
                   mode_margin: float, parameter_tolerance: float, log=print) -> dict[str, Any]:
    """Minimize the potential from every start under vmap; cluster the converged points into modes."""
    specs = state.parameter_specs
    model = posterior_model(specs, evaluator.log_likelihood)
    first = unconstrained_from_theta(specs, starts[0])
    _, unravel = ravel_pytree({k: jnp.asarray(v, dtype=jnp.float64) for k, v in first.items()})
    u0 = np.stack([np.asarray(ravel_pytree({k: jnp.asarray(v, dtype=jnp.float64) for k, v in unconstrained_from_theta(specs, s).items()})[0]) for s in starts])

    def potential(u):
        return potential_energy(model, (), {}, unravel(u))

    solver = jaxopt.LBFGS(fun=potential, maxiter=int(max_iter), tol=1.0e-6, linesearch="backtracking", history_size=20)
    started = time.perf_counter()
    run = jax.jit(jax.vmap(lambda u: solver.run(u)))
    result = run(jnp.asarray(u0))
    final_u = np.asarray(result.params)
    final_potential = np.asarray(jax.vmap(jax.jit(potential))(jnp.asarray(final_u)))
    iterations = np.asarray(result.state.iter_num)
    gradient_norm = np.asarray(result.state.error)
    elapsed = time.perf_counter() - started
    finite = np.isfinite(final_potential) & np.isfinite(final_u).all(axis=1)
    if not finite.any():
        raise RuntimeError("Every multi-start optimization ended at a non-finite potential.")
    # Scale for the mode distance: spread of the finite starts, per unconstrained coordinate.
    scale = np.where(finite.sum() > 1, np.std(u0[finite], axis=0), 1.0)
    scale = np.where(scale > 0, scale, 1.0)
    order = np.argsort(np.where(finite, final_potential, np.inf))
    modes: list[dict[str, Any]] = []
    assignment = np.full(len(starts), -1, dtype=int)
    for index in order:
        if not finite[index]:
            continue
        for mode_index, mode in enumerate(modes):
            if np.max(np.abs(final_u[index] - final_u[mode["representative"]]) / scale) < float(parameter_tolerance):
                assignment[index] = mode_index
                mode["members"].append(int(index))
                break
        else:
            assignment[index] = len(modes)
            modes.append({"representative": int(index), "members": [int(index)], "potential": float(final_potential[index])})
    best = modes[0]
    constrain = constrain_fn(prior_model(specs), (), {}, unravel(jnp.asarray(final_u[best["representative"]])))
    theta_best = theta_from_site_values({k: np.asarray(v) for k, v in constrain.items()}, specs)
    for mode in modes:
        values = constrain_fn(prior_model(specs), (), {}, unravel(jnp.asarray(final_u[mode["representative"]])))
        mode["theta"] = theta_from_site_values({k: np.asarray(v) for k, v in values.items()}, specs).tolist()
        mode["log_posterior"] = -mode["potential"]
        mode["count"] = len(mode["members"])
        mode["within_margin"] = bool(mode["potential"] - best["potential"] <= float(mode_margin))
    log(f"[search] {len(starts)} starts, {int(finite.sum())} finite, {len(modes)} modes in {elapsed:.1f}s; "
        f"best log posterior {-best['potential']:.2f}, modes within margin {sum(m['within_margin'] for m in modes)}")
    return dict(theta=theta_best, unconstrained=final_u[best["representative"]], modes=modes, assignment=assignment.tolist(),
                potential=final_potential.tolist(), iterations=iterations.tolist(), gradient_norm=gradient_norm.tolist(),
                elapsed_seconds=float(elapsed), n_finite=int(finite.sum()), starts=starts.tolist())


def chain_starts(search: dict[str, Any], chains: int, max_modes: int) -> np.ndarray:
    """One latent start per chain: the best mode first, then other modes within the margin."""
    candidates = [np.asarray(m["theta"], dtype=float) for m in search["modes"] if m["within_margin"]][:int(max_modes)]
    if not candidates:
        candidates = [np.asarray(search["theta"], dtype=float)]
    return np.stack([candidates[i % len(candidates)] for i in range(int(chains))])
