"""Bounded fresh-process CPU/NUTS timing on an existing smoothed-model artifact.

Use the lenstronomy interpreter. Run each CPU setting in a separate process;
--threads-per-chain omitted measures the inherited CPU/thread settings. Compare
against a previous --output .npz with --reference. Compilation is excluded from
synchronized timings. This fixed-step, short-trajectory probe measures execution
cost, not posterior convergence or effective samples per second.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="JSON report; state arrays use the same stem with .npz")
    parser.add_argument("--reference", type=Path, help="State .npz from the same chains, seed and transition settings")
    parser.add_argument("--chains", type=positive_int, default=4)
    parser.add_argument("--threads-per-chain", type=positive_int)
    parser.add_argument("--transitions", type=positive_int, default=8)
    parser.add_argument("--batches", type=positive_int, default=5)
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lenscluster.cpu_runtime import configure_cpu_runtime
    cpu_plan = configure_cpu_runtime(args.chains, "parallel", args.threads_per_chain)

    import numpy as np
    import jax
    import jax.numpy as jnp
    from numpyro.infer import NUTS
    from numpyro.infer.util import unconstrain_fn
    from lenscluster.smoothed_workflow import _load
    from lenscluster import cluster_solver as cs

    state, runtime, arrays, evaluator = _load(args.artifacts)
    theta = jnp.asarray(evaluator.reported_physical_to_latent_parameter_vector(arrays["best_fit"]))
    model = cs._posterior_model(state.parameter_specs, evaluator)
    sites = cs._parameter_sample_sites(state.parameter_specs)
    unconstrained = unconstrain_fn(cs._sample_site_model(state.parameter_specs), (), {}, {
        site.name: cs._site_value_from_theta(theta, site) for site in sites
    })
    global_names = tuple(sorted(site.name for site in sites if all(
        state.parameter_specs[i].component_family not in {"source_position", "member_scatter_offset"}
        for i in site.indices
    )))
    kernel = NUTS(model, dense_mass=[global_names] if global_names else [], step_size=0.0001,
                  adapt_step_size=False, adapt_mass_matrix=False, max_tree_depth=3)
    initial = jax.block_until_ready(jax.jit(jax.vmap(
        lambda key: kernel.init(key, num_warmup=0, init_params=unconstrained)
    ))(jax.random.split(jax.random.PRNGKey(491), args.chains)))

    def run(current):
        def step(current, _):
            result = kernel.sample(current, (), {})
            return result, (result.z, result.potential_energy, result.z_grad,
                            result.num_steps, result.diverging)
        return jax.lax.scan(step, current, None, length=args.transitions)

    started = time.perf_counter()
    compiled = jax.pmap(run).lower(initial).compile()
    compile_seconds = time.perf_counter() - started
    result = jax.block_until_ready(compiled(initial))
    values = {str(i): np.asarray(value) for i, value in enumerate(jax.tree.leaves(result))}
    for value in values.values():
        assert np.isfinite(value).all(), "nonfinite NUTS state in timing probe"
    report = dict(cpu_plan=cpu_plan.to_dict(), artifacts=str(args.artifacts.resolve()),
                  parameters=len(theta), exact_components=len(evaluator.smoothed_exact_components),
                  approximate_members=len(evaluator.smoothed_far_positions),
                  compile_seconds=compile_seconds, jax_version=jax.__version__,
                  transitions_per_chain=args.transitions, step_size=0.0001, max_tree_depth=3,
                  mass_matrix="identity in global dense and latent diagonal blocks",
                  adaptation=False, seed=491, x64=jax.config.jax_enable_x64,
                  scope="fixed-target execution timing only; not convergence or ESS evidence")
    if args.reference:
        max_scaled = 0.0
        bitwise_equal = True
        with np.load(args.reference) as reference:
            assert set(values) == set(reference.files), "incompatible reference state structure"
            for key, value in values.items():
                np.testing.assert_allclose(value, reference[key], rtol=1e-12, atol=1e-12)
                bitwise_equal &= value.dtype == reference[key].dtype and value.tobytes() == reference[key].tobytes()
                if value.dtype.kind == "f":
                    max_scaled = max(max_scaled, float(np.max(abs(value-reference[key])/(1+abs(reference[key])))))
        report.update(reference=str(args.reference.resolve()), max_scaled_difference=max_scaled,
                      bitwise_equal=bool(bitwise_equal))
    print(f"Compiled {args.chains} chains; CPU IDs {cpu_plan.selected_cpu_ids}", flush=True)
    report["load_start"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    timings = []
    for _ in range(args.batches):
        started = time.perf_counter()
        jax.block_until_ready(compiled(initial))
        timings.append(time.perf_counter() - started)
    report.update(batches_seconds=timings, median_seconds=float(np.median(timings)),
                  transitions_per_second=args.chains*args.transitions/float(np.median(timings)),
                  num_steps_range=[int(np.min(result[1][3])), int(np.max(result[1][3]))],
                  divergences=int(np.sum(result[1][4])))
    report["load_end"] = os.getloadavg() if hasattr(os, "getloadavg") else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **values)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
