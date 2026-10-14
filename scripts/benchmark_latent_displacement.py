"""Benchmark the new positional kernel and Hera evaluator without running inference.

Run with the lenstronomy interpreter and PYTHONPATH=src. Outputs stay in /tmp
unless --output is specified. The current run.xsh model supplies the inputs.
"""
import argparse
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from lenscluster import cluster_solver as cs
from lenscluster.latent_displacement import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/latent_displacement_benchmark.json'))
    args = parser.parse_args()
    script = Path('run.xsh').resolve()
    text = script.read_text().split('requested_cluster, campaign_name, covariance_mode, use_magnitude_likelihood = _parse_args')[0]
    namespace = {'__file__': str(script)}
    exec(compile('\n'.join(line for line in text.splitlines() if not line.startswith('$')), str(script), 'exec'), namespace)
    config = namespace['build_config']('HERA', cores=1, campaign_name='benchmark',
                                       covariance_mode='anisotropic', use_magnitude_likelihood=False)
    plan = namespace['compile_run_plan'](config)
    runtime = cs._clone_args(plan.runtime_args, sampling_engine='full_flat',
                            sample_likelihood_mode='isotropic-latent-displacement', quiet=True)
    # Read family IDs without building a second lens evaluator.
    _, images, _, _ = cs._declared_model_inputs(config.model)
    source_centers = {str(i): (0., 0.) for i in images.family_id.unique()}
    state = cs._build_state_from_inputs(runtime, fit_mode_override='joint', source_position_prior_values=source_centers)
    ev = cs.ClusterJAXEvaluator(state, .1, sampling_engine='full_flat',
                               sample_likelihood_mode='isotropic-latent-displacement',
                               latent_displacement_sigma_beta_arcsec=.03)
    theta = jnp.asarray(cs._default_theta(state.parameter_specs))
    report = dict(jax=jax.__version__, devices=str(jax.devices()), dtype=str(theta.dtype),
                  affinity=sorted(os.sched_getaffinity(0)),
                  cpu=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), ''),
                  images=len(ev.flat_critical_arc_data.x_obs), families=len(state.family_data),
                  components=len(state.packed_lens_spec.profile_type), parameters=len(theta),
                  sigma_beta_arcsec=.03, warmup_calls=10, batches=5, calls_per_batch=100)

    def bench(name, func, arg):
        start = time.perf_counter()
        compiled = jax.jit(func).lower(arg).compile()
        compilation = time.perf_counter() - start
        for _ in range(10):
            jax.block_until_ready(compiled(arg))
        batches = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(100):
                result = jax.block_until_ready(compiled(arg))
            batches.append((time.perf_counter()-start)*1000/100)
        assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(result))
        report[name] = dict(compile_s=compilation, median_ms=float(np.median(batches)), batch_ms=batches)
        print(name, report[name], flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')

    bench('full_flat_likelihood', ev._source_loglike_impl, theta)
    bench('full_flat_gradient', jax.grad(ev._source_loglike_impl), theta)
    bench('full_flat_value_and_gradient', jax.value_and_grad(ev._source_loglike_impl), theta)
    stats = ev._latent_displacement_statistics(theta)

    def kernel(jac):
        return statistics(stats['backprojected'], stats['source'], jac,
                          ev.flat_critical_arc_data.sigma_per_image, .03,
                          stats['member_covariance'], stats['constrained'])['loglike'].sum()

    bench('kernel_likelihood', kernel, stats['jacobian'])
    bench('kernel_gradient', jax.grad(kernel), stats['jacobian'])
    bench('kernel_value_and_gradient', jax.value_and_grad(kernel), stats['jacobian'])


if __name__ == '__main__':
    main()
