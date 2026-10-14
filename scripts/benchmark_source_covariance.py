"""Matched compiled benchmarks; run with the repository's lenstronomy Python.

--legacy-source points to a pre-change copy of cluster_solver.py. The integration
fixtures are synthetic: these timings are not a production cluster run.
"""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from lenscluster import cluster_solver as current
from test_source_covariance import make_evaluator


def measure(fn, x):
    results = {}
    for name, operation in [('likelihood', fn), ('gradient', jax.grad(fn)),
                            ('value_and_gradient', jax.value_and_grad(fn))]:
        start = time.perf_counter()
        compiled = jax.jit(operation).lower(x).compile()
        compile_seconds = time.perf_counter() - start
        result = jax.block_until_ready(compiled(x))
        if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(result)):
            raise RuntimeError(f'Nonfinite {name}')
        for _ in range(10):
            jax.block_until_ready(compiled(x))
        batches = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(100):
                jax.block_until_ready(compiled(x))
            batches.append((time.perf_counter() - start) * 1e6 / 100)
        results[name] = dict(compile_seconds=compile_seconds, median_us=float(np.median(batches)),
                             min_us=min(batches), max_us=max(batches), batches_us=batches)
    return results


def kernel(module, n, scatter):
    index = jnp.arange(n)
    def fn(t):
        a = .7 + .2 * jnp.sin(index + t[0])
        b = .2 * jnp.cos(index + t[1])
        c = -.1 * jnp.sin(index + t[2])
        d = 1.1 + .3 * jnp.cos(index + t[3])
        args = dict(beta_x=.1 * jnp.sin(index + t[4]), beta_y=.2 * jnp.cos(index + t[5]),
                    family_idx=(index // 3).astype(jnp.int32), n_families=n // 3,
                    sigma_per_image=jnp.full(n, .1) * jnp.exp(t[6]),
                    reliability_per_image=jnp.full(n, .95), image_has_constraint=jnp.ones(n, dtype=bool),
                    source_sigma_int=.02 * jnp.exp(t[7]),
                    scatter_var_x=jnp.full(n, .002 if scatter else 0.) * jnp.exp(t[8]),
                    scatter_var_y=jnp.full(n, .003 if scatter else 0.) * jnp.exp(t[9]),
                    covariance_floor=0., outlier_sigma_arcsec=10.)
        if module is current:
            args.update(jac_a00=a, jac_a01=b, jac_a10=c, jac_a11=d)
        else:
            args.update(inv_abs_mu=jnp.abs(a * d - b * c))
        return (module._full_covariance_source_plane_bin_loglike(**args) if module is current
                else module._source_plane_bin_loglike(**args))
    return fn, jnp.zeros(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('lenscluster._legacy_source_covariance', args.legacy_source)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    report = dict(platform=platform.platform(), python=platform.python_version(), jax=jax.__version__,
                  devices=[str(d) for d in jax.devices()], x64=jax.config.jax_enable_x64,
                  warmup_calls=10, batches=5, calls_per_batch=100,
                  legacy_source_sha256=hashlib.sha256(args.legacy_source.read_bytes()).hexdigest(),
                  current_source_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
                  cases={})
    for scatter in (False, True):
        for n in (6, 300):
            name = f'kernel_n{n}_scatter{scatter}'
            row = dict(images=n, families=n // 3, parameters=10)
            for label, module in [('before', legacy), ('after', current)]:
                row[label] = measure(*kernel(module, n, scatter))
            report['cases'][name] = row
            print(name, json.dumps(row), flush=True)
        for engine in ('full_flat', 'refreshing_surrogate_flat'):
            name = f'{engine}_scatter{scatter}'
            row = {}
            for label, module in [('before', legacy), ('after', current)]:
                evaluator, theta = make_evaluator(engine, scatter, module, vary_lens=True)
                row.update(images=len(evaluator.flat_critical_arc_data.x_obs),
                           families=evaluator.flat_critical_arc_data.n_families, parameters=len(theta),
                           lens_components=len(evaluator.state.packed_lens_spec.profile_type))
                row[label] = measure(evaluator._source_loglike_impl, theta)
            report['cases'][name] = row
            print(name, json.dumps(row), flush=True)
    for row in report['cases'].values():
        row['after_over_before'] = {k: row['after'][k]['median_us'] / row['before'][k]['median_us']
                                   for k in row['before']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
