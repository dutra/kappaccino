"""Synchronized, warmed JAX microbenchmarks of a saved smoothed model.

Standalone component timings are not additive (XLA fuses the full likelihood).
No inference is run and no saved model or membership is modified.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lenscluster import smoothed_likelihood as sl
from lenscluster.member_tiers import evaluate_farfield_sampling
from lenscluster.smoothed_workflow import _load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--gradient-breakdown', action='store_true')
    args = parser.parse_args()
    state, runtime, arrays, ev = _load(args.artifacts)
    x = jnp.asarray(ev.reported_physical_to_latent_parameter_vector(arrays['best_fit']))
    physical = ev._physical_parameter_vector(x)
    packed = ev._smoothed_packed(physical)
    data = ev.flat_critical_arc_data
    xy = jnp.stack((data.x_obs, data.y_obs), -1)
    fields = jax.jit(ev._smoothed_fields)(physical)
    stats = jax.jit(ev._smoothed_statistics)(x)
    absorbed = ev.smoothed_absorbed_positions
    ma, mu, ms = fields['member_alpha'][absorbed], fields['member_jacobian'][absorbed], fields['member_sigma'][absorbed]
    sigma = physical[state.smoothed_model['sigma_src_index']]
    report = dict(artifacts=str(args.artifacts.resolve()), devices=[str(d) for d in jax.devices()],
        jax_version=jax.__version__, x64=jax.config.jax_enable_x64, parameters=len(x),
        images=len(xy), families=data.n_families, exact_components=len(ev.smoothed_exact_components),
        approximate_members=len(ev.smoothed_far_positions), absorbed_members=len(absorbed),
        source_parameterization=ev.source_position_parameterization,
        repetitions=30, batches=5, measurements={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def measure(name, fn, inputs, gradient=False):
        for kind, operation in [('value', fn)] + ([('value_gradient', jax.value_and_grad(fn))] if gradient else []):
            start = time.perf_counter()
            compiled = jax.jit(operation).lower(inputs).compile()
            compile_seconds = time.perf_counter()-start
            value = jax.block_until_ready(compiled(inputs))
            assert all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(value)), name
            for _ in range(5):
                jax.block_until_ready(compiled(inputs))
            times = []
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(30):
                    jax.block_until_ready(compiled(inputs))
                times.append((time.perf_counter()-start)*1000/30)
            row = dict(median_ms=float(np.median(times)), batches_ms=times, compile_seconds=compile_seconds)
            report['measurements'][name+'_'+kind] = row
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(name, kind, json.dumps(row), flush=True)

    if args.gradient_breakdown:
        from lenscluster import member_tiers as mt
        original_exact = ev._smoothed_exact_rows
        original_far = mt.evaluate_farfield_sampling
        baseline = jax.jit(lambda p: ev._source_loglike_impl(p))(x)
        for freeze_exact, freeze_far in ((True, False), (False, True), (True, True), (False, False)):
            ev._smoothed_exact_rows = (
                (lambda xy, p, c: original_exact(xy, jax.tree.map(jax.lax.stop_gradient, p), c))
                if freeze_exact else original_exact)
            mt.evaluate_farfield_sampling = (
                (lambda *a: jax.tree.map(jax.lax.stop_gradient, original_far(*a)))
                if freeze_far else original_far)
            fn = lambda p: ev._source_loglike_impl(p)
            np.testing.assert_allclose(jax.jit(fn)(x), baseline, rtol=1e-12, atol=1e-9)
            measure(f'freeze_exact_{freeze_exact}_freeze_far_{freeze_far}', fn, x, True)
        ev._smoothed_exact_rows = original_exact
        mt.evaluate_farfield_sampling = original_far
        return
    measure('full_sampling_likelihood', ev._source_loglike_impl, x, True)
    # Controlled ablation: only the h evaluation is replaced, during tracing.
    original_h = sl.log_expected_absolute_normal
    try:
        sl.log_expected_absolute_normal = lambda mean, variance, parity=None: jnp.zeros_like(mean)
        measure('full_without_h_factor', lambda p: ev._source_loglike_impl(p), x, True)
    finally:
        sl.log_expected_absolute_normal = original_h
    def gaussian_only(p):
        s = ev._smoothed_statistics(p)
        return jnp.sum(s['gaussian_loglike']) + s['source_correction']
    measure('gaussian_covariance_and_sources_only', gaussian_only, x, True)
    measure('parameter_transform', ev._physical_parameter_vector, x)
    measure('pack_lens_state', ev._smoothed_packed, physical)
    measure('all_profile_fields_with_spatial_derivatives', ev._smoothed_fields, physical)
    measure('exact_profile_alpha_and_jacobian',
        lambda p: ev._smoothed_exact_rows(xy, p, ev.smoothed_exact_components), packed)
    def spatial(p):
        def total_u(pos):
            return ev._smoothed_exact_rows(pos, p, ev.smoothed_exact_components)[1].sum(axis=0)
        return jnp.stack([jax.jvp(total_u, (xy,), (jnp.broadcast_to(d, xy.shape),))[1]
                          for d in (jnp.array([1., 0.]), jnp.array([0., 1.]))], -1)
    measure('exact_profile_spatial_derivatives', spatial, packed)
    far = ev.smoothed_far_positions
    ci = ev.smoothed_member_components[far]
    geom = jax.tree.map(lambda a: a[far], ev.smoothed_geometry)
    measure('farfield_alpha_jacobian_and_spatial_derivatives',
        lambda p: evaluate_farfield_sampling(geom, p.sigma0[ci], p.Ra[ci], p.Rs[ci]), packed)
    covariance_inputs = (fields['jacobian'], sigma, ma, ms)
    def cov(q):
        return sl.covariance_statistics(q[0], data.sigma_per_image, q[1], q[2], q[3], data.image_has_constraint)['covariance']
    measure('covariance_construction', cov, covariance_inputs)
    measure('source_whitening', lambda c: sl.source_whitening(fields['backprojected'], c,
        data.global_family_index_per_image, data.n_families, data.image_has_constraint), stats['covariance'])
    def likelihood_fields(f):
        s = sl.statistics(f['backprojected'], stats['source_by_family'][data.global_family_index_per_image],
            f['jacobian'], f['determinant_gradient'], data.sigma_per_image, sigma,
            f['member_alpha'][absorbed], f['member_jacobian'][absorbed], f['member_sigma'][absorbed],
            data.image_has_constraint, parity=ev.smoothed_parity)
        return s['loglike'].sum()
    measure('likelihood_from_precomputed_fields', likelihood_fields, fields, True)
    measure('h_factor_only', lambda q: sl.log_expected_absolute_normal(q[0], q[1], ev.smoothed_parity).sum(),
        (stats['determinant_mean'], stats['determinant_variance']), True)
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
