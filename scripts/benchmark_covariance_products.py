"""Verify and benchmark geometry-product caching; never alter production kernels.

The member weights remain proposal-dependent. We cache all six symmetric
products of the three spatial terms, not a covariance frozen at a reference.
Summation over members still occurs at runtime because scaling slopes vary.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lenscluster import smoothed_likelihood as sl
from lenscluster import cluster_solver as cs
from lenscluster.member_tiers import _farfield_weights
from lenscluster.smoothed_workflow import _load


PAIRS = [(i, j) for i in range(3) for j in range(i, 3)]


def geometry_products(deflection):
    """Cache G_k G_l^T + G_l G_k^T, counting diagonal terms once."""
    g = np.asarray(deflection)
    products = []
    for i, j in PAIRS:
        product = np.einsum('gni,gnj->gnij', g[:, :, i], g[:, :, j])
        products.append(product if i == j else product + product.swapaxes(-1, -2))
    return jnp.asarray(np.stack(products, axis=2))


def contracted_covariance(products, weights, scatter):
    coefficients = jnp.stack([weights[..., i]*weights[..., j] for i, j in PAIRS], -1)
    return jnp.einsum('gnk,gnkij->nij', 4*scatter[:, None, None]**2*coefficients, products)


def direct_covariance(geometry, weights, scatter):
    response = 2*scatter[:, None, None]*jnp.einsum('gnk,gnki->gni', weights, geometry)
    return jnp.einsum('gni,gnj->nij', response, response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    state, _, arrays, ev = _load(args.artifacts)
    original_stats = ev._smoothed_statistics
    original_covariance = sl.covariance_statistics
    positions = np.intersect1d(ev.smoothed_far_positions, ev.smoothed_absorbed_positions)
    components = ev.smoothed_member_components[positions]
    absorbed_exact = np.flatnonzero(~np.isin(ev.smoothed_absorbed_positions, positions))
    geom = jax.tree.map(lambda a: a[positions], ev.smoothed_geometry)
    start = time.perf_counter()
    products = geometry_products(geom.deflection)
    jax.block_until_ready(products)
    report = dict(artifacts=str(args.artifacts.resolve()), devices=[str(d) for d in jax.devices()],
        x64=jax.config.jax_enable_x64, jax_version=jax.__version__,
        approximate_absorbed=len(positions), exact_absorbed=len(absorbed_exact),
        products_bytes=products.size*products.dtype.itemsize,
        precompute_seconds=time.perf_counter()-start, checks={}, timings={},
        scope='covariance geometry products only; cross-covariance and stable conditional variance unchanged',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    def check(name, actual, expected, rtol=2e-10, atol=2e-10):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert np.isfinite(actual).all() and np.isfinite(expected).all(), name
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=name)
        report['checks'][name] = dict(max_abs=float(np.max(np.abs(actual-expected), initial=0)),
            max_scaled=float(np.max(np.abs(actual-expected)/(1+np.abs(expected)), initial=0)),
            rtol=rtol, atol=atol)
        save()
    def weights_and_scatter(p):
        packed = ev._smoothed_packed(p)
        return _farfield_weights(geom, packed.sigma0[components], packed.Ra[components], packed.Rs[components]), p[jnp.asarray(state.smoothed_model['scatter_indices'])[components]]
    def cached_stats(params, **kwargs):
        assert not kwargs, 'Benchmark supports the sampling path only.'
        physical = ev._physical_parameter_vector(params)
        weights, scatter = weights_and_scatter(physical)
        approximate_covariance = contracted_covariance(products, weights, scatter)
        def replacement(jacobian, sigma_obs, sigma_src, member_alpha=None, member_sigma=0., mask=None):
            result = original_covariance(jacobian, sigma_obs, sigma_src, member_alpha, member_sigma, mask)
            # Exact absorbed members retain their exact profile responses.
            alpha, sigma = result['member_alpha'], result['member_sigma']
            response = 2*sigma[absorbed_exact, None, None]*alpha[absorbed_exact]
            member_cov = jnp.einsum('gni,gnj->nij', response, response) + approximate_covariance
            active = result['constrained']
            member_cov = jnp.where(active[:, None, None], member_cov, 0.)
            a, obs, src = result['jacobian'], result['sigma_obs'], result['sigma_src']
            cov = src[:, None, None]**2*jnp.eye(2) + obs[:, None, None]**2*(a@a.swapaxes(-1,-2)) + member_cov
            cov = .5*(cov+cov.swapaxes(-1,-2))
            chol = jnp.linalg.cholesky(cov)
            valid = (jnp.isfinite(cov).all(axis=(-2,-1)) & jnp.isfinite(chol).all(axis=(-2,-1))
                & (obs >= 0) & (src >= 0) & jnp.all(jnp.isfinite(sigma) & (sigma >= 0))
                & jnp.isfinite(a).all(axis=(-2,-1)))
            result.update(member_covariance=member_cov, covariance=cov, cholesky=chol,
                logdet=2*jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1),
                valid=jnp.where(active, valid, True))
            return result
        sl.covariance_statistics = replacement
        try:
            return original_stats(params)
        finally:
            sl.covariance_statistics = original_covariance
    # Compile the two variants separately; monkey patches exist only while tracing.
    def candidate(p):
        ev._smoothed_statistics = cached_stats
        try:
            return ev._source_loglike_impl(p)
        finally:
            ev._smoothed_statistics = original_stats
    def baseline(p):
        return ev._source_loglike_impl(p)
    latent = cs._convert_theta_to_latent(np.asarray(arrays['best_fit']), state.parameter_specs)
    ix, iy = ev.flat_critical_arc_data.global_family_source_x_param_index, ev.flat_critical_arc_data.global_family_source_y_param_index
    whiten = jax.jit(lambda p: original_stats(p)['whitening'])
    def convert(p):
        latent = cs._convert_theta_to_latent(np.asarray(p), state.parameter_specs)
        if ev.source_position_parameterization == 'likelihood-whitened':
            latent[ix], latent[iy] = 0., 0.
            w = whiten(jnp.asarray(latent))
            offsets = sl.inverse_source_transport(jnp.stack((p[ix], p[iy]), -1), w)
            latent[ix], latent[iy] = np.asarray(offsets[:, 0]), np.asarray(offsets[:, 1])
        return jnp.asarray(latent)
    samples = np.asarray(arrays['samples'])
    selected = np.linspace(0, len(samples)-1, min(16, len(samples)), dtype=int)
    physical = np.vstack([arrays['best_fit'], samples[selected]])
    inputs = [convert(p) for p in physical]
    report['posterior_indices'] = selected.tolist()
    # Include independent perturbations of every latent coordinate.
    rng = np.random.default_rng(9471)
    inputs += [inputs[0]+jnp.asarray(rng.normal(size=latent.shape)*scale) for scale in (.001,.01,.03)]
    print('Reference models prepared:', len(inputs), flush=True)
    baseline_vg = jax.jit(jax.value_and_grad(baseline)).lower(inputs[0]).compile()
    candidate_vg = jax.jit(jax.value_and_grad(candidate)).lower(inputs[0]).compile()
    def selected_statistics(fn, p):
        result = fn(p)
        return {k: result[k] for k in ('covariance','loglike','source_by_family','determinant_mean','determinant_variance')}
    bs = jax.jit(lambda p: selected_statistics(original_stats, p))
    opt = jax.jit(lambda p: selected_statistics(cached_stats, p))
    for i, x in enumerate(inputs):
        b, c = baseline_vg(x), candidate_vg(x)
        check(f'model_{i}_full_loglike', c[0], b[0])
        check(f'model_{i}_gradient', c[1], b[1])
        b, c = bs(x), opt(x)
        for key in b:
            check(f'model_{i}_{key}', c[key], b[key])
    # Kernel tests: off-reference weights/scatters, including strong cancellation.
    for i in range(8):
        w = jnp.asarray(rng.normal(size=geom.deflection.shape[:-1])*10**rng.uniform(-2,2))
        s = jnp.asarray(np.exp(rng.normal(size=len(positions))))
        check(f'random_covariance_{i}', contracted_covariance(products,w,s), direct_covariance(geom.deflection,w,s), rtol=2e-12, atol=2e-12)
    # Independent kernel gradients, without the downstream likelihood.
    kernel_probe = jnp.asarray(rng.normal(size=(geom.deflection.shape[1], 2, 2)))
    direct_grad = jax.jit(jax.grad(lambda pair: (direct_covariance(geom.deflection,*pair)*kernel_probe).sum()))
    cached_grad = jax.jit(jax.grad(lambda pair: (contracted_covariance(products,*pair)*kernel_probe).sum()))
    for i, (expected, actual) in enumerate(zip(direct_grad((w,s)), cached_grad((w,s)))):
        check(f'kernel_gradient_{i}', actual, expected, rtol=2e-11, atol=2e-11)
    print('All covariance, likelihood, and gradient checks passed.', flush=True)
    def time_pair(label, functions, x):
        compiled = [f if hasattr(f, 'runtime_executable') else jax.jit(f).lower(x).compile() for f in functions]
        for fn in compiled:
            for _ in range(10):
                jax.block_until_ready(fn(x))
        times = [[], []]
        for batch in range(9):
            for j in ((0,1) if batch%2 == 0 else (1,0)):
                start = time.perf_counter()
                for _ in range(50):
                    jax.block_until_ready(compiled[j](x))
                times[j].append((time.perf_counter()-start)*1000/50)
        row = dict(baseline_ms=float(np.median(times[0])), cached_products_ms=float(np.median(times[1])), batches_ms=times)
        row['speedup'] = row['baseline_ms']/row['cached_products_ms']
        report['timings'][label] = row
        save()
        print(label, json.dumps(row), flush=True)
    w, s = weights_and_scatter(ev._physical_parameter_vector(inputs[0]))
    direct = lambda pair: direct_covariance(geom.deflection, *pair)
    cached = lambda pair: contracted_covariance(products, *pair)
    time_pair('covariance_kernel', [direct,cached], (w,s))
    # Use a dense, nonconstant cotangent to exercise every covariance entry.
    cotangent = jnp.asarray(rng.normal(size=(geom.deflection.shape[1],2,2)))
    time_pair('covariance_kernel_value_gradient', [jax.value_and_grad(lambda q: (direct(q)*cotangent).sum()),
        jax.value_and_grad(lambda q: (cached(q)*cotangent).sum())], (w,s))
    time_pair('full_likelihood', [baseline,candidate], inputs[0])
    time_pair('full_likelihood_value_gradient', [baseline_vg,candidate_vg], inputs[0])
    report['status'] = 'verified_and_benchmarked'
    save()


if __name__ == '__main__':
    main()
