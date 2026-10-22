"""Benchmark saved latent-displacement states without running inference."""
from pathlib import Path
import argparse
import inspect
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import jax
import numpy as np
from lenscluster import cluster_solver as cs
from lenscluster.planning import SolverRuntime


def evaluator(state, saved, engine):
    keys=inspect.signature(cs.ClusterJAXEvaluator).parameters
    kwargs={k:v for k,v in saved.items() if k in keys and k!='state'}
    kwargs['sampling_engine']=engine
    kwargs['cab_likelihood_weight']=cs._effective_cab_likelihood_weight(saved.get('cab_likelihood_weight'),state)
    kwargs['image_presence_penalty_weight']=cs._effective_image_presence_penalty_weight(
        saved.get('image_presence_penalty_weight'),sample_likelihood_mode=saved['sample_likelihood_mode'],
        fit_mode=saved.get('fit_mode','joint'),image_plane_mode=saved.get('image_plane_mode','none'))
    kwargs.update(cs._normalized_saved_likelihood_stabilizer_args(saved))
    return cs.ClusterJAXEvaluator(state=state,**kwargs)


def run(artifact, output):
    state,saved,arrays,diagnostics=cs._rebuild_state_from_h5(artifact)
    if saved['sample_likelihood_mode']!='isotropic-latent-displacement':
        raise ValueError('Benchmark requires a latent-displacement artifact')
    arrays,_=cs._maybe_convert_loaded_posterior_arrays_to_physical(arrays,state.parameter_specs,diagnostics)
    exact=evaluator(state,saved,'full_flat')
    cached=evaluator(state,saved,'refreshing_surrogate_flat')
    reference=exact.reported_physical_to_latent_parameter_vector(arrays['best_fit'])
    start=time.perf_counter();cached.refresh_surrogate(reference)
    build_seconds=time.perf_counter()-start
    print(f'Cache built in {build_seconds:.3f}s',flush=True)
    functions=[jax.jit(jax.value_and_grad(e._source_loglike_impl)) for e in [exact,cached]]
    for label,fn in zip(['exact','cached'],functions):
        jax.block_until_ready(fn(reference)); print(f'{label} compiled',flush=True)
    timings=[]
    for fn in functions:
        elapsed=[]
        for _ in range(5):
            start=time.perf_counter();jax.block_until_ready(fn(reference));elapsed.append(time.perf_counter()-start)
        timings.append(float(np.median(elapsed)))
    rows=[]
    indices=sorted(set([0,len(arrays['samples'])//2,len(arrays['samples'])-1]))
    states=[('reference',reference)]+[(str(i),exact.reported_physical_to_latent_parameter_vector(arrays['samples'][i])) for i in indices]
    for label,theta in states:
        v,g=functions[0](theta);w,h=functions[1](theta)
        rows.append(dict(state=label,exact_loglike=float(v),cached_loglike=float(w),
            delta_loglike=float(w-v),gradient_max_abs_error=float(np.max(abs(np.asarray(h-g)))),
            gradient_relative_l2_error=float(np.linalg.norm(h-g)/max(np.linalg.norm(g),1e-30))))
    cache=cached.export_latent_surrogate_cache()
    report=dict(artifact=str(artifact.resolve()),cache_build_seconds=build_seconds,
        exact_seconds=timings[0],cached_seconds=timings[1],speedup=timings[0]/timings[1],
        cache_metadata=cache['metadata'],states=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != "cache_metadata"},indent=2),flush=True)
    # A temporary artifact exercises the same persistence path as production checkpoints.
    state.production_latent_surrogate_cache=cache
    args=SolverRuntime(dict(saved, sampling_engine='refreshing_surrogate_flat'))
    posterior=cs.PosteriorResults(samples=arrays['samples'],log_prob=arrays['log_prob'],
        accept_prob=arrays['accept_prob'],diverging=arrays['diverging'],num_steps=arrays['num_steps'],
        warmup_steps=0,sample_steps=len(arrays['samples']),num_chains=1,init_diagnostics=diagnostics)
    cs._save_plot_bundle_h5(output.with_suffix('.h5'),state,args,arrays['best_fit'],posterior)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args();run(options.artifact,options.output)
