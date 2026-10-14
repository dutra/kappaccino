"""Short fresh-process stage-2 workflow check; writes only to /tmp."""
from pathlib import Path
import json
from dataclasses import replace
import numpy as np
from lenscluster import cluster_solver as cs

def main():
    ns={'__file__':str(Path('run.xsh').resolve())}
    text=Path('run.xsh').read_text().split('requested_cluster, campaign_name, covariance_mode, use_magnitude_likelihood = _parse_args')[0]
    exec('\n'.join(l for l in text.splitlines() if not l.startswith('$')),ns)
    c=ns['build_config']('HERA',cores=1,campaign_name='temporary',covariance_mode='anisotropic',use_magnitude_likelihood=False)
    out=Path('/tmp/latent_stage2_replay');out.mkdir(exist_ok=True)
    cat=out/'images.cat';cat.write_text('\n'.join(Path('data/ff_sims/hera/hera_obs_arcs.cat').read_text().splitlines()[:7])+'\n')
    halos=tuple(replace(h,priors={'v_disp':h.priors['v_disp']}) for h in c.model.large_halos[:2])
    model=replace(c.model,large_halos=halos,independent_member_halos=(),image_constraints=replace(c.model.image_constraints,catalog_path=str(cat)))
    c=replace(c,model=model,members=replace(c.members,potfile_member_brightest_n=(4,)))
    plan=ns['compile_run_plan'](c)
    args=cs._clone_args(plan.runtime_args,output_dir=str(out),run_name='smoke',fit_method='svi+nuts',
        fit_mode='joint',svi_steps=30,refresh_every=15,warmup=32,samples=16,max_tree_depth=4,
        chains=1,nuts_chain_method='sequential',sampling_engine='full_flat',
        skip_plots=True,skip_validation=False,quick_diagnostics=True,numpyro_print_summary=False,
        debug_sampler_diagnostics=False,posterior_image_diagnostic_draws=0,
        resume=False,sampling_refresh_runs=1)
    p=cs._run_single_stage_in_fresh_process(args,'joint','smoke',sample_likelihood_mode='isotropic-latent-displacement',
        source_position_prior_values={'1':(0.,0.),'2':(0.,0.)},perturbation_discovery_independent_override_by_potfile=[set()])
    state,saved,arrays,diag=cs._rebuild_state_from_h5(p/'artifacts/plot_bundle.h5')
    print('ROUNDTRIP',len(state.parameter_specs),saved['latent_displacement_sigma_beta_arcsec'],flush=True)
    ev=cs.ClusterJAXEvaluator(state,.1,sampling_engine='full_flat',sample_likelihood_mode='isotropic-latent-displacement',latent_displacement_sigma_beta_arcsec=.03)
    theta=ev.reported_physical_to_latent_parameter_vector(arrays['best_fit'])
    exact = [ev._exact_family_prediction_details(theta,f) for f in state.family_data]
    report = dict(parameters=len(state.parameter_specs), images=sum(f.n_images for f in state.family_data),
                  families=len(state.family_data), sigma_beta_arcsec=saved['latent_displacement_sigma_beta_arcsec'],
                  finite_samples=bool(np.isfinite(arrays['samples']).all()),
                  divergences=int(np.sum(arrays['diverging'])),
                  recovered_images=[int(d['recovered_image_count']) for d in exact],
                  exact_failure_reasons=[d['multiplicity_failure_reason'] for d in exact],
                  convergence_claim=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('REPORT',report,flush=True)
if __name__=='__main__': main()
