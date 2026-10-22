"""Rerun only saved AS1063 stage-2 SVI and compare with retained NUTS."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from lenscluster import cluster_solver as cs
from lenscluster.planning import SolverRuntime

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/oct15a_isotropiclatent_fullflat/as1063/anisotropic-latent'
SOURCE = RUN / 'stages/stage2/artifacts/plot_bundle.h5'
OUT = RUN / 'comparisons/stage2_svi_replay'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'artifacts/svi/plot_bundle.h5').exists():
        raise FileExistsError('Replay already exists; refusing to overwrite it.')
    original_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    state, saved, arrays, old_diag = cs._load_artifacts(SOURCE.parent)
    args = SolverRuntime(dict(saved, output_dir=str(OUT), run_name='stage2_svi_replay',
                              skip_plots=True, skip_validation=True, debug_sampler_diagnostics=False))
    # Preserve fit_method and all optimizer settings; call SVI directly, never the NUTS driver.
    assert args.sampling_engine == 'full_flat'
    assert args.svi_steps == 20000 and args.seed == 12345
    evaluator = cs._build_cluster_evaluator_from_args(args, state)
    model = cs._posterior_model(state.parameter_specs, evaluator)
    result = cs._run_svi_fit(args, state, evaluator, model)
    center, posterior, diagnostics = result
    state, evaluator = result.state, result.evaluator
    cs._save_svi_checkpoint(args, state, evaluator, OUT, center, posterior)
    physical = cs._posterior_results_to_reported_physical(posterior, state.parameter_specs, evaluator)
    nuts = arrays['samples']
    rows = []
    for i, spec in enumerate(state.parameter_specs):
        row = dict(parameter=spec.name)
        for name, samples in [('svi', physical.samples), ('nuts', nuts)]:
            for q, value in zip([16, 50, 84], np.percentile(samples[:, i], [16, 50, 84])):
                row[f'{name}_p{q}'] = float(value)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'parameter_comparison.csv', index=False)
    indices = {s.field: i for i, s in enumerate(state.parameter_specs) if s.potential_id == 'potfile_1'}
    rows = []
    for mag in [16.18, 18.329466851303394, 20.]:
        lum = 10**(.4*(state.potfiles[0]['mag0']-mag))
        for name, samples in [('svi', physical.samples), ('nuts', nuts)]:
            alpha = samples[:, indices['alpha_sigma']]
            beta = samples[:, indices['beta_radius']]
            gamma = 2*alpha+beta-1
            quantities = dict(sigma_LT=samples[:, indices['sigma']]*lum**alpha,
                              rcut_kpc=samples[:, indices['cutkpc']]*lum**beta,
                              rcore_kpc=samples[:, indices['corekpc']]*lum**beta,
                              alpha=alpha, gamma=gamma, beta=beta)
            for key, values in quantities.items():
                q = np.percentile(values, [16,50,84])
                rows.append(dict(magnitude=mag, method=name, quantity=key,
                                 p16=q[0], p50=q[1], p84=q[2]))
    pd.DataFrame(rows).to_csv(OUT / 'potfile_comparison.csv', index=False)
    # Evaluate both methods in the same sampling coordinates; stored NUTS
    # potential energies include additional unconstraining Jacobians.
    latent_nuts = np.stack([evaluator.reported_physical_to_latent_parameter_vector(x) for x in nuts])
    nuts_logp = cs._posterior_logprob_matrix(state.parameter_specs, evaluator, latent_nuts)
    report = dict(source=str(SOURCE), source_sha256=original_hash,
                  svi_steps=args.svi_steps, learning_rate=args.svi_learning_rate,
                  refresh_every=args.refresh_every, seed=args.seed,
                  original_svi_final_loss=old_diag['svi_final_elbo_loss'],
                  replay_svi_final_loss=diagnostics['svi_final_elbo_loss'],
                  svi_runtime_sec=diagnostics['svi_runtime_sec'],
                  svi_draws=len(posterior.samples), nuts_draws=len(nuts),
                  svi_logp_percentiles=np.percentile(posterior.log_prob,[0,16,50,84,100]).tolist(),
                  nuts_logp_same_coordinates_percentiles=np.percentile(nuts_logp,[0,16,50,84,100]).tolist(),
                  source_unchanged=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original_hash)
    assert report['source_unchanged']
    (OUT / 'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2), flush=True)


if __name__ == '__main__':
    main()
