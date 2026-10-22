"""Short synthetic three-stage execution check, never a convergence study.

Run with the lenstronomy Python interpreter and PYTHONPATH=src. Outputs go only
to the requested scratch directory; this does not use any cluster campaign.
"""
from dataclasses import replace
import argparse
import json
from pathlib import Path

from lenscluster.config import (
    CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig,
    LensModelConfig, MemberPopulationConfig, PriorConfig, RunPathsConfig,
    RuntimeConfig, StageScheduleConfig, WorkflowConfig,
)
from lenscluster.planning import compile_run_plan
from lenscluster.runner import LensClusterRunner


def configuration(root):
    root.mkdir(parents=True, exist_ok=True)
    images, members = root/'images.cat', root/'members.cat'
    images.write_text('#REFERENCE 3\n'
        '1.a 3 5 1 1 0 2 24 1 .05\n1.b -4 -5 1 1 0 2 24 1 .05\n'
        '2.a 8 -6 1 1 0 3 24 1 .05\n2.b -7 9 1 1 0 3 24 1 .05\n')
    members.write_text('#REFERENCE 3\n'
        'g1 2 3 1 .9 20 20 1 1\ng2 -6 7 1 .8 30 21 1 1\n'
        'g3 100 70 1 1 0 21 1 1\ng4 -100 -80 1 1 0 23 1 1\n')
    mode = 'smoothed-local-image-plane'
    return LensClusterSolverConfig(
        model=LensModelConfig(cosmology=CosmologyConfig(H0=70., Om0=.3, Ode0=.7),
            large_halos=(DPIEHaloConfig(id='halo', x_centre=0., y_centre=0., angle_pos=0.,
                v_disp=600., core_radius_kpc=5., cut_radius_kpc=1000., z_lens=.3, ellipticite=.2,
                priors={'v_disp': PriorConfig('uniform', lower=500., upper=700.)}),),
            member_populations=(MemberPopulationConfig(id='members', catalog_path=str(members),
                mag0=20., corekpc=.2, sigma=100., cutkpc=10., z_lens=.3),),
            image_constraints=ImageConstraintsConfig(catalog_path=str(images), sigma_arcsec=.1)),
        paths=RunPathsConfig(output_dir=str(root/'output'), run_name='smoothed', layout='results-v1'),
        runtime=RuntimeConfig(chains=1, nuts_chain_method='sequential', dense_mass='lens-and-hyperparameters',
            skip_plots=True, skip_validation=True, quiet=True),
        workflow=WorkflowConfig(stage0_likelihood=mode, stage1_likelihood=mode, stage2_forward_mode=mode,
            sampling_engine='tiered_members',
            stage1_sampling_engine='tiered_members', stage2_sampling_engine='tiered_members', stage2_fresh_process=True,
            source_position_parameterization='likelihood-whitened'),
        schedule=StageScheduleConfig(svi_steps=(2,2,2), refresh_every=(None,None,None),
            fit_method=('svi','svi+nuts'), warmup=(1,5), samples=(16,5),
            sampling_refresh_runs=(1,1), max_tree_depth=(3,3)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    options = parser.parse_args()
    config = configuration(options.output.resolve())
    if options.resume:
        config = replace(config, runtime=replace(config.runtime, resume='all'))
    result = LensClusterRunner().run(compile_run_plan(config))
    from lenscluster import cluster_solver as cs
    import numpy as np
    report = dict(completed=result.completed, scope='synthetic_three_stage_execution_only',
                  convergence_test=False, repeated_mock_coverage_test=False, stages={})
    for stage in ('stage0', 'stage1', 'stage2'):
        path = result.run_dir/'stages'/stage/'artifacts'
        state, runtime, arrays, diagnostics = cs._load_artifacts(path)
        report['stages'][stage] = dict(sampling_engine=runtime['sampling_engine'], samples=len(arrays['samples']),
            parameters=len(state.parameter_specs), finite_samples=bool(np.isfinite(arrays['samples']).all()),
            finite_log_prob=bool(np.isfinite(arrays['log_prob']).all()),
            divergences=int(np.asarray(arrays.get('diverging', [])).sum()),
            source_prior_mean=state.smoothed_model['source_prior_mean'],
            source_prior_sigma=state.smoothed_model['source_prior_sigma'],
            phase=state.smoothed_model['phase'],
            sampler='numpyro_nuts' if stage == 'stage2' else 'svi', artifact=str(path))
        if (path/'member_tier_audit.json').exists():
            report['stages'][stage]['audit'] = json.loads((path/'member_tier_audit.json').read_text())
    (options.output/'validation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
