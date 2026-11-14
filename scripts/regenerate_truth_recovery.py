"""Recompute truth maps and dependent figures with the shared conditional GRF.

Uses the saved posterior and the reporting draws (reporting.draws), or --draws.
No inference.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    from _covariance_plot_cli import add_catalogs_argument
    add_catalogs_argument(parser)
    parser.add_argument('--draws', type=int, default=None,
                        help='posterior draws behind the truth maps, profiles and footprint numbers (default: the run config)')
    args = parser.parse_args()
    from dataclasses import replace
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pandas as pd
    from _covariance_plot_cli import load_run
    from kappaccino.plotting import plots
    from kappaccino.diagnostics.truth_recovery import compute_truth_recovery
    from kappaccino.diagnostics.exact_images import astropy_cosmology
    from kappaccino.plotting.magnification_recovery import write_magnification_recovery
    from kappaccino.plotting.aperture_mass_histogram import write_aperture_mass_histogram
    from kappaccino.plotting.zeta_masses import write_zeta_masses
    from kappaccino.reporting.results_tex import write_run_results
    for run in args.runs:
        config,state,evaluator,posterior,paths=load_run(run,catalogs=args.catalogs,ffsims_dir=args.ffsims_dir)
        if args.draws is not None:
            config=replace(config,reporting=replace(config.reporting,draws=args.draws))
        from kappaccino.reporting.draws import reporting_draws
        draws=reporting_draws(config,evaluator,posterior)
        tables,figures=paths.tables,paths.figures
        images=pd.read_csv(tables/'image_fit_quality.csv',dtype={'family_id':str,'image_label':str})
        truth=compute_truth_recovery(state,draws,config.truth,run/'final/fits',tables,log=lambda text:print(text,flush=True))
        if truth is None:
            raise ValueError(f'No truth maps configured for {run}')
        prefix=config.paths.figure_prefix
        plots.plot_truth_recovery_residuals(truth,images,plots.figure_path(figures,'truth_recovery_residuals.pdf',prefix))
        if truth['profile'] is not None:
            plots.plot_m2d_aperture_ratio(truth['profile'],truth['center'],images,
                truth['mass_per_kappa_pixel_msun'],plots.figure_path(figures,'mass_profile.pdf',prefix),
                literature=truth['literature_m2d'])
            plots.plot_truth_ratio_panels(truth['profile'],truth['kappa_profile'],truth['center'],images,
                plots.figure_path(figures,'kappa_mass_residuals.pdf',prefix),
                literature_mass=truth['literature_m2d'],literature_kappa=truth['literature_kappa'])
        write_magnification_recovery(truth,images,state.reference,state.z_lens,
            astropy_cosmology(state.cosmo_config),config.truth,figures,tables,prefix)
        from kappaccino.plotting.cumulative_source_area import write_cumulative_source_area
        write_cumulative_source_area(truth,draws,astropy_cosmology(state.cosmo_config),figures,tables,prefix)
        from kappaccino.plotting.scatter_summary import write_magnification_scatter
        write_magnification_scatter(truth,draws,images,tables,log=lambda text:print(text,flush=True))
        write_aperture_mass_histogram(config,state,draws,paths,images,log=lambda text:print(text,flush=True))
        write_zeta_masses(config,state,draws,paths,images,log=lambda text:print(text,flush=True))
        plots.plot_critical_curves(state,draws,truth['z_source'],plots.figure_path(figures,'critical_curves.pdf',prefix),
            log=lambda text:print(text,flush=True))
        summary_path=tables/'run_summary.json'
        if summary_path.exists() and truth['profile'] is not None:
            summary=json.loads(summary_path.read_text())
            summary['truth']={'m2d_ratio_at_max_radius':float(truth['profile']['m2d_ratio_median'].iloc[-1]),
                              'n_radii':len(truth['profile'])}
            summary_path.write_text(json.dumps(summary,indent=2)+'\n')
        write_run_results(run/'final',name=prefix)
        print(f'Regenerated truth and critical-curve figures: {figures}',flush=True)


if __name__=='__main__':
    main()
