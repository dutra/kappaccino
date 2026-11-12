"""Recompute the truth-recovery maps and profiles of saved runs and redraw the enclosed-mass figure and the
mass/convergence ratio panels, without the other truth products (magnification, aperture histogram, critical curves).

Same draws and statistic as the pipeline (compute_truth_recovery with the configured draw count and the run's seed).
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    from _covariance_plot_cli import add_catalogs_argument
    add_catalogs_argument(parser)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    import pandas as pd
    from _covariance_plot_cli import load_run
    from kappaccino.diagnostics.truth_recovery import compute_truth_recovery
    from kappaccino.plotting import plots
    for run in args.runs:
        config, state, evaluator, posterior, paths = load_run(run, catalogs=args.catalogs, ffsims_dir=args.ffsims_dir)
        images = pd.read_csv(paths.tables/'image_fit_quality.csv', dtype={'family_id': str, 'image_label': str})
        from kappaccino.reporting.draws import reporting_draws
        truth = compute_truth_recovery(state, reporting_draws(config, evaluator, posterior), config.truth, run/'final/fits', paths.tables,
                                       log=lambda text: print(text, flush=True))
        prefix = config.paths.figure_prefix
        print(plots.plot_m2d_aperture_ratio(truth['profile'], truth['center'], images, truth['mass_per_kappa_pixel_msun'],
                                            plots.figure_path(paths.figures, 'mass_profile.pdf', prefix),
                                            literature=truth['literature_m2d']), flush=True)
        print(plots.plot_truth_ratio_panels(truth['profile'], truth['kappa_profile'], truth['center'], images,
                                            plots.figure_path(paths.figures, 'kappa_mass_residuals.pdf', prefix),
                                            literature_mass=truth['literature_m2d'], literature_kappa=truth['literature_kappa']), flush=True)


if __name__ == '__main__':
    main()
