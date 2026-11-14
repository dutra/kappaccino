"""Regenerate only the cumulative source-area figure of a simulated cluster from the saved posterior.

Uses the reporting draws (reporting.draws), or --draws.  No inference; no other product of the run is rewritten.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    from _covariance_plot_cli import add_catalogs_argument, load_run
    add_catalogs_argument(parser)
    parser.add_argument('--draws', type=int, default=None, help='posterior draws behind the bands (default: the run config)')
    args = parser.parse_args()
    from dataclasses import replace
    import matplotlib
    matplotlib.use('Agg')
    from kappaccino.diagnostics.exact_images import astropy_cosmology
    from kappaccino.diagnostics.truth_recovery import truth_maps_on_grid
    from kappaccino.plotting.cumulative_source_area import write_cumulative_source_area
    from kappaccino.reporting.draws import reporting_draws
    for run in args.runs:
        config, state, evaluator, posterior, paths = load_run(run, catalogs=args.catalogs, ffsims_dir=args.ffsims_dir)
        if args.draws is not None:
            config = replace(config, reporting=replace(config.reporting, draws=args.draws))
        if not (config.truth.kappa_true_fits and config.truth.gammax_true_fits and config.truth.gammay_true_fits):
            raise ValueError(f'No kappa/shear truth maps configured for {run}')
        _, _, x, y, maps = truth_maps_on_grid(state, config.truth)
        truth = dict(x=x, y=y, z_source=float(config.truth.caustic_source_redshift), truth=maps)
        write_cumulative_source_area(truth, reporting_draws(config, evaluator, posterior), astropy_cosmology(state.cosmo_config),
                                     paths.figures, paths.tables, config.paths.figure_prefix, log=lambda text: print(text, flush=True))


if __name__ == '__main__':
    main()
