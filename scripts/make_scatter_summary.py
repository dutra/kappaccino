"""Draw scatter_summary.pdf: the scatter of the model magnification about the truth at mu = 3 and 10 for every given
simulated-cluster run, beside the scatter between teams on the Hubble Frontier Fields (Raney et al. 2020).

Uses the saved posterior and the reporting draws (reporting.draws), or --draws.  No inference.  Each run also gets
final/tables/magnification_scatter.csv; --reuse reads that table instead of evaluating the draws again.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='run directories of clusters with kappa/shear truth, in the order drawn')
    parser.add_argument('--output', type=Path, default=Path('results/scatter_summary.pdf'))
    parser.add_argument('--draws', type=int, default=None, help='posterior draws per run (default: the run config)')
    parser.add_argument('--reuse', action='store_true', help='read magnification_scatter.csv of runs that already have it')
    args = parser.parse_args()
    from dataclasses import replace
    import json
    import matplotlib
    matplotlib.use('Agg')
    import pandas as pd
    from kappaccino.plotting.scatter_summary import render_scatter_summary, run_scatter_table
    tables = {}
    for run in args.runs:
        saved = run / 'final/tables/magnification_scatter.csv'
        name = json.loads((run / 'final/artifacts/config.json').read_text())['paths']['figure_prefix'].capitalize()
        if args.reuse and saved.is_file():
            tables[name] = pd.read_csv(saved)
            continue
        from _covariance_plot_cli import load_run
        from kappaccino.diagnostics.truth_recovery import truth_maps_on_grid
        from kappaccino.reporting.draws import reporting_draws
        config, state, evaluator, posterior, paths = load_run(run)
        if args.draws is not None:
            config = replace(config, reporting=replace(config.reporting, draws=args.draws))
        if not (config.truth.kappa_true_fits and config.truth.gammax_true_fits and config.truth.gammay_true_fits):
            raise ValueError(f'No kappa/shear truth maps configured for {run}')
        _, _, x, y, maps = truth_maps_on_grid(state, config.truth)
        truth = dict(x=x, y=y, z_source=float(config.truth.caustic_source_redshift), truth=maps)
        images = pd.read_csv(paths.tables / 'image_fit_quality.csv')
        table = run_scatter_table(truth, reporting_draws(config, evaluator, posterior), images[['x_obs_arcsec', 'y_obs_arcsec']].to_numpy(float),
                                  log=lambda text: print(text, flush=True))
        table.to_csv(saved, index=False)
        tables[name] = table
    for name, table in tables.items():
        print(name)
        print(table.to_string(index=False), flush=True)
    print(f'Wrote {render_scatter_summary(tables, args.output)}')


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    main()
