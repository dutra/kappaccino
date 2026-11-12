"""Rerun the whole output pass of saved runs over their reporting draws; no inference.

Every figure, table and results macro is rebuilt from one set of posterior draws (``reporting.draws`` of the run, or
--draws), each draw evaluated once and shared by all products.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    from _covariance_plot_cli import add_catalogs_argument
    add_catalogs_argument(parser)
    parser.add_argument('--draws', type=int, default=None, help='reporting draws (default: the run config)')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    from _covariance_plot_cli import load_run
    from kappaccino.pipeline import StagePaths
    from kappaccino.reporting.outputs import write_outputs
    for run in args.runs:
        config, state, evaluator, posterior, _ = load_run(run, catalogs=args.catalogs, ffsims_dir=args.ffsims_dir)
        if args.draws is not None:
            config = replace(config, reporting=replace(config.reporting, draws=args.draws))
        final = StagePaths(run.resolve() / 'final')
        audit = json.loads((final.tables / 'audit.json').read_text()) if (final.tables / 'audit.json').exists() else {}
        summary = write_outputs(config, state, evaluator, posterior, audit, final, log=lambda text: print(text, flush=True))
        print(json.dumps(summary.get('fit', {}), indent=2, default=float), flush=True)


if __name__ == '__main__':
    main()
