"""Add the lens-only best-fit residuals (the parametric lens without the GRF conditional mean) to a saved run: writes
final/tables/image_fit_quality_lens_only.csv, adds the lens_only_* keys to the fit block of run_summary.json and
rewrites the results macros (LensOnlyRMS, LensOnlyMedianResidual, NLensOnlyRecovered).  No inference.

    regenerate_lens_only_rms.py RUN [RUN ...] [--catalogs IMAGES MEMBERS] [--ffsims-dir DIR]
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path)
    from _covariance_plot_cli import add_catalogs_argument, load_run
    add_catalogs_argument(parser)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from kappaccino.diagnostics.image_recovery import fit_summary, lens_only_summary, lens_only_table
    from kappaccino.diagnostics.exact_images import ExactImageSolver
    from kappaccino.reporting.results_tex import write_run_results
    for run in args.runs:
        config, state, evaluator, posterior, paths = load_run(run, catalogs=args.catalogs, ffsims_dir=args.ffsims_dir)
        theta = np.asarray(posterior.samples[int(np.nanargmax(posterior.log_prob))])
        frame = lens_only_table(ExactImageSolver(evaluator, config.image_diagnostics), state, evaluator, theta)
        if frame is None:
            print(f'{run}: no GRF conditional mean; the quoted residuals are already lens-only')
            continue
        frame.to_csv(paths.tables / 'image_fit_quality_lens_only.csv', index=False)
        summary_path = paths.tables / 'run_summary.json'
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
        else:                                   # a run whose write_outputs stopped early: rebuild the fit block from its tables
            summary = json.loads((run / 'final/artifacts/summary.json').read_text())
            summary['fit'] = fit_summary(state, pd.read_csv(paths.tables / 'image_fit_quality.csv', dtype={'family_id': str, 'image_label': str}),
                                         pd.read_csv(paths.tables / 'image_count_recovery.csv', dtype={'family_id': str}))
        summary.setdefault('fit', {}).update(lens_only_summary(frame))
        summary_path.write_text(json.dumps(summary, indent=2, default=float))
        fit = summary['fit']
        print(f"{run}: with field rms {fit.get('image_rms_arcsec'):.3f}, lens only rms {fit['lens_only_image_rms_arcsec']:.3f} "
              f"({fit['lens_only_recovered_image_count']}/{fit['observed_image_count']} recovered)")
        print(write_run_results(run / 'final', name=config.paths.figure_prefix))


if __name__ == '__main__':
    main()
