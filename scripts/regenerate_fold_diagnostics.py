"""Regenerate the model-identified fold-pair PDFs over the reporting draws (needs the image recovery tables)."""
import argparse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from _covariance_plot_cli import load_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    parser.add_argument('--imaging', choices=('saved', 'jwst'), default='saved')
    parser.add_argument('--recompute', action='store_true', help='Recompute fold geometry instead of reusing saved diagnostics')
    parser.add_argument('--draws', type=int, default=None, help='reporting draws (default: the run config)')
    from _covariance_plot_cli import add_catalogs_argument
    add_catalogs_argument(parser)
    args = parser.parse_args()
    from kappaccino.imaging.cutouts import imaging_variants, load_cluster_imaging
    from kappaccino.plotting.fold_pairs import (
        compute_fold_diagnostics, save_fold_data, render_fold_diagnostics, load_fold_data,
    )
    from kappaccino.io import artifacts
    log = lambda message: print(message, flush=True)
    final = Path(args.run_directory).resolve() / 'final'
    paths = SimpleNamespace(tables=final/'tables', figures=final/'figures')
    cached = paths.tables / 'fold_pairs_source_caustics.json'
    data = load_fold_data(paths.tables) if cached.exists() and not args.recompute else None
    if data is not None:
        # A rendering operation must not reinterpret saved latent source coordinates.
        # The cache already contains image positions, sources, curves and bands.
        from _covariance_plot_cli import with_ffsims_dir
        config = with_ffsims_dir(artifacts.load_config(final/'artifacts'), args.ffsims_dir)
        reference = config.model.reference
        state = SimpleNamespace(reference=(reference.reference, reference.ra0_deg, reference.dec0_deg))
        log(f'[outputs] rendering saved fold cache: {len(data["draw_indices"])} draws; gp_blocks={config.likelihood.gp_blocks}')
    else:
        config, state, evaluator, posterior, paths = load_run(args.run_directory, catalogs=args.catalogs, ffsims_dir=args.ffsims_dir)
        from kappaccino.reporting.draws import reporting_draws
        if args.draws is not None:
            config = replace(config, reporting=replace(config.reporting, draws=args.draws))
        data = compute_fold_diagnostics(config, state, reporting_draws(config, evaluator, posterior), paths.tables, log=log)
        save_fold_data(data, paths.tables)
    for image_config in imaging_variants(config, overview_only=True):
        if args.imaging == 'jwst' and image_config.figure_suffix != '_jwst':
            continue
        try:
            imaging = load_cluster_imaging(image_config, config.model, config.paths.run_name)
        except FileNotFoundError as error:
            print(f'[outputs] fold RGB skipped: {error}')
            continue
        render_fold_diagnostics(replace(config, image_catalog=image_config), state, data, paths, imaging)


if __name__ == '__main__':
    main()
