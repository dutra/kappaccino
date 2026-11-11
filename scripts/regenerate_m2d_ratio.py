"""Redraw the enclosed-mass figure and the mass/convergence ratio panels of saved runs from their tables, without
recomputing the truth maps.

Reads tables/truth_recovery_m2d_aperture_profile.csv, truth_recovery_kappa_annulus_profile.csv and image_fit_quality.csv
of each run, and the published FF-SIMS ratios beside the truth map when they exist (truth_recovery.literature_ratios).
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
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    from _covariance_plot_cli import load_run
    from kappaccino.diagnostics.truth_recovery import literature_ratios, mass_per_kappa_pixel_msun
    from kappaccino.plotting import plots
    for run in args.runs:
        config, state, _, _, paths = load_run(run, catalogs=args.catalogs, ffsims_dir=args.ffsims_dir)
        profile = pd.read_csv(paths.tables/'truth_recovery_m2d_aperture_profile.csv')
        images = pd.read_csv(paths.tables/'image_fit_quality.csv', dtype={'family_id': str, 'image_label': str})
        center = {key: profile[key].iloc[0] for key in ('center_x_arcsec', 'center_y_arcsec')}
        wcs = WCS(fits.getheader(run/'final/fits/truth_recovery_kappa_model_median.fits'))
        scale = mass_per_kappa_pixel_msun(state, wcs, float(config.truth.caustic_source_redshift))
        print(plots.plot_m2d_aperture_ratio(profile, center, images, scale,
                                            plots.figure_path(paths.figures, 'mass_profile.pdf', config.paths.figure_prefix),
                                            literature=literature_ratios(config.truth, 'mass')), flush=True)
        kappa_profile = pd.read_csv(paths.tables/'truth_recovery_kappa_annulus_profile.csv')
        print(plots.plot_truth_ratio_panels(profile, kappa_profile, center, images,
                                            plots.figure_path(paths.figures, 'kappa_mass_residuals.pdf', config.paths.figure_prefix),
                                            literature_mass=literature_ratios(config.truth, 'mass'),
                                            literature_kappa=literature_ratios(config.truth, 'kappa')), flush=True)


if __name__ == '__main__':
    main()
