"""Compare saved fitted-member summaries with matched TNG Subfind truth.

Uses the same satellites on both sides; excludes the independently modeled BCG.
The SHMF histogram uses individual posterior median masses, not posterior
median bin counts. No inference or saved-run compatibility code is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import yaml

from kappaccino.plotting.style import PAPER_STYLE


def plot_scaling_truth(matched, final, figures, prefix):
    from kappaccino.plotting.plots import population_scaling_draws
    plt.rcParams.update(PAPER_STYLE)

    summary = pd.read_csv(final / 'tables/posterior_summary.csv')
    with h5py.File(final / 'artifacts/posterior.h5') as handle:
        physical = handle['physical'][:]
    # Confirm the saved table identifies the physical-array columns in order.
    if physical.shape[1] != len(summary) or not np.allclose(
            np.percentile(physical, [16, 50, 84], axis=0).T,
            summary[['p16', 'median', 'p84']].to_numpy(), rtol=1e-7, atol=1e-9):
        raise ValueError('Saved posterior columns do not match the summary table.')
    specs = [SimpleNamespace(name=name) for name in summary.parameter]
    relations = population_scaling_draws(matched, physical, specs)
    fields = [
        ('v_disp', matched.veldisp, r'$\sigma$ (km s$^{-1}$)', True,
         'Truth: Subfind dispersion'),
        ('cut_radius_kpc', matched.rhalf_pkpc, 'Radius (kpc)', True,
         r'Truth: Subfind $r_{1/2}$'),
        ('log10_mass_msun', np.log10(matched.mtot), r'$\log_{10}(M/M_\odot)$', False,
         'Truth: Subfind bound mass'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), sharex=True)
    for ax, (field, truth, label, logscale, truth_label) in zip(axes, fields):
        for magnitude, curves in relations.values():
            low, median, high = np.percentile(curves[field], [16, 50, 84], axis=0)
            ax.fill_between(magnitude, low, high, color='0.2', alpha=.12, linewidth=0)
            ax.plot(magnitude, median, color='0.2', lw=1.3,
                    label='Population relation (16–84%)')
        low, median, high = (matched[f'{field}_{q}'].to_numpy()
                             for q in ('p16', 'median', 'p84'))
        if not (np.isfinite([low, median, high]).all() and
                ((low <= median) & (median <= high)).all()):
            raise ValueError(f'Invalid saved quantiles for {field}')
        ax.vlines(matched.catalog_mag, np.minimum(truth, median),
                  np.maximum(truth, median), color='0.55', linewidth=.65,
                  alpha=.7, zorder=1.5)
        ax.errorbar(matched.catalog_mag, median, yerr=[median-low, high-median],
                    fmt='o', color='#3976a8', ecolor='#3976a880', markersize=3,
                    elinewidth=.7, capsize=1.5,
                    label=(r'Fit: dPIE $r_{\rm cut}$' if field == 'cut_radius_kpc'
                           else 'Fit: dPIE median (16–84%)'))
        ax.plot(matched.catalog_mag, truth, linestyle='none', marker='x',
                color='#ba5a2a', markersize=4, markeredgewidth=.9, label=truth_label)
        ax.set(xlabel='F125W magnitude', ylabel=label)
        if logscale:
            ax.set_yscale('log')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), fontsize=7.5,
                  handlelength=1.8)
    low, high = matched.catalog_mag.min(), matched.catalog_mag.max()
    axes[0].set_xlim(high + .15, low - .15)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(figures / f'{prefix}_scaling_relation_vs_truth.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path(
        'results/oct22_single_stage/tngc014_small/smoothed-local-image-plane'))
    parser.add_argument('--catalog', type=Path, default=Path('catalogs/tngc014/small/members.yaml'))
    parser.add_argument('--truth', type=Path, default=Path('catalogs/tngc014/source/members_truth.csv'))
    args = parser.parse_args()
    final = args.run / 'final'
    source = final / 'tables/member_scaling_relation.csv'
    fit = pd.read_csv(source, dtype={'catalog_id': str})
    truth = pd.read_csv(args.truth, dtype={'id': str})
    catalog = yaml.safe_load(args.catalog.read_text())['members']
    satellite_ids = {str(k) for k, v in catalog.items() if v.get('role') != 'bcg'}
    if set(fit.catalog_id) != satellite_ids:
        raise ValueError('Saved fitted satellites do not match the selected catalog.')
    matched = fit.merge(truth, left_on='catalog_id', right_on='id',
                        how='left', validate='one_to_one', indicator=True)
    if not matched['_merge'].eq('both').all() or not matched.role.eq('member').all():
        raise ValueError('Every fitted satellite must have one non-BCG truth match.')
    matched = matched.drop(columns='_merge')
    numeric = ['mtot', 'veldisp', 'v_disp_p16', 'v_disp_median', 'v_disp_p84',
               'log10_mass_msun_median']
    if not np.isfinite(matched[numeric].to_numpy()).all():
        raise ValueError('Nonfinite mass or velocity summary.')
    if not (matched[['mtot', 'veldisp', 'v_disp_p16']] > 0).all().all():
        raise ValueError('Masses and dispersions must be positive.')
    lo, med, hi = (matched[f'v_disp_{q}'].to_numpy() for q in ('p16', 'median', 'p84'))
    if not ((lo <= med) & (med <= hi)).all():
        raise ValueError('Invalid velocity percentile ordering.')
    figures = final / 'figures'
    figures.mkdir(exist_ok=True)
    # The run's own name (run.yaml `name`), so every catalog size labels its files.
    prefix = json.loads((final / 'artifacts/config.json').read_text())['paths']['figure_prefix']
    plt.style.use(Path(__file__).resolve().parents[1] / 'src/kappaccino/plotting/style.mplstyle')
    plt.rcParams.update(PAPER_STYLE)

    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    masses = [np.log10(matched.mtot.to_numpy()), matched.log10_mass_msun_median.to_numpy()]
    bin_width = .5
    bins = np.arange(np.floor(min(v.min() for v in masses) / bin_width) * bin_width,
                     np.ceil(max(v.max() for v in masses) / bin_width) * bin_width + bin_width / 2,
                     bin_width)
    for values, color, label in zip(masses, ['0.2', '#3976a8'],
            ['Truth: Subfind bound mass', 'Fit: dPIE median mass']):
        counts, _ = np.histogram(values, bins=bins)
        assert counts.sum() == len(matched)
        ax.stairs(counts, bins, color=color, label=label)
    ax.set(xlim=(bins[0], bins[-1]),
           xlabel=r'$\log_{10}(M/M_\odot)$', ylabel='Number of satellites per 0.5 dex')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / f'{prefix}_shmf_truth_vs_catalog.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    actual = matched.veldisp.to_numpy()
    limits = (min(actual.min(), lo.min()) * .85, max(actual.max(), hi.max()) * 1.15)
    ax.plot(limits, limits, color='0.35', ls='--', lw=1, label='Equality')
    ax.errorbar(actual, med, yerr=[med-lo, hi-med], fmt='o', markersize=3,
                color='#3976a8', elinewidth=.7, capsize=1.5,
                label='Median and 16–84% interval')
    ax.set(xscale='log', yscale='log', xlim=limits, ylim=limits,
           xlabel=r'Truth: Subfind $\sigma$ (km s$^{-1}$)',
           ylabel=r'Fit: dPIE $\sigma$ (km s$^{-1}$)')
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='upper left', fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / f'{prefix}_velocity_dispersion_vs_truth.pdf')
    plt.close(fig)

    plot_scaling_truth(matched, final, figures, prefix)
    matched.to_csv(final / f'tables/{prefix}_members_truth_matched.csv', index=False)
    metadata = {
        'inputs': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (source, args.catalog, args.truth,
                             final / 'tables/posterior_summary.csv',
                             final / 'artifacts/posterior.h5')},
        'n_matched_satellites': len(matched), 'bcg_excluded': True,
        'shmf': 'Histogram counts in common 0.5 dex bins for the same selected satellites; fitted masses are individual posterior medians, not posterior bin-count quantiles.',
        'shmf_bin_edges_log10_msun': bins.tolist(),
        'mass_definitions': 'Subfind bound total mass versus saved dPIE total mass. These are different mass definitions.',
        'velocity_definitions': 'Subfind particle velocity dispersion versus fitted dPIE fiducial velocity parameter; equality is a reference, not an assumed physical identity.',
        'radius_definitions': 'Subfind three-dimensional total half-mass radius versus dPIE truncation radius; these are different quantities, not a truth r_cut measurement.',
        'selection': f'Members of {args.catalog} ({len(matched)} satellites); the small and medium catalogs are selected partly by truth-mass deflection and are not a complete cluster SHMF.',
        'median_velocity_ratio': float(np.median(med / actual)),
    }
    (final / f'tables/{prefix}_truth_comparison.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
