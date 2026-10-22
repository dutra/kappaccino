"""Population scaling relations, not individual-member posterior predictions.

Uses the same literature calibration as make_as1063_potfile_comparison.py.
Run with the lenstronomy interpreter; no inference is performed.
"""
import hashlib
import json

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from make_as1063_potfile_comparison import ROOT, BUNDLE, OUT, BLUE, ORANGE, GREEN

STEM = 'as1063_member_scaling_comparison'


def main():
    comparison = json.loads((OUT / 'as1063_potfile_comparison.json').read_text())
    digest = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
    assert digest == comparison['sha256'], 'Regenerate the corner comparison first.'
    with h5py.File(BUNDLE) as handle:
        samples = handle['posterior/samples'][()]
        meta = json.loads(handle['state/build_state_meta_json'][()])
    if meta.get('scaling_relation_mode') != 'velocity-radius-exponents':
        raise ValueError('Historical alpha/gamma runs are unsupported; select a fresh beta_radius run.')
    fields = {p['field']: samples[:, i] for i, p in enumerate(meta['parameter_specs'])
              if p['potential_id'] == 'potfile_1'}
    mref = comparison['reference_magnitude_F160W_AB']
    mag = np.linspace(mref, 24, 350)
    relative_luminosity = 10**(-.4*(mag-mref))
    original_luminosity = 10**(-.4*(mag-meta['potfiles'][0]['mag0']))
    alpha, beta = fields['alpha_sigma'], fields['beta_radius']
    gamma = 2*alpha+beta-1
    relations = [fields['sigma'][:, None]*original_luminosity**alpha[:, None],
                 fields['cutkpc'][:, None]*original_luminosity**beta[:, None],
                 relative_luminosity**gamma[:, None]]
    slopes = [alpha, beta, gamma]
    quantiles = [np.percentile(values, [2.5, 16, 50, 84, 97.5], axis=0)
                 for values in relations]
    # Verify normalization and the sign/convention of every luminosity exponent.
    for values, slope in zip(relations, slopes):
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        measured = np.log(values[:, -1]/values[:, 0])/np.log(relative_luminosity[-1])
        np.testing.assert_allclose(measured, slope, atol=1e-12)
    np.testing.assert_allclose(relations[2][:, 0], 1)
    for panel in range(2):
        np.testing.assert_allclose(quantiles[panel][1:4, 0],
                                   comparison['our_percentiles_16_50_84'][panel])
    lit = [np.asarray(comparison['published_median_p16_p84']),
           np.asarray(comparison['caminha2016']['median_p16_p84_at_common_magnitude'])]
    lit_slopes = [[.27, .66, .2], [.25, .5, 0.]]
    colors, styles = [ORANGE, GREEN], ['--', '-.']
    members = np.loadtxt(ROOT / meta['potfiles'][0]['catalog_path'])[:, 6]
    member_mag = members[(members >= mref) & (members <= mag[-1])]

    plt.style.use(ROOT / 'src/lenscluster/style.mplstyle')
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), sharex=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.26, top=.77, wspace=.29)
    labels = [r'$\sigma_{\rm LT}\;[\mathrm{km\,s^{-1}}]$',
              r'$r_{\rm cut}\;[\mathrm{kpc}]$',
              r'$(M/L)\, /\, (M/L)_{\rm ref}$']
    symbols = [r'\alpha', r'\beta', r'\gamma']
    ticks = [[10, 30, 100, 300], [.1, 1, 10, 100, 1000], [.1, 1, 10, 100]]
    for panel, ax in enumerate(axes):
        q = quantiles[panel]
        ax.set_yscale('log')
        ax.fill_between(mag, q[0], q[4], color=BLUE, alpha=.10, linewidth=0)
        ax.fill_between(mag, q[1], q[3], color=BLUE, alpha=.23, linewidth=0)
        ax.plot(mag, q[2], color=BLUE, lw=2.4, zorder=5)
        for values, slope, color, ls in zip(lit, lit_slopes, colors, styles):
            norm = values[panel] if panel < 2 else np.ones(3)
            curve = norm[:, None]*relative_luminosity**slope[panel]
            if panel < 2:
                ax.fill_between(mag, curve[1], curve[2], color=color, alpha=.13, linewidth=0)
            ax.plot(mag, curve[0], color=color, ls=ls, lw=2.1, zorder=6)
        ax.set_xlim(24, mref)
        ax.set_xticks([17, 19, 21, 23])
        ax.set_xlabel(r'$m_{\rm F160W}$ (AB)', labelpad=7)
        ax.set_ylabel(labels[panel], labelpad=8)
        ax.yaxis.set_major_locator(FixedLocator(ticks[panel]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        # Rug shows the catalog sampling, not fitted velocity measurements.
        ax.plot(member_mag, np.full(len(member_mag), .013), '|', color='#73777B',
                alpha=.35, ms=5, mew=.7, transform=ax.get_xaxis_transform(), clip_on=False)
        top = ax.secondary_xaxis('top', functions=(
            lambda m: 10**(-.4*(m-mref)),
            lambda lum: mref-2.5*np.log10(np.maximum(lum, 1e-100))))
        top.set_xticks([1, .1, .01, .001])
        top.set_xticklabels(['1', '0.1', '0.01', '0.001'])
        top.set_xlabel(r'$L/L_{\rm ref}$', labelpad=7)
        top.minorticks_off()
        lo, mid, hi = np.percentile(slopes[panel], [16, 50, 84])
        # Place slope labels below each panel to keep curves unobstructed.
        ax.text(.5, -.25,
                rf'${symbols[panel]}={mid:.2f}^{{+{hi-mid:.2f}}}_{{-{mid-lo:.2f}}}$'
                rf'  /  ${lit_slopes[0][panel]:.2f}$  /  ${lit_slopes[1][panel]:.2f}$',
                transform=ax.transAxes, ha='center', va='top', fontsize=13)
    axes[0].set_ylim(8, 500)
    axes[1].set_ylim(.07, max(500, 1.12*quantiles[1][4].max()))
    axes[2].set_ylim(.1, 250)
    handles = [Line2D([], [], color=BLUE, lw=2.4),
               Line2D([], [], color=ORANGE, ls='--', lw=2.1),
               Line2D([], [], color=GREEN, ls='-.', lw=2.1)]
    fig.legend(handles, ['This work', 'Bergamini et al. (2019)', 'Caminha et al. (2016), F2'],
               loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=3,
               handlelength=2.8, handletextpad=.7, columnspacing=2.5)
    fig.text(.5, .88, r'$m_{\rm ref}=16.18$ (AB)'
             '     |     This work: 68% and 95% bands; literature: 68% normalization intervals',
             ha='center', fontsize=12)
    fig.text(.5, .035, 'Exponents below panels follow the legend order; literature exponents are fixed. '
             'Gray ticks mark member magnitudes.', ha='center', fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{STEM}.pdf', metadata={'Subject': comparison['status']})
    fig.savefig(OUT / f'{STEM}.png', dpi=240)
    plt.close(fig)
    provenance = {
        'input': str(BUNDLE.relative_to(ROOT)), 'sha256': digest,
        'stylesheet': 'src/lenscluster/style.mplstyle',
        'reference_magnitude_F160W_AB': mref,
        'our_exponents_alpha_beta_gamma_p16_p50_p84':
            [np.percentile(s, [16, 50, 84]).tolist() for s in slopes],
        'literature_exponents_alpha_beta_gamma': lit_slopes,
        'literature_calibration': comparison['caminha2016'],
        'bergamini_source': comparison['source'],
        'bands': 'Pointwise intervals for population scaling laws. No intrinsic member scatter or individual free-member deviations included. Literature normalization intervals propagated with fixed slopes; no joint posterior reconstructed.',
        'mass_to_light': 'Relative to each draw or model at mref, not absolute M/L. Defined by gamma, so normalization and its uncertainty cancel.',
        'status': comparison['status'],
    }
    (OUT / f'{STEM}.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (OUT / f'{STEM}_caption.tex').write_text(
        r'Luminosity scaling of AS1063 cluster members. '
        r'From left to right, the panels show $\sigma_{\rm LT}\propto L^\alpha$, '
        r'$r_{\rm cut}\propto L^\beta$, and $M/L\propto L^\gamma$, where $\beta=1+\gamma-2\alpha$. '
        r'Blue curves show our pointwise posterior medians; dark and light bands enclose 68\% and 95\% of the population-relation posterior. '
        r'These bands exclude intrinsic member scatter and individually fitted member deviations. '
        r'Orange dashed and green dot-dashed curves use the reported median normalizations of Bergamini et al. (2019) and Caminha et al. (2016, reference model F2); '
        r'their bands propagate marginal 68\% normalization intervals with fixed exponents. '
        r'The right panel normalizes each relation to its own $M/L$ at $m_{\rm F160W}=16.18$, so it compares slopes rather than absolute mass-to-light ratios. '
        r'Gray ticks mark member magnitudes; exponents below the panels follow the legend order. '
        r'Bergamini et al. fix the kinematically measured $\alpha=0.27$ and adopt $\gamma=0.2$; '
        r'Caminha et al. adopt $\alpha=0.25$ and $\gamma=0$. '
        r'For Caminha et al., we convert $\sigma_{\rm LT}=\sqrt{2/3}\,\sigma_v$ and use the reference magnitude 18.236799 from the later released configuration, assuming unchanged reference photometry. '
        r'The Caminha relation is evaluated at BCG luminosity for comparison; their BCG was fitted separately.'+'\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
