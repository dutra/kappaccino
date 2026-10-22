"""Compare saved AS1063 draws with Bergamini (2019) and Caminha (2016).

Run with the lenstronomy interpreter. No inference is performed.
Published marginal intervals are NOT treated as a joint posterior.
"""
from pathlib import Path
import hashlib
import json

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from astropy.cosmology import FlatLambdaCDM

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'results/oct15a_isotropiclatent_fullflat/as1063/anisotropic-latent/stages/stage2/artifacts/plot_bundle.h5'
OUT = BUNDLE.parent.parent / 'figures'
STEM = 'as1063_potfile_comparison'
BLUE, ORANGE, INK = '#28799B', '#C26935', '#24272B'
GREEN = '#408568'


def _load_posterior(bundle):
    from lenscluster.output_paths import stage_name
    selected_stage = stage_name(bundle.parent.parent)
    with h5py.File(bundle) as f:
        runtime = json.loads(f['runtime_args_json'][()])
        if stage_name(Path(runtime['run_name'])) != selected_stage:
            raise ValueError('Saved posterior does not match the selected stage')
        samples = f['posterior/samples'][()]
        meta = json.loads(f['state/build_state_meta_json'][()])
        div = int(np.sum(f['posterior/diverging'][()]))
        diagnostics = json.loads(f['init_diagnostics_json'][()])['nuts_quality_metrics']
    summary = dict(diagnostics, stage_semantic_name=selected_stage)
    return samples, meta, div, summary


def main():
    samples, meta, div, summary = _load_posterior(BUNDLE)
    if meta.get('scaling_relation_mode') != 'velocity-radius-exponents':
        raise ValueError('Historical alpha/gamma runs are unsupported; select a fresh beta_radius run.')
    fields = {p['field']: samples[:, i] for i, p in enumerate(meta['parameter_specs'])
              if p['potential_id'] == 'potfile_1'}
    mref = 16.18
    lum = 10**(-0.4 * (mref - meta['potfiles'][0]['mag0']))
    alpha, beta = fields['alpha_sigma'], fields['beta_radius']
    gamma = 2 * alpha + beta - 1
    physical = np.column_stack([fields['sigma'] * lum**alpha,
                                fields['cutkpc'] * lum**beta,
                                fields['corekpc'] * lum**beta, alpha, gamma])
    assert meta['core_radius_scaling'] == 'shared_beta_radius'
    assert np.all(np.isfinite(physical)) and np.all(physical[:, :3] > 0)
    scale = FlatLambdaCDM(H0=70, Om0=0.3).kpc_proper_per_arcmin(.348).value / 60
    published = np.array([[299.4, 285.2, 313.7],
                          [6.83*scale, 5.51*scale, 8.52*scale],
                          [.05*scale]*3, [.27]*3, [.2]*3])
    # C16 Table 5, reference model F2 (not the later v2 archive chain).
    # Eq. (2) uses sigma_v in Sigma=sigma_v**2/(2G)*(...):
    # sigma_LT=sqrt(2/3)*sigma_v, consistent with B19 Appendix C.
    # Reference galaxy 5170 is identified in C16 Fig. 2; its magnitude is
    # recorded by the released configuration and member catalogue.
    c16_mag0 = 18.236799
    c16_lum = 10**(-0.4*(mref-c16_mag0))
    c16 = np.array([[128., 114., 142.], [16.7, 10.6, 27.9],
                    [0.]*3, [.25]*3, [0.]*3])
    c16[0] *= np.sqrt(2/3)*c16_lum**.25
    c16[1] *= scale*c16_lum**.5
    c16_display = c16.copy()
    c16_display[1] = np.log10(c16[1])
    # C16 assumes a zero core: do not place it at an arbitrary positive floor.
    c16_display[2] = np.nan
    data, paper = physical.copy(), published.copy()
    data[:, 1:3] = np.log10(data[:, 1:3])
    paper[1:3] = np.log10(paper[1:3])
    limits = []
    for j in range(5):
        lo, hi = np.percentile(data[:, j], [.1, 99.9])
        lo, hi = min(lo, paper[j].min()), max(hi, paper[j].max())
        if j != 2:
            lo, hi = min(lo, c16_display[j].min()), max(hi, c16_display[j].max())
        pad = .075*(hi-lo)
        limits.append((lo-pad, hi+pad))
    limits[3] = (.05, .51)
    limits[4] = (-.85, .65)
    ticks = [[200, 300, 400], np.log10([10, 30, 100, 300]),
             np.log10([.03, .1, .3, 1]), [.1, .3, .5], [-.6, 0, .6]]
    ticklabels = [['200', '300', '400'], ['10', '30', '100', '300'],
                  ['0.03', '0.1', '0.3', '1'], ['0.1', '0.3', '0.5'], ['−0.6', '0', '0.6']]
    labels = [r'$\sigma_{\rm LT}\;[\mathrm{km\,s^{-1}}]$',
              r'$r_{\rm cut}\;[\mathrm{kpc}]$', r'$r_{\rm core}\;[\mathrm{kpc}]$',
              r'$\alpha$', r'$\gamma$']
    plt.style.use(ROOT / 'src/lenscluster/style.mplstyle')
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(5, 5, figsize=(12, 11.7))
    fig.subplots_adjust(left=.10, right=.99, bottom=.085, top=.99, wspace=.07, hspace=.07)
    comparison_handle = None
    c16_handle = None
    for i in range(5):
        for j in range(5):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            ax.set_xlim(limits[j]); ax.set_xticks(ticks[j]); ax.set_xticklabels(ticklabels[j])
            if i == j:
                counts, edges = np.histogram(data[:, j], bins=100, range=limits[j])
                density = gaussian_filter1d(counts.astype(float), 1.5)
                density /= density.max()
                x = (edges[1:] + edges[:-1])/2
                ax.fill_between(x, density, color=BLUE, alpha=.12)
                q16, q50, q84 = np.percentile(data[:, j], [16, 50, 84])
                ax.fill_between(x, density, where=(x >= q16) & (x <= q84), color=BLUE, alpha=.3)
                ax.plot(x, density, color=BLUE)
                ax.axvline(q50, color=BLUE, lw=.8, ls=':')
                if j < 2:
                    ax.axvspan(paper[j, 1], paper[j, 2], color=ORANGE, alpha=.15)
                    ax.axvline(paper[j, 0], color=ORANGE, lw=1.3)
                else:
                    ax.axvline(paper[j, 0], color=ORANGE, lw=1.3, ls='--')
                if j < 2:
                    ax.axvspan(c16_display[j, 1], c16_display[j, 2], color=GREEN, alpha=.12)
                    ax.axvline(c16_display[j, 0], color=GREEN, lw=1.3)
                elif j != 2:
                    ax.axvline(c16_display[j, 0], color=GREEN, lw=1.3, ls='--')
                ax.set_ylim(0, 1.12); ax.set_yticks([])
                ax.spines[['top', 'right', 'left']].set_visible(False)
                ax.tick_params(axis='x', which='both', top=False)
            else:
                ax.set_ylim(limits[i]); ax.set_yticks(ticks[i]); ax.set_yticklabels(ticklabels[i])
                hist, xe, ye = np.histogram2d(data[:, j], data[:, i], bins=100,
                                              range=[limits[j], limits[i]])
                density = gaussian_filter(hist, 1.4)
                ordered = np.sort(density.ravel())[::-1]
                cdf = np.cumsum(ordered)/ordered.sum()
                levels = sorted([ordered[np.searchsorted(cdf, p)] for p in [.68, .95]])
                x, y = (xe[1:]+xe[:-1])/2, (ye[1:]+ye[:-1])/2
                ax.contourf(x, y, density.T, levels=[*levels, density.max()+1],
                            colors=['#DCEAF0', '#91BDCE'])
                ax.contour(x, y, density.T, levels=levels, colors=BLUE, linewidths=[1.2, 1.8])
                def error(k):
                    return np.array([[paper[k, 0]-paper[k, 1]], [paper[k, 2]-paper[k, 0]]]) if k < 2 else None
                h = ax.errorbar(paper[j, 0], paper[i, 0], xerr=error(j), yerr=error(i),
                                fmt='D', ms=5.6, color=ORANGE, mfc='white', mew=1.65,
                                elinewidth=1.9, capsize=4.1, zorder=10)
                if i == 1 and j == 0:
                    comparison_handle = h
                if i != 2 and j != 2:
                    def c16_error(k):
                        return np.array([[c16_display[k, 0]-c16_display[k, 1]],
                                         [c16_display[k, 2]-c16_display[k, 0]]]) if k < 2 else None
                    h16 = ax.errorbar(c16_display[j, 0], c16_display[i, 0],
                                     xerr=c16_error(j), yerr=c16_error(i), fmt='s',
                                     ms=5.6, color=GREEN, mfc='white', mew=1.65,
                                     elinewidth=1.9, capsize=4.1, zorder=9)
                    if i == 1 and j == 0:
                        c16_handle = h16
            if i < 4:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(labels[j], labelpad=5)
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], labelpad=5)
            elif i != j:
                ax.tick_params(labelleft=False)
    legend_ax = fig.add_axes([.45, .69, .54, .29]); legend_ax.axis('off')
    legend_ax.legend([Patch(facecolor='#91BDCE', edgecolor=BLUE), comparison_handle,
                      c16_handle, Line2D([], [], color=INK, ls='--', lw=1.3)],
                     ['This work: 68% and 95% contours',
                      'Bergamini et al. (2019): median and 68% interval',
                      'Caminha et al. (2016), F2: median and 68% interval',
                      'Dashed lines: fixed parameters (matching colors)'],
                     loc='upper left', frameon=False, labelspacing=1.2,
                     handlelength=2.2, borderaxespad=0)
    legend_ax.text(0, .28, r'$m_{\rm F160W}=16.18$ (AB)')
    legend_ax.text(0, .13, r'Caminha et al.: $r_{\rm core}=0$ (not shown)')
    # Calibration notes occupy the unused upper triangle, outside data panels.
    note_x = .655
    fig.text(note_x, .675, 'Bergamini et al. (2019)', color=ORANGE,
             fontsize=12, fontweight='bold', va='top')
    fig.text(note_x, .651,
             r'MUSE kinematics: measure $\alpha$, then fix $0.27$.' '\n'
             r'Adopt $\gamma=0.2$ (fundamental plane).' '\n'
             r'Aperture-corrected $\sigma_{\rm LT}$ prior: $310\pm15$ km s$^{-1}$' '\n'
             r'(width narrowed from the measured $\simeq35$ km s$^{-1}$).',
             color=INK, fontsize=11, linespacing=1.5, va='top')
    fig.text(note_x, .555, 'Caminha et al. (2016), F2', color=GREEN,
             fontsize=12, fontweight='bold', va='top')
    fig.text(note_x, .531,
             r'Adopt $\alpha=0.25$ (Faber-Jackson) and' '\n'
             r'$\gamma=0$ (constant $M/L$); no kinematic prior.' '\n'
             r'Lensing fits $\sigma_{\rm LT}$ and $r_{\rm cut}$ with flat priors.',
             color=INK, fontsize=11, linespacing=1.5, va='top')
    fig.text(note_x, .455,
             r'Both fix $\beta=1+\gamma-2\alpha$: $0.66$ and $0.50$, respectively.',
             color=INK, fontsize=11, va='top')
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{STEM}.pdf', metadata={'Title': 'AS1063 member scaling comparison',
                                            'Author': '', 'Subject': 'Provisional saved posterior; see provenance JSON.'})
    fig.savefig(OUT / f'{STEM}.png', dpi=240)
    plt.close(fig)
    provenance = {'stylesheet': 'src/lenscluster/style.mplstyle',
                  'input': str(BUNDLE.relative_to(ROOT)),
                  'sha256': hashlib.sha256(BUNDLE.read_bytes()).hexdigest(),
                  'reference_magnitude_F160W_AB': mref, 'draws': len(samples), 'divergences': div,
                  'parameters': ['sigma_LT_km_s', 'rcut_kpc', 'rcore_kpc', 'alpha', 'gamma'],
                  'our_percentiles_16_50_84': np.percentile(physical, [16,50,84], axis=0).T.tolist(),
                  'published_median_p16_p84': published.tolist(),
                  'source': 'https://arxiv.org/html/1905.13236v3, Table 3 and Section 5.3',
                  'caminha2016': {
                      'source': 'https://arxiv.org/pdf/1512.04555, Eq. 2-3, Tables 4-5, model F2',
                      'reference_magnitude': c16_mag0,
                      'reference_source': 'data/Bergamini/A1063_Caminha16/A1063_lenstool_Caminha2016.par; member 5170 in members_95_160_nonan_cuts.cat, identified with C16 Fig. 2',
                      'reference_caveat': 'Numerical reference magnitude is taken from the later released v2 configuration; the paper identifies the reference galaxy but does not tabulate its magnitude. Assumes the same reference photometry.',
                      'sigma_table_to_LT_factor': float(np.sqrt(2/3)),
                      'luminosity_ratio': c16_lum,
                      'median_p16_p84_at_common_magnitude': c16.tolist(),
                      'archive_chain_used': False,
                      'archive_warning': 'The local bayes.dat is an updated model with Karman 2017 redshifts, not the published F2 posterior.',
                      'zero_core_display': 'Omitted from logarithmic core panels; alpha=0.25 and gamma=0 fixed.'},
                  'contours': 'Smoothed histograms in displayed coordinates; 68% and 95% enclosed probability; all chains retained.',
                  'comparison': 'Published marginal error bars only. No reconstructed joint posterior or assumed correlations.',
                  'status': f"Provisional: {div} divergences; global Rhat maximum={summary['rhat_max']:.3f}; minimum ESS={summary['ess_min']:.2f}.",
                  'stage': summary['stage_semantic_name']}
    (OUT / f'{STEM}.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (OUT / f'{STEM}_caption.tex').write_text(
        r'AS1063 cluster-member scaling relations. '
        r'All normalizations are evaluated at $m_{\rm F160W}=16.18$ (AB). '
        r'Blue contours enclose 68\% and 95\% of our marginalized two-dimensional distributions; '
        r'the diagonal panels show the one-dimensional distributions, with darker shading between the 16th and 84th percentiles. '
        r'Orange diamonds and error bars show the medians and marginal 68\% intervals reported by Bergamini et al. (2019). '
        r'Green squares show the corresponding intervals for the reference model F2 of Caminha et al. (2016). '
        r'Dashed lines mark fixed parameters in the matching colors; the zero core radius of Caminha et al. is omitted from the logarithmic core panels. '
        r'For Caminha et al., we use the reference magnitude 18.236799 from the released model configuration, assuming unchanged reference photometry, '
        r'and convert their tabulated dispersion using $\sigma_{\rm LT}=\sqrt{2/3}\,\sigma_v$. '
        r'The published marginal intervals do not specify joint credible regions. '
        r'The velocity dispersion uses the Lenstool convention, and both radii are shown on logarithmic axes. '
        r'The exponents satisfy $\sigma_{\rm LT}\propto L^\alpha$ and $M/L\propto L^\gamma$, '
        r'with $r_{\rm cut}\propto L^{1+\gamma-2\alpha}$. '
        r'Bergamini et al. measure $\alpha$ from MUSE stellar kinematics and fix it to 0.27 in the lensing fit; '
        r'they adopt $\gamma=0.2$ from the fundamental-plane scaling. '
        r'After correcting the aperture dispersions to the Lenstool convention, they impose a Gaussian normalization prior of '
        r'$310\pm15\,\mathrm{km\,s^{-1}}$, narrowing the measured uncertainty of approximately $35\,\mathrm{km\,s^{-1}}$. '
        r'Caminha et al. adopt $\alpha=0.25$ and $\gamma=0$, corresponding to classical Faber--Jackson scaling and constant total $M/L$, '
        r'and fit the normalizations with flat priors and no stellar-kinematic constraint. '
        r'The resulting radius exponents are fixed to 0.66 and 0.50, respectively.' + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
