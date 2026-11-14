"""Publication macros from recorded results; no inference or plot generation."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from ..io.hdf5 import load_arrays

_DIGITS = ('Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine')
_ALIASES = {'ares': 'Ares', 'hera': 'Hera', 'm0416': 'MJZeroFourOneSix',
            'mj0416': 'MJZeroFourOneSix', 'macs0416': 'MJZeroFourOneSix',
            'm1206': 'MJOneTwoZeroSix', 'mj1206': 'MJOneTwoZeroSix',
            'as1063': 'ASOneZeroSixThree', 'mock': 'Mock'}


def command_prefix(name: str) -> str:
    key = re.sub(r'[^a-z0-9]', '', name.lower())
    if key in _ALIASES:
        return _ALIASES[key]
    words = re.findall(r'[A-Za-z]+|[0-9]', name)
    prefix = ''.join(_DIGITS[int(w)] if w.isdigit() else w[:1].upper() + w[1:] for w in words)
    if not prefix:
        raise ValueError('Results name must contain letters or digits')
    return prefix


def number(value, decimals=3) -> str:
    if value is None or not math.isfinite(float(value)):
        return r'\textnormal{not available}'
    return f'{float(value):.{decimals}f}'


def duration(seconds) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return r'\textnormal{not available}'
    hours, remainder = divmod(int(math.floor(float(seconds) + .5)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return rf'\ensuremath{{{hours}\,\mathrm{{h}}\;{minutes}\,\mathrm{{min}}\;{seconds}\,\mathrm{{s}}}}'


def _read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _write(path, prefix, metrics, comments):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['% Generated results. Do not edit; regenerate from saved artifacts.',
             '% Residuals are in arcseconds; fractions are percentages only when named Percent.',
             *['% ' + text for text in comments]]
    for suffix, value in metrics.items():
        command = prefix + suffix
        if not re.fullmatch('[A-Za-z]+', command):
            raise ValueError(f'Invalid TeX command: {command}')
        lines.append(rf'\newcommand{{\{command}}}{{{value}}}')
    path.write_text('\n'.join(lines) + '\n')
    return path


def interval(median, lower_error, upper_error, decimals=1):
    """TeX numerical interval in braces, to be used in math mode.  The braces make it one group, so a unit written as a
    superscript right after it (``$\\HeraRMS''$``, arcsec) does not become a double superscript."""
    return ('{' + number(median, decimals) + '_{-' + number(lower_error, decimals)
            + '}^{+' + number(upper_error, decimals) + '}}')


def m0416_table_metrics(final):
    """Saved-draw model comparison; R25 published Tables 4--5, A&A 696, A15 (2025)."""
    import numpy as np
    import pandas as pd
    summary_path = final / 'tables/posterior_summary.csv'
    state_path = final.parent / 'init/artifacts/state.json'
    if not summary_path.exists() or not state_path.exists():
        return {}
    table = pd.read_csv(summary_path).set_index('parameter')
    state = _read(state_path)
    scale = state['kpc_per_arcsec']
    result = {}
    components = [('HaloOne','1'), ('HaloTwo','2'), ('HaloThree','3'), ('HaloFour','4'),
                  ('Foreground','5'), ('BCGNorth','bcg_2526'), ('BCGSouth','bcg_3158'),
                  ('SouthernGalaxy','los736'), ('SouthernDark','dark_s')]
    for label, component in components:
        for suffix, field, factor, decimals in [('Sigma','v_disp',1,0),
                ('Core','core_radius_kpc',scale,1), ('Cut','cut_radius_kpc',scale,1)]:
            key = component+'.'+field
            if key in table.index:
                row = table.loc[key]
                result[label+suffix] = interval(row['median']/factor,
                    (row['median']-row.p16)/factor, (row.p84-row['median'])/factor, decimals)
    for label, field, decimals in [('ScalingAlpha','alpha_sigma',3), ('ScalingBeta','beta_radius',3),
            ('ScalingSigmaScatter','velocity_log_scatter',3), ('ScalingRadiusScatter','radius_log_scatter',2)]:
        key = 'potfile_1.'+field
        if key in table.index:
            row = table.loc[key]
            result[label] = interval(row['median'],row['median']-row.p16,row.p84-row['median'],decimals)
    result['ComparisonMagnitude'] = '17.02'
    posterior_path = final / 'artifacts/posterior.h5'
    members_path = final / 'tables/member_diagnostics.csv'
    if posterior_path.exists() and members_path.exists():
        archive = load_arrays(posterior_path)
        physical = archive['physical']
        index = {spec['name']:i for i,spec in enumerate(state['parameter_specs'])}
        for key in table.index:
            if key in index:
                np.testing.assert_allclose(np.median(physical[:,index[key]]), table.loc[key,'median'],rtol=1e-9,atol=1e-10)
        def v(field):
            return physical[:,index['potfile_1.'+field]]
        def quantile(key, values, decimals):
            low, median, high = np.quantile(values,[.16,.5,.84])
            result[key] = interval(median,median-low,high-median,decimals)
        members = pd.read_csv(members_path)
        group = members[members.potfile_id == 'potfile_1']
        pivots = group.catalog_mag + 2.5*np.log10(group.luminosity_ratio)
        pivot = float(pivots.median())
        np.testing.assert_allclose(pivots,pivot,atol=1e-8)
        quantile('ScalingSigma',v('sigma')*10**(.4*v('alpha_sigma')*(pivot-17.02)),0)
        quantile('ScalingCut',v('cutkpc')/scale*10**(.4*v('beta_radius')*(pivot-17.02)),1)
        deviations_path = final / 'tables/member_diagnostic_draws.h5'
        match = np.flatnonzero(members.member_id.to_numpy() == 'potfile_1:8971')
        if len(match)==1 and deviations_path.exists():
            j = int(match[0])
            archive = load_arrays(deviations_path)
            deviations = archive['deviations']
            if deviations.shape != (len(physical),len(members),2):
                raise ValueError('Member deviation draws do not align with saved posterior')
            # Saved deviations already include centering/scatter transformations.
            row = members.iloc[j]
            for axis, prefix in [(0,'velocity'),(1,'radius')]:
                np.testing.assert_allclose(np.quantile(deviations[:,j,axis],[.16,.5,.84]),
                    [row[prefix+'_p16'],row[prefix+'_median'],row[prefix+'_p84']],atol=1e-9)
            lum = float(row.luminosity_ratio)
            quantile('GalEightNineSevenOneSigma',v('sigma')*lum**v('alpha_sigma')*np.exp(deviations[:,j,0]),0)
            quantile('GalEightNineSevenOneCut',v('cutkpc')/scale*lum**v('beta_radius')*np.exp(deviations[:,j,1]),1)
    return result


def footprint_metrics(final, cfg):
    """Mean model/true ratios within the image footprint (``truth_recovery.footprint_mean_ratios``) from the saved
    profile tables, for a simulated cluster with a truth map; empty otherwise.  This work's numbers are the medians over
    the saved posterior draws (``truth_recovery_*_ratio_draws.csv``), with ``...Interval`` macros carrying the 16-84 per
    cent range; without those tables they come from the median profile and the intervals are not available."""
    import numpy as np
    import pandas as pd
    from ..config import TruthRecoveryConfig
    from ..diagnostics.truth_recovery import footprint_mean_ratios, literature_ratios
    tables = final / 'tables'
    paths = [tables / f for f in ('truth_recovery_m2d_aperture_profile.csv', 'truth_recovery_kappa_annulus_profile.csv',
                                  'image_fit_quality.csv')]
    if not all(path.exists() for path in paths) or not cfg.get('truth', {}).get('kappa_true_fits'):
        return {}
    mass, kappa, images = (pd.read_csv(path) for path in paths)
    draws = [tables / f for f in ('truth_recovery_m2d_aperture_ratio_draws.csv', 'truth_recovery_kappa_annulus_ratio_draws.csv')]
    mass_draws, kappa_draws = (pd.read_csv(path) for path in draws) if all(path.exists() for path in draws) else (None, None)
    radii = np.hypot(images['x_obs_arcsec'] - mass['center_x_arcsec'].iloc[0], images['y_obs_arcsec'] - mass['center_y_arcsec'].iloc[0])
    truth = TruthRecoveryConfig(**cfg['truth'])
    found = footprint_mean_ratios(mass, kappa, radii.to_numpy(float), literature_ratios(truth, 'mass'),
                                  literature_ratios(truth, 'kappa'), mass_draws, kappa_draws)
    result = {'FootprintInnerArcsec': number(found['inner_arcsec'], 1), 'FootprintOuterArcsec': number(found['outer_arcsec'], 1),
              'FootprintNDraws': number(None if mass_draws is None else mass_draws['draw'].nunique(), 0)}
    for quantity, label in (('mass', 'Mass'), ('kappa', 'Kappa')):
        for model, value in found[quantity].items():
            ours = model == 'This work'
            suffix = '' if ours else re.sub(r'[^A-Za-z]', '', model)
            for index, (name, scale, decimals) in enumerate((('Ratio', 1, 3), ('MeanAbsPercent', 100, 1), ('RMSPercent', 100, 1))):
                key = f'Footprint{label}{name}{suffix}'
                if value is None:
                    result[key] = number(None)
                elif ours and np.ndim(value) == 2:
                    median, low, high = scale * value[:, index]
                    result[key] = number(median, decimals)
                    result[key + 'Interval'] = interval(median, median - low, high - median, decimals)
                    result[key + 'Minus'] = number(median - low, decimals)
                    result[key + 'Plus'] = number(high - median, decimals)
                else:
                    result[key] = number(scale * value[index], decimals)
                    if ours:
                        for extra in ('Interval', 'Minus', 'Plus'):
                            result[key + extra] = number(None)
    return result


_REFERENCE_NAMES = {3: 'Three', 10: 'Ten', 30: 'Thirty'}


def truth_magnification_metrics(final):
    """Magnification against the truth of a simulated cluster, from ``magnification_scatter.csv`` and
    ``cumulative_source_area.csv``; every macro is "not available" without its table.

    ``MagnificationScatterMu<Three|Ten>Percent`` and ``MagnificationBiasMu...Percent``: scatter and bias of the model
    magnification over the reporting draws at the map pixels where the truth has that magnification
    (``plotting.scatter_summary.conditional_scatter``), in per cent.  ``SourceAreaRatioMu<Three|Ten|Thirty>``: median
    over the draws of the source-plane area magnified above that threshold, over the true area (interpolated in
    log mu).  No prefix is the lens of each draw, parametric plus its realization of the GRF (the parametric lens
    when the run has no field); ``LensOnly...`` the same draws without the GRF."""
    import numpy as np
    import pandas as pd
    result = {}
    scatter_path, area_path = final / 'tables/magnification_scatter.csv', final / 'tables/cumulative_source_area.csv'
    scatter = pd.read_csv(scatter_path).set_index(['model', 'mu_ref']) if scatter_path.exists() else None
    area = pd.read_csv(area_path) if area_path.exists() else None
    for prefix, model in (('', 'model'), ('LensOnly', 'parametric')):
        for mu in (3, 10):
            for suffix, column in (('Scatter', 'scatter'), ('Bias', 'bias')):
                value = None
                if scatter is not None:
                    key = (model if (model, float(mu)) in scatter.index else 'parametric', float(mu))
                    value = 100 * scatter.loc[key, column] if key in scatter.index else None
                result[f'{prefix}Magnification{suffix}Mu{_REFERENCE_NAMES[mu]}Percent'] = number(
                    value if value is None or not math.isfinite(value) else round(value) + 0., 0)   # + 0. turns -0 into 0
        for mu in (3, 10, 30):
            value = None
            if area is not None:
                column = f'area_{model}_median_arcsec2'
                column = column if column in area else 'area_parametric_median_arcsec2'
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = area[column].to_numpy(float) / area['area_true_arcsec2'].to_numpy(float)
                value = float(np.interp(np.log(mu), np.log(area['mu'].to_numpy(float)), ratio, left=np.nan, right=np.nan))
            result[f'{prefix}SourceAreaRatioMu{_REFERENCE_NAMES[mu]}'] = number(value, 2)
    return result


def count_metrics(suffix, values, decimals=0):
    """``<suffix>`` (the median over draws) and ``<suffix>Interval`` (median with its 16-84% range, for math mode)
    from ``{"median", "q16", "q84"}``; "not available" without it."""
    if not values:
        return {suffix: number(None, decimals), suffix + 'Interval': number(None)}
    median = values['median']
    return {suffix: number(median, decimals),
            suffix + 'Interval': interval(median, median - values['q16'], values['q84'] - median, decimals)}


def posterior_residual_metrics(path, prefix):
    """``<prefix>RMS`` and ``<prefix>MedianResidual`` (arcsec, three decimals, for math mode): the median and 16th-84th
    percentile range over the posterior draws in ``path`` (``posterior_image_residual_draws*.csv``) of each draw's RMS
    and median residual of its recovered images; "not available" without the table."""
    import numpy as np
    import pandas as pd
    result = {prefix + 'RMS': number(None), prefix + 'MedianResidual': number(None)}
    if not Path(path).exists():
        return result
    draws = pd.read_csv(path)
    values = pd.to_numeric(draws['image_residual_arcsec'], errors='coerce')
    draws = draws.assign(r=values)[draws['recovered'].astype(bool) & np.isfinite(values) & (values >= 0)]
    if draws.empty:
        return result
    grouped = draws.groupby('draw_index')['r']
    for name, per_draw in (('RMS', grouped.apply(lambda r: float(np.sqrt(np.mean(r ** 2))))), ('MedianResidual', grouped.median())):
        low, median, high = np.percentile(per_draw.to_numpy(float), [16, 50, 84])
        result[prefix + name] = interval(median, median - low, high - median, 3)
    return result


def write_run_results(final_dir: str | Path, *, name: str | None = None) -> Path:
    r"""Write LaTeX macros of a finished run's recorded results.

    Reads only saved outputs under ``final_dir`` (no inference, no plots) and
    writes one ``\newcommand`` per metric to
    ``final/figures/<name>_results.tex``. Each command is the prefix of
    :func:`command_prefix` (``name`` with known clusters aliased, e.g. ``m0416``
    to ``MJZeroFourOneSix``, digits spelled out) followed by a suffix. A quantity
    that is missing from the saved files is written as ``\textnormal{not
    available}``; nothing is inferred.

    Files read (relative to the run root; all except the first are optional):
    ``final/artifacts/summary.json``, ``final/artifacts/config.json``,
    ``final/artifacts/posterior_diagnostics.json``,
    ``final/tables/run_summary.json``, ``final/tables/runtime.json``,
    ``final/tables/posterior_summary.csv``,
    ``final/tables/posterior_image_rms.csv``,
    ``final/tables/image_fit_quality.csv`` and
    ``final/tables/aperture_mass_histogram.json``. For an M0416 run (``name``
    containing ``m0416`` or ``mj0416``) it also reads
    ``final/artifacts/posterior.h5``, ``final/tables/member_diagnostics.csv``,
    ``final/tables/member_diagnostic_draws.h5`` and
    ``init/artifacts/state.json`` to add halo and scaling-relation macros
    comparable with Rihtarsic et al. (2025).

    Main suffixes: counts (``NImages``, ``NRecovered``, ``NMissed``,
    ``NFamilies``, ``NFamiliesRecovered``, ``NParameters``, ``NMembers``,
    ``NExactMembers``, ``NFarFieldMembers``, ``NDivergences``, ``NChains``,
    ``NWarmupPerChain``, ``NSamplesPerChain``, ``NExtraImages``); image
    residuals in arcsec as posterior median and 16-84% range over the saved draws
    (``RMS``, ``MedianResidual``, and ``LensOnlyRMS``, ``LensOnlyMedianResidual``
    for the same draws without the GRF; ``posterior_residual_metrics``),
    ``RecoveredPercent`` and ``NPosteriorRMSDraws`` (counts and percentages are medians over the reporting draws,
    with ``...Interval`` the 16-84% range); position coverage (``PositionCoverageOneSigmaPercent``,
    ``PositionCoverageTwoSigmaPercent`` and ``NPositionCoverage``); likelihood-noise posterior
    medians and intervals (``SigmaGP``, ``SigmaMeas``,
    ``EllGP``, ``SigmaGRF``, ``LambdaB``;
    angular scales in arcsec); convergence (``MaxRhat``, ``MinESS``); run time
    (``Runtime``, ``SamplingRuntime`` as h/min/s, plus ``...Seconds`` and
    ``...Hours``); and ``ApertureMassRatio...`` macros when the aperture
    histogram exists; and, for a simulated cluster, the magnification against
    the truth (``MagnificationScatterMuThreePercent``, ``SourceAreaRatioMuTen``
    and the like, with ``LensOnly...``; ``truth_magnification_metrics``).

    Parameters
    ----------
    final_dir : str or pathlib.Path
        The run's ``final`` directory (``<run root>/final``).
    name : str or None, optional
        Name used for the file name (non-alphanumeric characters other than
        ``_`` and ``-`` replaced by ``_``) and the macro prefix. None (default)
        uses ``paths.figure_prefix`` from ``final/artifacts/config.json``, or
        else the name of the run root directory.

    Returns
    -------
    pathlib.Path
        The written file, ``final/figures/<name>_results.tex``.

    Raises
    ------
    FileNotFoundError
        If ``final/artifacts/summary.json`` does not exist.
    ValueError
        If ``name`` yields no valid TeX command name (letters only), or, for an
        M0416 run, the saved member deviation draws do not align with the saved
        posterior.
    AssertionError
        For an M0416 run, if the saved posterior draws disagree with
        ``posterior_summary.csv`` or ``member_diagnostics.csv``.
    """
    final = Path(final_dir)
    if not (final / 'artifacts/summary.json').is_file():
        raise FileNotFoundError(f'No saved run summary under {final / "artifacts"}')
    cfg = _read(final / 'artifacts/config.json')
    name = name or cfg.get('paths', {}).get('figure_prefix') or final.parent.name
    filename = re.sub(r'[^A-Za-z0-9_-]', '_', name)
    summary = _read(final / 'artifacts/summary.json')
    fit = _read(final / 'tables/run_summary.json').get('fit', {})
    diag = _read(final / 'artifacts/posterior_diagnostics.json') or summary.get('nuts', {})
    timing = _read(final / 'tables/runtime.json')
    metrics = {}
    for suffix, value in {
        'NImages': fit.get('observed_image_count', summary.get('images')),
        'NFamilies': summary.get('families'), 'NFamiliesRecovered': fit.get('families_fully_recovered'),
        'NParameters': summary.get('n_parameters'), 'NMembers': summary.get('members'),
        'NExactMembers': summary.get('exact_members'), 'NDivergences': summary.get('divergences'),
        'NChains': cfg.get('runtime', {}).get('chains'),
        'NWarmupPerChain': cfg.get('sampling', {}).get('warmup'),
        'NSamplesPerChain': cfg.get('sampling', {}).get('samples'),
    }.items():
        metrics[suffix] = number(value, 0)
    members, exact = summary.get('members'), summary.get('exact_members')
    metrics['NFarFieldMembers'] = number(members-exact if members is not None and exact is not None else None, 0)
    # Image residuals: median and 16-84% range over the saved posterior draws (each draw's own lens, and the parametric lens without its GRF realization).
    for prefix, table in (('', 'posterior_image_residual_draws.csv'), ('LensOnly', 'posterior_image_residual_draws_lens_only.csv')):
        metrics.update(posterior_residual_metrics(final / 'tables' / table, prefix))
    # Counts over the reporting draws: median, with the 16-84% range in ...Interval (knee-GP runs have LensOnly...).
    total = fit.get('observed_image_count')
    for suffix, key in (('NRecovered', 'recovered_image_count'), ('NLensOnlyRecovered', 'lens_only_recovered_image_count'),
                        ('NExtraImages', 'extra_image_count')):
        metrics.update(count_metrics(suffix, fit.get(key)))
    recovered = fit.get('recovered_image_count')
    metrics.update(count_metrics('NMissed', None if recovered is None or total is None else
                                 {'median': total - recovered['median'], 'q16': total - recovered['q84'], 'q84': total - recovered['q16']}))
    metrics.update(count_metrics('RecoveredPercent', None if recovered is None or not total else
                                 {k: 100 * v / total for k, v in recovered.items()}, decimals=1))
    # Keep the median individual residual distinct from the median of draw RMSs.
    import numpy as np
    import pandas as pd
    parameter_path = final / 'tables/posterior_summary.csv'
    parameters = pd.read_csv(parameter_path).set_index('parameter') if parameter_path.exists() else pd.DataFrame()
    for suffix, parameter, decimals in [
        ('SigmaGP', 'images.sigma_gp', 3),
        ('SigmaMeas', 'images.sigma_meas', 3),
        ('EllGP', 'images.ell_gp', 1),
        ('SigmaGRF', 'images.sigma_grf', 2),
        ('LambdaB', 'images.lambda_b', 1),
    ]:
        value = parameters.loc[parameter, 'median'] if parameter in parameters.index else None
        metrics[suffix] = number(value, decimals)
        metrics[suffix+'Interval'] = number(None)
        if parameter in parameters.index and {'p16', 'p84'}.issubset(parameters.columns):
            row = parameters.loc[parameter]
            metrics[suffix+'Interval'] = interval(value, value-row.p16, row.p84-value, decimals)
    metrics['GRFReferenceRedshift'] = number(cfg.get('likelihood', {}).get('gp_reference_redshift'), 1)
    rms_path = final / 'tables/posterior_image_rms.csv'
    metrics['NPosteriorRMSDraws'] = number(None, 0)
    if rms_path.exists():
        metrics['NPosteriorRMSDraws'] = str(int(np.isfinite(pd.read_csv(rms_path)['image_rms_arcsec'].to_numpy(float)).sum()))
    coverage_path = final / 'tables/position_coverage.csv'
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else pd.DataFrame()
    levels = coverage['credible_level'].to_numpy(float) if 'credible_level' in coverage else np.empty(0)
    levels = levels[np.isfinite(levels)]
    metrics['NPositionCoverage'] = str(len(levels)) if len(coverage) else number(None, 0)
    for suffix, probability in (('OneSigma', .682689492), ('TwoSigma', .954499736)):
        metrics['PositionCoverage'+suffix+'Percent'] = number(100*np.mean(levels <= probability) if len(levels) else None, 0)
    metrics['MaxRhat'] = number(diag.get('max_r_hat'), 4)
    metrics['MinESS'] = number(diag.get('min_n_eff'), 1)
    for suffix, seconds in [('Runtime', timing.get('total')), ('SamplingRuntime', diag.get('elapsed_seconds'))]:
        metrics[suffix] = duration(seconds)
        metrics[suffix+'Seconds'] = number(seconds, 1)
        metrics[suffix+'Hours'] = number(seconds/3600 if seconds is not None else None, 3)
    if 'm0416' in name.lower() or 'mj0416' in name.lower():
        metrics.update(m0416_table_metrics(final))
    aperture = _read(final / 'tables/aperture_mass_histogram.json')
    if aperture:
        metrics['ApertureEffectiveRadiusNormalizationKpc'] = number(aperture['normalization_kpc'], 1)
        for sample, label in [('all', 'All'), ('image_hull', 'ImageFootprint')]:
            row = aperture['samples'][sample]
            stem = 'ApertureMassRatio' + label
            metrics[stem + 'NMembers'] = str(int(row['n']))
            for field, suffix in [('median','Median'), ('p25','LowerQuartile'),
                                  ('p75','UpperQuartile'), ('iqr','IQR'), ('Q','Score')]:
                metrics[stem + suffix] = number(row[field], 3)
    metrics.update(footprint_metrics(final, cfg))
    metrics.update(truth_magnification_metrics(final))
    return _write(final / 'figures' / f'{filename}_results.tex', command_prefix(name), metrics,
                  ['Image recovery over the reporting posterior draws: counts are medians, ...Interval the 16-84% range; no reduced chi-square exported.',
                   'This work: median over the saved posterior draws (FootprintNDraws), ...Interval with the 16-84% range,',
                   '...Minus and ...Plus the distances from the median to the 16th and 84th percentiles.',
                   'Footprint...Ratio: mean model/true ratio; Footprint...MeanAbsPercent and ...RMSPercent: mean |model/true - 1|',
                   'and rms of model/true - 1, in per cent,',
                   'of the enclosed mass (Mass) or the azimuthally averaged',
                   'convergence (Kappa), uniform in log R from FootprintInnerArcsec to FootprintOuterArcsec (the outermost image, cut',
                   'at the edge of the published figure); no suffix is this work (median over draws), a model name suffix is that',
                   'published parametric reconstruction of Meneghetti et al. (2017), over the part of the range its curve covers.',
                   'RMS and MedianResidual (and LensOnly...): median and 16-84% range over posterior draws of each draw RMS and median residual of its recovered images, for math mode.',
                   'RMS and MedianResidual use the lens of each draw, parametric plus its realization of the GRF; LensOnlyRMS and LensOnlyMedianResidual the same draws without the GRF.',
                   'MagnificationScatterMu...Percent and MagnificationBiasMu...Percent: scatter (half the 16-84% range over the median) and bias (median - 1)',
                   'of model/true magnification over all reporting draws, at map pixels inside the image hull whose true magnification is within 10% of 3 or 10',
                   '(after Raney et al. 2020); SourceAreaRatioMu...: median over draws of the source-plane area magnified above 3, 10 or 30, over the true area.',
                   'Both at the truth-map source redshift; no prefix is parametric plus the GRF realization of each draw, LensOnly the same draws without the GRF.',
                   'GP and GRF parameter macros are posterior medians; angular scales are in arcseconds.',
                   'ApertureMassRatio macros, when present: galaxy-distribution median and quartiles, not posterior intervals.',
                   'Apertures use twice the estimated effective radius; normalization is Re at F160W=20 in kpc.',
                   'ImageFootprint means galaxy centers inside the observed-image convex hull. Score is -log10(IQR*abs(median-1)).',
                   'NPosteriorRMSDraws counts the draws with a finite RMS.',
                   'PositionCoverageOneSigma/TwoSigmaPercent: catalog positions inside the 68.3% and 95.4% regions of their predicted observed positions '
                   '(each draw lens position plus one draw of the image-plane GP and measurement error); NPositionCoverage images have a region.',
                   'These are in-sample checks of the size of the fitted scatter, not held-out predictive errors.',
                   'Runtime is recorded full-run wall time, including initialization and outputs.',
                   'SamplingRuntime is the saved sampler elapsed time, not total run time.',
                   'NSamplesPerChain is configured sampling iterations before thinning.',
                   'Unavailable values are not inferred from modification timestamps.'])


def spread_interval(values, decimals=3):
    """Median and 16-84% range of ``values`` as a math-mode interval; "not available" when empty."""
    import numpy as np
    if not len(values):
        return number(None)
    low, median, high = np.percentile(values, [16, 50, 84])
    return interval(median, median - low, high - median, decimals)


def write_mock_results(campaign, output: Path, *, rhat_cut: float, min_ess: float) -> Path:
    r"""Write LaTeX macros summarizing a mock-validation campaign.

    Writes ``output/mock_results.tex`` with ``\newcommand`` macros prefixed
    ``Mock``. Aggregates follow the residual figure rather than pooling all
    images: for each (seed, posterior draw) group of
    ``campaign.residual_draws``, the RMS and median of the finite, nonnegative
    residuals of recovered images are computed, and the macros report the
    median of each across groups.

    Macros: ``MockNCompleted`` (realizations in ``campaign.summary``),
    ``MockNIncluded`` (``campaign.seeds``), ``MockNExcluded``, ``MockRhatCut``,
    ``MockMinESSCut``, ``MockRMS`` and ``MockMedianResidual`` (arcsec, median and 16-84% range over the groups, math mode),
    ``MockNImages``, ``MockNRecovered``, ``MockNMissed`` (sums over included
    seeds; not available if any value is missing), and ``MockRuntime``,
    ``MockRuntimeSeconds``, ``MockRuntimeHours`` (always not available: a
    campaign's run time is not inferred).

    Parameters
    ----------
    campaign : kappaccino.mock_validation.aggregate_plots.Campaign
        The loaded campaign. Uses ``summary`` (DataFrame with ``seed``,
        ``observed_image_count``, ``recovered_image_count`` (the median over the
        reporting draws)), ``seeds`` (the included seeds) and
        ``residual_draws`` (DataFrame with ``seed``, ``draw_index``,
        ``image_residual_arcsec``, ``recovered``).
    output : pathlib.Path
        Directory to write into, normally ``<campaign root>/aggregate_plots``
        (created if needed).
    rhat_cut : float
        Convergence cut recorded as ``MockRhatCut`` (dimensionless; a seed
        converges with max R-hat strictly below it). Only reported here.
    min_ess : float
        Minimum effective sample size recorded as ``MockMinESSCut``. Only
        reported here.

    Returns
    -------
    pathlib.Path
        The written file, ``output / "mock_results.tex"``.
    """
    import numpy as np
    import pandas as pd
    selected = campaign.summary[campaign.summary.seed.isin(campaign.seeds)]
    rms, medians = [], []
    for _, group in campaign.residual_draws.groupby(['seed', 'draw_index']):
        values = pd.to_numeric(group.image_residual_arcsec, errors='coerce').to_numpy(float)
        values = values[group.recovered.astype(bool).to_numpy() & np.isfinite(values) & (values >= 0)]
        if len(values):
            rms.append(np.sqrt(np.mean(values**2)))
            medians.append(np.median(values))
    metrics = {'NCompleted': str(len(campaign.summary)), 'NIncluded': str(len(campaign.seeds)),
               'NExcluded': str(len(campaign.summary)-len(campaign.seeds)),
               'RhatCut': number(rhat_cut, 2), 'MinESSCut': number(min_ess, 0),
               'RMS': spread_interval(rms), 'MedianResidual': spread_interval(medians),
               'Runtime': duration(None), 'RuntimeSeconds': number(None),
               'RuntimeHours': number(None)}
    totals = {}
    for suffix, col in [('NImages','observed_image_count'),('NRecovered','recovered_image_count')]:
        values = pd.to_numeric(selected[col], errors='coerce') if col in selected else None
        totals[suffix] = values.sum() if values is not None and values.notna().all() else None
        metrics[suffix] = number(totals[suffix], 0)
    missed = None if None in totals.values() else totals['NImages'] - totals['NRecovered']
    metrics['NMissed'] = number(missed, 0)
    return _write(output / 'mock_results.tex', 'Mock', metrics,
                  ['Counts sum over included mocks the per-mock median count over its reporting draws.',
                   'RMS and MedianResidual: median and 16-84% range across recovered-image mock/draw groups, for math mode.',
                   'Convergence diagnostics: max Rhat strictly below RhatCut; min ESS at least MinESSCut.',
                   'Aggregates include all completed mocks by default; optional converged-only selection is recorded in campaign_selection.json.',
                   'No campaign runtime inferred: concurrent/resumed runs cannot be timed by summing draws.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('final_dirs', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.final_dirs:
        print(write_run_results(path))


if __name__ == '__main__':
    main()
