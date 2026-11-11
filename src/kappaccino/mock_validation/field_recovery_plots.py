"""Campaign-level GRF/GP field recovery from saved posterior field diagnostics."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .aggregate_plots import STYLE, PAPER_STYLE, parameter_recovery_bins, _save, _polish
from ..io.hdf5 import load_arrays


@plt.rc_context(fname=STYLE, rc=PAPER_STYLE)
def plot_field_recovery(campaign, output: Path):
    """Pool image components, keeping mocks as independent units for binned SEM."""
    rows = []
    for seed in campaign.seeds:
        path = campaign.root/'aggregate_plots'/f'seed_{seed:08d}_posterior_fields.h5'
        if not path.exists():
            continue
        data = load_arrays(path)
        n = len(data['xy'])
        center = data['mean_images']
        sigma = np.sqrt(data['within_variance_images']+data['between_variance_images'])
        for field, start in [('GRF', 0), ('GP', 2*n)]:
            for i in range(n):
                for axis in range(2):
                    index = start+2*i+axis
                    rows.append(dict(seed=seed, image_index=i, field=field, component='xy'[axis],
                                     truth=float(data['truth_images'][index]), posterior_mean=float(center[index]),
                                     posterior_sd=float(sigma[index])))
    if not rows:
        return None
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output/'mock_grf_gp_field_recovery.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 6.5))
    handles = {}
    bins_saved = []
    for row, field in enumerate(('GRF', 'GP')):
        for col, component in enumerate('xy'):
            ax = axes[row, col]
            data = table[(table.field == field) & (table.component == component)]
            finite = np.isfinite(data[['truth','posterior_mean','posterior_sd']]).all(axis=1)
            data = data[finite]
            if data.empty:
                ax.set_axis_off(); continue
            raw = ax.errorbar(data.truth, data['posterior_mean'], yerr=data.posterior_sd,
                              fmt='.', markersize=2.5, color='tab:blue', alpha=.22,
                              elinewidth=.45, capsize=0, zorder=2)
            handles.setdefault('Images (posterior SD)', raw)
            bins = parameter_recovery_bins(data.rename(columns={'posterior_mean': 'median'}), bins=8)
            if not bins.empty:
                binned = ax.errorbar(bins.truth_mean, bins.recovered_mean, yerr=bins.recovered_sem,
                                     fmt='o', markersize=4.5, color='tab:orange',
                                     elinewidth=1., capsize=2, zorder=4)
                handles.setdefault('Binned images', binned)
                bins_saved.append(bins.assign(field=field, component=component))
            limit = 1.06*max(np.max(np.abs(data.truth)), np.max(np.abs(data['posterior_mean'])+data.posterior_sd),1e-8)
            ax.plot([-limit,limit],[-limit,limit],ls='--',lw=.8,color='.3',zorder=1)
            ax.set(xlim=(-limit,limit),ylim=(-limit,limit),
                   xlabel=rf'Injected $\delta {component}$ (arcsec)',
                   ylabel=rf'Recovered $\delta {component}$ (arcsec)')
            ax.set_box_aspect(1)
            ax.text(.04,.96,'GRF deflection' if field=='GRF' else 'Image-plane GP',
                    transform=ax.transAxes,va='top',fontsize=9,color='.25')
            _polish(ax)
    handles['One-to-one'] = Line2D([],[],color='.3',ls='--',lw=.8)
    fig.legend(list(handles.values()),list(handles),loc='upper center',ncol=3,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.96),h_pad=1.2,w_pad=1.)
    if bins_saved:
        pd.concat(bins_saved,ignore_index=True).to_csv(output/'mock_grf_gp_field_recovery_bins.csv',index=False)
    (output/'mock_grf_gp_field_recovery.txt').write_text(
        'Points: posterior mean field, the conditional mean of each posterior draw averaged over the draws.\n'
        'Bars: total posterior SD, including conditional variance and variation between joint parameter draws; not central credible intervals about these points.\n'
        'Binned points: equal-weight mean of within-mock means per truth bin. Bars: SEM across contributing mocks (at least two).\n'
        'GRF: at reference redshift, with fitted low-order modes removed from truth and model. GP: image-plane displacements.\n'
        f'Included seeds with saved posterior fields: {sorted(table.seed.unique().tolist())}\n'
        f'Seeds missing field diagnostics: {sorted(set(campaign.seeds)-set(table.seed.unique()))}\n')
    return _save(fig,output/'mock_grf_gp_field_recovery.pdf')
