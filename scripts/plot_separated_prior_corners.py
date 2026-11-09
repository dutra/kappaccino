"""One-page-per-block saved-posterior corners with marginal saved-prior overlays."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
from scipy.stats import norm
import corner
from kappaccino.io.hdf5 import load_arrays
from kappaccino.plotting.block_corner import posterior_corner_data
from kappaccino.plotting.corner import DisplayPrior
from kappaccino.plotting.style import PAPER_STYLE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args(); run=args.run.resolve()
    state=json.loads((run/'init/artifacts/state.json').read_text())
    specs=[SimpleNamespace(**s) for s in state['parameter_specs']]
    samples=np.asarray(load_arrays(run/'final/artifacts/posterior.h5')['physical'])
    assert samples.shape[1]==len(specs)
    data=posterior_corner_data(samples,specs)
    names=data['parameters']; values=data['values']; weights=data['weights']
    lookup={s.name:s for s in specs}
    blocks=[('Halo 1',[i for i,n in enumerate(names) if n.startswith('1.')],'#0072B2'),
            ('Halo 2',[i for i,n in enumerate(names) if n.startswith('2.')],'#0072B2'),
            ('Member scaling relation',[i for i,g in enumerate(data['groups']) if g=='Scaling relation'],'#D55E00'),
            ('GRF + GP',[i for i,g in enumerate(data['groups']) if g=='GRF + GP'],'#CC3311'),
            ('Shear and softening',[i for i,n in enumerate(names) if n.startswith(('S1.','softening_length.'))],'#AA4499'),
            ('Pooled sources',[i for i,g in enumerate(data['groups']) if g=='Sources'],'#009E73')]
    symbols={'x_centre':'x','y_centre':'y','e1':'e_1','e2':'e_2','core_radius_kpc':r'r_{\rm core}',
             'cut_radius_kpc':r'r_{\rm cut}','v_disp':r'\sigma_0','gamma1':r'\gamma_1','gamma2':r'\gamma_2',
             'log_softening_length_kpc':r'\ln\epsilon','sigma':r'\sigma_\star','cutkpc':r'r_{{\rm cut},\star}',
             'corekpc':r'r_{{\rm core},{\rm piv}}','alpha_sigma':r'\alpha_\sigma','beta_radius':r'\beta_r',
             'velocity_log_scatter':r's_\sigma','radius_log_scatter':'s_r','sigma_grf':r'\sigma_{\rm GRF}',
             'lambda_b':r'\lambda_{\rm b}','gp_knee_gamma':r'\gamma_\kappa','sigma_gp':r'\sigma_{\rm GP}',
             'ell_gp':r'\ell_{\rm GP}','sigma_meas':r'\sigma_{\rm meas}'}
    path=run/'final/figures/hera_corner_separated_priors.pdf'
    metadata={'run':str(run),'prior_overlay':'Marginal normalized densities only; joint panels show posterior only',
              'layout':'one block per page','contour_probabilities':[.68,.95],'blocks':{},'source_prior':'Physical Gaussian transformed with each source posterior median and s68; equal source weights; displayed over posterior range'}
    style=Path(__file__).resolve().parents[1]/'src/kappaccino/plotting/style.mplstyle'
    with plt.rc_context(fname=style,rc=PAPER_STYLE), PdfPages(path) as pdf:
        for b,(block,indices,color) in enumerate(blocks):
            n=len(indices)
            if not n:continue
            fig=plt.figure(figsize=(max(6,1.55*n),max(6,1.55*n)))
            inner=fig.add_gridspec(n,n,left=.09,right=.98,bottom=.09,top=.94,wspace=.04,hspace=.04)
            axes=np.array([[fig.add_subplot(inner[i,j]) for j in range(n)] for i in range(n)])
            ranges=[]; labels=[]; priors=[]
            for index in indices:
                name=names[index]; v=values[:,index]
                if name.startswith('pooled_source.'):
                    axis=0 if name.endswith('beta_x') else 1
                    entries=list(data['source_scales'].items())
                    def density(x,axis=axis,entries=entries):
                        curves=[]
                        for family,s in entries:
                            spec=lookup[f'source.{family}.beta_{"xy"[axis]}']
                            scale=s['s68'][axis]; center=s['center'][axis]
                            curves.append(norm.pdf(center+scale*x,loc=spec.physical_mean,scale=spec.physical_std)*scale)
                        return np.mean(curves,axis=0)
                    lo,hi=np.quantile(v,[.001,.999]); label=r'\Delta\beta_'+('x' if axis==0 else 'y')+r'/s_{68}'
                else:
                    spec=lookup[name]; prior=DisplayPrior(spec); density=prior.pdf
                    pl,ph=prior.ppf([.001,.999]);lo=min(v.min(),pl);hi=max(v.max(),ph)
                    label=symbols.get(spec.field,spec.field.replace('_',r'\_'))
                    if spec.transform_kind.startswith('log_'):
                        if spec.transform_kind=='log_offset_positive':label+=f'-{spec.transform_offset:g}'
                        label=r'\log_{10}('+label+')'
                ranges.append((lo-.03*(hi-lo),hi+.03*(hi-lo)));labels.append('$'+label+'$');priors.append(density)
            for i,index in enumerate(indices):
                for j,jindex in enumerate(indices):
                    ax=axes[i,j]
                    if j>i:ax.set_axis_off();continue
                    if i==j:
                        ax.hist(values[:,index],bins=45,range=ranges[i],weights=weights,density=True,histtype='step',color=color,lw=1.6)
                        x=np.linspace(*ranges[i],1000);ax.plot(x,priors[i](x),color='.2',ls='--',lw=1.3)
                        ax.set_yticks([]);ax.set_xlim(ranges[i]);ax.set_ylim(bottom=0)
                    else:
                        corner.hist2d(values[:,jindex],values[:,index],weights=weights,bins=40,range=[ranges[j],ranges[i]],
                                      smooth=1,color=color,levels=(.68,.95),plot_datapoints=False,plot_density=False,
                                      fill_contours=True,contour_kwargs={'linewidths':1.2},ax=ax)
                    ax.tick_params(top=True,right=True,labelsize=7,labelbottom=i==n-1,labelleft=j==0 and i>0)
                    ax.xaxis.set_major_locator(MaxNLocator(3,prune='both'))
                    if i!=j:ax.yaxis.set_major_locator(MaxNLocator(3,prune='both'))
                    if i==n-1:ax.set_xlabel(labels[j],fontsize=9)
                    if j==0 and i>0:ax.set_ylabel(labels[i],fontsize=9)
                    ax.grid(False)
            # Block identification inside the unused upper triangle, not a plot title.
            axes[0,-1].text(.95,.9,block,transform=axes[0,-1].transAxes,ha='right',va='top',fontsize=13,color=color)
            if block=='Pooled sources':
                axes[0,-1].text(.95,.65,'Prior extends beyond\nthe displayed range',transform=axes[0,-1].transAxes,
                                ha='right',va='top',fontsize=9,color='.3')
            metadata['blocks'][block]={'parameters':[names[i] for i in indices],'ranges':ranges}
            fig.legend(handles=[Line2D([],[],color=color,lw=1.6,label='Posterior'),
                                Line2D([],[],color='.2',ls='--',lw=1.3,label='Prior (diagonal panels)')],
                       loc='upper center',ncol=2,frameon=False,fontsize=10)
            pdf.savefig(fig);plt.close(fig)
    path.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print(path)

if __name__=='__main__':main()
