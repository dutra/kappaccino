"""Compare saved-run truth residuals with and without its shared conditional GRF."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.lines import Line2D
from _covariance_plot_cli import load_run
from kappaccino.diagnostics.truth_recovery import load_truth_fits, grid_arcsec
from kappaccino.io.hdf5 import load_arrays, save_arrays
from kappaccino.plotting.style import PAPER_STYLE


def render(data, images, path):
    plt.style.use(Path(__file__).resolve().parents[1]/'src/kappaccino/plotting/style.mplstyle')
    plt.rcParams.update(PAPER_STYLE)
    cmap=LinearSegmentedColormap.from_list('cropped_RdBu',plt.get_cmap('RdBu_r')(np.linspace(.25,1,768)),N=768)
    cmap.set_bad('#e8e8e8');norm=Normalize(-1,2,clip=True)
    fig,axes=plt.subplots(2,3,figsize=(7.2,5.7),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.99,bottom=.09,top=.86,wspace=.06,hspace=.06)
    x,y=data['x'],data['y'];meshes=[]
    for row,mode in enumerate(['parametric','with_grf']):
        for col,q in enumerate(['kappa','abs_mu']):
            truth=data[f'truth_{q}'];model=data[f'{mode}_{q}']
            residual=np.divide(model-truth,truth,out=np.full_like(truth,np.nan),where=np.isfinite(truth)&(truth>0)&np.isfinite(model))
            mesh=axes[row,col].pcolormesh(x,y,np.ma.masked_invalid(residual),cmap=cmap,norm=norm,shading='nearest',rasterized=True)
            if row==0:meshes.append(mesh)
        for key,color in [('truth_detA','#252525'),(f'{mode}_detA','#d55e00')]:
            axes[row,2].contour(x,y,data[key],levels=[0],colors=[color],linewidths=[.75])
        for ax in axes[row]:
            ax.scatter(images.x_obs_arcsec,images.y_obs_arcsec,s=4,facecolors='none',edgecolors='white',linewidths=.9,zorder=5)
            ax.scatter(images.x_obs_arcsec,images.y_obs_arcsec,s=4,facecolors='none',edgecolors='#333333',linewidths=.35,zorder=6)
            ax.set_aspect('equal');ax.tick_params(which='both',top=False,right=False);ax.grid(False)
        axes[row,0].set_ylabel('y (arcsec)')
        axes[row,0].text(.03,.97,['Parametric','Parametric + GRF'][row],transform=axes[row,0].transAxes,
                         va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=2))
    for ax in axes[-1]:ax.set_xlabel('x (arcsec)')
    handles=[Line2D([],[],color='#252525',lw=.85,label='Truth'),Line2D([],[],color='#d55e00',lw=.75,label='Model'),
             Line2D([],[],marker='o',markersize=3,mfc='none',mec='#333333',ls='none',label='Images')]
    axes[0,2].legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,1.10),ncol=3,frameon=False,fontsize=7,handlelength=1.1,columnspacing=.6)
    fig.canvas.draw()
    for ax,mesh,label in zip(axes[0,:2],meshes,[r'$(\kappa_{\rm model}-\kappa_{\rm true})/\kappa_{\rm true}$',r'$(|\mu_{\rm model}|-|\mu_{\rm true}|)/|\mu_{\rm true}|$']):
        box=ax.get_position();cax=fig.add_axes([box.x0,box.y1+.025,box.width,.016])
        cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[-1,0,1,2],extend='both')
        cb.set_label(label,fontsize=8,labelpad=5);cb.ax.xaxis.set_label_position('top');cb.ax.xaxis.set_ticks_position('top')
        cb.ax.minorticks_off();cb.ax.tick_params(labelsize=8,length=2.5)
    fig.savefig(path,dpi=400,bbox_inches='tight',pad_inches=.04);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--render-only',action='store_true');a=p.parse_args()
    final=a.run/'final';cache=final/'tables/truth_recovery_grf_comparison.h5'
    cfg=json.loads((final/'artifacts/config.json').read_text());prefix=cfg['paths']['figure_prefix']
    output=final/'figures'/f'{prefix}_truth_recovery_residuals_grf_comparison.pdf'
    images=pd.read_csv(final/'tables/image_fit_quality.csv')
    if not a.render_only:
        config,state,evaluator,posterior,paths=load_run(a.run)
        from kappaccino.likelihood.grf_fields import shared_field_evaluator
        from kappaccino.reporting.draws import reporting_draws
        if shared_field_evaluator(evaluator) is None:
            raise ValueError('A single cluster field requires a knee mass field')
        # The reporting draws and grid of compute_truth_recovery; each map is the per-pixel median over the draws, with
        # each draw's field conditioned at that draw.
        draws=reporting_draws(config,evaluator,posterior)
        z=float(config.truth.caustic_source_redshift)
        data={}
        for q in ['kappa','abs_mu','detA']:
            data[f'truth_{q}'],wcs=load_truth_fits(final/f'fits/truth_recovery_{q}_true.fits')
        shape=data['truth_kappa'].shape;x,y=grid_arcsec(state,wcs,shape)
        maps=draws.grid_maps('truth',x,y,z,log=lambda text:print(text,flush=True))
        with np.errstate(divide='ignore'):
            parametric_abs_mu=np.abs(1/maps['detA_parametric'].astype(float))
        grids={'parametric_kappa':maps['kappa_parametric'],'parametric_detA':maps['detA_parametric'],'parametric_abs_mu':parametric_abs_mu,
               'with_grf_kappa':maps['kappa'],'with_grf_detA':maps['detA'],'with_grf_abs_mu':maps['abs_mu']}
        data.update({k:np.nanmedian(np.asarray(v,dtype=float),axis=0) for k,v in grids.items()})
        grf_kappa=np.nanmedian(maps['kappa_grf'].astype(float),axis=0)
        differences={}
        for q in ['kappa','abs_mu','detA']:
            saved,_=load_truth_fits(final/f'fits/truth_recovery_{q}_model_median.fits')
            from astropy.io import fits
            header = fits.getheader(final/f'fits/truth_recovery_{q}_model_median.fits')
            mode = 'with_grf' if header.get('GRFMEAN', False) else 'parametric'
            np.testing.assert_allclose(data[f"{mode}_{q}"],saved,rtol=1e-6,atol=1e-8)
            differences[q]=float(np.nanmax(abs(data[f'{mode}_{q}']-saved)))
        data.update(x=x,y=y,grf_kappa=grf_kappa)
        save_arrays(cache,**data)
        metadata=dict(run=str(a.run.resolve()),z_source=z,model=f'per-pixel median over {len(draws)} reporting draws, matching the truth-recovery maps',
            grf='conditional mean of each draw at that draw; grf_kappa is the per-pixel median of the GRF convergence',
            grid_shape=list(shape),baseline_max_absolute_difference=differences,
            magnification='1 / det(I - U_parametric - U_GRF), recomputed from the total Jacobian',
            normalization='display [-1,2], cropped from symmetric [-2,2] RdBu_r',
            grf_kappa_min=float(np.nanmin(data['grf_kappa'])),grf_kappa_max=float(np.nanmax(data['grf_kappa'])))
        (cache.with_suffix('.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    data = load_arrays(cache)
    render(data,images,output)
    print(output,flush=True)


if __name__=='__main__':main()
