"""RGB overviews with matched posterior z=9 critical curves and bands."""
from pathlib import Path
from .style import CRITICAL_COLOR
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .grf import plot_deflection_correction, PLOT_FIELD_VERSION
from ..io.hdf5 import load_arrays, save_arrays
from ..likelihood.grf_fields import shared_field_evaluator, shared_grf_label, shared_theta


def overview_grf_correction(evaluator, theta, geometry):
    """Cover the full overview with the same smooth field used by fold plots."""
    base=evaluator._padded_evaluator() if evaluator.gp_groups is not None else evaluator
    corners=np.array([(x,y) for x in geometry['xlim'] for y in geometry['ylim']])
    observed=np.asarray(base.gp['xy'])[np.asarray(base.gp['mask'],bool)]
    extent=float(np.linalg.norm(corners[:,None]-observed[None],axis=-1).max())
    return plot_deflection_correction(evaluator,theta,max_separation=extent)


def marker_photometry(catalog, reference, photpath):
    """Use measured F200W magnitudes only for matching IDs and sky positions."""
    from ..imaging.cutouts import arcsec_to_radec
    ra,dec=arcsec_to_radec(catalog.x_obs_arcsec.to_numpy(),catalog.y_obs_arcsec.to_numpy(),reference)
    result=pd.DataFrame({'image_label':catalog.image_label.to_numpy(),'phot_id':-1,
                         'match_separation_arcsec':np.nan,'m_F200W_AB':np.nan})
    if photpath is not None and Path(photpath).exists():
        saved=pd.read_csv(photpath).set_index('image_label').reindex(result.image_label)
        sep=np.hypot((saved.ra.to_numpy()-ra)*3600*np.cos(np.deg2rad(dec)),(saved.dec.to_numpy()-dec)*3600)
        valid=(sep<.05)&saved.valid_photometry.fillna(False).to_numpy(dtype=bool)&np.isfinite(saved.m_F200W_AB.to_numpy())
        for key in ('phot_id','match_separation_arcsec','m_F200W_AB'):
            result.loc[valid,key]=saved[key].to_numpy()[valid]
    valid=np.isfinite(result.m_F200W_AB.to_numpy())
    result['valid_photometry']=valid
    result['marker_area_pt2']=9.
    result.loc[valid,'marker_area_pt2']=np.clip(16*10**(-.2*(result.loc[valid,'m_F200W_AB']-26)),9,64)
    return result


def rgb_overview_stem(image_config, bands):
    """Name the imaging product by its instrument, independently of cluster."""
    jwst = str(getattr(image_config, 'figure_suffix', '')).lower() == '_jwst'
    jwst |= any(str(b).lower().split('_')[-1] in {'f090w', 'f115w', 'f150w', 'f200w', 'f277w', 'f356w', 'f410m', 'f444w'} for b in bands)
    return ('jwst' if jwst else 'hst') + '_rgb_critical_z9'


def overview_marker_photometry(config, catalog, reference, photpath, stem):
    """Attach catalog photometry by ID; prefer matched F200W measurements."""
    result = marker_photometry(catalog, reference, photpath)
    result['magnitude_AB'] = result['m_F200W_AB']
    result['magnitude_band'] = np.where(result.valid_photometry, 'F200W', '')
    constraints = getattr(getattr(config, 'model', None), 'image_constraints', None)
    path = getattr(constraints, 'catalog_path', None)
    if path:
        from ..io.image_catalog import load_image_catalog
        phot = load_image_catalog(path).set_index('image_label').reindex(catalog.image_label)
    else:
        phot = catalog
    preferred = ('f200w', 'f814w') if stem.startswith('jwst') else ('f814w', 'f200w')
    columns = [c for band in preferred for c in phot if c.lower() in {f'mag_{band}', f'mag_hst_{band}', f'mag_jwst_{band}'}]
    columns += [c for c in phot if c.startswith('mag_') and not c.startswith('mag_err_') and c not in columns]
    if 'catalog_mag' in phot:
        columns.append('catalog_mag')
    for column in columns:
        values = pd.to_numeric(phot[column], errors='coerce').to_numpy()
        use = ~result.valid_photometry.to_numpy() & np.isfinite(values)
        result.loc[use, 'magnitude_AB'] = values[use]
        result.loc[use, 'magnitude_band'] = column.removeprefix('mag_')
        result.loc[use, 'valid_photometry'] = True
    valid = result.valid_photometry.to_numpy()
    result.loc[valid, 'marker_area_pt2'] = np.clip(16 * 10**(-.2 * (result.loc[valid, 'magnitude_AB'] - 26)), 9, 64)
    return result


@plt.rc_context()
def render_rgb_critical(config, state, evaluator, theta, catalog, extra, imaging,
                             output, tables, *, best_index=None, posterior=None, log=print):
    """Render a median determinant contour and interval from the same draws."""
    output,tables=Path(output),Path(tables)
    if posterior is None:
        raise ValueError("Median critical curves require the saved posterior")
    from .mass_figure import overview_geometry, aligned_rgb, grid_pixels
    from ..imaging import cutouts
    from ..lens.cosmology import dpie_sigma0_factor_from_config
    import hashlib
    images,bands,display,cluster=imaging
    simulated=cluster.lower() in {'hera','ares'}
    stem=rgb_overview_stem(config.image_catalog, bands)
    m0416=cluster.lower()=='m0416'
    reference_image=images[cutouts.rgb_reference_band(bands)]
    geometry=overview_geometry(reference_image,catalog,extra,state.reference,resolution=768)
    tables.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    shared_grf=shared_field_evaluator(evaluator) is not None
    band,band_info=critical_band(config,state,evaluator,posterior,best_index,geometry,tables,log=log,cache_stem=stem)
    saved = load_arrays(tables/f'{stem}_band.h5')
    lower,det,upper=critical_quantile_summary(saved['determinants'])
    fingerprint=hashlib.sha256(str(saved['fingerprint']).encode()+b'median-determinant-contour-v1').hexdigest()
    band_info['definition']='Zero contour of pointwise median determinant; shaded where its 16-84% interval contains zero; same parametric + conditional-mean GRF draws'
    band_info['selection']='Chain containing best draw; not all-chain uncertainty'
    (tables/f'{stem}_band.json').write_text(json.dumps(band_info,indent=2)+'\n')
    save_arrays(tables/f'{stem}.h5',determinant=det,lower=lower,upper=upper,
                        fingerprint=fingerprint,z_source=9.,draw_indices=band_info['draw_indices'],
                        central_statistic='pointwise posterior median determinant')
    log(f'[outputs] rendering {cluster} RGB critical overview')
    rgb=cutouts.rgb_cutout(images,bands,display,geometry['center'],geometry['size'])
    rgb=aligned_rgb(rgb,reference_image,geometry)
    plt.style.use(str(Path(__file__).with_name('style.mplstyle')))
    from ..diagnostics.position_contours import load_or_compute_contours
    recovered_contours = load_or_compute_contours(config, state, evaluator, posterior, extra, tables, log=log)
    main_output = output
    for show_candidates in ((False, True) if m0416 else (False,)):
        output = main_output.with_name(main_output.stem + '_candidates.pdf') if show_candidates else main_output
        # Match the M0416 candidate overview's page and map footprint for
        # side-by-side paper figures. Expand the sky crop, never stretch pixels.
        matched_page = cluster.lower() == 'as1063'
        page_width, page_height = 612.8157348632812, 551.6261596679688
        map_left, map_top, map_right, map_bottom = 113.94043731689453, 6.9825439453125, 608.6768798828125, 506.11749267578125
        if matched_page:
            fig=plt.figure(figsize=(page_width/72.,page_height/72.))
            ax=fig.add_axes([map_left/page_width, (page_height-map_bottom)/page_height,
                             (map_right-map_left)/page_width, (map_bottom-map_top)/page_height],
                            projection=geometry['wcs'])
        else:
            fig=plt.figure(figsize=(9,9))
            ax=fig.add_subplot(111,projection=geometry['wcs'])
        n=geometry['resolution']
        ax.imshow(rgb,origin='lower',extent=(-.5,n-.5,-.5,n-.5),interpolation='none')
        if band is not None:
            draw_critical_interval(ax,lower,upper,CRITICAL_COLOR,(-.5,n-.5,-.5,n-.5))
        ax.contour(det,levels=[0.],colors=[CRITICAL_COLOR],linewidths=.65,
                   origin='lower',extent=(-.5,n-.5,-.5,n-.5))
        px,py=grid_pixels(geometry,catalog.x_obs_arcsec.to_numpy(),catalog.y_obs_arcsec.to_numpy())
        photpath=Path(__file__).parents[1]/'data'/'m0416_image_f200w_photometry.csv' if m0416 else None
        marker_table=overview_marker_photometry(config, catalog, state.reference, photpath, stem)
        sizes=marker_table.marker_area_pt2.to_numpy()
        valid=marker_table.valid_photometry.to_numpy()
        candidate_path=getattr(config.image_catalog,'candidate_catalog_path',None)
        candidate_path=(Path(candidate_path) if candidate_path else Path(__file__).parents[1]/'data'/'m0416_candidates.json') if m0416 and show_candidates else None
        candidates=json.loads(candidate_path.read_text())['candidates'] if candidate_path is not None and candidate_path.exists() else []
        family_colors={'K66':'#40DFFF','K67':'#FF70C5','K20':'#FF6060',
                       'K28':'#BD9AFF','K34':'#609FFF','K115':'#FFFFFF','A':'#32E875'}
        systems=catalog.family_id.astype(str).str.extract(r'^(K\d+|C\d+)',expand=False)
        selected={r['system'] for r in candidates if r['kind']=='counterimage'}
        colors=[family_colors.get(system,'#FF9500') if system in selected else '#FF9500' for system in systems]
        ax.scatter(px,py,s=sizes,marker='s',
                   facecolors='none',edgecolors=colors,linewidths=.8,zorder=8)
        for curve in recovered_contours['curves']:
            if curve['kind'] != 'recovered' or not np.isclose(curve['probability'], .954499736):
                continue
            for segment in curve['segments']:
                sx, sy = grid_pixels(geometry, segment[:, 0], segment[:, 1])
                line, = ax.plot(sx, sy, color='cyan', linewidth=.45,
                                antialiased=True, solid_joinstyle='round', zorder=20)
                line.get_path().should_simplify = False
        marker_table['system']=systems.to_numpy()
        marker_table['color']=colors
        overlay_rows=[]
        all_px,all_py=list(px),list(py)
        for system in dict.fromkeys(r['system'] for r in candidates):
            color=family_colors.get(system,'#42A5FF')
            label='New family A' if system=='A' else f'{system} candidates'
            rows=[r for r in candidates if r['system']==system]
            if not rows:
                continue
            ra0,dec0=state.reference[1:]
            x=-(np.array([r['ra'] for r in rows])-ra0)*3600*np.cos(np.deg2rad(dec0))
            y=(np.array([r['dec'] for r in rows])-dec0)*3600
            cx,cy=grid_pixels(geometry,x,y)
            if not np.all((cx>=0)&(cx<n)&(cy>=0)&(cy<n)):
                raise ValueError('Candidate overlay extends beyond the RGB grid')
            ax.scatter(cx,cy,s=48,marker='D',facecolors='none',edgecolors=color,linewidths=1.,zorder=8,label=label)
            all_px.extend(cx);all_py.extend(cy)
            overlay_rows.extend(rows)
        if overlay_rows:
            pd.DataFrame(overlay_rows).to_csv(tables/f'{stem}_candidate_overlays.csv',index=False)
        # Include both fitted images and proposed candidates, with 3 arcsec padding.
        pad=3./geometry['size']*n
        limits=[float(min(all_px)-pad),float(max(all_px)+pad),float(min(all_py)-pad),float(max(all_py)+pad)]
        if matched_page:
            target_ratio=(map_right-map_left)/(map_bottom-map_top)
            width,height=limits[1]-limits[0],limits[3]-limits[2]
            cx,cy=(limits[0]+limits[1])/2.,(limits[2]+limits[3])/2.
            width=max(width,height*target_ratio)
            height=width/target_ratio
            limits=[cx-width/2.,cx+width/2.,cy-height/2.,cy+height/2.]
        ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:])
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.legend_handler import HandlerTuple
        handles = [
            Line2D([], [], marker='s', linestyle='none', markerfacecolor='none',
                   markeredgecolor='#FF9500', markeredgewidth=.8, markersize=5),
            Line2D([], [], marker='o', linestyle='none', markerfacecolor='none',
                   markeredgecolor='cyan', markeredgewidth=.45, markersize=5),
            (Patch(facecolor=CRITICAL_COLOR, alpha=.25, edgecolor='none'),
             Line2D([], [], color=CRITICAL_COLOR, linewidth=.65)),
        ]
        labels = ['Observed images', 'Recovered positions', r'Critical curve ($z=9$)']
        candidate_handles, candidate_labels = ax.get_legend_handles_labels()
        if show_candidates:
            candidate_handles = [
                Line2D([], [], marker='D', linestyle='none', markerfacecolor='none',
                       markeredgecolor='white', markersize=5),
                Line2D([], [], marker='D', linestyle='none', markerfacecolor='none',
                       markeredgecolor=family_colors['A'], markersize=5),
            ]
            candidate_labels = ['Candidate counterimages', 'New family A']
        legend = ax.legend(handles + candidate_handles, labels + candidate_labels,
                           handler_map={tuple: HandlerTuple(ndivide=1)}, loc='lower right',
                           frameon=True, facecolor='black', edgecolor='none',
                           framealpha=.7, labelcolor='white', borderpad=.6)
        legend.set_zorder(40)
        bar_length = 5. * n / abs(geometry['xlim'][1] - geometry['xlim'][0])
        bar_right = limits[0] + .05 * (limits[1] - limits[0]) + bar_length
        bar_y = limits[3] - .07 * (limits[3] - limits[2])
        ax.plot([bar_right - bar_length, bar_right], [bar_y, bar_y],
                color='white', linewidth=2.5, solid_capstyle='butt', zorder=30)
        ax.annotate(r'$5^{\prime\prime}$', (bar_right - bar_length / 2, bar_y),
                    xytext=(0, 5), textcoords='offset points', ha='center',
                    va='bottom', color='white', fontsize=12, zorder=30)
        assert np.all((px>limits[0])&(px<limits[1])&(py>limits[2])&(py<limits[3]))
        marker_table.to_csv(tables/(('hst_rgb_markers' if stem.startswith('hst') else 'jwst_rgb_marker_magnitudes') + ('_candidates' if show_candidates else '') + '.csv'),index=False)
        log(f'[outputs] magnitude-scaled RGB markers: {valid.sum()}/{len(valid)}')
        ra,dec=ax.coords
        ra.set_axislabel('R.A. (J2000)');dec.set_axislabel('Dec. (J2000)')
        ra.set_major_formatter('hh:mm:ss');dec.set_major_formatter('dd:mm:ss')
        ra.set_ticks(number=5,color='white',direction='in',size=6,width=1)
        dec.set_ticks(number=5,color='white',direction='in',size=6,width=1)
        ra.set_ticks_position('bt');dec.set_ticks_position('lr')
        ra.set_ticklabel_position('b');dec.set_ticklabel_position('l')
        ra.display_minor_ticks(True);dec.display_minor_ticks(True)
        inset_info=None
        if matched_page:
            from matplotlib.transforms import Bbox
            fig.savefig(output,dpi=300,bbox_inches=Bbox.from_bounds(0,0,page_width/72.,page_height/72.),pad_inches=0)
        else:
            fig.savefig(output,dpi=300,bbox_inches='tight',pad_inches=.06)
        plt.close(fig)
        output.with_suffix('.json').write_text(json.dumps(dict(run=config.paths.run_name,image_position_marker='square',recovered_position_probability=.954499736,insets=inset_info,candidate_marker='diamond' if show_candidates else None,best_index=best_index,central_statistic='zero contour of pointwise posterior median determinant; same draws as band',z_source=9.,model='parametric + conditional mean GRF; no GP displacement' if shared_grf else 'parametric only; no shared GRF available',grf_field=shared_grf_label(evaluator),grf_derivatives=PLOT_FIELD_VERSION,bands=list(bands),catalog_images=len(catalog),resolution=n,determinant_resolution=int(det.shape[0]),critical_definition='median over draws of det(I - Hessian_draw) = 0',cache_fingerprint=fingerprint,marker_photometry=str(photpath) if photpath else None,magnitude_band=sorted(set(marker_table.loc[valid, 'magnitude_band'])),marker_photometry_catalog=str(getattr(getattr(getattr(config, 'model', None), 'image_constraints', None), 'catalog_path', '')),matched_magnitudes=int(valid.sum()),missing_magnitude_policy='minimum marker area 9 pt^2',marker_area='clip(16*10**(-0.2*(m-26)),9,64) pt^2',crop_padding_arcsec=3.,crop_pixel_limits=limits,candidate_catalog=str(candidate_path) if candidate_path else None,candidate_overlays=[r['id'] for r in overlay_rows]),indent=2)+'\n')
        log(f"[outputs] RGB critical overview written: {output}")
    return main_output


def best_chain_draw_indices(posterior, best_index, count=32):
    """Evenly cover only the chain containing the displayed best draw."""
    grouped=np.asarray(posterior.grouped_samples)
    samples=np.asarray(posterior.samples)
    if grouped.ndim!=3 or not np.array_equal(grouped.reshape(samples.shape),samples):
        raise ValueError('Critical band requires verified chain-major saved samples')
    per_chain=grouped.shape[1]
    chain=int(best_index)//per_chain
    indices=np.unique(np.linspace(chain*per_chain,(chain+1)*per_chain-1,min(count,per_chain)).round().astype(int))
    if best_index not in indices:
        indices[np.argmin(abs(indices-best_index))]=best_index
    return np.sort(indices),chain


def draw_critical_interval(ax, lower, upper, color, extent=None, *, coordinates=None):
    """Shade the intersection lower<=0 and upper>=0, preserving subpixel bands.

    Interpolating min(-lower, upper) first can erase a narrow interval whose
    two boundaries cross a pixel. Intersect their separately contoured regions.
    """
    style=dict(origin='lower',extent=extent) if coordinates is None else {}
    args=() if coordinates is None else tuple(coordinates)
    if np.min(lower)>=0 or np.max(upper)<=0:
        return None
    bound=max(float(np.max(np.abs(lower))),float(np.max(np.abs(upper))),1.)+1.
    fill=ax.contourf(*args,lower,levels=[-bound,0.],colors=[color],alpha=.25,zorder=2,**style)
    clip=ax.contourf(*args,upper,levels=[0.,bound],colors='none',**style)
    fill.set_clip_path(clip.get_paths()[0],ax.transData)
    fill.set_clip_box(ax.bbox)  # Keep the interval clipped to the displayed RGB crop too.
    clip.remove()
    return fill


def critical_quantile_summary(determinants):
    """Matched 16/50/84 percentiles of the same determinant draws."""
    return np.quantile(determinants, [.16, .5, .84], axis=0)


def critical_quantile_band(determinants):
    """Pointwise region where the central 68% determinant interval straddles zero."""
    lower,_,upper=critical_quantile_summary(determinants)
    return lower,upper,(lower<0)&(upper>0)


def critical_band(config,state,evaluator,posterior,best_index,geometry,tables,log=print,*,cache_stem='jwst_rgb_critical_z9'):
    import hashlib
    from ..lens.cosmology import dpie_sigma0_factor_from_config
    indices,chain=best_chain_draw_indices(posterior,best_index,count=int(config.image_catalog.critical_band_draws))
    samples=np.asarray(posterior.samples)[indices]
    resolution=512
    xe=np.linspace(*geometry['xlim'],resolution+1);ye=np.linspace(*geometry['ylim'],resolution+1)
    x,y=np.meshgrid((xe[:-1]+xe[1:])/2,(ye[:-1]+ye[1:])/2)
    xy=np.column_stack([x.ravel(),y.ravel()])
    fingerprint=hashlib.sha256(samples.tobytes()+indices.tobytes()+xy.tobytes()+
        json.dumps({'model':config.to_json_dict()['model'],'likelihood':config.to_json_dict()['likelihood']},sort_keys=True,default=str).encode()+PLOT_FIELD_VERSION.encode()+b'critical-band-z9-best-chain-smooth-v2'+shared_grf_label(evaluator).encode()).hexdigest()
    path=tables/f'{cache_stem}_band.h5'
    if path.exists():
        saved = load_arrays(path)
        if str(saved['fingerprint'])==fingerprint:
            log(f'[outputs] cached critical band: chain {chain}, {len(indices)} draws')
            return saved['band'].copy(),dict(chain=chain,draw_indices=indices.tolist(),resolution=resolution,cache_reused=True)
    twin=shared_field_evaluator(evaluator)
    shared=twin is not None
    factor=float(dpie_sigma0_factor_from_config(state.z_lens,9.,state.cosmo_config))/float(dpie_sigma0_factor_from_config(state.z_lens,state.gp_reference_redshift,state.cosmo_config))/float(twin.gp_weights_per_family[0]) if shared else 0.
    determinants=[]
    for number,theta in enumerate(samples):
        packed=evaluator.packed_state_for_z(evaluator.physical(theta),9.)
        _,hessian=evaluator.total_fields_at(xy,packed,chunk=1024)
        hessian=np.array(hessian,copy=True)
        if shared:
            correction=overview_grf_correction(twin,shared_theta(evaluator,theta),geometry)
            for start in range(0,len(xy),384):
                q=xy[start:start+384]
                c=correction(q[:,0],q[:,1],np.zeros(len(q),dtype=int))
                hessian[start:start+len(q)]+=np.stack(c[2:],axis=-1).reshape(-1,2,2)*factor
        det=np.linalg.det(np.eye(2)-hessian).reshape(x.shape)
        if not np.isfinite(det).all():raise ValueError('Nonfinite posterior determinant grid')
        determinants.append(det)
        log(f'[outputs] critical band chain {chain}: {number+1}/{len(indices)} draws')
    lower,upper,band=critical_quantile_band(determinants)
    save_arrays(path,band=band,lower=lower,upper=upper,determinants=np.asarray(determinants),
                        draw_indices=indices,chain=chain,fingerprint=fingerprint)
    return band,dict(chain=chain,draw_indices=indices.tolist(),resolution=resolution,cache_reused=False)


render_jwst_rgb_critical = render_rgb_critical
