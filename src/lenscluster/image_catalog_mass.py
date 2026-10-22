"""Physical surface-density overlay and finite-window strip-mass integration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from astropy.wcs import WCS

import h5py
import numpy as np
import pandas as pd
from astropy import constants as const, units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import map_coordinates

from . import plotting
from .output_paths import output_paths

def integrate_strips(surface_density, x_edges, y_edges, bounds):
    """Integrate a piecewise-constant map over a rectangle, including partial pixels.

    Surface density is in Msun/arcsec^2; edges and bounds are in arcsec.
    Edges must be increasing. Returns strip centers, widths, dM/dx, total mass.
    """
    values = np.asarray(surface_density, dtype=float)
    xe, ye = np.asarray(x_edges), np.asarray(y_edges)
    xmin, xmax, ymin, ymax = bounds
    if values.shape != (len(ye) - 1, len(xe) - 1) or not np.isfinite(values).all():
        raise ValueError("Invalid surface-density grid")
    if np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0):
        raise ValueError("Integration edges must increase")
    if not (xe[0] <= xmin < xmax <= xe[-1] and ye[0] <= ymin < ymax <= ye[-1]):
        raise ValueError("Displayed rectangle must be fully covered by the mass map")
    lo, hi = np.maximum(xe[:-1], xmin), np.minimum(xe[1:], xmax)
    dx = np.maximum(0, hi - lo)
    dy = np.maximum(0, np.minimum(ye[1:], ymax) - np.maximum(ye[:-1], ymin))
    keep = dx > 0
    profile = (values * dy[:, None]).sum(axis=0)
    total = float(np.sum(values * dy[:, None] * dx[None, :]))
    np.testing.assert_allclose(np.dot(profile[keep], dx[keep]), total, rtol=1e-12)
    return (lo[keep] + hi[keep]) / 2, dx[keep], profile[keep], total


def overview_geometry(helpers, image, catalog, extra, reference, resolution=512):
    """A rectangular solver-coordinate grid, independent of simulation truth."""
    xc, yc, size = plotting._image_catalog_cluster_overview_geometry(catalog, extra)
    center = plotting._arcsec_to_skycoord(xc, yc, reference)
    window = helpers._cutout_window(image, center, cutout_size_arcsec=size)
    if window is None:
        raise ValueError('Image overview is outside the imaging footprint')
    # Retain the native handedness while aligning the marginal axes to lens x/y.
    origin = np.array(image.wcs.world_to_pixel(center))
    dx = np.array(image.wcs.world_to_pixel(plotting._arcsec_to_skycoord(xc + 1, yc, reference))) - origin
    dy = np.array(image.wcs.world_to_pixel(plotting._arcsec_to_skycoord(xc, yc + 1, reference))) - origin
    if abs(dx[0]) < abs(dx[1]) or abs(dy[1]) < abs(dy[0]):
        raise ValueError('Mass overview requires imaging with x predominantly horizontal and y vertical')
    size = window.npix * image.pixel_scale_arcsec
    xlim = np.array([xc-size/2, xc+size/2])[::1 if dx[0] > 0 else -1]
    ylim = np.array([yc-size/2, yc+size/2])[::1 if dy[1] > 0 else -1]
    xe, ye = np.linspace(*xlim, resolution+1), np.linspace(*ylim, resolution+1)
    x, y = np.meshgrid((xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2)
    # CAR with zero reference latitude represents the same linear RA/Dec offsets.
    wcs = WCS(naxis=2)
    dra = -(xe[1]-xe[0]) / (3600*np.cos(np.deg2rad(reference[2])))
    ddec = (ye[1]-ye[0])/3600
    wcs.wcs.ctype = ['RA---CAR', 'DEC--CAR']
    wcs.wcs.crval = [reference[1], 0]
    wcs.wcs.cdelt = [dra, ddec]
    wcs.wcs.crpix = [1 - (-x[0,0]/(3600*np.cos(np.deg2rad(reference[2]))))/dra,
                     1 - (reference[2]+y[0,0]/3600)/ddec]
    wcs.wcs.radesys = 'ICRS'
    return dict(x=x, y=y, xe=xe, ye=ye, xlim=xlim, ylim=ylim, wcs=wcs,
                resolution=resolution, window=window, reference=reference)


def grid_pixels(geometry, x, y):
    n = geometry['resolution']
    return ((np.asarray(x)-geometry['xlim'][0])/np.diff(geometry['xlim'])[0]*n-.5,
            (np.asarray(y)-geometry['ylim'][0])/np.diff(geometry['ylim'])[0]*n-.5)


def aligned_rgb(rgb, image, geometry, resolution=1600):
    """Reproject the displayed RGB cutout, preserving its sky-frame conversion."""
    xedge = np.linspace(*geometry['xlim'], resolution+1)
    yedge = np.linspace(*geometry['ylim'], resolution+1)
    x, y = np.meshgrid((xedge[:-1]+xedge[1:])/2, (yedge[:-1]+yedge[1:])/2)
    ra, dec = plotting._solver_arcsec_offsets_to_radec(x,y,geometry['reference'])
    px, py = image.wcs.world_to_pixel(SkyCoord(ra*u.deg, dec*u.deg, frame='icrs'))
    window = geometry['window']
    return np.stack([map_coordinates(rgb[:,:,c], [py-window.y0, px-window.x0],
                                    order=1, mode='constant', cval=0)
                     for c in range(rgb.shape[-1])], axis=-1)


def prepare_mass_maps(evaluator, best_fit, args, geometry):
    """Evaluate physical best-fit parameters on the common model/truth grid.

    The plotting grid evaluator performs the physical-to-latent conversion.
    Truth is optional, but a configured truth map must cover the full rectangle
    and declare its convergence source redshift in the FITS header.
    """
    x, y = geometry['x'], geometry['y']
    state = evaluator.state
    zs = float(getattr(args, 'caustic_source_redshift', 9.0))
    zl = float(state.z_lens)
    if zs <= zl:
        raise ValueError('Mass map source redshift must exceed lens redshift')
    config = dict(state.cosmo_config)
    if getattr(evaluator, 'fit_cosmology_flat_wcdm', False):
        latent = plotting._reported_physical_to_latent_vector(evaluator, best_fit)
        physical = evaluator._physical_parameter_vector(latent)
        om0, w0 = evaluator._cosmology_parameters_from_physical(physical)
        config.update(Om0=float(om0), w0=float(w0), **{'class':'FlatwCDM'})
    from .cluster_solver import _build_cosmology_from_config
    cosmo = _build_cosmology_from_config(config)
    dl, ds, dls = (cosmo.angular_diameter_distance(zl), cosmo.angular_diameter_distance(zs),
                   cosmo.angular_diameter_distance(zl,zs))
    sigma_crit = (const.c**2/(4*np.pi*const.G)*ds/(dl*dls)).to_value(u.Msun/u.kpc**2)
    scale = dl.to_value(u.kpc)*np.pi/(180*3600)
    model = plotting._lens_model_quantity_on_arcsec_grid(evaluator, best_fit, x,y,zs,'kappa')
    if not np.isfinite(model).all():
        raise ValueError('Model convergence is nonfinite in the displayed rectangle')
    truth = None
    source = getattr(args,'kappa_true_fits',None)
    if source:
        source = Path(source)
        native, wcs = plotting._load_kappa_true_fits(source)
        header = fits.getheader(source)
        truth_z = header.get('ZSOURCE', header.get('REDSHIFT'))
        if truth_z is None:
            raise ValueError('Truth convergence FITS must specify ZSOURCE or REDSHIFT')
        if not np.isclose(float(truth_z),zs):
            raise ValueError('Truth convergence source redshift differs from model map')
        ra,dec = plotting._solver_arcsec_offsets_to_radec(x,y,state.reference)
        px,py = wcs.world_to_pixel(SkyCoord(ra*u.deg,dec*u.deg,frame='icrs'))
        # Round-off at an exactly coincident edge pixel is not missing coverage.
        px = np.where((px >= -1e-7) & (px <= native.shape[1]-1+1e-7),
                      np.clip(px,0,native.shape[1]-1),px)
        py = np.where((py >= -1e-7) & (py <= native.shape[0]-1+1e-7),
                      np.clip(py,0,native.shape[0]-1),py)
        truth = map_coordinates(native,[py,px],order=1,mode='constant',cval=np.nan)
        if not np.isfinite(truth).all():
            raise ValueError('Truth convergence does not cover the displayed rectangle with finite values')
    return model, truth, dict(cosmology=config,z_lens=zl,z_source=zs,
                             sigma_crit_msun_per_kpc2=sigma_crit,kpc_per_arcsec=scale,
                             truth_available=truth is not None,truth_path=str(source) if source else None)


def add_mass_panels(fig, paths, evaluator, best_fit, args, geometry, catalog, provenance):
    model, truth, physical = prepare_mass_maps(evaluator,best_fit,args,geometry)
    provenance.update(physical)
    sigma_crit = physical['sigma_crit_msun_per_kpc2']
    sigma_angle = sigma_crit * physical['kpc_per_arcsec']**2
    x,y,xe,ye = [geometry[k] for k in ('x','y','xe','ye')]
    bounds = (float(xe.min()),float(xe.max()),float(ye.min()),float(ye.max()))
    maps = {'model':model}
    if truth is not None:
        maps['truth']=truth
    profiles = {}
    profile = pd.DataFrame()
    yprofile = pd.DataFrame()
    for name,values in maps.items():
        integral = values[::1 if ye[-1]>ye[0] else -1, ::1 if xe[-1]>xe[0] else -1]
        centers,widths,px,total = integrate_strips(integral*sigma_angle,np.sort(xe),np.sort(ye),bounds)
        yc,yw,py,ytotal = integrate_strips(integral.T*sigma_angle,np.sort(ye),np.sort(xe),
                                         (bounds[2],bounds[3],bounds[0],bounds[1]))
        np.testing.assert_allclose(total,ytotal,rtol=1e-12)
        profile['x_arcsec'],profile['width_arcsec']=centers,widths
        profile[f'{name}_dMdx_msun_per_arcsec']=px
        yprofile['y_arcsec'],yprofile['width_arcsec']=yc,yw
        yprofile[f'{name}_dMdy_msun_per_arcsec']=py
        profiles[name]=(px,py)
        provenance[f'{name}_mass_in_window_msun']=total
        provenance[f'{name}_y_integral_msun']=ytotal
    paths.tables.mkdir(parents=True,exist_ok=True)
    paths.fits.mkdir(parents=True,exist_ok=True)
    profile.to_csv(paths.tables/'image_catalog_mass_profile.csv',index=False)
    yprofile.to_csv(paths.tables/'image_catalog_mass_profile_y.csv',index=False)
    header=geometry['wcs'].to_header()
    header['ZSOURCE']=physical['z_source']; header['MODSTAT']=str(provenance['reference_sample_kind'])
    fits.writeto(paths.fits/'image_catalog_mass_kappa_model.fits',model,header,overwrite=True)
    ax=fig.axes[0]
    grid=ax.get_subplotspec().get_gridspec()
    lower=fig.add_subplot(grid[1,0]); right=fig.add_subplot(grid[0,1]); corner=fig.add_subplot(grid[1,1])
    levels=np.array([.5,1,2,4])*1e9
    def available(values):
        return levels[(levels>np.min(values)) & (levels<np.max(values))]
    rx,ry=grid_pixels(geometry,x,y)
    selected=available(model*sigma_crit)
    if len(selected):
        contours=ax.contour(rx,ry,model*sigma_crit/1e9,levels=selected/1e9,
                            colors='#D2BC82',linewidths=.65,alpha=.85,zorder=2)
        # One interior label per level keeps the RGB and image markers readable.
        n = geometry['resolution']
        positions = []
        for level, segments in zip(contours.levels, contours.allsegs):
            interior = [segment[(segment[:,0] > .12*n) & (segment[:,0] < .88*n)
                                & (segment[:,1] > .15*n) & (segment[:,1] < .85*n)]
                        for segment in segments]
            longest = max(interior, key=len, default=np.empty((0,2)))
            if len(longest):
                positions.append(tuple(longest[len(longest)//3]))
        if positions:
            ax.clabel(contours, manual=positions, fmt='%g', fontsize=7, inline_spacing=3)
    styles={'model':('#2878A5','-',1.5),'truth':('0.2','--',1.2)}
    for name,values in maps.items():
        color,ls,lw=styles[name]; px,py=profiles[name]
        lower.plot(centers,px/1e12,color=color,ls=ls,lw=lw,label=name.title())
        right.plot(py/1e12,yc,color=color,ls=ls,lw=lw)
        selected_corner=available(values*sigma_crit)
        if len(selected_corner):
            corner.contour(x,y,values*sigma_crit,levels=selected_corner,colors=color,linestyles=ls,linewidths=.65)
    lower.set_xlim(geometry['xlim']); lower.set_ylim(bottom=0)
    right.set_ylim(geometry['ylim']); right.set_xlim(left=0)
    corner.set_xlim(lower.get_xlim()); corner.set_ylim(right.get_ylim()); corner.set_aspect('equal',adjustable='box')
    image_x=catalog.x_obs_arcsec.to_numpy(float); image_y=catalog.y_obs_arcsec.to_numpy(float)
    lower.vlines(image_x,0,.10,transform=lower.get_xaxis_transform(),color='black',linewidth=.85,zorder=5)
    right.hlines(image_y,0,.10,transform=right.get_yaxis_transform(),color='black',linewidth=.85,zorder=5)
    lower.set_xlabel(r'$x$ (arcsec)',fontsize=10)
    lower.set_ylabel(r'$dM/dx$ ($10^{12}\,M_\odot\,{\rm arcsec}^{-1}$)',fontsize=9)
    right.set_xlabel(r'$dM/dy$ ($10^{12}\,M_\odot\,{\rm arcsec}^{-1}$)',fontsize=8,labelpad=6)
    right.set_ylabel(r'$y$ (arcsec)',fontsize=9)
    right.xaxis.set_label_position('top'); right.xaxis.tick_top()
    right.yaxis.set_label_position('right'); right.yaxis.tick_right()
    corner.set_xlabel(r'$x$ (arcsec)',fontsize=9); corner.set_ylabel(r'$y$ (arcsec)',fontsize=9)
    corner.yaxis.set_label_position('right'); corner.yaxis.tick_right()
    lower.legend(loc='upper right',fontsize=9,frameon=True,facecolor='white',edgecolor='none',framealpha=.85)
    lower.xaxis.set_major_locator(MaxNLocator(5,prune='both'));lower.yaxis.set_major_locator(MaxNLocator(4))
    right.xaxis.set_major_locator(MaxNLocator(3));right.yaxis.set_major_locator(MaxNLocator(5))
    corner.xaxis.set_major_locator(MaxNLocator(3));corner.yaxis.set_major_locator(MaxNLocator(3))
    lower.ticklabel_format(axis='y',style='plain',useOffset=False)
    right.ticklabel_format(axis='x',style='plain',useOffset=False)
    for panel in (lower,right,corner):
        panel.grid(False,which='both')
        panel.tick_params(which='both',direction='in',top=True,right=True,labelsize=8,width=.7,length=3)
        panel.tick_params(which='minor',length=1.5)
        for spine in panel.spines.values(): spine.set_linewidth(.7)
    fig.canvas.draw()
    np.testing.assert_allclose([ax.get_position().x0,ax.get_position().x1],
                              [lower.get_position().x0,lower.get_position().x1],atol=1e-6)
    np.testing.assert_allclose([ax.get_position().y0,ax.get_position().y1],
                              [right.get_position().y0,right.get_position().y1],atol=1e-6)
    ra, dec = plotting._solver_arcsec_offsets_to_radec(image_x, image_y, geometry['reference'])
    wx, wy = geometry['wcs'].world_to_pixel(SkyCoord(ra*u.deg, dec*u.deg, frame='icrs'))
    gx, gy = grid_pixels(geometry, image_x, image_y)
    n = geometry['resolution']
    np.testing.assert_allclose(wx, gx, atol=1e-6, rtol=0)
    np.testing.assert_allclose(wy, gy, atol=1e-6, rtol=0)
    overview_display = ax.transData.transform(np.column_stack((gx, gy)))
    marginal_x = lower.transData.transform(np.column_stack((image_x, np.zeros(len(image_x)))))[:,0]
    marginal_y = right.transData.transform(np.column_stack((np.zeros(len(image_y)), image_y)))[:,1]
    np.testing.assert_allclose(overview_display[:,0], marginal_x, atol=1e-6)
    np.testing.assert_allclose(overview_display[:,1], marginal_y, atol=1e-6)
    provenance.update(bounds_arcsec=dict(zip(('xmin','xmax','ymin','ymax'),bounds)),
                      grid=dict(shape=list(model.shape),wcs=dict(header)),
                      contour_levels_msun_per_kpc2=selected.tolist(),
                      requested_contour_levels_msun_per_kpc2=levels.tolist(),
                      coordinate_alignment_max_fractional_error_x=float(np.max(np.abs(wx-gx))/n),
                      coordinate_alignment_max_fractional_error_y=float(np.max(np.abs(wy-gy))/n),
                      x_profile_units='Msun/arcsec',y_profile_units='Msun/arcsec',
                      integration='piecewise constant surface density with fractional boundary pixels')
    return provenance


def render_mass_companion(stage, evaluator, best_fit, args, helpers, images, bands, rgb,
                          display_image, catalog, extra, render_config, *, sample_index=None,
                          sample_kind='saved_best_fit'):
    """Write the four-panel companion and numerical products without inference.

    Both live plotting and saved-artifact rendering supply the physical best-fit
    vector associated with their image catalog. A saved bundle, when present,
    provides an independent check of that association and provenance.
    """
    paths=output_paths(stage)
    geometry=overview_geometry(helpers,display_image,catalog,extra,evaluator.state.reference)
    fig=plotting._image_catalog_point_overview_figure(
        helpers,images,bands,rgb,display_image,catalog,extra,evaluator.state.reference,
        render_config=render_config,mass_panel=True,mass_geometry=geometry)
    artifact=paths.artifacts/'plot_bundle.h5'
    if artifact.exists():
        with h5py.File(artifact) as handle:
            saved = np.asarray(handle['posterior/best_fit'])
            np.testing.assert_allclose(best_fit, saved, rtol=0, atol=1e-10,
                                       err_msg='Mass map and saved image predictions must use the same sample')
            diagnostics = json.loads(handle['init_diagnostics_json'][()])
        saved_index = diagnostics.get('best_value_selected_sample_index')
        if sample_index is not None and sample_index != saved_index:
            raise ValueError('Mass map sample index differs from saved image predictions')
        sample_index = saved_index
        sample_kind = diagnostics.get('best_value_selected', sample_kind)
    provenance=dict(reference_sample_kind=sample_kind,reference_sample_index=sample_index,
                    best_fit=np.asarray(best_fit).tolist(),sample_likelihood_mode=args.sample_likelihood_mode,
                    arc_supported_images=sum(plotting._image_catalog_arc_recovered(row)
                                             for _, row in catalog.iterrows()),
                    artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else None)
    try:
        provenance=add_mass_panels(fig,paths,evaluator,best_fit,args,geometry,catalog,provenance)
        output=plotting._plot_path(stage,'image_catalog_family_cluster_mass.pdf')
        fig.savefig(output,dpi=300,bbox_inches='tight',pad_inches=.04,facecolor='white')
        (paths.tables/'image_catalog_mass_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    finally:
        plotting.plt.close(fig)
    return output


def load_saved_model(artifact, summary):
    """Load the same physical sample used for the saved image predictions, once."""
    from . import cluster_solver as cs
    from .planning import SolverRuntime
    state,saved,arrays,diagnostics=cs._load_artifacts(Path(artifact).parent)
    arrays,_=cs._maybe_convert_loaded_posterior_arrays_to_physical(arrays,state.parameter_specs,diagnostics)
    if diagnostics.get('best_value_selected_sample_index') != summary.get('fit_quality_reference_sample_index'):
        raise ValueError('Mass map and image diagnostics refer to different samples')
    evaluator=cs._build_cluster_evaluator_from_args(SolverRuntime(saved),state,sampling_engine='full_flat')
    return evaluator,np.asarray(arrays['best_fit'],float)
