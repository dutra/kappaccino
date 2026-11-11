"""Covariance posterior and exact-field uncertainty figures for run outputs."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter

from ..likelihood.smoothed import source_covariance
from .style import PAPER_STYLE

STYLE = Path(__file__).with_name('style.mplstyle')


def plot_covariance_errors(config, state, draws, paths, *, plane='both'):
    """Exact covariance components over the reporting draws: per image, the median over draws of each component width
    and of the singular values, and the 16-84% range of the total."""
    if plane not in {'source', 'image', 'both'}:
        raise ValueError('Unknown covariance display plane')
    plane_mode = plane
    evaluator = draws.evaluator
    @jax.jit
    def evaluate(theta):
        p = evaluator.physical(theta)
        a = evaluator.fields(p, force_exact=True, derivative=False)['jacobian']
        scales = evaluator.error_scales(p)
        sel, lam, model = scales['sigma_sel'], scales['lambda_sel'], scales['sigma_def']
        u, s, _ = jnp.linalg.svd(a)
        # The image-plane scale of each image's diagonal block: sigma_meas, plus sigma_gp under gp_image_plane.
        meas = jnp.broadcast_to(scales['sigma_img'], s.shape[:1])
        # Ascending singular values: maximum image stretch first.
        s, u = s[:, ::-1], u[:, :, ::-1]
        measurement = meas[:, None] * s
        selection = sel / jnp.sqrt(1 + (s / lam)**2)
        deflection = jnp.broadcast_to(jnp.broadcast_to(model, s.shape[:1])[:, None], s.shape)   # sigma_def is per image under the knee kernel
        total = jnp.sqrt(measurement**2 + selection**2 + deflection**2)
        cov = source_covariance(a, meas, model, sel, lam)['covariance']
        projected = jnp.einsum('nij,nik,nkl->njl', u, cov, u)
        return s, jnp.stack([measurement, selection, deflection, total]), projected

    source_draws, image_draws, singular = [], [], []
    for theta in draws.samples:
        draw_s, draw_widths, projected = map(np.asarray, evaluate(theta))
        np.testing.assert_allclose(draw_widths[-1]**2, np.diagonal(projected, axis1=-2, axis2=-1), rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(projected[:, 0, 1], 0, atol=1e-12)
        if np.any(draw_s <= 0):
            raise ValueError('Image-plane covariance is undefined at a singular Jacobian')
        singular.append(draw_s)
        source_draws.append(draw_widths)
        image_draws.append(draw_widths / draw_s[None, :, :])
    s = np.median(singular, axis=0)
    plane_data = {plane: (np.median(stack, axis=0), stack) for plane, stack in
                  (('source', np.stack(source_draws)), ('image', np.stack(image_draws)))}
    planes = ['image', 'source'] if plane == 'both' else [plane]
    prefix = config.paths.figure_prefix or 'cluster'
    stem = 'covariance_errors' + {'image': '_image_plane', 'source': '', 'both': '_both_planes'}[plane]
    output = paths.figures / f'{prefix}_{stem}.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ['Measurement', 'Selection', 'Model', 'Total']
    colors = ['tab:blue', 'tab:green', 'tab:red', 'black']
    table = []
    with plt.rc_context(fname=STYLE, rc=PAPER_STYLE):
        fig, axes = plt.subplots(len(planes), 2, figsize=(7.1, 5.5 if len(planes) == 2 else 3.25),
                                 sharey='row', sharex='col', squeeze=False)
        for row, plane in enumerate(planes):
            widths, stacked = plane_data[plane]
            component_lo, component_median, component_hi = np.percentile(stacked, [16, 50, 84], axis=0)
            lo, median, hi = np.percentile(stacked[:, -1], [16, 50, 84], axis=0)
            for k, ax in enumerate(axes[row]):
                order = np.argsort(s[:, k])
                for j in range(3):
                    positive = widths[j, order, k] > 0
                    ax.plot(s[order, k][positive], widths[j, order, k][positive], color=colors[j], lw=1.5, label=names[j])
                ax.plot(s[order, k], widths[-1, order, k], color='black', lw=1.25, label='Total: median')
                ax.errorbar(s[:, k], median[:, k], yerr=[median[:, k]-lo[:, k], hi[:, k]-median[:, k]],
                            fmt='o', ms=2.5, color='.35', ecolor='.65', elinewidth=.65, capsize=0,
                            label='Total: posterior 16–84%', zorder=1)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
                ax.xaxis.set_minor_formatter(NullFormatter())
                if row == len(planes)-1:
                    ax.set_xlabel(r'$|\lambda_{\rm arc}|$' if k == 0 else r'$|\lambda_{\perp}|$')
                ax.text(.04, .96, 'Along image stretch' if k == 0 else 'Perpendicular', transform=ax.transAxes,
                        va='top', fontsize=9)
                for i, label in enumerate(state.images.image_labels):
                    table.append(dict(plane=plane, image_label=label, direction='arc' if k == 0 else 'perpendicular',
                                      singular_value_median=s[i, k], **{name.lower()+'_arcsec': widths[j, i, k] for j, name in enumerate(names)},
                                      **{f'{name.lower()}_{q}_arcsec': values[j, i, k]
                                         for j, name in enumerate(names[:3])
                                         for q, values in [('p16', component_lo), ('median', component_median), ('p84', component_hi)]},
                                      total_p16_arcsec=lo[i,k], total_median_arcsec=median[i,k], total_p84_arcsec=hi[i,k]))
            axes[row, 0].set_ylabel(f'{plane.capitalize()} plane\nUncertainty (arcsec)')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.52, 1.01))
        fig.subplots_adjust(left=.10, right=.98, bottom=.10 if len(planes) == 2 else .17,
                            top=.86 if len(planes) == 2 else .77, wspace=0, hspace=0)
        fig.savefig(output, bbox_inches='tight', pad_inches=.04)
        plt.close(fig)
    tables = paths.tables
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(tables / f'{stem}.csv', index=False)
    (tables / f'{stem}.json').write_text(json.dumps(dict(posterior_draw_indices=np.asarray(draws.indices).tolist(),
        plane=plane_mode, transformation={p: 'A^-1 C A^-T' if p == 'image' else 'C' for p in planes},
        kernels='exact', positions='observed images at their family redshifts',
        x_axis='median over draws of each singular value',
        lines='median over draws of each component width',
        intervals='16th–84th percentiles of component and total widths in each draw\u0027s singular basis',
        units='arcsec; standard deviations, not covariance eigenvalues'), indent=2))
    return output


def plot_covariance_corner(config, state, posterior, paths):
    """Covariance hyperparameters in the common prior-overlaid corner style."""
    from .corner import plot_parameter_corner
    fields = ['sigma_meas', 'sigma_def', 'sigma_sel', 'lambda_sel', 'sigma_gp', 'ell_gp',
              'sigma_grf', 'lambda_b', 'gp_knee_gamma']
    symbols = [r'$\sigma_{\rm meas}$', r'$\sigma_{\rm def}$',
               r'$\sigma_{\rm sel}$', r'$\lambda_{\rm sel}$', r'$\sigma_{\rm GP}$', r'$\ell_{\rm GP}$',
               r'$\sigma_{\rm GRF}$', r'$\lambda_{\rm b}$', r'$\gamma_\kappa$']
    selection = [(i, spec, label) for field, label in zip(fields, symbols)
                 for i, spec in enumerate(state.parameter_specs) if spec.name == 'images.'+field]
    grouped = None
    if getattr(posterior, 'grouped_samples', None) is not None:
        np.testing.assert_array_equal(posterior.grouped_samples.reshape(posterior.samples.shape), posterior.samples)
        grouped = np.asarray(posterior.physical).reshape(posterior.grouped_samples.shape)
    path = paths.figures / f'{config.paths.figure_prefix or "cluster"}_covariance_corner.pdf'
    result = plot_parameter_corner(posterior.physical, selection, path, grouped_physical=grouped)
    if result is not None:
        paths.tables.mkdir(parents=True, exist_ok=True)
        (paths.tables / 'covariance_corner.json').write_text(path.with_suffix('.json').read_text())
    return result


def write_covariance_plots(config, state, draws, posterior, paths, *, log=print):
    outputs = []
    corner = plot_covariance_corner(config, state, posterior, paths)
    if corner is not None:
        outputs.append(corner)
    else:
        log('[outputs] covariance corner skipped: insufficient draws or no sampled covariance parameters')
    outputs.append(plot_covariance_errors(config, state, draws, paths))
    for path in outputs:
        log(f'[outputs] covariance plot: {path}')
    return outputs
