"""Median critical curves at each cutout's source redshift, over the reporting draws."""
import json
from pathlib import Path
import numpy as np
from ..io.hdf5 import save_arrays
from ..likelihood.grf_fields import shared_grf_label


def median_cutout_determinants(draws, centers, fovs, redshifts, path, *, resolution=96, log=print):
    """Per-pixel 16th, 50th and 84th percentiles over the reporting draws (``reporting.draws.ReportingDraws``) of
    det A on a ``resolution``-square grid per cutout (centre, field of view in arcsec, source redshift), each draw with
    its own GRF conditional mean.  Returns ``(grids, median, lower, upper)``; also written to ``path`` (h5 and json)."""
    centers, fovs, redshifts = map(np.asarray, (centers, fovs, redshifts))
    axis = np.linspace(-.5, .5, resolution)
    grids = np.array([np.stack(np.meshgrid(c[0] + axis * f, c[1] + axis * f), axis=-1) for c, f in zip(centers, fovs)])
    determinants = np.stack([draws.grid_maps(f'cutout {i}', grid[..., 0], grid[..., 1], float(z), log=log)['detA']
                             for i, (grid, z) in enumerate(zip(grids, redshifts))], axis=1)
    lower, median, upper = np.quantile(determinants, [.16, .5, .84], axis=0)
    path = Path(path)
    save_arrays(path, median_determinant=median, lower=lower, upper=upper, draw_indices=draws.indices, redshifts=redshifts,
                centers=centers, fovs=fovs, fingerprint=draws.key)
    path.with_suffix('.json').write_text(json.dumps(dict(
        definition='Zero contour of pointwise posterior median determinant; band where the same draws 16-84% determinant interval contains zero',
        model='parametric + conditional mean GRF of each draw' if draws.has_field else 'parametric',
        grf_field=shared_grf_label(draws.evaluator), draw_indices=draws.indices.tolist(),
        redshifts=redshifts.tolist(), resolution=resolution), indent=2) + '\n')
    return grids, median, lower, upper
