"""The shared GRF of one posterior draw, for every reported product: a conditional realization of the field.

Given a draw's parameters the mass field is still uncertain: the images constrain it where they are and leave it free
elsewhere.  Reporting the conditional mean of every draw would drop that variance, so each draw carries one sample of
the field conditioned on the images, by Matheron's rule:

    r_cond(x) = r_prior(x) + M[v - C^-1 d_sim](x),      d_sim = W r_prior(images) + n,

with ``r_prior`` an unconditional draw of the field (random Fourier modes of the knee spectrum, the low-order modes
removed as in the likelihood), ``n`` a draw of everything else in the data covariance (image-plane GP, measurement
error, floor: N(0, C - W K_r W)), ``v = C^-1 (b - J beta)`` the draw's conditioning coefficients and ``M`` the
conditional-mean operator.  The median over the draws is close to the mean field; their spread is the uncertainty of
the lens parameters and of the field together.

``M`` keeps the saved conditioning coefficients and low-order subtraction of gp_knee.MeanDeflectionKnee (which sampling
and image finding retain) and replaces its radial interpolation with a C2, curl-free Hermite interpolant.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import BPoly, CubicSpline, PPoly
from scipy.linalg import cho_solve

from ..likelihood import gp_knee

PLOT_FIELD_VERSION = 'conditional-realization-hermite-v4'   # v2: all-block twin; v3: JAX evaluation; v4: realization, not the mean


CHUNK = 4096   # points per compiled call; a shorter chunk is padded to one of SIZES, so few shapes are compiled
SIZES = (64, 512, CHUNK)


def _padded_size(n: int) -> int:
    """The compiled size that holds ``n`` points: an image solver calls with a handful of points per iteration."""
    return next(size for size in SIZES if n <= size)
N_MODES = 4096 # random Fourier modes of one prior draw of the field


@jax.jit
def _smooth_fields(points, xy, u, coefficients, breaks, origin_second):
    """Unscaled field sum_m [c_T(r) u_m + c_T'(r)/r (d.u_m) d] and its Jacobian at ``points`` from the images ``xy``
    with coefficients ``u``; ``coefficients`` (6, interval) are the power-basis pieces of the quintic Hermite c_T on
    ``breaks``.  Also returns the largest separation, for the table-range check."""
    d = points[:, None] - xy[None]
    r = jnp.sqrt(jnp.sum(d * d, axis=-1))
    index = jnp.clip(jnp.searchsorted(breaks, r, side='right') - 1, 0, coefficients.shape[1] - 1)
    t = r - breaks[index]
    c = coefficients[:, index]                                  # (6, point, image), highest power first
    ct = ((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]
    first = (((5 * c[0] * t + 4 * c[1]) * t + 3 * c[2]) * t + 2 * c[3]) * t + c[4]
    second = ((20 * c[0] * t + 12 * c[1]) * t + 6 * c[2]) * t + 2 * c[3]
    near = r > 1e-10
    safe = jnp.maximum(r, 1e-10)
    f = jnp.where(near, first / safe, origin_second)
    g = jnp.where(near, (second - first / safe) / safe ** 2, 0.)
    du = jnp.einsum('pmi,mi->pm', d, u)
    value = jnp.einsum('pm,ma->pa', ct, u) + jnp.einsum('pm,pma->pa', f * du, d)
    deriv = (jnp.einsum('pm,ma,pmc->pac', f, u, d) + jnp.einsum('pm,pma,mc->pac', f, d, u)
             + jnp.sum(f * du, axis=1)[:, None, None] * jnp.eye(2) + jnp.einsum('pm,pma,pmc->pac', g * du, d, d))
    return value, deriv, jnp.max(r)


@jax.jit
def _mode_fields(points, kvec, a, b):
    """Deflection (P, 2) and Jacobian (P, 2, 2) of the curl-free plane waves sum_m khat_m [a_m cos(k_m.x) + b_m sin(k_m.x)]."""
    phase = points @ kvec.T
    cosine, sine = jnp.cos(phase), jnp.sin(phase)
    norm = jnp.sqrt(jnp.sum(kvec * kvec, axis=1))
    khat = kvec / norm[:, None]
    return (cosine * a + sine * b) @ khat, jnp.einsum('pm,mi,mj->pij', (cosine * b - sine * a) * norm, khat, khat)


def _chunked(function, points, *arguments):
    """The arrays ``function(points, *arguments)`` returns per point, evaluated in padded chunks of ``CHUNK`` points."""
    outputs = None
    for start in range(0, len(points), CHUNK):
        chunk = points[start:start + CHUNK]
        n = len(chunk)
        result = function(jnp.asarray(np.pad(chunk, ((0, _padded_size(n) - n), (0, 0)), mode='edge')), *arguments)
        if outputs is None:
            outputs = [np.empty((len(points),) + r.shape[1:]) for r in result]
        for output, r in zip(outputs, result):
            output[start:start + n] = np.asarray(r)[:n]
    return outputs


def image_field_covariance(xy, mask, r_grid, cl, ct, amplitude):
    """Unweighted field covariance between the images of one block, (M, M, 2, 2), with the likelihood's linear
    interpolation of the tabulated kernel; padded slots are zero."""
    d = xy[:, None] - xy[None, :]
    radius = np.linalg.norm(d, axis=-1)
    n = d / np.maximum(radius[..., None], 1e-30)
    longitudinal, transverse = np.interp(radius, r_grid, cl), np.interp(radius, r_grid, ct)
    k = amplitude * (transverse[..., None, None] * np.eye(2) + (longitudinal - transverse)[..., None, None] * n[..., :, None] * n[..., None, :])
    return k * (mask[:, None] & mask[None, :])[..., None, None]


def _affine(patterns_at_probe):
    """Offset at the origin and Jacobian of a low-order (affine) pattern from its values at (0,0), (1,0), (0,1)."""
    return patterns_at_probe[0], (patterns_at_probe[1:] - patterns_at_probe[0]).T


PROBE = np.array([[0., 0.], [1., 0.], [0., 1.]])


class SmoothPlotDeflection:
    """The conditional-mean operator M[v] of the shared field, called with (x, y, family indices): deflection and
    Jacobian of ``sum_j K(x - x_j) w_j v_j`` with the low-order modes removed, times the family weight."""

    def __init__(self, layout, tables, coefficients, lam, sigma_grf2, weights_per_family,
                 gamma=None, modes=()):
        cl, ct = map(np.asarray, gp_knee.knee_curves(tables, jnp.asarray(lam), None if gamma is None else jnp.asarray(gamma)))
        self.r = np.asarray(tables['r_grid'])
        # For a potential field c_L = c_T + r*c_T'. Preserve both tabulated
        # kernel values by prescribing c_T' at every node. Quintic Hermite
        # interpolation also matches second derivatives, so the deflection
        # Jacobian, which contains c_T'', is continuous.
        slope = np.zeros_like(ct)
        slope[1:] = (cl[1:]-ct[1:])/self.r[1:]
        second = CubicSpline(self.r, ct, bc_type=((1, 0.), 'not-a-knot'))(self.r, 2)
        self.ct = BPoly.from_derivatives(self.r, np.column_stack([ct,slope,second]))
        self.ct1, self.ct2 = self.ct.derivative(1), self.ct.derivative(2)
        power = PPoly.from_bernstein_basis(self.ct)
        self.coefficients, self.breaks = jnp.asarray(power.c), jnp.asarray(power.x)
        self.origin_second = float(self.ct2(0.))
        self.xy = np.asarray(layout['xy'])
        self.mask = np.asarray(layout['mask'], bool)
        self.blocks = np.asarray(layout['block_of_family'], int)
        self.weights = np.asarray(weights_per_family)
        self.amplitude = float(sigma_grf2)
        n_blocks, size = self.mask.shape
        u = (np.asarray(coefficients)*(self.mask*np.asarray(layout['weights']))[...,None]).reshape(n_blocks,2*size)
        self.affine_offset = np.zeros((n_blocks,2))
        self.affine_jacobian = np.zeros((n_blocks,2,2))
        if modes:
            q, r, centre = map(np.asarray,gp_knee.low_order_projector(layout['xy'],layout['mask'],modes))
            u = u-np.einsum('sin,sn->si',q,np.einsum('sin,si->sn',q,u))
            # Exactly the saved model's low-order subtraction, including its
            # original covariance interpolation. Do not refit the coefficients.
            for block in range(n_blocks):
                k=image_field_covariance(self.xy[block],self.mask[block],self.r,cl,ct,self.amplitude)
                ku=np.einsum('imab,mb->ia',k,u[block].reshape(size,2)).reshape(-1)
                g=np.linalg.solve(r[block],q[block].T@ku)
                pattern=np.asarray(gp_knee.low_order_patterns(PROBE,centre[block],modes))@g
                self.affine_offset[block],self.affine_jacobian[block]=_affine(pattern)
        self.u=u.reshape(n_blocks,size,2)

    def __call__(self,x,y,families):
        points=np.column_stack([np.asarray(x,float).ravel(),np.asarray(y,float).ravel()])
        families=np.asarray(families,int).ravel()
        delta=np.empty_like(points);jac=np.empty((len(points),2,2))
        for block in np.unique(self.blocks[families]):
            rows=np.flatnonzero(self.blocks[families]==block)
            xy=jnp.asarray(self.xy[block,self.mask[block]]);u=jnp.asarray(self.u[block,self.mask[block]])
            value=np.empty((len(rows),2));deriv=np.empty((len(rows),2,2))
            for start in range(0,len(rows),CHUNK):
                chunk=points[rows[start:start+CHUNK]];n=len(chunk)
                padded=np.pad(chunk,((0,_padded_size(n)-n),(0,0)),mode='edge')
                v,j,reach=_smooth_fields(jnp.asarray(padded),xy,u,self.coefficients,self.breaks,self.origin_second)
                if float(reach)>self.r[-1]:
                    raise ValueError('Plotting GRF separation exceeds the saved covariance table')
                value[start:start+n],deriv[start:start+n]=np.asarray(v)[:n],np.asarray(j)[:n]
            value=self.amplitude*value-self.affine_offset[block]-points[rows]@self.affine_jacobian[block].T
            deriv=self.amplitude*deriv-self.affine_jacobian[block]
            weight=self.weights[families[rows]]
            delta[rows]=value*weight[:,None]
            jac[rows]=deriv*weight[:,None,None]
        return delta[:,0],delta[:,1],jac[:,0,0],jac[:,0,1],jac[:,1,0],jac[:,1,1]


class FieldRealization:
    """One conditional realization of the shared field (see the module docstring), called like the mean operator with
    (x, y, family indices) in arcsec; returns the deflection components and the four Jacobian components, each (point,)."""

    def __init__(self, layout, tables, draw, weights_per_family, modes=()):
        self.conditioned = SmoothPlotDeflection(layout, tables, draw['coefficients'], draw['lam'], draw['sigma'] ** 2, weights_per_family,
                                                gamma=draw['curve_gamma'], modes=modes)
        self.kvec, self.a, self.b = (jnp.asarray(draw[key]) for key in ('kvec', 'a', 'b'))
        self.offset, self.affine = np.asarray(draw['prior_offset']), np.asarray(draw['prior_affine'])
        self.weights = np.asarray(weights_per_family)

    def __call__(self, x, y, families):
        points = np.column_stack([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
        families = np.asarray(families, int).ravel()
        conditioned = np.stack(self.conditioned(points[:, 0], points[:, 1], families), axis=-1)
        if not len(points):
            return tuple(conditioned.T)
        value, derivative = _chunked(_mode_fields, points, self.kvec, self.a, self.b)
        prior = np.column_stack([value - self.offset - points @ self.affine.T, (derivative - self.affine).reshape(-1, 4)])
        return tuple((conditioned + self.weights[families][:, None] * prior).T)


def draw_field(layout, tables, cholesky, coefficients, sigma, lam, gamma, *, curve_gamma=None, modes=(), rng, n_modes=N_MODES):
    """The random ingredients of one conditional realization (they do not depend on how far the field is evaluated).

    Parameters
    ----------
    layout : dict
        Block layout of the shared field (one block over all images): ``xy`` (1, M, 2), ``mask`` (1, M), ``weights`` (1, M).
    tables : dict
        Kernel tables of the likelihood (``gp_knee.knee_tables``).
    cholesky : numpy.ndarray
        Lower Cholesky factor of the data covariance C of the block, shape (1, 2M, 2M).
    coefficients : numpy.ndarray
        ``v = C^-1 (b - J beta)`` of the draw, shape (1, M, 2).
    sigma, lam, gamma : float
        Field amplitude (arcsec), knee wavelength (arcsec) and spectral index of the draw.
    curve_gamma : float or None
        ``gamma`` when the tables are tabulated in it, else None.
    modes : tuple of str
        Low-order modes removed from the field.
    rng : numpy.random.Generator
        Source of the realization.
    n_modes : int, optional
        Random Fourier modes of the prior draw.

    Returns
    -------
    dict
        ``coefficients`` (v - C^-1 d_sim, shape (1, M, 2)), the prior modes ``kvec`` (n_modes, 2), ``a``, ``b``, the
        low-order pattern removed from the prior draw (``prior_offset`` (2,), ``prior_affine`` (2, 2)), and ``sigma``,
        ``lam``, ``curve_gamma``.
    """
    xy, mask, weights = (np.asarray(layout[key]) for key in ('xy', 'mask', 'weights'))
    if mask.shape[0] != 1:
        raise ValueError('A field realization needs one shared block over all images')
    xy, mask = xy[0], mask[0].astype(bool)
    size = len(mask)
    w2, m2 = np.repeat(np.where(mask, weights[0], 0.), 2), np.repeat(mask, 2)
    cl, ct = map(np.asarray, gp_knee.knee_curves(tables, jnp.asarray(lam), None if curve_gamma is None else jnp.asarray(curve_gamma)))
    k = image_field_covariance(xy, mask, np.asarray(tables['r_grid']), cl, ct, sigma ** 2).transpose(0, 2, 1, 3).reshape(2 * size, 2 * size)
    # Prior draw: |k| from P_kappa dlnk over the scales of the tables, isotropic directions; per-component variance sigma^2.
    scale_min, scale_max = map(float, tables['scales'])
    nodes = np.geomspace(2 * np.pi / scale_max, 2 * np.pi / scale_min, 3000)
    probability = 1. / (1. + (nodes * lam / (2 * np.pi)) ** gamma)
    magnitude = rng.choice(nodes, size=n_modes, p=probability / probability.sum())
    angle = rng.uniform(0, 2 * np.pi, n_modes)
    kvec = magnitude[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
    a, b = rng.normal(0, sigma * np.sqrt(2. / n_modes), (2, n_modes))
    prior = (np.asarray(_mode_fields(jnp.asarray(xy), jnp.asarray(kvec), jnp.asarray(a), jnp.asarray(b))[0]) * mask[:, None]).reshape(-1)
    offset, affine = np.zeros(2), np.zeros((2, 2))
    if modes:
        q, r_factor, centre = (np.asarray(value)[0] for value in gp_knee.low_order_projector(layout['xy'], layout['mask'], modes))
        a_ = k - q @ (q.T @ k)
        k = a_ - (a_ @ q) @ q.T
        k = .5 * (k + k.T)
        fit = q.T @ prior
        offset, affine = _affine(np.asarray(gp_knee.low_order_patterns(PROBE, centre, modes)) @ np.linalg.solve(r_factor, fit))
        prior = prior - q @ fit
    # Everything in the data covariance that is not the field: image-plane GP, measurement error, floor.
    cholesky = np.asarray(cholesky)[0]
    eigenvalue, eigenvector = np.linalg.eigh(cholesky @ cholesky.T - k * w2[:, None] * w2[None, :])
    noise = eigenvector @ (np.sqrt(np.clip(eigenvalue, 0, None)) * rng.normal(size=len(eigenvalue)))
    simulated = np.where(m2, w2 * prior + noise, 0.)
    v_sim = np.where(m2, cho_solve((cholesky, True), simulated), 0.).reshape(1, size, 2)
    return dict(coefficients=np.asarray(coefficients) - v_sim, kvec=kvec, a=a, b=b, prior_offset=offset, prior_affine=affine,
                sigma=float(sigma), lam=float(lam), curve_gamma=None if curve_gamma is None else float(curve_gamma))


def field_draw(base, theta, rng):
    """:func:`draw_field` for the latent vector ``theta`` of the shared-field evaluator ``base`` (padded layout)."""
    if not hasattr(base, '_field_draw_inputs_jit'):
        base._field_draw_inputs_jit = jax.jit(lambda t: (base.statistics(t)['family_cholesky'], *base._knee_mean_inputs(t)))
    cholesky, coefficients, scales = base._field_draw_inputs_jit(jnp.asarray(theta, dtype=jnp.float64))
    sigma, lam, gamma = map(float, np.asarray(scales))
    return draw_field(base.gp, base.gp_tables, cholesky, coefficients, sigma, lam, gamma,
                      curve_gamma=gamma if base.gp_knee_gamma_index >= 0 else None,
                      modes=tuple(getattr(base, 'gp_project_modes', ())), rng=rng)


def display_tables(base, max_separation=None):
    """The kernel tables of ``base``, extended to ``max_separation`` arcsec when a map reaches farther from the images
    than they do from each other.  The saved nodes are kept exactly; conditioning always uses the original tables."""
    tables = base.gp_tables
    if max_separation is None or max_separation <= float(tables['r_grid'][-1]):
        return tables
    cached = getattr(base, '_plot_extended_tables', None)
    if cached is None or float(cached['r_grid'][-1]) < max_separation:
        gam = np.asarray(tables['gamma_grid'])
        cached = gp_knee.knee_tables(r_max=float(max_separation) + tables['r_step'], r_step=tables['r_step'],
                                     lam_min=tables['lam_range'][0], lam_max=tables['lam_range'][1], n_lam=len(tables['ln_lam']),
                                     scale_min=tables['scales'][0], scale_max=tables['scales'][1],
                                     gamma=float(gam[0]) if len(gam) == 1 else None,
                                     gamma_grid=None if len(gam) == 1 else (float(gam[0]), float(gam[-1]), len(gam)))
        n = len(tables['r_grid'])
        for key in ('c_l', 'c_t', 'd2_l', 'd2_t'):
            cached[key] = cached[key].at[:n].set(tables[key])
        base._plot_extended_tables = cached
    return cached


def field_realization(base, draw, max_separation=None):
    """The realization ``draw`` (:func:`field_draw`) as a field evaluable out to ``max_separation`` arcsec from the images."""
    return FieldRealization(base.gp, display_tables(base, max_separation), draw, base.gp_weights_per_family,
                            modes=tuple(getattr(base, 'gp_project_modes', ())))
