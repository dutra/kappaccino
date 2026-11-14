"""The posterior draws behind every reported product, chosen once and evaluated once.

``reporting_draws`` picks ``config.reporting.draws`` random posterior draws over all chains with the run seed.  The
returned :class:`ReportingDraws` evaluates each draw lazily and keeps what it computed, so the figures and tables of one
output pass share the work.  Each draw carries one conditional realization of the shared GRF (``plotting.grf``), drawn
once with a seed fixed by the run seed and the draw's posterior row, and a grid of lens maps is evaluated once per draw
however many products read it.  Every product reports the median over these draws and their 16th to 84th percentile
range, which therefore holds the uncertainty of the lens parameters and of the field; none uses the single best sample.
"""
from __future__ import annotations

import hashlib
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..core.utils import progress
from ..likelihood.grf_fields import _sigma0_factor, shared_field_evaluator, shared_theta


class ReportingDraws:
    """The reporting draws of one run and their cached lens evaluations.

    Attributes
    ----------
    evaluator : Evaluator
        Evaluator of the run.
    indices : numpy.ndarray
        Rows of the posterior behind the maps and other products, shape (draw,), sorted.
    samples, physical : numpy.ndarray
        Latent and physical parameter vectors of those draws, shape (draw, parameter).
    image_indices, image_samples : numpy.ndarray
        Rows and latent vectors of the (more numerous) draws behind the image positions and residuals, each solved in
        its own lens (:meth:`image_correction`); ``indices`` is a subset of ``image_indices``, both the leading entries
        of one seeded shuffle of the posterior.
    seed : int
        Seed of the choice.
    """

    def __init__(self, evaluator, posterior, count: int, seed: int, image_count: int | None = None, rows=None):
        self.evaluator = evaluator
        self.seed = int(seed)
        if rows is None:
            order = np.random.default_rng(self.seed).permutation(len(posterior.samples))
            self.indices = np.sort(order[:int(count)])
            self.image_indices = np.sort(order[:max(int(count), int(image_count or count))])
        else:        # chosen posterior rows, in the given order (scripts that put the best draw first)
            self.indices = self.image_indices = np.asarray(rows, dtype=int)
        self.samples = np.asarray(posterior.samples)[self.indices]
        self.image_samples = np.asarray(posterior.samples)[self.image_indices]
        self.physical = np.asarray(posterior.physical)[self.indices]
        self._field_draws: dict[int, dict] = {}
        self._corrections: dict[int, Any] = {}
        self._grids: dict[tuple, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def key(self) -> str:
        """A fingerprint of the draws (seed, indices and parameter values), for caches written to disk."""
        return hashlib.sha256(str(self.seed).encode() + self.indices.tobytes() + self.samples.tobytes()).hexdigest()

    @property
    def has_field(self) -> bool:
        """Whether the run has one shared GRF mass field (its own, or the all-block twin of a system-block run)."""
        return shared_field_evaluator(self.evaluator) is not None

    @property
    def field_is_rebuilt(self) -> bool:
        """Whether the shared field is rebuilt for a run that did not sample it (``gp_blocks`` other than ``"all"``).
        The posterior gave each block its own field, so a realization of one shared field is not a posterior sample of
        the lens: residuals, RMS and position scatter in the draws' own lenses come out too large."""
        twin = shared_field_evaluator(self.evaluator)
        return twin is not None and twin is not self.evaluator

    def shared(self):
        """The shared-field evaluator (the run's, or the all-block twin of a system-block run) and the draws in its
        latent coordinates (same physical parameters and sources), converted once."""
        cached = getattr(self, '_shared', None)
        if cached is None:
            twin = shared_field_evaluator(self.evaluator)
            if twin is None or twin is self.evaluator:
                cached = (self.evaluator, np.asarray(self.samples))
            else:
                cached = (twin, np.stack([shared_theta(self.evaluator, theta) for theta in self.samples]))
            self._shared = cached
        return cached

    def correction(self, k: int, max_separation: float):
        """Draw ``k``'s conditional realization of the shared field (``plotting.grf.FieldRealization``, its distance
        tables reaching ``max_separation`` arcsec); None without a shared field.  The realization is drawn once per
        draw, so every product and every grid sees the same field."""
        from ..plotting.grf import field_draw, field_realization
        twin = shared_field_evaluator(self.evaluator)
        if twin is None:
            return None
        base = twin._padded_evaluator()
        cached = self._corrections.get(k)
        if cached is None or cached[0] < max_separation:
            if k not in self._field_draws:
                rng = np.random.default_rng([self.seed, int(self.indices[k])])
                self._field_draws[k] = field_draw(base, shared_theta(self.evaluator, self.samples[k]), rng)
            cached = (float(max_separation), field_realization(base, self._field_draws[k], float(max_separation)))
            self._corrections[k] = cached
        return cached[1]

    def image_correction(self, j: int, max_separation: float):
        """The conditional realization of the shared field of image draw ``j`` (``image_indices[j]``): the product
        draw's own (:meth:`correction`) when the row is one, otherwise drawn with the same seeding and not kept."""
        from ..plotting.grf import field_draw, field_realization
        row = int(self.image_indices[j])
        k = np.flatnonzero(self.indices == row)
        if len(k):
            return self.correction(int(k[0]), max_separation)
        twin = shared_field_evaluator(self.evaluator)
        if twin is None:
            return None
        base = twin._padded_evaluator()
        draw = field_draw(base, shared_theta(self.evaluator, self.image_samples[j]), np.random.default_rng([self.seed, row]))
        return field_realization(base, draw, float(max_separation))

    def grf_weight(self, z_source: float) -> float:
        """Factor from the field's units (family zero at the reference redshift) to a source at ``z_source``."""
        twin = shared_field_evaluator(self.evaluator)
        state = twin.state
        return _sigma0_factor(state, z_source) / _sigma0_factor(state, state.gp_reference_redshift) / float(twin.gp_weights_per_family[0])

    def reach(self, xy: np.ndarray) -> float:
        """Largest distance (arcsec) from the bounding box of ``xy`` to an image of the shared field: what the
        realization's distance tables must reach to be evaluated at ``xy``."""
        base = shared_field_evaluator(self.evaluator)._padded_evaluator()
        observed = np.asarray(base.gp['xy'])[np.asarray(base.gp['mask'], bool)]
        corners = np.array([(a, b) for a in (xy[:, 0].min(), xy[:, 0].max()) for b in (xy[:, 1].min(), xy[:, 1].max())])
        return float(np.linalg.norm(corners[:, None] - observed[None], axis=-1).max())

    def lens(self, k: int, xy: np.ndarray, z_source: float, *, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Draw ``k``'s lens at ``xy`` (arcsec, shape (point, 2)) for a source at ``z_source``: the total deflection
        (point, 2) of parametric plus the draw's GRF realization, and the parametric and GRF deflection Jacobians, each
        (point, 2, 2).  The GRF part is zero without a shared field."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        e = self.evaluator
        alpha, parametric = e.total_fields_at(xy, e.packed_state_for_z(e.physical(jnp.asarray(self.samples[k])), z_source), chunk=chunk)
        alpha, parametric = np.array(alpha, dtype=float, copy=True), np.asarray(parametric, dtype=float)
        grf = np.zeros_like(parametric)
        if self.has_field and len(xy):
            weight = self.grf_weight(z_source)
            values = self.correction(k, self.reach(xy))(xy[:, 0], xy[:, 1], np.zeros(len(xy), dtype=int))     # chunked inside, compiled
            alpha += np.stack(values[:2], axis=-1) * weight
            grf[:] = np.stack(values[2:], axis=-1).reshape(-1, 2, 2) * weight
        return alpha, parametric, grf

    def jacobians(self, k: int, xy: np.ndarray, z_source: float, *, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Parametric and GRF deflection Jacobians of draw ``k`` at ``xy`` (arcsec, shape (point, 2)) for a source at
        ``z_source``, each of shape (point, 2, 2); the GRF one is zero without a shared field."""
        _, parametric, grf = self.lens(k, xy, z_source, chunk=chunk)
        if not (np.isfinite(parametric).all() and np.isfinite(grf).all()):
            raise ValueError("Nonfinite lens Jacobian of a reporting draw")
        return parametric, grf

    def deflections(self, k: int, xy: np.ndarray, z_source: float) -> np.ndarray:
        """Total deflection (arcsec, shape (point, 2)) of draw ``k`` at ``xy`` for a source at ``z_source``: parametric
        plus its realization of the shared GRF."""
        return self.lens(k, xy, z_source)[0]

    def grid_maps(self, name: str, x: np.ndarray, y: np.ndarray, z_source: float, *, log=print) -> dict[str, np.ndarray]:
        """Per-draw lens maps on a grid, evaluated once per (grid, redshift) and kept: ``kappa_parametric``,
        ``kappa_grf`` (the convergence of the draw's GRF realization), ``detA_parametric``, and ``kappa``, ``gamma1``, ``gamma2``,
        ``detA``, ``mu`` and ``abs_mu`` of parametric plus GRF, each of shape (draw,) + x.shape, float32.  Nonfinite input points stay NaN."""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        key = (name, float(z_source), hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest())
        if key in self._grids:
            return self._grids[key]
        finite = np.isfinite(x) & np.isfinite(y)
        xy = np.column_stack((x[finite], y[finite]))
        names = ("kappa_parametric", "kappa_grf", "detA_parametric", "kappa", "gamma1", "gamma2", "detA", "mu", "abs_mu")
        maps = {q: np.full((len(self),) + x.shape, np.nan, dtype=np.float32) for q in names}
        for k in progress(range(len(self)), desc=f"{name} draws"):
            parametric, grf = self.jacobians(k, xy, z_source)
            total = parametric + grf
            a = np.eye(2) - total
            det = np.linalg.det(a)
            with np.errstate(divide="ignore"):
                mu = 1.0 / det
            values = dict(kappa_parametric=.5 * np.trace(parametric, axis1=-2, axis2=-1), kappa_grf=.5 * np.trace(grf, axis1=-2, axis2=-1),
                          detA_parametric=np.linalg.det(np.eye(2) - parametric),
                          kappa=.5 * np.trace(total, axis1=-2, axis2=-1), gamma1=.5 * (a[:, 1, 1] - a[:, 0, 0]),
                          gamma2=-.5 * (a[:, 0, 1] + a[:, 1, 0]), detA=det, mu=mu, abs_mu=np.abs(mu))
            for q in names:
                maps[q][k][finite] = values[q]
        log(f"[draws] {name}: {len(self)} posterior draws on a {x.shape[0]}x{x.shape[-1]} grid at z = {z_source:g}")
        self._grids[key] = maps
        return maps


def reporting_draws(config, evaluator, posterior) -> ReportingDraws:
    """The reporting draws of a run (``config.reporting.draws`` product draws within ``config.reporting.image_draws``
    image draws, ``config.runtime.seed``), cached on the posterior object so every product of one output pass shares
    them and their evaluations."""
    count, seed = int(config.reporting.draws), int(config.runtime.seed)
    image_count = int(getattr(config.reporting, 'image_draws', count))
    cached = getattr(posterior, "_reporting_draws", None)
    if (cached is not None and cached.evaluator is evaluator and len(cached) == min(count, len(posterior.samples)) and cached.seed == seed
            and len(cached.image_indices) == min(max(count, image_count), len(posterior.samples))):
        return cached
    draws = ReportingDraws(evaluator, posterior, count, seed, image_count)
    try:
        object.__setattr__(posterior, "_reporting_draws", draws)
    except (AttributeError, TypeError):
        pass
    return draws


def percentiles(values: np.ndarray, axis: int = 0) -> dict[str, np.ndarray]:
    """``{"median", "q16", "q84"}`` of ``values`` along ``axis``, ignoring NaN."""
    q16, q50, q84 = np.nanpercentile(np.asarray(values, dtype=float), [16, 50, 84], axis=axis)
    return {"median": q50, "q16": q16, "q84": q84}
