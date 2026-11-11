"""The posterior draws behind every reported product, chosen once and evaluated once.

``reporting_draws`` picks ``config.reporting.draws`` random posterior draws over all chains with the run seed.  The
returned :class:`ReportingDraws` evaluates each draw lazily and keeps what it computed, so the figures and tables of one
output pass share the work: the shared-field GRF conditional mean of a draw is built once, and a grid of lens maps is
evaluated once per draw however many products read it.  Every product reports the median over these draws and their
16th to 84th percentile range; none uses the single best sample.
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
        Rows and latent vectors of the (more numerous) draws behind the image positions and residuals; ``indices`` is a
        subset of ``image_indices``, both the leading entries of one seeded shuffle of the posterior.
    seed : int
        Seed of the choice.
    """

    def __init__(self, evaluator, posterior, count: int, seed: int, image_count: int | None = None):
        self.evaluator = evaluator
        self.seed = int(seed)
        order = np.random.default_rng(self.seed).permutation(len(posterior.samples))
        self.indices = np.sort(order[:int(count)])
        self.image_indices = np.sort(order[:max(int(count), int(image_count or count))])
        self.samples = np.asarray(posterior.samples)[self.indices]
        self.image_samples = np.asarray(posterior.samples)[self.image_indices]
        self.physical = np.asarray(posterior.physical)[self.indices]
        self._corrections: dict[int, Any] = {}
        self._grids: dict[tuple, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def key(self) -> str:
        """A fingerprint of the draws (indices and parameter values), for caches written to disk."""
        return hashlib.sha256(self.indices.tobytes() + self.samples.tobytes()).hexdigest()

    @property
    def has_field(self) -> bool:
        """Whether the run has one shared GRF mass field (its own, or the all-block twin of a system-block run)."""
        return shared_field_evaluator(self.evaluator) is not None

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
        """Draw ``k``'s shared-field conditional-mean deflection for maps (the smooth display interpolation, its distance
        tables reaching ``max_separation`` arcsec), built once per draw; None without a shared field."""
        from ..plotting.grf import plot_deflection_correction
        twin = shared_field_evaluator(self.evaluator)
        if twin is None:
            return None
        cached = self._corrections.get(k)
        if cached is None or cached[0] < max_separation:
            cached = (float(max_separation), plot_deflection_correction(twin, shared_theta(self.evaluator, self.samples[k]),
                                                                        max_separation=float(max_separation)))
            self._corrections[k] = cached
        return cached[1]

    def grf_weight(self, z_source: float) -> float:
        """Factor from the conditional mean's units (family zero at the reference redshift) to a source at ``z_source``."""
        twin = shared_field_evaluator(self.evaluator)
        state = twin.state
        return _sigma0_factor(state, z_source) / _sigma0_factor(state, state.gp_reference_redshift) / float(twin.gp_weights_per_family[0])

    def jacobians(self, k: int, xy: np.ndarray, z_source: float, *, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Parametric and GRF deflection Jacobians of draw ``k`` at ``xy`` (arcsec, shape (point, 2)) for a source at
        ``z_source``, each of shape (point, 2, 2); the GRF one is zero without a shared field."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        e = self.evaluator
        packed = e.packed_state_for_z(e.physical(jnp.asarray(self.samples[k])), z_source)
        _, parametric = e.total_fields_at(xy, packed, chunk=chunk)
        parametric = np.asarray(parametric, dtype=float)
        grf = np.zeros_like(parametric)
        if self.has_field and len(xy):
            # The display tables must reach from every grid point to every image: the farthest box corner bounds it.
            correction = self.correction(k, self._grf_reach(xy))
            weight = self.grf_weight(z_source)
            values = correction(xy[:, 0], xy[:, 1], np.zeros(len(xy), dtype=int))     # chunked inside, compiled
            grf[:] = np.stack(values[2:], axis=-1).reshape(-1, 2, 2) * weight
        if not (np.isfinite(parametric).all() and np.isfinite(grf).all()):
            raise ValueError("Nonfinite lens Jacobian of a reporting draw")
        return parametric, grf

    def _grf_reach(self, xy: np.ndarray) -> float:
        base = shared_field_evaluator(self.evaluator)._padded_evaluator()
        observed = np.asarray(base.gp['xy'])[np.asarray(base.gp['mask'], bool)]
        corners = np.array([(a, b) for a in (xy[:, 0].min(), xy[:, 0].max()) for b in (xy[:, 1].min(), xy[:, 1].max())])
        return float(np.linalg.norm(corners[:, None] - observed[None], axis=-1).max())

    def deflections(self, k: int, xy: np.ndarray, z_source: float) -> np.ndarray:
        """Total deflection (arcsec, shape (point, 2)) of draw ``k`` at ``xy`` for a source at ``z_source``: parametric
        plus the shared-field GRF conditional mean."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        e = self.evaluator
        alpha, _ = e.total_fields_at(xy, e.packed_state_for_z(e.physical(jnp.asarray(self.samples[k])), z_source))
        alpha = np.array(alpha, dtype=float, copy=True)
        if self.has_field and len(xy):
            correction, weight = self.correction(k, self._grf_reach(xy)), self.grf_weight(z_source)
            values = correction(xy[:, 0], xy[:, 1], np.zeros(len(xy), dtype=int))     # chunked inside, compiled
            alpha += np.stack(values[:2], axis=-1) * weight
        return alpha

    def grid_maps(self, name: str, x: np.ndarray, y: np.ndarray, z_source: float, *, log=print) -> dict[str, np.ndarray]:
        """Per-draw lens maps on a grid, evaluated once per (grid, redshift) and kept: ``kappa_parametric``,
        ``kappa_grf`` (the conditional-mean GRF convergence), ``detA_parametric``, and ``kappa``, ``gamma1``, ``gamma2``,
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
