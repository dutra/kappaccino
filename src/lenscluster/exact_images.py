"""Exact image prediction for diagnostics: solve the lens equation per family and match images."""
from __future__ import annotations

import time
from typing import Any

import jax.numpy as jnp
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel as NPLensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver as NPLensEquationSolver
from scipy.optimize import linear_sum_assignment

from .config import ImageDiagnosticsConfig
from .evaluator import Evaluator
from .model import DP_IE_PROFILE, SHEAR_PROFILE, FamilyData, PackedLensState

_TRUST_RADIUS_DEFAULT = 0.3
_LM_DAMPING_RELATIVE = 1.0e-3
_LM_DAMPING_ABSOLUTE = 1.0e-6
_DET_GUARD = 1.0e-30


def astropy_cosmology(cosmo_config: dict[str, Any]):
    from astropy.cosmology import FlatLambdaCDM, FlatwCDM
    h0, om0 = float(cosmo_config["H0"]), float(cosmo_config["Om0"])
    if str(cosmo_config.get("class", "FlatLambdaCDM")) == "FlatwCDM":
        return FlatwCDM(H0=h0, Om0=om0, w0=float(cosmo_config.get("w0", -1.0)))
    return FlatLambdaCDM(H0=h0, Om0=om0)


def packed_to_kwargs_lens(packed: PackedLensState) -> list[dict[str, float]]:
    kwargs: list[dict[str, float]] = []
    for idx, profile in enumerate(np.asarray(packed.profile_type, dtype=int).tolist()):
        if profile == DP_IE_PROFILE:
            kwargs.append({"sigma0": float(packed.sigma0[idx]), "Ra": float(packed.Ra[idx]), "Rs": float(packed.Rs[idx]),
                           "e1": float(packed.e1[idx]), "e2": float(packed.e2[idx]),
                           "center_x": float(packed.center_x[idx]), "center_y": float(packed.center_y[idx])})
        elif profile == SHEAR_PROFILE:
            kwargs.append({"gamma1": float(packed.gamma1[idx]), "gamma2": float(packed.gamma2[idx])})
        else:
            raise ValueError(f"Unsupported profile type {profile}.")
    return kwargs


class ExactImageSolver:
    """Solve the lens equation for every family at a latent vector and match the model images."""

    def __init__(self, evaluator: Evaluator, config: ImageDiagnosticsConfig) -> None:
        self.evaluator = evaluator
        self.config = config
        self.models_by_z: dict[float, tuple[NPLensModel, NPLensEquationSolver]] = {}
        self.timing: dict[str, float] = {"exact_solver": 0.0}
        self.all_components = np.arange(evaluator.n_components, dtype=np.int32)

    # ------------------------------------------------------------------ lens fields at arbitrary points
    def ray_shoot_and_jacobian(self, packed: PackedLensState, x: np.ndarray, y: np.ndarray):
        xy = jnp.stack((jnp.asarray(x, dtype=jnp.float64), jnp.asarray(y, dtype=jnp.float64)), -1)
        alpha_rows, u_rows = self.evaluator.component_rows(xy, packed, self.all_components)
        alpha = np.asarray(jnp.sum(alpha_rows, axis=0))
        a = np.asarray(jnp.eye(2) - jnp.sum(u_rows, axis=0))
        return (np.asarray(xy[:, 0]) - alpha[:, 0], np.asarray(xy[:, 1]) - alpha[:, 1],
                a[:, 0, 0], a[:, 0, 1], a[:, 1, 0], a[:, 1, 1])

    def magnification(self, packed: PackedLensState, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        _, _, a00, a01, a10, a11 = self.ray_shoot_and_jacobian(packed, x, y)
        return 1.0 / (a00 * a11 - a01 * a10)

    def lenstronomy_model(self, z_source: float) -> tuple[NPLensModel, NPLensEquationSolver]:
        key = float(z_source)
        if key not in self.models_by_z:
            model = NPLensModel(lens_model_list=self.evaluator.state.lens_model_list, z_lens=self.evaluator.state.z_lens,
                                z_source=key, cosmo=astropy_cosmology(self.evaluator.state.cosmo_config))
            self.models_by_z[key] = (model, NPLensEquationSolver(model))
        return self.models_by_z[key]

    # ------------------------------------------------------------------ local Levenberg--Marquardt solver
    def _lm_step(self, fx, fy, a00, a01, a10, a11):
        finite = np.isfinite(fx) & np.isfinite(fy) & np.isfinite(a00) & np.isfinite(a01) & np.isfinite(a10) & np.isfinite(a11)
        n00 = a00 ** 2 + a10 ** 2
        n01 = a00 * a01 + a10 * a11
        n11 = a01 ** 2 + a11 ** 2
        lam = _LM_DAMPING_ABSOLUTE + _LM_DAMPING_RELATIVE * np.maximum(n00 + n11, 0.0)
        h00, h11, h01 = n00 + lam, n11 + lam, n01
        rhs0 = -(a00 * fx + a10 * fy)
        rhs1 = -(a01 * fx + a11 * fy)
        det = np.maximum(h00 * h11 - h01 ** 2, _DET_GUARD)
        dx = (h11 * rhs0 - h01 * rhs1) / det
        dy = (-h01 * rhs0 + h00 * rhs1) / det
        radius = np.sqrt(dx ** 2 + dy ** 2 + 1.0e-30)
        trust = float(self.config.exact_image_lm_trust_radius_arcsec)
        if trust > 0.0:
            scaled = radius / trust
            scale = np.tanh(scaled) / np.maximum(scaled, 1.0e-12)
            dx, dy = dx * scale, dy * scale
        return dx, dy, finite & np.isfinite(dx) & np.isfinite(dy)

    def _refine(self, packed, source_x, source_y, start_x, start_y) -> dict[str, np.ndarray]:
        cfg = self.config
        x = np.asarray(start_x, dtype=float).copy()
        y = np.asarray(start_y, dtype=float).copy()
        n = x.size
        converged = np.zeros(n, dtype=bool)
        finite = np.isfinite(x) & np.isfinite(y)
        iterations = np.zeros(n, dtype=int)
        final_step = np.full(n, np.inf)
        final_source = np.full(n, np.inf)
        for step in range(int(cfg.exact_image_lm_max_iter)):
            bx, by, a00, a01, a10, a11 = self.ray_shoot_and_jacobian(packed, x, y)
            rx, ry = bx - source_x, by - source_y
            dx, dy, ok = self._lm_step(rx, ry, a00, a01, a10, a11)
            step_norm = np.sqrt(dx ** 2 + dy ** 2)
            source_norm = np.sqrt(rx ** 2 + ry ** 2)
            active = finite & ok & ~converged
            x = np.where(active, x + dx, x)
            y = np.where(active, y + dy, y)
            final_step = np.where(active | converged, step_norm, final_step)
            final_source = np.where(active | converged, source_norm, final_source)
            iterations = np.where(active, step + 1, iterations)
            converged |= (active & np.isfinite(step_norm) & np.isfinite(source_norm)
                          & (step_norm <= cfg.exact_image_displacement_tol_arcsec)
                          & (source_norm <= max(cfg.exact_image_identification_tol_arcsec, cfg.exact_image_precision_limit)))
            finite &= ok & np.isfinite(x) & np.isfinite(y)
            if np.all(converged | ~finite):
                break
        bx, by, a00, a01, a10, a11 = self.ray_shoot_and_jacobian(packed, x, y)
        rx, ry = bx - source_x, by - source_y
        cx, cy, ok = self._lm_step(rx, ry, a00, a01, a10, a11)
        root_error = np.sqrt(cx ** 2 + cy ** 2)
        source_norm = np.sqrt(rx ** 2 + ry ** 2)
        finite_final = finite & ok & np.isfinite(root_error) & np.isfinite(source_norm) & np.isfinite(x) & np.isfinite(y)
        identified = finite_final & (root_error <= cfg.exact_image_identification_tol_arcsec)
        converged |= identified & (root_error <= cfg.exact_image_displacement_tol_arcsec)
        return dict(x=x, y=y, converged=converged, finite=finite_final, identification_pass=identified,
                    iterations=iterations, final_step_arcsec=root_error, final_source_residual_arcsec=source_norm)

    def solve_local_lm(self, family: FamilyData, packed, source_x, source_y, *, adaptive: bool):
        cfg = self.config
        start_x, start_y = np.asarray(family.x_obs, dtype=float), np.asarray(family.y_obs, dtype=float)
        result = self._refine(packed, source_x, source_y, start_x, start_y)
        ok = np.asarray(result["identification_pass"], dtype=bool)
        if adaptive and not ok.all():
            radius = min(float(cfg.exact_image_lm_trust_radius_arcsec), max(float(cfg.exact_image_min_distance_arcsec), 1.0e-3))
            for idx in np.flatnonzero(~ok):
                best, best_score, r = None, np.inf, radius
                for _ in range(int(cfg.exact_image_adaptive_max_levels)):
                    offsets = np.asarray([-r, 0.0, r])
                    gx, gy = np.meshgrid(offsets, offsets)
                    refined = self._refine(packed, source_x, source_y, start_x[idx] + gx.ravel(), start_y[idx] + gy.ravel())
                    score = np.where(refined["finite"], refined["final_step_arcsec"] + 0.01 * refined["final_source_residual_arcsec"], np.inf)
                    k = int(np.argmin(score))
                    if np.isfinite(score[k]) and score[k] < best_score:
                        best_score, best = float(score[k]), {key: np.asarray(value)[k] for key, value in refined.items()}
                    if best is not None and bool(best["identification_pass"]):
                        break
                    r *= 0.5
                if best is not None and bool(best["identification_pass"]):
                    for key in result:
                        result[key][idx] = best[key]
            ok = np.asarray(result["identification_pass"], dtype=bool)
        if not ok.any():
            raise RuntimeError("local exact image solver failed identification tolerance")
        return np.asarray(result["x"])[ok], np.asarray(result["y"])[ok]

    def solve_lenstronomy(self, family: FamilyData, packed, source_x, source_y):
        model, solver = self.lenstronomy_model(family.z_source)
        x, y = solver.image_position_from_source(
            source_x, source_y, packed_to_kwargs_lens(packed), solver="lenstronomy",
            min_distance=float(self.config.exact_image_min_distance_arcsec), search_window=family.search_window,
            x_center=family.x_center, y_center=family.y_center, num_iter_max=int(self.config.exact_image_num_iter_max),
            precision_limit=float(self.config.exact_image_precision_limit))
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def solve(self, family: FamilyData, packed, source_x: float, source_y: float):
        started = time.perf_counter()
        try:
            finder = self.config.exact_image_finder
            if finder == "lenstronomy":
                return self.solve_lenstronomy(family, packed, source_x, source_y)
            return self.solve_local_lm(family, packed, source_x, source_y, adaptive=finder == "local-lm-adaptive")
        finally:
            self.timing["exact_solver"] += time.perf_counter() - started

    # ------------------------------------------------------------------ matching
    def match(self, x_pred: np.ndarray, y_pred: np.ndarray, family: FamilyData) -> dict[str, Any]:
        x_pred, y_pred = np.asarray(x_pred, dtype=float).ravel(), np.asarray(y_pred, dtype=float).ravel()
        n_obs, n_pred = family.n_images, int(min(x_pred.size, y_pred.size))
        matched_x, matched_y = np.full(n_obs, np.nan), np.full(n_obs, np.nan)
        recovered = np.zeros(n_obs, dtype=bool)
        out = dict(produced_image_count=n_pred, recovered_image_count=0, missing_image_count=n_obs, extra_image_count=n_pred,
                   multiplicity_failed=True, multiplicity_failure_reason="no_model_images" if n_pred == 0 else "no_matches",
                   matched_model_x_arcsec=matched_x, matched_model_y_arcsec=matched_y, recovered_image_mask=recovered,
                   extra_model_x_arcsec=np.asarray([], dtype=float), extra_model_y_arcsec=np.asarray([], dtype=float))
        if n_pred == 0 or n_obs == 0:
            return out
        pred = np.column_stack([x_pred[:n_pred], y_pred[:n_pred]])
        obs = np.column_stack([family.x_obs, family.y_obs])
        cost = np.linalg.norm(pred[:, None, :] - obs[None, :, :], axis=2)
        cost = np.where(np.isfinite(pred).all(axis=1)[:, None] & np.isfinite(obs).all(axis=1)[None, :], cost, np.inf)
        if not np.isfinite(cost).any():
            out["multiplicity_failure_reason"] = "nonfinite_prediction"
            return out
        tolerance = float(self.config.match_tolerance_arcsec)
        assignment_cost = np.where(np.isfinite(cost), cost, cost[np.isfinite(cost)].max() + tolerance + 1.0)
        rows, cols = linear_sum_assignment(assignment_cost)
        accepted = np.isfinite(cost[rows, cols]) & (cost[rows, cols] <= tolerance)
        for p, o in zip(rows[accepted], cols[accepted]):
            matched_x[o], matched_y[o], recovered[o] = x_pred[p], y_pred[p], True
        extra = np.ones(n_pred, dtype=bool)
        extra[rows[accepted]] = False
        n_rec = int(accepted.sum())
        missing, surplus = n_obs - n_rec, n_pred - n_rec
        reason = "" if missing == 0 and surplus == 0 else ("missing_and_extra_model_images" if missing and surplus and n_pred != n_obs
                                                            else "match_tolerance_exceeded" if missing and surplus
                                                            else "missing_model_images" if missing else "extra_model_images")
        out.update(recovered_image_count=n_rec, missing_image_count=missing, extra_image_count=surplus,
                   multiplicity_failed=bool(missing or surplus), multiplicity_failure_reason=reason,
                   matched_model_x_arcsec=matched_x, matched_model_y_arcsec=matched_y, recovered_image_mask=recovered,
                   extra_model_x_arcsec=x_pred[:n_pred][extra], extra_model_y_arcsec=y_pred[:n_pred][extra])
        return out

    # ------------------------------------------------------------------ per family prediction
    def predict_family(self, family: FamilyData, packed: PackedLensState, source_x: float, source_y: float) -> dict[str, Any]:
        """Model images of one family: matched positions, recovery mask, extra images, magnifications."""
        details: dict[str, Any] = {"failed": True, "exact_image_rms": np.nan}
        mu = self.magnification(packed, family.x_obs, family.y_obs)
        try:
            x_pred, y_pred = self.solve(family, packed, source_x, source_y)
        except Exception as error:
            details.update(self.match(np.asarray([]), np.asarray([]), family))
            details["multiplicity_failure_reason"] = f"exact_image_prediction_failed: {error}"
            details["magnification"] = mu
            return details
        details.update(self.match(x_pred, y_pred, family))
        recovered = details["recovered_image_mask"]
        if recovered.any():
            residual = np.sqrt((details["matched_model_x_arcsec"] - family.x_obs) ** 2 + (details["matched_model_y_arcsec"] - family.y_obs) ** 2)
            details["exact_image_rms"] = float(np.sqrt(np.nanmean(residual[recovered] ** 2)))
            details["failed"] = not bool(recovered.all())
        details["magnification"] = mu
        return details
