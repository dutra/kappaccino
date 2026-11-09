"""Model comparison: BIC at the init stage's best fit and WBIC from a tempered NUTS run.

Both numbers estimate :math:`-2 \\ln Z`, minus twice the log evidence, so they read the same way: lower is better, and
the difference between two models fitted to the same data estimates :math:`-2 \\ln B`, minus twice the log Bayes
factor.

BIC (the screen)
    :math:`\\mathrm{BIC} = -2 \\ln L(\\hat\\theta) + k \\ln n` at the init stage's best fit :math:`\\hat\\theta`, with
    :math:`k` the number of sampled parameters and :math:`n` the number of data values (``n_data``).  It needs no
    NUTS, so the init stage alone (minutes) gives BIC.  That makes BIC the number for screening many model variants
    and for comparing with published BIC values.  The best fit maximizes the posterior, not the likelihood, which is the
    usual convention of cluster-lens BIC values.  Compare nested models only (one model is the other with parameters
    added), where :math:`\\Delta k` is the number of added parameters.  BIC is not defined for the uncertain-image
    mixture (a singular model).  The init stage has no uncertain images, so its BIC covers the trusted images only.

WBIC (the reported number)
    :math:`\\mathrm{WBIC} = -2\\, \\mathrm{E}_\\beta[\\ln L]`, the posterior mean of :math:`-2 \\ln L` under the
    tempered posterior :math:`p_\\beta(\\theta) \\propto L(\\theta)^\\beta \\pi(\\theta)` with :math:`\\beta = 1/\\ln n`
    (Watanabe 2013, JMLR 14, 867).  WBIC needs no parameter count and stays valid for hierarchical and mixture models
    (the member deviations, the GRF, the uncertain-image slots), where BIC's :math:`k \\ln n` is wrong.  It costs one
    more NUTS run (``SamplingConfig.wbic``).  Its Monte Carlo error is the standard deviation of :math:`-2 \\ln L`
    over the tempered draws divided by the square root of the draws' effective sample size.  WBIC agrees with
    :math:`-2 \\ln Z` to leading order in :math:`n`.  The remainder can be several units, so calibrate a claimed
    preference with mocks generated from the simpler model.

:math:`\\ln L` is the data log-likelihood (``Evaluator.data_log_likelihood``): the image-position terms, the
uncertain-image mixture and the member kinematics, with every normalization term.  The source-transport correction,
the far-field barrier and the ellipticity caps are part of the prior and are not tempered.  Neither number is available
with enumerated uncertain members (``MemberConfig.membership="marginalize"`` with ``membership_max_enumerated > 0`` and
uncertain members near the images), whose configuration sum mixes the data and prior terms.

Compare two runs only when they fit the same data: :func:`compare` refuses runs whose data fingerprints differ (other
image catalogs, other uncertain images or kinematics, or another trusted/uncertain split).  From the command line::

    python -m kappaccino.inference.model_comparison results/m0416/nfw results/m0416/gnfw --csv comparison.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

MODEL_COMPARISON_FILE = "model_comparison.json"
WBIC_DRAWS_FILE = "wbic_draws.h5"


def n_data(state) -> dict[str, int]:
    """The number of data values :math:`n` of BIC and of :math:`\\beta = 1/\\ln n`: two coordinates per trusted image,
    two per uncertain image and one per member velocity dispersion.

    Parameters
    ----------
    state : kappaccino.core.model.ModelState
        The model, with the source slots when the run has uncertain images.

    Returns
    -------
    dict
        ``image_coordinates``, ``uncertain_image_coordinates``, ``kinematics`` and their sum ``n``.
    """
    images = 2 * int(state.images.n_images)
    uncertain = 2 * len(state.slots.labels) if getattr(state, "slots", None) is not None else 0
    kinematics = len((getattr(state, "member_kinematics", None) or {}).get("member_ids", ()))
    return dict(image_coordinates=images, uncertain_image_coordinates=uncertain, kinematics=kinematics, n=images + uncertain + kinematics)


def wbic_beta(n: int) -> float:
    """The WBIC temperature :math:`\\beta = 1/\\ln n` (Watanabe 2013).  ``n`` must exceed :math:`e`."""
    if n <= math.e:
        raise ValueError(f"WBIC needs more than e data values; got n={n}.")
    return 1.0 / math.log(n)


def parameter_counts(state) -> dict[str, int]:
    """Sampled parameters per group (``sampling.parameter_group``: member offsets and source coordinates pooled, the
    rest by name), so that :math:`\\Delta k` between two models can be read off by group."""
    from .sampling import parameter_group
    counts: dict[str, int] = {}
    for spec in state.parameter_specs:
        group = parameter_group(spec)
        counts[group] = counts.get(group, 0) + 1
    return counts


def data_fingerprint(state) -> str:
    """A hash of the data a run fits: the trusted image positions, errors, families and redshifts, the uncertain image
    labels and the measured dispersions.  Runs with the same fingerprint fit the same data and can be compared."""
    images = state.images
    z = np.asarray(state.z_source_values, dtype=float)[np.asarray(images.z_index_per_image)]
    payload: dict[str, Any] = dict(
        labels=list(images.image_labels), family=np.asarray(images.family_index_per_image).tolist(),
        x=np.round(np.asarray(images.x_obs, dtype=float), 9).tolist(), y=np.round(np.asarray(images.y_obs, dtype=float), 9).tolist(),
        sigma=np.round(np.asarray(images.sigma_inst_per_image, dtype=float), 9).tolist(), z=np.round(z, 9).tolist())
    if getattr(state, "slots", None) is not None:
        payload["uncertain"] = list(state.slots.labels)
    kinematics = getattr(state, "member_kinematics", None) or {}
    if kinematics:
        payload["kinematics"] = dict(ids=list(kinematics["member_ids"]), sigma=np.round(np.asarray(kinematics["sigma"]), 9).tolist(),
                                     err=np.round(np.asarray(kinematics["err"]), 9).tolist())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def bic(state, evaluator, theta: np.ndarray) -> dict[str, Any]:
    """BIC at ``theta``, the init stage's best fit.

    Parameters
    ----------
    state : kappaccino.core.model.ModelState
        The model.
    evaluator : kappaccino.likelihood.evaluator.Evaluator
        Its evaluator.
    theta : numpy.ndarray
        Latent parameter vector of the best fit, shape (parameter,).

    Returns
    -------
    dict
        ``bic``, ``log_likelihood`` (:math:`\\ln L(\\hat\\theta)`), ``k``, ``n`` and the parts of ``n`` (``n_data``),
        ``parameter_counts`` and ``data_fingerprint``.

    Raises
    ------
    NotImplementedError
        With enumerated uncertain members (``Evaluator.data_log_likelihood``).
    """
    log_like = float(jax.jit(evaluator.data_log_likelihood)(jnp.asarray(theta, dtype=jnp.float64)))
    counts, k = n_data(state), int(state.n_parameters)
    return dict(bic=-2.0 * log_like + k * math.log(counts["n"]), log_likelihood=log_like, k=k, n_data=counts, n=counts["n"],
                parameter_counts=parameter_counts(state), data_fingerprint=data_fingerprint(state), point="init best fit")


def log_likelihood_draws(evaluator, samples: np.ndarray, *, batch: int = 8) -> np.ndarray:
    """``Evaluator.data_log_likelihood`` at each row of ``samples`` (draw, parameter), in batches of ``batch``."""
    function = jax.jit(jax.vmap(evaluator.data_log_likelihood))
    samples = np.asarray(samples, dtype=float)
    out = np.empty(samples.shape[0])
    for start in range(0, samples.shape[0], batch):
        chunk = samples[start:start + batch]
        padded = np.concatenate([chunk, np.repeat(chunk[-1:], batch - chunk.shape[0], axis=0)]) if chunk.shape[0] < batch else chunk
        out[start:start + chunk.shape[0]] = np.asarray(function(jnp.asarray(padded)))[:chunk.shape[0]]
    return out


def wbic_from_draws(log_likelihood: np.ndarray, *, beta: float, n: int) -> dict[str, Any]:
    """WBIC and its Monte Carlo error from :math:`\\ln L` at the tempered draws.

    Parameters
    ----------
    log_likelihood : numpy.ndarray
        :math:`\\ln L` per draw, shape (chain, draw).  Draws that are not finite (rejected proposals) are dropped, and
        the effective sample size is then computed with the chains pooled.
    beta, n : float, int
        The temperature and the number of data values, recorded with the result.

    Returns
    -------
    dict
        ``wbic`` (:math:`-2\\,\\mathrm{E}_\\beta[\\ln L]`), ``wbic_error``, ``ess``, ``draws``, ``dropped_draws``,
        ``beta`` and ``n``.
    """
    from numpyro.diagnostics import effective_sample_size
    values = -2.0 * np.asarray(log_likelihood, dtype=float)
    if values.ndim == 1:
        values = values[None]
    finite = np.isfinite(values) & (np.abs(values) < 1e29)
    dropped = int((~finite).sum())
    grouped = values if finite.all() else values[finite][None]
    if grouped.size < 4:
        raise ValueError(f"WBIC needs at least four finite tempered draws; got {grouped.size}.")
    ess = float(effective_sample_size(grouped[..., None])[0]) if grouped.shape[1] >= 4 else float(grouped.size)
    ess = max(min(ess, float(grouped.size)), 1.0)
    return dict(wbic=float(grouped.mean()), wbic_error=float(grouped.std(ddof=1) / math.sqrt(ess)), ess=ess, draws=int(grouped.size),
                dropped_draws=dropped, beta=float(beta), n=int(n))


def run_wbic(state, evaluator, posterior, config, *, log: Callable[[str], None] = print) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The WBIC run: NUTS on the tempered posterior from the last draw of each main chain, then :math:`\\ln L` at
    every tempered draw.

    Parameters
    ----------
    state, evaluator
        The final stage's model and evaluator.
    posterior : kappaccino.core.model.PosteriorResults
        The main posterior, whose last draw per chain starts the tempered chains.
    config : kappaccino.config.LensClusterSolverConfig
        The run configuration, whose ``sampling`` (with ``wbic_warmup`` and ``wbic_samples``) and ``runtime`` settings
        the WBIC run uses.

    Returns
    -------
    tuple
        The WBIC summary (``wbic_from_draws`` plus the data fingerprint, the tempered run's sampler diagnostics and its
        time) and the arrays to save (``log_likelihood`` (chain, draw) and the tempered ``samples``).
    """
    from .sampling import run_nuts
    runtime, sampling = config.runtime, config.sampling
    counts = n_data(state)
    beta = wbic_beta(counts["n"])
    chains = int(runtime.chains)
    grouped = posterior.grouped_samples
    if grouped is not None and np.asarray(grouped).shape[0] == chains:
        starts = np.asarray(grouped)[:, -1]
    else:
        flat = np.asarray(posterior.samples)
        starts = flat[np.linspace(0, flat.shape[0] - 1, chains).astype(int)]
    warmup = int(sampling.wbic_warmup or sampling.warmup)
    samples = int(sampling.wbic_samples or sampling.samples)
    log(f"[wbic] tempered NUTS at beta = 1/ln({counts['n']}) = {beta:.4f}, chains={chains} warmup={warmup} samples={samples}")
    started = time.perf_counter()
    tempered = run_nuts(state, evaluator, starts, warmup=warmup, samples=samples, chains=chains, chain_method=runtime.nuts_chain_method,
                        max_tree_depth=sampling.max_tree_depth, target_accept=sampling.target_accept,
                        initial_step_size=sampling.initial_step_size, seed=int(runtime.seed) + 1, thin=sampling.thin,
                        mass_matrix=sampling.mass_matrix, dense_member_offsets=sampling.dense_member_offsets,
                        dense_offset_members=sampling.dense_offset_members, far_field_offsets=sampling.far_field_offsets,
                        stiffness=False, log_likelihood=lambda theta: evaluator.tempered_log_likelihood(theta, beta), log=log)
    log_like = log_likelihood_draws(evaluator, tempered.samples).reshape(chains, -1)
    result = wbic_from_draws(log_like, beta=beta, n=counts["n"])
    diagnostics = tempered.diagnostics or {}
    result.update(n_data=counts, data_fingerprint=data_fingerprint(state), seconds=time.perf_counter() - started,
                  sampler=dict(divergences=int(np.asarray(tempered.diverging).sum()), max_r_hat=diagnostics.get("max_r_hat"),
                               min_ess=diagnostics.get("min_n_eff"), mean_accept=float(np.mean(tempered.accept_prob))))
    log(f"[wbic] WBIC {result['wbic']:.2f} +- {result['wbic_error']:.2f} (ESS {result['ess']:.0f} of {result['draws']} draws, "
        f"{result['sampler']['divergences']} divergences, {result['seconds']:.0f}s)")
    return result, dict(log_likelihood=log_like, samples=np.asarray(tempered.samples))


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """The model-comparison record of a run root (``<output_dir>/<run_name>``): the final stage's when it exists, else
    the init stage's (BIC only).  ``run`` is the directory name."""
    run_dir = Path(run_dir)
    for stage in ("final", "init"):
        path = run_dir / stage / "artifacts" / MODEL_COMPARISON_FILE
        if path.is_file():
            record = json.loads(path.read_text())
            record.setdefault("run", run_dir.name)
            return record
    raise FileNotFoundError(f"No {MODEL_COMPARISON_FILE} under {run_dir}/final/artifacts or {run_dir}/init/artifacts; "
                            "run the init stage (BIC) or the final stage with sampling.wbic (WBIC) first.")


def compare(runs: Sequence[str | Path | dict[str, Any]]):
    """Table of BIC and WBIC for runs fitted to the same data, with differences to the best (lowest) value.

    Parameters
    ----------
    runs : sequence
        Run roots (``<output_dir>/<run_name>``) or records from :func:`load_run`.

    Returns
    -------
    pandas.DataFrame
        One row per run: ``run``, ``k``, ``n``, ``log_likelihood`` (at the best fit), ``bic``, ``delta_bic``,
        ``wbic``, ``wbic_error``, ``delta_wbic`` and ``delta_wbic_error`` (the two runs' errors in quadrature).
        Missing values are NaN.

    Raises
    ------
    ValueError
        If two runs' BIC (or WBIC) records come from different data or, for WBIC, different temperatures.
    """
    import pandas as pd
    records = [r if isinstance(r, dict) else load_run(r) for r in runs]
    rows = []
    for record in records:
        b, w = record.get("bic") or {}, record.get("wbic") or {}
        rows.append(dict(run=record.get("run"), k=b.get("k", np.nan), n=b.get("n", w.get("n", np.nan)),
                         log_likelihood=b.get("log_likelihood", np.nan), bic=b.get("bic", np.nan), wbic=w.get("wbic", np.nan),
                         wbic_error=w.get("wbic_error", np.nan), _bic_data=b.get("data_fingerprint"),
                         _wbic_data=w.get("data_fingerprint"), _beta=w.get("beta")))
    for key, label in (("_bic_data", "BIC"), ("_wbic_data", "WBIC")):
        seen = {r[key] for r in rows if r[key] is not None}
        if len(seen) > 1:
            raise ValueError(f"The runs' {label} values come from different data (fingerprints {sorted(seen)}): "
                             "compare only runs fitted to the same images, uncertain images and kinematics.")
    betas = {round(r["_beta"], 12) for r in rows if r["_beta"] is not None}
    if len(betas) > 1:
        raise ValueError(f"The runs' WBIC values use different temperatures {sorted(betas)}.")
    table = pd.DataFrame(rows).drop(columns=["_bic_data", "_wbic_data", "_beta"])
    table["delta_bic"] = table["bic"] - table["bic"].min()
    table["delta_wbic"] = table["wbic"] - table["wbic"].min()
    best = table["wbic"].idxmin() if table["wbic"].notna().any() else None
    table["delta_wbic_error"] = (np.hypot(table["wbic_error"], table.loc[best, "wbic_error"]) if best is not None else np.nan)
    if best is not None:
        table.loc[best, "delta_wbic_error"] = 0.0
    return table[["run", "k", "n", "log_likelihood", "bic", "delta_bic", "wbic", "wbic_error", "delta_wbic", "delta_wbic_error"]]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare runs fitted to the same data by BIC and WBIC (lower is better).")
    parser.add_argument("runs", nargs="+", help="run roots, <output_dir>/<run_name>")
    parser.add_argument("--csv", help="also write the table to this CSV file")
    args = parser.parse_args(argv)
    table = compare(args.runs)
    with __import__("pandas").option_context("display.width", 160, "display.max_columns", None):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if args.csv:
        table.to_csv(args.csv, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
