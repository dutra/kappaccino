"""Priors are ``numpyro.distributions`` objects of the physical value of each sampled quantity.

A prior is sampled in its own base coordinate: a distribution transformed by ``exp`` (``dist.LogNormal``,
``dist.LogUniform``, ``truncated_lognormal``, ``lognormal_from_moments``) is sampled as the log of the value with the
base distribution's density, any other distribution as the value itself.  That coordinate is the "latent" value of the
parameter (``ParameterSpec.prior`` and ``model.latent_to_physical``).

Supported base distributions: ``dist.Uniform``, ``dist.Normal``, ``dist.TruncatedNormal`` (bounded below, or on both
sides) and ``dist.StudentT``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

LATENT_FAMILIES = ("Uniform", "Normal", "TruncatedNormal", "StudentT")


def _dist():
    import numpyro.distributions as dist
    return dist


def _float(value: Any) -> float | None:
    return None if value is None else float(np.asarray(value))


def is_distribution(value: Any) -> bool:
    """True for a NumPyro distribution (recognized without importing NumPyro)."""
    return type(value).__module__.startswith("numpyro.distributions") and hasattr(value, "log_prob")


# ----------------------------------------------------------------------------- constructors
def truncated_lognormal(median: float, log_sigma: float, *, low: float | None = None, high: float | None = None):
    """Log-normal with median ``median`` and log-space width ``log_sigma``, truncated to [low, high] (physical units):
    ``exp`` of ``dist.TruncatedNormal(ln median, log_sigma, ln low, ln high)``.  NumPyro cannot truncate a positive
    distribution directly, so this is the truncated log-normal."""
    return _lognormal(float(np.log(float(median))), float(log_sigma), low=low, high=high)


def _lognormal(loc: float, scale: float, *, low: float | None, high: float | None):
    """exp of N(loc, scale), truncated to [low, high] in physical units when either is given."""
    dist = _dist()
    log = lambda v: None if v is None else float(np.log(float(v)))
    base = dist.TruncatedNormal(loc, scale, low=log(low), high=log(high)) if (low, high) != (None, None) else dist.Normal(loc, scale)
    return dist.TransformedDistribution(base, dist.transforms.ExpTransform())


def lognormal_from_moments(mean: float, std: float, *, low: float | None = None, high: float | None = None):
    """The (truncated) log-normal whose untruncated mean and standard deviation are ``mean`` and ``std``: a positive
    quantity with a normal-like prior, sampled in log space."""
    mean, std = float(mean), float(std)
    sigma2 = float(np.log1p(std * std / (mean * mean)))
    return _lognormal(float(np.log(mean) - 0.5 * sigma2), float(np.sqrt(sigma2)), low=low, high=high)


# ----------------------------------------------------------------------------- structure
def split(prior: Any) -> tuple[Any, bool]:
    """(base distribution, sampled in log space): the latent distribution of ``prior`` and whether the physical value is
    exp(latent).  Raises for an unsupported distribution."""
    transforms = getattr(prior, "transforms", None)
    if transforms is not None:
        names = [type(t).__name__ for t in transforms]
        if names != ["ExpTransform"]:
            raise ValueError(f"unsupported prior {type(prior).__name__} with transforms {names}: only exp-transformed priors "
                             "(LogNormal, LogUniform, truncated_lognormal) are sampled in a transformed coordinate")
        base, log = prior.base_dist, True
    else:
        base, log = prior, False
    if family(base) not in LATENT_FAMILIES:
        raise ValueError(f"unsupported prior {type(prior).__name__}: use a Uniform, Normal, TruncatedNormal or StudentT, or one of "
                         "these transformed by exp (LogUniform, LogNormal, truncated_lognormal)")
    return base, log


def family(latent: Any) -> str:
    """'Uniform', 'Normal', 'TruncatedNormal' or 'StudentT' (any other class name is returned as is)."""
    name = type(latent).__name__
    if name.endswith("TruncatedDistribution") and type(getattr(latent, "base_dist", None)).__name__ == "Normal":
        return "TruncatedNormal"
    return name


def parameters(latent: Any) -> dict[str, float]:
    """The scalar parameters of a latent distribution: loc, scale, low, high, df (whichever apply; bounds may be inf)."""
    kind = family(latent)
    if kind == "Uniform":
        return {"low": _float(latent.low), "high": _float(latent.high)}
    if kind == "Normal":
        return {"loc": _float(latent.loc), "scale": _float(latent.scale)}
    if kind == "StudentT":
        return {"df": _float(latent.df), "loc": _float(latent.loc), "scale": _float(latent.scale)}
    if kind == "TruncatedNormal":
        low, high = _float(getattr(latent, "low", None)), _float(getattr(latent, "high", None))
        return {"loc": _float(latent.base_dist.loc), "scale": _float(latent.base_dist.scale),
                "low": -math.inf if low is None else low, "high": math.inf if high is None else high}
    raise ValueError(f"unsupported latent distribution {type(latent).__name__}")


def bounds(latent: Any) -> tuple[float, float]:
    """Support of a latent distribution as (low, high), infinite where unbounded."""
    values = parameters(latent)
    return float(values.get("low", -math.inf)), float(values.get("high", math.inf))


def center(latent: Any) -> float:
    """A representative latent value: the midpoint of a uniform, the location of the others."""
    values = parameters(latent)
    return 0.5 * (values["low"] + values["high"]) if family(latent) == "Uniform" else values["loc"]


def make(kind: str, *, validate_args: bool | None = None, **values: float):
    """The latent distribution of family ``kind`` from ``parameters``-style values (array values give a batch)."""
    dist = _dist()
    if kind == "Uniform":
        return dist.Uniform(values["low"], values["high"], validate_args=validate_args)
    if kind == "Normal":
        return dist.Normal(values["loc"], values["scale"], validate_args=validate_args)
    if kind == "StudentT":
        return dist.StudentT(values["df"], values["loc"], values["scale"], validate_args=validate_args)
    if kind == "TruncatedNormal":
        low, high = np.asarray(values["low"], dtype=float), np.asarray(values["high"], dtype=float)
        return dist.TruncatedNormal(values["loc"], values["scale"], low=None if np.all(np.isneginf(low)) else low,
                                    high=None if np.all(np.isposinf(high)) else high, validate_args=validate_args)
    raise ValueError(f"unsupported latent distribution {kind}")


def rebuild(latent: Any, log: bool):
    """The physical prior of a latent distribution (inverse of ``split``)."""
    dist = _dist()
    return dist.TransformedDistribution(latent, dist.transforms.ExpTransform()) if log else latent


def scaled(prior: Any, factor: float):
    """The prior of ``factor`` times the quantity: a shift of the log for a log-sampled prior, a rescaling otherwise."""
    latent, log = split(prior)
    values = parameters(latent)
    if log:
        shift = math.log(float(factor))
        moved = {k: (v + shift if k in ("loc", "low", "high") else v) for k, v in values.items()}
    else:
        moved = {k: (v * float(factor) if k in ("loc", "scale", "low", "high") else v) for k, v in values.items()}
    return rebuild(make(family(latent), **moved), log)


def physical_bounds(prior: Any) -> tuple[float, float]:
    latent, log = split(prior)
    low, high = bounds(latent)
    return (math.exp(low), math.exp(high)) if log else (low, high)


def physical_center(prior: Any) -> float:
    """The physical value at the latent ``center`` (the median of a log-normal)."""
    latent, log = split(prior)
    value = center(latent)
    return math.exp(value) if log else value


def to_scipy(latent: Any):
    """The latent distribution as a frozen scipy.stats distribution (quantiles and densities in NumPy)."""
    from scipy import stats
    values, kind = parameters(latent), family(latent)
    if kind == "Uniform":
        return stats.uniform(loc=values["low"], scale=values["high"] - values["low"])
    if kind == "Normal":
        return stats.norm(loc=values["loc"], scale=values["scale"])
    if kind == "StudentT":
        return stats.t(df=values["df"], loc=values["loc"], scale=values["scale"])
    loc, scale = values["loc"], values["scale"]
    return stats.truncnorm((values["low"] - loc) / scale, (values["high"] - loc) / scale, loc=loc, scale=scale)


# ----------------------------------------------------------------------------- description
def to_dict(prior: Any) -> dict[str, Any]:
    """A JSON-ready description: {"family": ..., "log": bool, parameters...} (infinite bounds as "inf" strings)."""
    latent, log = split(prior)
    clean = {k: (("inf" if v > 0 else "-inf") if math.isinf(v) else v) for k, v in parameters(latent).items()}
    return {"family": family(latent), "log": log, **clean}


def from_dict(payload: dict[str, Any]):
    """Inverse of ``to_dict``."""
    values = {k: float(v) for k, v in payload.items() if k not in ("family", "log")}
    return rebuild(make(str(payload["family"]), **values), bool(payload["log"]))


def describe(prior: Any, digits: int = 4) -> str:
    """Short text form, e.g. 'TruncatedNormal(6.9, 0.3, [5.6, inf]) in log' (latent parameters)."""
    latent, log = split(prior)
    values = parameters(latent)
    number = lambda v: f"{v:.{digits}g}"
    if family(latent) == "Uniform":
        text = f"Uniform({number(values['low'])}, {number(values['high'])})"
    elif family(latent) == "StudentT":
        text = f"StudentT(df {number(values['df'])}, {number(values['loc'])}, {number(values['scale'])})"
    else:
        text = f"{family(latent)}({number(values['loc'])}, {number(values['scale'])}"
        if family(latent) == "TruncatedNormal":
            text += f", [{number(values['low'])}, {number(values['high'])}]"
        text += ")"
    return text + (" in log" if log else "")


# ----------------------------------------------------------------------------- shapes
def _angle_grid(lower_rad: float, upper_rad: float) -> np.ndarray:
    if not np.isfinite(lower_rad) or not np.isfinite(upper_rad) or lower_rad >= upper_rad:
        raise ValueError("angle bounds must be finite and ordered")
    step = 0.5 * math.pi
    values = [float(lower_rad), float(upper_rad)]
    for index in range(int(math.floor(lower_rad / step)) - 1, int(math.ceil(upper_rad / step)) + 2):
        value = float(index * step)
        if lower_rad <= value <= upper_rad:
            values.append(value)
    return np.asarray(sorted(set(values)), dtype=float)


def _polar_box(radii: tuple[float, float], angle_deg: tuple[float, float], names: tuple[str, str]) -> dict[str, Any]:
    """Uniform priors on the Cartesian pair (r cos 2 phi, r sin 2 phi): the smallest box holding the polar region
    radii x angles (angles in degrees, the doubled angle of a spin-2 quantity)."""
    dist = _dist()
    theta = _angle_grid(math.radians(2.0 * angle_deg[0]), math.radians(2.0 * angle_deg[1]))
    c1 = np.asarray([r * math.cos(t) for r in radii for t in theta])
    c2 = np.asarray([r * math.sin(t) for r in radii for t in theta])
    return {names[0]: dist.Uniform(float(c1.min()), float(c1.max())), names[1]: dist.Uniform(float(c2.min()), float(c2.max()))}


def ellipticity_box(upper: float = 0.8, *, lower: float = 0.0, angle_deg: tuple[float, float] = (-180.0, 180.0)) -> dict[str, Any]:
    """Priors {"e1": ..., "e2": ...} of a halo whose Lenstool ellipticite ((a^2 - b^2) / (a^2 + b^2)) lies in [lower, upper]
    and angle_pos in ``angle_deg``: uniform on the smallest (e1, e2) box around that region, e = (1 - q) / (1 + q)."""
    ellipticite = np.clip(np.asarray([lower, upper], dtype=float), 0.0, 1.0 - 1.0e-9)
    q = np.clip(np.sqrt((1.0 - ellipticite) / (1.0 + ellipticite)), 1.0e-3, 1.0)
    modulus = (1.0 - q) / (1.0 + q)
    return _polar_box((float(modulus.min()), float(modulus.max())), angle_deg, ("e1", "e2"))


def shear_box(upper: float = 0.3, *, lower: float = 0.0, angle_deg: tuple[float, float] = (-180.0, 180.0)) -> dict[str, Any]:
    """Priors {"gamma1": ..., "gamma2": ...} of an external shear of amplitude in [lower, upper] and angle in ``angle_deg``."""
    return _polar_box((max(float(lower), 0.0), max(float(upper), 0.0)), angle_deg, ("gamma1", "gamma2"))


def cut_excess_from_moments(mean: float, std: float, *, low: float, high: float, core: float, floor: float = 1.0e-3):
    """Prior of a dPIE cut radius above its effective core (the sampled ``cut_excess_kpc``) for a cut radius of about
    ``mean`` +- ``std`` on [low, high] and a core of ``core`` (kpc): the log-normal of the excess with those moments,
    truncated to [low - core, high - core] (at least ``floor``)."""
    gap_low = max(float(low) - float(core), float(floor))
    gap_high = max(float(high) - float(core), gap_low * (1.0 + 1.0e-9)) if math.isfinite(float(high)) else None
    return lognormal_from_moments(max(float(mean) - float(core), gap_low), float(std), low=gap_low, high=gap_high)
