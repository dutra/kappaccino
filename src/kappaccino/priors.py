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
    """Return whether ``value`` is a NumPyro distribution.

    The check uses the class's module name and a ``log_prob`` attribute, so it
    does not import NumPyro.

    Parameters
    ----------
    value : object
        Any object.

    Returns
    -------
    bool
        True if ``value`` is an instance of a class defined in
        ``numpyro.distributions`` that has a ``log_prob`` method.
    """
    return type(value).__module__.startswith("numpyro.distributions") and hasattr(value, "log_prob")


# ----------------------------------------------------------------------------- constructors
def truncated_lognormal(median: float, log_sigma: float, *, low: float | None = None, high: float | None = None):
    """Build a log-normal prior, optionally truncated, from its median and log width.

    The result is ``exp`` of ``dist.TruncatedNormal(ln median, log_sigma, ln low,
    ln high)`` (``dist.Normal`` when neither bound is given). NumPyro cannot
    truncate a positive distribution directly, so the truncation is applied in ln
    space. The prior is sampled in ln of the value.

    Parameters
    ----------
    median : float
        Median of the untruncated log-normal, in the units of the quantity
        (e.g. kpc, km s^-1); must be positive.
    log_sigma : float
        Standard deviation of the natural log of the quantity (dimensionless).
    low : float or None, optional
        Lower truncation bound in the units of the quantity (positive). None
        (default) leaves the lower side untruncated.
    high : float or None, optional
        Upper truncation bound in the units of the quantity. None (default)
        leaves the upper side untruncated.

    Returns
    -------
    numpyro.distributions.TransformedDistribution
        The physical prior: a ``Normal`` or ``TruncatedNormal`` in ln space
        transformed by ``ExpTransform``.
    """
    return _lognormal(float(np.log(float(median))), float(log_sigma), low=low, high=high)


def _lognormal(loc: float, scale: float, *, low: float | None, high: float | None):
    """exp of N(loc, scale), truncated to [low, high] in physical units when either is given."""
    dist = _dist()
    log = lambda v: None if v is None else float(np.log(float(v)))
    base = dist.TruncatedNormal(loc, scale, low=log(low), high=log(high)) if (low, high) != (None, None) else dist.Normal(loc, scale)
    return dist.TransformedDistribution(base, dist.transforms.ExpTransform())


def lognormal_from_moments(mean: float, std: float, *, low: float | None = None, high: float | None = None):
    r"""Build a log-normal prior with a given mean and standard deviation.

    For a positive quantity with a normal-like prior, sampled in ln space. The
    ln-space parameters are :math:`\sigma^2 = \ln(1 + \mathrm{std}^2/\mathrm{mean}^2)`
    and :math:`\mu = \ln \mathrm{mean} - \sigma^2/2`, so ``mean`` and ``std`` are
    the moments of the untruncated distribution; truncation then shifts them.

    Parameters
    ----------
    mean : float
        Mean of the untruncated log-normal, in the units of the quantity; must be
        positive.
    std : float
        Standard deviation of the untruncated log-normal, in the units of the
        quantity.
    low : float or None, optional
        Lower truncation bound in the units of the quantity (positive). None
        (default) leaves it untruncated below.
    high : float or None, optional
        Upper truncation bound in the units of the quantity. None (default)
        leaves it untruncated above.

    Returns
    -------
    numpyro.distributions.TransformedDistribution
        ``exp`` of a ``Normal`` (no bounds) or ``TruncatedNormal`` in ln space.
    """
    mean, std = float(mean), float(std)
    sigma2 = float(np.log1p(std * std / (mean * mean)))
    return _lognormal(float(np.log(mean) - 0.5 * sigma2), float(np.sqrt(sigma2)), low=low, high=high)


# ----------------------------------------------------------------------------- structure
def split(prior: Any) -> tuple[Any, bool]:
    """Split a physical prior into its latent distribution and coordinate.

    Parameters
    ----------
    prior : numpyro.distributions.Distribution
        A supported physical prior: ``Uniform``, ``Normal``, ``TruncatedNormal``
        or ``StudentT``, or one of these transformed by a single
        ``ExpTransform`` (``LogUniform``, ``LogNormal``,
        :func:`truncated_lognormal`, :func:`lognormal_from_moments`).

    Returns
    -------
    latent : numpyro.distributions.Distribution
        The base distribution that is sampled (of the ln of the value when
        ``log`` is True, of the value otherwise).
    log : bool
        True if the physical value is ``exp(latent)``.

    Raises
    ------
    ValueError
        If ``prior`` has transforms other than one ``ExpTransform``, or its base
        family is not supported.
    """
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
    """Return the family name of a latent distribution.

    Parameters
    ----------
    latent : numpyro.distributions.Distribution
        A latent (untransformed) distribution.

    Returns
    -------
    str
        ``"Uniform"``, ``"Normal"``, ``"TruncatedNormal"`` (any truncated
        distribution whose base is a ``Normal``) or ``"StudentT"``; any other
        class name is returned unchanged.
    """
    name = type(latent).__name__
    if name.endswith("TruncatedDistribution") and type(getattr(latent, "base_dist", None)).__name__ == "Normal":
        return "TruncatedNormal"
    return name


def parameters(latent: Any) -> dict[str, float]:
    """Return the scalar parameters of a latent distribution.

    Parameters
    ----------
    latent : numpyro.distributions.Distribution
        A ``Uniform``, ``Normal``, ``TruncatedNormal`` or ``StudentT``.

    Returns
    -------
    dict of str to float
        In the latent coordinate (ln of the value for a log-sampled prior):
        ``Uniform``: ``low``, ``high``; ``Normal``: ``loc``, ``scale``;
        ``StudentT``: ``df``, ``loc``, ``scale``; ``TruncatedNormal``: ``loc``,
        ``scale`` (of the untruncated normal), ``low``, ``high`` (``-inf`` /
        ``inf`` where that side is untruncated).

    Raises
    ------
    ValueError
        If the family is not one of the four supported.
    """
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
    """Return the support of a latent distribution.

    Parameters
    ----------
    latent : numpyro.distributions.Distribution
        A supported latent distribution (see :func:`parameters`).

    Returns
    -------
    tuple of float
        ``(low, high)`` in the latent coordinate; ``-inf`` / ``inf`` where
        unbounded (always for ``Normal`` and ``StudentT``).
    """
    values = parameters(latent)
    return float(values.get("low", -math.inf)), float(values.get("high", math.inf))


def center(latent: Any) -> float:
    """Return a representative latent value.

    Parameters
    ----------
    latent : numpyro.distributions.Distribution
        A supported latent distribution (see :func:`parameters`).

    Returns
    -------
    float
        The midpoint ``(low + high) / 2`` of a ``Uniform``; ``loc`` otherwise
        (for a ``TruncatedNormal``, the location of the untruncated normal).
    """
    values = parameters(latent)
    return 0.5 * (values["low"] + values["high"]) if family(latent) == "Uniform" else values["loc"]


def make(kind: str, *, validate_args: bool | None = None, **values: float):
    """Build a latent distribution from its family name and parameters.

    Inverse of :func:`family` plus :func:`parameters`. Array values give a batched
    distribution.

    Parameters
    ----------
    kind : str
        ``"Uniform"`` (needs ``low``, ``high``), ``"Normal"`` (``loc``,
        ``scale``), ``"StudentT"`` (``df``, ``loc``, ``scale``) or
        ``"TruncatedNormal"`` (``loc``, ``scale``, ``low``, ``high``; a bound that
        is ``-inf`` / ``inf`` everywhere means that side is untruncated).
    validate_args : bool or None, optional
        Passed to the NumPyro constructor. None (default) uses NumPyro's global
        setting.
    **values : float or array_like
        The parameters named above, in the latent coordinate.

    Returns
    -------
    numpyro.distributions.Distribution
        The latent distribution.

    Raises
    ------
    ValueError
        If ``kind`` is not one of the four supported families.
    KeyError
        If a parameter the family needs is missing from ``values``.
    """
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
    """Return the physical prior of a latent distribution (inverse of :func:`split`).

    Parameters
    ----------
    latent : numpyro.distributions.Distribution
        The latent distribution.
    log : bool
        True if the physical value is ``exp(latent)``.

    Returns
    -------
    numpyro.distributions.Distribution
        ``latent`` transformed by ``ExpTransform`` if ``log``, else ``latent``
        itself.
    """
    dist = _dist()
    return dist.TransformedDistribution(latent, dist.transforms.ExpTransform()) if log else latent


def scaled(prior: Any, factor: float):
    """Return the prior of ``factor`` times the quantity.

    For a log-sampled prior, ``ln(factor)`` is added to ``loc``, ``low`` and
    ``high``; otherwise ``loc``, ``scale``, ``low`` and ``high`` are multiplied by
    ``factor`` (``df`` is unchanged). Used, for example, to move a member-relation
    prior from one reference magnitude to another.

    Parameters
    ----------
    prior : numpyro.distributions.Distribution
        A supported physical prior (see :func:`split`).
    factor : float
        Multiplicative factor (dimensionless); must be positive for a
        log-sampled prior.

    Returns
    -------
    numpyro.distributions.Distribution
        The rescaled physical prior, same family and coordinate as ``prior``.
    """
    latent, log = split(prior)
    values = parameters(latent)
    if log:
        shift = math.log(float(factor))
        moved = {k: (v + shift if k in ("loc", "low", "high") else v) for k, v in values.items()}
    else:
        moved = {k: (v * float(factor) if k in ("loc", "scale", "low", "high") else v) for k, v in values.items()}
    return rebuild(make(family(latent), **moved), log)


def physical_bounds(prior: Any) -> tuple[float, float]:
    """Return the support of a prior in physical units.

    Parameters
    ----------
    prior : numpyro.distributions.Distribution
        A supported physical prior (see :func:`split`).

    Returns
    -------
    tuple of float
        ``(low, high)`` in the units of the quantity: the latent bounds, or their
        ``exp`` for a log-sampled prior (so ``(0, inf)`` for an untruncated
        log-normal and ``(-inf, inf)`` for an untruncated normal).
    """
    latent, log = split(prior)
    low, high = bounds(latent)
    return (math.exp(low), math.exp(high)) if log else (low, high)


def physical_center(prior: Any) -> float:
    """Return the physical value at the latent :func:`center`.

    Parameters
    ----------
    prior : numpyro.distributions.Distribution
        A supported physical prior (see :func:`split`).

    Returns
    -------
    float
        In the units of the quantity: the midpoint of a uniform, the location of
        a (truncated) normal or Student-t, or ``exp`` of the latent center for a
        log-sampled prior (the median of an untruncated log-normal, the geometric
        midpoint of a log-uniform).
    """
    latent, log = split(prior)
    value = center(latent)
    return math.exp(value) if log else value


def to_scipy(latent: Any):
    """Return a latent distribution as a frozen ``scipy.stats`` distribution.

    Useful for quantiles and densities in NumPy without JAX.

    Parameters
    ----------
    latent : numpyro.distributions.Distribution
        A supported latent distribution (see :func:`parameters`).

    Returns
    -------
    scipy.stats.rv_continuous frozen distribution
        ``stats.uniform``, ``stats.norm``, ``stats.t`` or ``stats.truncnorm``
        with the same parameters, in the latent coordinate.
    """
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
    """Return a JSON-ready description of a prior.

    Parameters
    ----------
    prior : numpyro.distributions.Distribution
        A supported physical prior (see :func:`split`).

    Returns
    -------
    dict
        ``family`` (str, see :func:`family`), ``log`` (bool, True if sampled in
        ln of the value), and the latent parameters of :func:`parameters`, with
        infinite bounds written as the strings ``"inf"`` / ``"-inf"``.
    """
    latent, log = split(prior)
    clean = {k: (("inf" if v > 0 else "-inf") if math.isinf(v) else v) for k, v in parameters(latent).items()}
    return {"family": family(latent), "log": log, **clean}


def from_dict(payload: dict[str, Any]):
    """Rebuild a prior from :func:`to_dict` output (inverse of :func:`to_dict`).

    Parameters
    ----------
    payload : dict
        Must contain ``family`` and ``log``; every other key is converted to
        float (so ``"inf"`` / ``"-inf"`` strings are accepted) and passed to
        :func:`make`.

    Returns
    -------
    numpyro.distributions.Distribution
        The physical prior.
    """
    values = {k: float(v) for k, v in payload.items() if k not in ("family", "log")}
    return rebuild(make(str(payload["family"]), **values), bool(payload["log"]))


def describe(prior: Any, digits: int = 4) -> str:
    """Return a short text form of a prior, in its latent parameters.

    Parameters
    ----------
    prior : numpyro.distributions.Distribution
        A supported physical prior (see :func:`split`).
    digits : int, optional
        Significant digits of each number (``g`` format). Default 4.

    Returns
    -------
    str
        ``"Uniform(low, high)"``, ``"Normal(loc, scale)"``,
        ``"TruncatedNormal(loc, scale, [low, high])"`` or
        ``"StudentT(df df, loc, scale)"``, followed by ``" in log"`` for a
        log-sampled prior (whose numbers are then in ln of the value), e.g.
        ``"TruncatedNormal(6.9, 0.3, [5.6, inf]) in log"``.
    """
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
    r"""Return uniform (e1, e2) priors covering a range of halo ellipticity and angle.

    The region of Lenstool ellipticity ``ellipticite`` :math:`(a^2 - b^2)/(a^2 + b^2)`
    in ``[lower, upper]`` and position angle ``angle_pos`` in ``angle_deg`` is
    mapped to the solver's shape modulus :math:`|e| = (1 - q)/(1 + q)` (axis ratio
    :math:`q = b/a`) and doubled angle, :math:`e_1 = |e| \cos 2\phi`,
    :math:`e_2 = |e| \sin 2\phi`. The priors are uniform on the smallest
    Cartesian box holding that region, so they also admit corners outside it.

    Parameters
    ----------
    upper : float, optional
        Largest Lenstool ellipticity (dimensionless, clipped to below 1).
        Default 0.8.
    lower : float, optional
        Smallest Lenstool ellipticity (dimensionless). Default 0.0.
    angle_deg : tuple of float, optional
        ``(low, high)`` position angle range in degrees. Default
        ``(-180.0, 180.0)`` (every orientation).

    Returns
    -------
    dict
        ``{"e1": dist.Uniform, "e2": dist.Uniform}`` (dimensionless bounds).

    Raises
    ------
    ValueError
        If the angle bounds are not finite and increasing.
    """
    ellipticite = np.clip(np.asarray([lower, upper], dtype=float), 0.0, 1.0 - 1.0e-9)
    q = np.clip(np.sqrt((1.0 - ellipticite) / (1.0 + ellipticite)), 1.0e-3, 1.0)
    modulus = (1.0 - q) / (1.0 + q)
    return _polar_box((float(modulus.min()), float(modulus.max())), angle_deg, ("e1", "e2"))


def shear_box(upper: float = 0.3, *, lower: float = 0.0, angle_deg: tuple[float, float] = (-180.0, 180.0)) -> dict[str, Any]:
    r"""Return uniform (gamma1, gamma2) priors covering a range of external shear.

    The region of shear amplitude in ``[lower, upper]`` and angle in
    ``angle_deg`` is mapped to :math:`\gamma_1 = \gamma \cos 2\phi`,
    :math:`\gamma_2 = \gamma \sin 2\phi`, and the priors are uniform on the
    smallest Cartesian box holding it.

    Parameters
    ----------
    upper : float, optional
        Largest shear amplitude (dimensionless; negative values become 0).
        Default 0.3.
    lower : float, optional
        Smallest shear amplitude (dimensionless; negative values become 0).
        Default 0.0.
    angle_deg : tuple of float, optional
        ``(low, high)`` shear angle range in degrees. Default
        ``(-180.0, 180.0)``.

    Returns
    -------
    dict
        ``{"gamma1": dist.Uniform, "gamma2": dist.Uniform}`` (dimensionless
        bounds).

    Raises
    ------
    ValueError
        If the angle bounds are not finite and increasing.
    """
    return _polar_box((max(float(lower), 0.0), max(float(upper), 0.0)), angle_deg, ("gamma1", "gamma2"))


def cut_excess_from_moments(mean: float, std: float, *, low: float, high: float, core: float, floor: float = 1.0e-3):
    r"""Return the prior of a dPIE cut radius above its effective core.

    The solver samples ``cut_excess_kpc``, the cut radius minus the effective core
    :math:`\sqrt{r_\mathrm{core}^2 + \mathrm{softening}^2}`, so the cut always
    exceeds the core. This turns a cut-radius prior of about ``mean`` :math:`\pm`
    ``std`` on ``[low, high]`` into the :func:`lognormal_from_moments` prior of the
    excess with mean ``max(mean - core, low - core)`` and standard deviation
    ``std``, truncated to ``[low - core, high - core]``.

    Parameters
    ----------
    mean : float
        Mean cut radius in kpc.
    std : float
        Standard deviation of the cut radius in kpc.
    low : float
        Lower bound of the cut radius in kpc.
    high : float
        Upper bound of the cut radius in kpc; ``inf`` leaves the excess
        untruncated above.
    core : float
        Effective core radius in kpc, subtracted from the cut radius.
    floor : float, optional
        Smallest allowed lower bound of the excess, in kpc. Default ``1e-3``.

    Returns
    -------
    numpyro.distributions.TransformedDistribution
        Truncated log-normal prior of ``cut_excess_kpc`` (kpc), sampled in ln.
    """
    gap_low = max(float(low) - float(core), float(floor))
    gap_high = max(float(high) - float(core), gap_low * (1.0 + 1.0e-9)) if math.isfinite(float(high)) else None
    return lognormal_from_moments(max(float(mean) - float(core), gap_low), float(std), low=gap_low, high=gap_high)
