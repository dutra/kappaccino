"""A solver configuration from a Lenstool parameter file (``.par``).

Reads the blocks the solver has an equivalent for and returns a ``LensClusterSolverConfig`` with library defaults
elsewhere:

* ``runmode`` ``reference``: the reference frame;
* ``cosmology`` / ``cosmologie``: ``H0``, ``omega``, ``lambda``, ``wX``;
* ``potentiel`` / ``limit``: dPIE halos (``profil 81``) and external shear (``profil 14``); radii in arcsec
  (``core_radius``, ``cut_radius``) are converted to kpc at the halo redshift, ``_kpc`` radii are taken as they are;
* ``potfile``: the member population (``mag0``, ``core``/``corekpc``, ``sigma``, ``cut``/``cutkpc``, ``vdslope``,
  ``slope``), with the slopes converted to the relation exponents alpha = 1 / vdslope and beta = 2 / slope;
* ``image``: ``multfile`` when it is a YAML image catalog, and ``sigposArcsec``.

Limits become numpyro priors by their Lenstool flag (0 fixed, 1 uniform, 3 normal, 9 truncated normal), in the
coordinate the solver samples: positions as given; the radii and dispersions in log (a uniform limit is log-uniform, a
normal one the log-normal with its mean and width); the cut radius as its excess above the core; the ellipticity and angle
(and the shear amplitude and angle) together, as uniform priors on the Cartesian (e1, e2) (or (gamma1, gamma2)) box of
their limits.  A field without a limit is fixed, and Lenstool fixes what a ``.par`` file leaves without a limit, so the
relation slopes and the member core are fixed when the file gives them with flag 0; relax them on the returned
configuration with ``dataclasses.replace``.

The solver reads YAML image catalogs only: pass ``image_catalog`` (convert a Lenstool multiple-image file with
scripts/convert_lenstool_catalog.py).  The potfile's ``filein`` catalog is resolved next to the ``.par`` file unless
``member_catalog`` is given.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpyro.distributions as dist

from ..config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, LensClusterSolverConfig, LensModelConfig,
                     MemberPopulationConfig, ReferenceFrameConfig, RunPathsConfig, ScalingModelConfig, ShearHaloConfig)
from ..core.model import DP_IE_PROFILE, SAFE_RADIUS_MARGIN_KPC, SHEAR_PROFILE
from ..priors import cut_excess_from_moments, ellipticity_box, lognormal_from_moments, physical_bounds, shear_box

ALIASES = {"ellipticity": "ellipticite", "profile": "profil", "zlens": "z_lens"}


def _number(token: str) -> float | str:
    try:
        return float(token)
    except ValueError:
        return token


def read_lenstool_par(path: str | Path) -> list[tuple[str, str | None, dict[str, list]]]:
    """Parse a Lenstool ``.par`` file into its blocks, in file order.

    Comments (``#`` to end of line) and blank lines are dropped. A block starts
    with a line ``name [id]`` and ends with ``end``; reading stops at ``fini``. A
    repeated key keeps its last value, as in Lenstool. The keys ``ellipticity``,
    ``profile`` and ``zlens`` are renamed to ``ellipticite``, ``profil`` and
    ``z_lens``.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.par`` file to read.

    Returns
    -------
    list of tuple
        One ``(name, block_id, values)`` per block: ``name`` (str, e.g.
        ``"potentiel"``), ``block_id`` (str, the rest of the opening line, or
        None if empty) and ``values`` (dict mapping each key to its list of
        tokens, numeric tokens as float and others as str).

    Raises
    ------
    ValueError
        If the last block has no ``end`` line.
    """
    path = Path(path)
    blocks: list[tuple[str, str | None, dict[str, list]]] = []
    current: dict[str, list] | None = None
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        tokens = raw.split("#", 1)[0].split()
        if not tokens:
            continue
        if current is None:
            if tokens[0] == "fini":
                break
            current = {}
            blocks.append((tokens[0], " ".join(tokens[1:]) or None, current))
        elif tokens[0] == "end":
            current = None
        else:
            current[ALIASES.get(tokens[0], tokens[0])] = [_number(token) for token in tokens[1:]]
    if current is not None:
        raise ValueError(f"{path}: block {blocks[-1][0]!r} is missing 'end'.")
    return blocks


def limit_bounds(values: list, *, scale: float = 1.0) -> tuple[float, float]:
    """Return the bounds of a Lenstool ``limit`` entry.

    Parameters
    ----------
    values : list
        The entry's tokens, flag first: ``[1, low, high]`` (uniform),
        ``[3, mean, std]`` (normal), ``[9, mean, std, low[, high]]`` (truncated
        normal), or ``[0, ...]`` (fixed).
    scale : float, optional
        Factor applied to every number, e.g. kpc per arcsec to turn radii in
        arcsec into kpc. Default 1.0.

    Returns
    -------
    tuple of float
        ``(low, high)`` times ``scale``, in the units of the entry: the
        uniform range for flag 1, the truncation range for flag 9 (``high`` is
        ``inf`` when absent), and ``(-inf, inf)`` for any other flag.
    """
    flag, numbers = int(values[0]), [float(v) * scale for v in values[1:]]
    if flag == 1:
        return numbers[0], numbers[1]
    if flag == 9:
        return numbers[2], numbers[3] if len(numbers) > 3 else math.inf
    return -math.inf, math.inf


def prior_from_limit(values: list, *, scale: float = 1.0, log: bool = False, floor: float = SAFE_RADIUS_MARGIN_KPC,
                     normal_bounds: tuple[float, float | None] | None = None):
    """Convert a Lenstool ``limit`` entry into a NumPyro prior.

    With ``log=False`` the prior is in the entry's coordinate: flag 1 gives
    ``dist.Uniform(low, high)``, flag 3 ``dist.Normal(mean, std)``, flag 9
    ``dist.TruncatedNormal(mean, std, low, high)``. With ``log=True`` (a positive
    quantity sampled in ln): flag 1 gives ``dist.LogUniform(max(low, floor),
    high)``; flags 3 and 9 give :func:`~kappaccino.priors.lognormal_from_moments`
    with that mean and standard deviation, truncated to ``[max(low, floor),
    high]`` for flag 9 and to ``normal_bounds`` (default ``(floor, None)``) for
    flag 3, whose entry has no bounds.

    Parameters
    ----------
    values : list
        The entry's tokens, flag first (see :func:`limit_bounds`).
    scale : float, optional
        Factor applied to every number including the width, e.g. kpc per arcsec
        for a radius given in arcsec. Default 1.0.
    log : bool, optional
        Sample the quantity in ln (radii, velocity dispersions). Default False.
    floor : float, optional
        Smallest lower bound of a log prior, in the units of the scaled entry.
        Default ``SAFE_RADIUS_MARGIN_KPC`` (``1e-3`` kpc).
    normal_bounds : tuple or None, optional
        ``(low, high)`` truncation of a log prior from a flag-3 entry, in the
        units of the scaled entry; either may be None for no bound on that side,
        so ``(None, None)`` leaves it untruncated. None (default) means
        ``(floor, None)``. Ignored unless ``log`` and flag 3.

    Returns
    -------
    numpyro.distributions.Distribution or None
        The physical prior, or None for flag 0 (the field stays fixed).

    Raises
    ------
    ValueError
        If the flag is not 0, 1, 3 or 9.
    """
    flag, numbers = int(values[0]), [float(v) * scale for v in values[1:]]
    if flag == 0:
        return None
    if flag == 1:
        return dist.LogUniform(max(numbers[0], floor), numbers[1]) if log else dist.Uniform(numbers[0], numbers[1])
    if flag in (3, 9):
        mean, std = numbers[0], numbers[1]
        if not log:
            return dist.Normal(mean, std) if flag == 3 else dist.TruncatedNormal(mean, std, low=numbers[2], high=numbers[3] if len(numbers) > 3 else None)
        low, high = (normal_bounds or (floor, None)) if flag == 3 else (max(numbers[2], floor), numbers[3] if len(numbers) > 3 else None)
        return lognormal_from_moments(max(mean, low if low is not None else floor), std, low=low, high=high)
    raise ValueError(f"Unsupported Lenstool limit flag {flag}; expected 0, 1, 3 or 9.")


def _cut_excess_prior(values: list, *, scale: float, core: float):
    """The prior of the cut radius above the core (``cut_excess_kpc``) from the cut radius's limit entry."""
    flag, numbers = int(values[0]), [float(v) * scale for v in values[1:]]
    if flag == 0:
        return None
    if flag == 1:
        low = max(numbers[0] - core, SAFE_RADIUS_MARGIN_KPC)
        return dist.LogUniform(low, max(numbers[1] - core, low * (1.0 + 1.0e-9)))
    if flag == 3:
        return lognormal_from_moments(max(numbers[0] - core, SAFE_RADIUS_MARGIN_KPC), numbers[1])
    if flag == 9:
        return cut_excess_from_moments(numbers[0], numbers[1], low=numbers[2], high=numbers[3] if len(numbers) > 3 else math.inf, core=core)
    raise ValueError(f"Unsupported Lenstool limit flag {flag}; expected 0, 1, 3 or 9.")


def _polar_priors(limit: dict[str, list], amplitude: str, box, context: str) -> dict[str, Any]:
    """The Cartesian box priors of an (amplitude, angle_pos) pair limited together with uniform limits."""
    given = {name for name in (amplitude, "angle_pos") if int(limit.get(name, [0])[0])}
    if not given:
        return {}
    if given != {amplitude, "angle_pos"} or any(int(limit[name][0]) != 1 for name in given):
        raise ValueError(f"{context}: {amplitude} and angle_pos must both have uniform limits (flag 1) to be sampled.")
    (a_lo, a_hi), angle = limit_bounds(limit[amplitude]), limit_bounds(limit["angle_pos"])
    return box(a_hi, lower=a_lo, angle_deg=angle)


def _inside(value: float, bounds: tuple[float, float]) -> float:
    """The start moved inside the limit's bounds (Lenstool files often start a halo outside its own limit)."""
    return min(max(value, bounds[0]), bounds[1])


def _value(block: dict[str, list], key: str, context: str) -> float:
    if key not in block or not block[key]:
        raise ValueError(f"{context}: missing {key!r}.")
    return float(block[key][0])


def _dpie(potential_id: str, block: dict[str, list], limit: dict[str, list], kpc_per_arcsec) -> DPIEHaloConfig:
    context = f"potentiel {potential_id}"
    z_lens = _value(block, "z_lens", context)
    scale = kpc_per_arcsec(z_lens)
    fields = {name: _value(block, name, context) for name in ("x_centre", "y_centre", "ellipticite", "angle_pos", "v_disp")}
    for name in ("core_radius", "cut_radius"):
        fields[f"{name}_kpc"] = _value(block, f"{name}_kpc", context) if f"{name}_kpc" in block else _value(block, name, context) * scale
    scales = {"core_radius": scale, "cut_radius": scale}
    for name in fields:
        key = name.removesuffix("_kpc")
        if key in limit:
            fields[name] = _inside(fields[name], limit_bounds(limit[key], scale=scales.get(key, 1.0)))
    priors = {}
    for key in limit:                                     # in the file's order, which is the order of the sample sites
        if key in ("x_centre", "y_centre"):
            priors[key] = prior_from_limit(limit[key])
        elif key in ("ellipticite", "angle_pos"):
            priors.update(_polar_priors(limit, "ellipticite", ellipticity_box, context))
        elif key == "core_radius":
            priors["core_radius_kpc"] = prior_from_limit(limit[key], scale=scale, log=True, normal_bounds=(None, None))
        elif key == "cut_radius":
            priors["cut_excess_kpc"] = _cut_excess_prior(limit[key], scale=scale, core=fields["core_radius_kpc"])
        elif key == "v_disp":
            priors["v_disp"] = prior_from_limit(limit[key], log=True, floor=1.0e-4)
    return DPIEHaloConfig(id=potential_id, z_lens=z_lens, priors={k: v for k, v in priors.items() if v is not None}, **fields)


def _shear(potential_id: str, block: dict[str, list], limit: dict[str, list]) -> ShearHaloConfig:
    context = f"potentiel {potential_id}"
    values = {name: _value(block, name, context) for name in ("gamma", "angle_pos")}
    values = {name: _inside(value, limit_bounds(limit[name])) if name in limit else value for name, value in values.items()}
    return ShearHaloConfig(id=potential_id, z_lens=_value(block, "z_lens", context), priors=_polar_priors(limit, "gamma", shear_box, context),
                           **values)


def _slope_prior(values: list | None, exponent: float) -> tuple[float, float | None, tuple[float, float] | None]:
    """A Lenstool slope (vdslope, or slope) as the relation exponent ``exponent / slope``: (mean, std or None when
    fixed, bounds or None)."""
    flag, numbers = int(values[0]), [float(v) for v in values[1:]]
    if flag == 0:
        return exponent / numbers[0], None, None
    if flag == 1:
        lower, upper = exponent / numbers[1], exponent / numbers[0]
        return 0.5 * (lower + upper), 0.5 * (upper - lower), (lower, upper)
    if flag in (3, 9):
        mean = exponent / numbers[0]
        return mean, exponent * numbers[1] / numbers[0] ** 2, None
    raise ValueError(f"Unsupported Lenstool limit flag {flag} for a potfile slope.")


def _population(block: dict[str, list], catalog: Path, kpc_per_arcsec) -> tuple[MemberPopulationConfig, dict[str, Any]]:
    context = "potfile"
    z_lens = _value(block, "z_lens", context)
    scale = kpc_per_arcsec(z_lens)
    core = _value(block, "corekpc", context) if "corekpc" in block else _value(block, "core", context) * scale
    cut_key, cut_scale = ("cutkpc", 1.0) if "cutkpc" in block else ("cut", scale)
    # A flag-3 (normal) limit has no bounds; it takes those of the library's default prior.
    sigma_prior = prior_from_limit(block["sigma"], log=True, floor=1.0e-4, normal_bounds=(20.0, 500.0))
    cut_prior = prior_from_limit(block[cut_key], scale=cut_scale, log=True, normal_bounds=(20.0, 1000.0))
    sigma = float(block["sigma"][1])
    cut = float(block[cut_key][1]) * cut_scale
    if int(block["sigma"][0]) == 1:
        sigma = 0.5 * sum(limit_bounds(block["sigma"]))
    if int(block[cut_key][0]) == 1:
        cut = 0.5 * sum(limit_bounds(block[cut_key], scale=cut_scale))
    population = MemberPopulationConfig(
        id="potfile", catalog_path=str(catalog), mag0=_value(block, "mag0", context), corekpc=core,
        sigma=max(sigma, 1.0), cutkpc=max(cut, 1.0e-3), z_lens=z_lens, profile_type=int(_value(block, "type", context)),
        **{name: prior for name, prior in (("sigma_prior", sigma_prior), ("cutkpc_prior", cut_prior)) if prior is not None})
    scaling: dict[str, Any] = {"fit_corekpc": False}
    defaults = ScalingModelConfig()
    fixed = []
    for key, name, exponent in (("vdslope", "alpha_sigma", 1.0), ("slope", "beta_radius", 2.0)):
        if key not in block:
            continue
        mean, std, bounds = _slope_prior(block[key], exponent)
        default = getattr(defaults, f"{name}_prior")
        low, high = bounds if bounds is not None else physical_bounds(default)
        if std is None:
            fixed.append(name)
            std = float(default.base_dist.scale)
        scaling[f"{name}_prior"] = dist.TruncatedNormal(mean, std, low=low, high=high)
    if sigma_prior is None:
        fixed.append("sigma")
    if cut_prior is None:
        fixed.append("cutkpc")
    scaling["fixed_relation_fields"] = tuple(fixed)
    return population, scaling


def config_from_lenstool_par(path: str | Path, *, image_catalog: str | Path | None = None, member_catalog: str | Path | None = None,
                             output_dir: str | Path = "results", run_name: str | None = None) -> LensClusterSolverConfig:
    """Build a solver configuration from the lens model of a Lenstool ``.par`` file.

    Reads the blocks listed in the module docstring and returns a
    :class:`~kappaccino.config.LensClusterSolverConfig` with library defaults
    elsewhere. Each ``potentiel`` block (``profil 81`` dPIE or ``profil 14``
    external shear) takes its priors from the ``limit`` block with the same id; a
    start value outside its limit is moved inside it. Radii given in arcsec are
    converted to kpc with the ``.par`` cosmology at the halo redshift. At most one
    ``potfile`` block is read. The member catalog must exist; the image catalog
    is only recorded by path and is read later by the run.

    Parameters
    ----------
    path : str or pathlib.Path
        The Lenstool ``.par`` file.
    image_catalog : str or pathlib.Path or None, optional
        The YAML multiple-image catalog (see :mod:`kappaccino.io.image_catalog`).
        None (default) uses the ``image`` block's ``multfile``, resolved relative
        to the ``.par`` file's directory, when it ends in ``.yaml`` or ``.yml``.
    member_catalog : str or pathlib.Path or None, optional
        The cluster-member catalog (see :mod:`kappaccino.io.member_catalog`).
        None (default) uses the ``potfile`` block's ``filein``, resolved relative
        to the ``.par`` file's directory. Unused without a ``potfile`` block.
    output_dir : str or pathlib.Path, optional
        Parent directory of the run directory. Default ``"results"``.
    run_name : str or None, optional
        Name of the run directory ``output_dir/run_name``. None (default) uses
        the ``.par`` file's stem. The figure prefix is always the file's stem.

    Returns
    -------
    LensClusterSolverConfig
        The configuration: reference frame (``runmode reference``; default
        reference 3 at (0, 0) degrees), cosmology (``H0`` in km s^-1 Mpc^-1,
        default 70; ``omega`` default 0.3; ``lambda``; ``wX`` default -1), the
        halos, the member population with its scaling relation (the ``vdslope``
        and ``slope`` limits as priors on alpha = 1 / vdslope and
        beta = 2 / slope), and the image constraints with ``sigposArcsec`` as
        the instrumental position error in arcsec (default 0.1 arcsec).

    Raises
    ------
    ValueError
        If no YAML image catalog is given or found, a ``potentiel`` has an
        unsupported ``profil`` or lacks a required key, an ellipticity (or
        shear) and its angle are not both limited with flag 1, a limit flag is
        unsupported, there is more than one ``potfile``, or the ``potfile`` has
        no ``filein`` and ``member_catalog`` is None.
    FileNotFoundError
        If the member catalog does not exist.
    """
    from ..lens.cosmology import kpc_per_arcsec_from_config

    path = Path(path)
    blocks = read_lenstool_par(path)
    first = {}
    for name, _, data in blocks:
        first.setdefault(name, data)

    reference = ReferenceFrameConfig()
    if "reference" in first.get("runmode", {}):
        kind, ra0, dec0 = (first["runmode"]["reference"] + [0.0, 0.0])[:3]
        reference = ReferenceFrameConfig(reference=int(kind), ra0_deg=float(ra0), dec0_deg=float(dec0))
    cosmo_block = first.get("cosmology", first.get("cosmologie", {}))
    cosmology = CosmologyConfig(H0=float(cosmo_block.get("H0", [70.0])[0]), Om0=float(cosmo_block.get("omega", [0.3])[0]),
                                Ode0=float(cosmo_block["lambda"][0]) if "lambda" in cosmo_block else None,
                                w0=float(cosmo_block.get("wX", [-1.0])[0]))
    cosmo_dict = {"H0": cosmology.H0, "Om0": cosmology.Om0, "w0": cosmology.w0}

    def kpc_per_arcsec(z_lens: float) -> float:
        return float(kpc_per_arcsec_from_config(z_lens, cosmo_dict))

    limits = {block_id: data for name, block_id, data in blocks if name == "limit" and block_id is not None}
    halos = []
    for name, block_id, data in blocks:
        if name not in {"potentiel", "potential"}:
            continue
        potential_id = block_id or str(len(halos) + 1)
        profile = int(_value(data, "profil", f"potentiel {potential_id}"))
        limit = limits.get(potential_id, {})
        if profile == DP_IE_PROFILE:
            halos.append(_dpie(potential_id, data, limit, kpc_per_arcsec))
        elif profile == SHEAR_PROFILE:
            halos.append(_shear(potential_id, data, limit))
        else:
            raise ValueError(f"{path}: potentiel {potential_id} has profil {profile}; only dPIE (81) and shear (14) are read "
                             "from .par files. Configure other profiles in Python (kappaccino.config).")

    image_block = first.get("image", {})
    if image_catalog is None:
        multfile = image_block.get("multfile")
        candidate = path.parent / str(multfile[-1]) if multfile else None
        if candidate is None or candidate.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"{path}: the solver reads YAML image catalogs; convert {candidate or 'the multfile'} with "
                             "scripts/convert_lenstool_catalog.py and pass image_catalog=.")
        image_catalog = candidate
    sigma_inst = float(image_block["sigposArcsec"][0]) if "sigposArcsec" in image_block else ImageConstraintsConfig.sigma_inst_arcsec

    populations, scaling_fields = (), {}
    potfiles = [data for name, _, data in blocks if name == "potfile"]
    if len(potfiles) > 1:
        raise ValueError(f"{path}: {len(potfiles)} potfile blocks; the reader takes one. Configure several populations in Python.")
    if potfiles:
        potfile = potfiles[0]
        if member_catalog is None:
            if "filein" not in potfile:
                raise ValueError(f"{path}: the potfile has no filein; pass member_catalog=.")
            member_catalog = path.parent / str(potfile["filein"][-1])
        if not Path(member_catalog).is_file():
            raise FileNotFoundError(f"{path}: member catalog {member_catalog} not found; pass member_catalog=.")
        population, scaling_fields = _population(potfile, Path(member_catalog), kpc_per_arcsec)
        populations = (population,)

    model = LensModelConfig(reference=reference, cosmology=cosmology, large_halos=tuple(halos), member_populations=populations,
                            image_constraints=ImageConstraintsConfig(catalog_path=str(image_catalog), sigma_inst_arcsec=sigma_inst))
    config = LensClusterSolverConfig(model=model, paths=RunPathsConfig(output_dir=output_dir, run_name=run_name or path.stem,
                                                                       figure_prefix=path.stem))
    if scaling_fields:
        config = replace(config, scaling=replace(ScalingModelConfig(), **scaling_fields))
    return config
