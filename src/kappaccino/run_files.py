"""A run declared in two files in the cluster's catalog folder: ``priors.yaml`` and ``run.yaml``.

``priors.yaml`` declares what is fitted: the reference point, the cosmology, the lens redshift, the image catalog,
the cluster-scale halos, the member populations and the priors of the member scaling relation.  ``run.yaml`` declares
how: the likelihood, the search, the sampler, the member classification, the reported products and the figures, with
named ``profiles`` (for example ``fast``) layered on top.  :func:`load_run` turns the two into a
:class:`~kappaccino.config.LensClusterSolverConfig`.

Values
------
A number is a fixed value.  A string that is an arithmetic expression of numbers, ``inf`` and (in ``priors.yaml``)
``arcsec``, the kpc per arcsec at the lens redshift, is evaluated: ``2.3 / 0.72``, ``13.5871 * arcsec``.  A string
that calls one of the prior functions is a prior on the physical value:

``uniform(low, high)``
    Uniform, sampled linearly.
``loguniform(low, high)``
    Log-uniform, sampled in ln.
``normal(loc, scale)``, ``normal(loc, scale) in [low, high]``
    Normal, or a truncated normal; sampled linearly.
``student_t(df, loc, scale)``
    Student-t, sampled linearly.
``lognormal(median, width)``, ``lognormal(mean=m, std=s)``, either ``in [low, high]``
    Log-normal of a median and ln-width, or of its untruncated mean and standard deviation; sampled in ln.

``inf`` leaves a side of ``in [low, high]`` open.  Any other string stays a string.

Paths in either file are relative to the file's folder.  A file may name a base with ``extends:``; mappings merge
key by key into the base and every other value replaces the base's.

``priors.yaml``
---------------
``reference``, ``cosmology``
    Fields of :class:`~kappaccino.config.ReferenceFrameConfig` and :class:`~kappaccino.config.CosmologyConfig`.
``z_lens``
    Redshift of every halo and member population.
``images``
    The image catalog, or the fields of :class:`~kappaccino.config.ImageConstraintsConfig`.
``halos``
    Cluster-scale components keyed by id.  ``profile`` is ``dpie`` (default), ``nfw``, ``shear`` or ``sheet``.
    ``preset`` builds the component with a constructor of its class (``cluster_scale`` for a dPIE, ``free`` for a
    shear) from the keys that constructor takes.  Every other key is a parameter of the component: a number fixes
    it, a prior samples it starting from the prior's centre (:func:`kappaccino.priors.physical_center`), and a
    mapping with ``start`` and ``prior`` samples it from ``start`` (write it in block style: a prior's commas and
    brackets end a YAML flow mapping).  ``e1``, ``e2``, ``cut_excess_kpc``, ``gamma1`` and
    ``gamma2`` are sampled only.  ``shape: {ellipticity: [low, high], angle: [low, high], soft_cap: c}`` samples the
    shape in the box of :func:`kappaccino.priors.ellipticity_box`, starting from the middle of the box (below
    ``soft_cap``) unless ``ellipticite`` and ``angle_pos`` are given.
``members``
    Member populations keyed by id: ``catalog_path``, an optional ``preset`` (``bergamini2019`` or ``neutral``) with
    the keys it takes, fields of :class:`~kappaccino.config.MemberPopulationConfig`, and ``independent``: members
    taken off the relation, keyed by catalog id, each with an optional ``id`` and priors that replace the defaults
    (``fixed`` holds a parameter at its start).
``scaling``
    An optional ``preset`` (``bergamini2019`` or ``neutral``) and fields of
    :class:`~kappaccino.config.ScalingModelConfig`.  A ``cluster_scale`` halo takes its softening from the centre of
    ``softening_length_prior``.

``run.yaml``
------------
``name``
    The cluster's name in outputs: the figure prefix and the output folder.
``priors``
    The ``priors.yaml`` of the run.
``profiles``
    Named overlays, applied in the order asked for, each with the sections below.
Any other key is a section of :class:`~kappaccino.config.LensClusterSolverConfig` (``likelihood``, ``search``,
``sampling``, ``members``, ``reporting``, ``image_catalog``, ...) with its field names; ``scaling`` here changes
the fields that ``priors.yaml`` set.  An unknown key raises.
"""
from __future__ import annotations

import ast
import inspect
import math
import operator
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from . import config as _config
from .config import (CosmologyConfig, DPIEHaloConfig, ImageConstraintsConfig, IndependentMemberHaloConfig,
                     LensClusterSolverConfig, LensModelConfig, MassSheetConfig, MemberPopulationConfig, NFWHaloConfig,
                     ReferenceFrameConfig, ScalingModelConfig, ShearHaloConfig)
from .priors import ellipticity_box, is_distribution, lognormal_from_moments, physical_center, truncated_lognormal

PRIORS_FILE = "priors.yaml"
RUN_FILE = "run.yaml"

PROFILES = {"dpie": DPIEHaloConfig, "nfw": NFWHaloConfig, "shear": ShearHaloConfig, "sheet": MassSheetConfig}
# Sampled parameters with no field of their own.
PRIOR_ONLY = {"dpie": {"e1", "e2", "cut_excess_kpc"}, "nfw": {"e1", "e2"}, "shear": {"gamma1", "gamma2"}, "sheet": set()}
SAMPLED_FIELDS = {"dpie": {"x_centre", "y_centre", "core_radius_kpc", "v_disp"},
                  "nfw": {"x_centre", "y_centre", "log10_m200", "scale_radius_kpc"}, "shear": set(), "sheet": {"kappa"}}
HALO_PRESETS = {"dpie": {"cluster_scale"}, "shear": {"free"}, "nfw": set(), "sheet": set()}
MEMBER_PRESETS = {"bergamini2019", "neutral"}
SCALING_PRESETS = {"bergamini2019", "neutral"}
# Implicit fields of every component.
_IMPLICIT = {"id", "z_lens", "priors", "profile_type"}

PRIORS_KEYS = {"extends", "reference", "cosmology", "z_lens", "images", "halos", "members", "scaling"}
RUN_KEYS = {"extends", "name", "priors", "profiles"}
# Paths in run.yaml sections, resolved against the file's folder.
RUN_PATHS = {"truth": ("kappa_true_fits", "gammax_true_fits", "gammay_true_fits"),
             "image_catalog": ("image_dir", "candidate_catalog_path"), "partition": ("catalog_path",)}


# ----------------------------------------------------------------------------- values
_OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
              ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _uniform(low, high):
    import numpyro.distributions as dist
    return dist.Uniform(float(low), float(high))


def _loguniform(low, high):
    import numpyro.distributions as dist
    return dist.LogUniform(float(low), float(high))


def _student_t(df, loc, scale):
    import numpyro.distributions as dist
    return dist.StudentT(float(df), float(loc), float(scale))


def _normal(loc, scale, *, low=None, high=None):
    import numpyro.distributions as dist
    if low is None and high is None:
        return dist.Normal(float(loc), float(scale))
    return dist.TruncatedNormal(float(loc), float(scale), low=low, high=high)


def _lognormal(*args, low=None, high=None, **keywords):
    if args and keywords or len(args) not in (0, 2):
        raise ValueError("lognormal takes (median, width), median=, width= or mean=, std=")
    if args:
        keywords = {"median": args[0], "width": args[1]}
    if set(keywords) == {"median", "width"}:
        return truncated_lognormal(float(keywords["median"]), float(keywords["width"]), low=low, high=high)
    if set(keywords) == {"mean", "std"}:
        return lognormal_from_moments(float(keywords["mean"]), float(keywords["std"]), low=low, high=high)
    raise ValueError(f"lognormal takes median= and width=, or mean= and std=; got {sorted(keywords)}")


PRIOR_FUNCTIONS = {"uniform": _uniform, "loguniform": _loguniform, "normal": _normal, "student_t": _student_t,
                   "lognormal": _lognormal}
TRUNCATABLE = {"normal", "lognormal"}


def _arithmetic(node: ast.AST, names: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in names:
        return float(names[node.id])
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_arithmetic(node.left, names), _arithmetic(node.right, names))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_arithmetic(node.operand, names))
    raise _NotArithmetic


class _NotArithmetic(Exception):
    pass


def _prior_call(node: ast.AST) -> str | None:
    """The prior function a call node names, else None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in PRIOR_FUNCTIONS:
        return node.func.id
    return None


def _prior(node: ast.AST, names: dict[str, float], text: str):
    bounds = {}
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In) or not isinstance(node.comparators[0], ast.List) \
                or len(node.comparators[0].elts) != 2:
            raise ValueError(f"{text!r}: write a truncation as `in [low, high]`")
        low, high = (_arithmetic(e, names) for e in node.comparators[0].elts)
        if not low < high:
            raise ValueError(f"{text!r}: the truncation needs low < high")
        bounds = {"low": None if math.isinf(low) or (low <= 0 and _prior_call(node.left) == "lognormal") else low,
                  "high": None if math.isinf(high) else high}
        node = node.left
        if _prior_call(node) not in TRUNCATABLE:
            raise ValueError(f"{text!r}: only normal and lognormal priors take `in [low, high]`")
    name = _prior_call(node)
    try:
        args = [_arithmetic(a, names) for a in node.args]
        keywords = {k.arg: _arithmetic(k.value, names) for k in node.keywords}
    except _NotArithmetic:
        raise ValueError(f"{text!r}: prior arguments must be numbers or arithmetic") from None
    try:
        return PRIOR_FUNCTIONS[name](*args, **keywords, **bounds)
    except TypeError as error:
        raise ValueError(f"{text!r}: {error}") from None


def parse_value(value: Any, names: dict[str, float] | None = None) -> Any:
    """A YAML value with its strings evaluated: a prior call becomes a prior, arithmetic a float, lists and mappings
    are parsed element by element, and anything else is returned unchanged.

    Parameters
    ----------
    value : object
        A value read from YAML.
    names : dict of str to float, optional
        Names arithmetic may use besides ``inf`` (``arcsec`` in ``priors.yaml``).

    Returns
    -------
    object
        The parsed value.

    Raises
    ------
    ValueError
        If a prior call is malformed.
    """
    names = {"inf": math.inf, **(names or {})}
    if isinstance(value, list):
        return [parse_value(item, names) for item in value]
    if isinstance(value, dict):
        return {key: parse_value(item, names) for key, item in value.items()}
    if not isinstance(value, str):
        return value
    try:
        node = ast.parse(value.strip(), mode="eval").body
    except SyntaxError:
        return value
    if _prior_call(node) or (isinstance(node, ast.Compare) and _prior_call(node.left)):
        return _prior(node, names, value)
    try:
        return _arithmetic(node, names)
    except (_NotArithmetic, ZeroDivisionError, OverflowError):
        return value


# ----------------------------------------------------------------------------- files
def _resolve(path: Any, folder: Path) -> Any:
    """A path written relative to ``folder``, relative to the working directory when it lies inside it."""
    if path is None:
        return None
    full = Path(os.path.normpath(folder / str(path)))
    try:
        return str(full.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(full.resolve())


def _merge(base: dict, overlay: dict) -> dict:
    """``overlay`` merged into ``base``: mappings key by key; a prior (or any other value) replaces."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and not {"family", "log"} <= value.keys():
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path, resolve_paths) -> dict:
    """A YAML mapping with its paths resolved against its folder, merged onto the file it ``extends``."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    resolve_paths(data, path.parent)
    base = data.pop("extends", None)
    return _merge(_read(base, resolve_paths), data) if base is not None else data


def _resolve_priors_paths(data: dict, folder: Path) -> None:
    if "extends" in data:
        data["extends"] = _resolve(data["extends"], folder)
    images = data.get("images")
    if isinstance(images, dict) and "catalog_path" in images:
        images["catalog_path"] = _resolve(images["catalog_path"], folder)
    elif images is not None:
        data["images"] = _resolve(images, folder)
    for population in (data.get("members") or {}).values():
        if isinstance(population, dict) and "catalog_path" in population:
            population["catalog_path"] = _resolve(population["catalog_path"], folder)


def _resolve_run_paths(data: dict, folder: Path) -> None:
    for key in ("extends", "priors"):
        if key in data:
            data[key] = _resolve(data[key], folder)
    layers = [data, *((data.get("profiles") or {}).values())]
    for layer in layers:
        if "priors" in layer and layer is not data:
            layer["priors"] = _resolve(layer["priors"], folder)
        for section, keys in RUN_PATHS.items():
            for key in keys:
                if isinstance(layer.get(section), dict) and layer[section].get(key) is not None:
                    layer[section][key] = _resolve(layer[section][key], folder)


def _unknown(where: str, keys, allowed) -> None:
    extra = set(keys) - set(allowed)
    if extra:
        raise ValueError(f"{where}: unknown keys {sorted(extra)}; expected {sorted(allowed)}")


# ----------------------------------------------------------------------------- priors.yaml
def _build(cls, values: dict):
    """A dataclass from parsed values, through the JSON path so lists become tuples and priors stay priors."""
    return _config._from_jsonable(_config._jsonable(values), cls)


def _start_and_prior(key: str, value: Any, where: str):
    if isinstance(value, dict):
        _unknown(f"{where}.{key}", value, {"start", "prior"})
        if not is_distribution(value.get("prior")):
            raise ValueError(f"{where}.{key}: `prior` must be a prior")
        return float(value["start"]), value["prior"]
    if is_distribution(value):
        return physical_center(value), value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), None
    raise ValueError(f"{where}.{key}: expected a number, a prior or {{start, prior}}; got {value!r}")


def _halo(halo_id: str, entry: dict, z_lens: float, softening: float):
    where = f"halos.{halo_id}"
    entry = dict(entry)
    profile = entry.pop("profile", "dpie")
    if profile not in PROFILES:
        raise ValueError(f"{where}: profile must be one of {sorted(PROFILES)}; got {profile!r}")
    cls = PROFILES[profile]
    field_names = {f.name for f in fields(cls)} - _IMPLICIT
    values, priors = {}, {}
    preset = entry.pop("preset", None)
    if preset is not None:
        if preset not in HALO_PRESETS[profile]:
            raise ValueError(f"{where}: a {profile} preset must be one of {sorted(HALO_PRESETS[profile])}; got {preset!r}")
        method = getattr(cls, preset)
        takes = set(inspect.signature(method).parameters) - {"id", "z_lens"}
        arguments = {key: float(entry.pop(key)) for key in list(entry) if key in takes}
        if "softening_kpc" in takes and "softening_kpc" not in arguments:
            arguments["softening_kpc"] = softening
        built = method(id=halo_id, z_lens=z_lens, **arguments)
        values = {name: getattr(built, name) for name in field_names}
        priors = dict(built.priors)
    shape = entry.pop("shape", None)
    for key, value in entry.items():
        if key in SAMPLED_FIELDS[profile]:
            values[key], prior = _start_and_prior(key, value, where)
            if prior is None:
                priors.pop(key, None)
            else:
                priors[key] = prior
        elif key in PRIOR_ONLY[profile]:
            if not is_distribution(value):
                raise ValueError(f"{where}.{key}: is sampled only; give a prior")
            priors[key] = value
        elif key in field_names:
            if is_distribution(value) or isinstance(value, dict):
                raise ValueError(f"{where}.{key}: cannot be sampled; give a number")
            values[key] = value
        else:
            raise ValueError(f"{where}: unknown key {key!r} for a {profile} component")
    if shape is not None:
        if profile not in ("dpie", "nfw"):
            raise ValueError(f"{where}: only dpie and nfw components take a shape")
        _unknown(f"{where}.shape", shape, {"ellipticity", "angle", "soft_cap"})
        (low, high), angle = shape["ellipticity"], tuple(shape.get("angle", (-180.0, 180.0)))
        priors.update(ellipticity_box(float(high), lower=float(low), angle_deg=(float(angle[0]), float(angle[1]))))
        start = 0.5 * (low + high)
        if shape.get("soft_cap") is not None:
            values["ellipticity_soft_cap"] = float(shape["soft_cap"])
            start = min(start, 0.5 * (low + shape["soft_cap"]))       # start below the smooth cap
        values.setdefault("ellipticite", start)
        values.setdefault("angle_pos", 0.5 * (angle[0] + angle[1]))
    return cls(id=str(halo_id), z_lens=z_lens, priors=priors, **values)


def _population(population_id: str, entry: dict, z_lens: float):
    where = f"members.{population_id}"
    entry = dict(entry)
    independent = entry.pop("independent", None) or {}
    preset = entry.pop("preset", None)
    if "catalog_path" not in entry:
        raise ValueError(f"{where}: needs catalog_path")
    catalog = entry.pop("catalog_path")
    if preset is not None:
        if preset not in MEMBER_PRESETS:
            raise ValueError(f"{where}: preset must be one of {sorted(MEMBER_PRESETS)}; got {preset!r}")
        method = getattr(MemberPopulationConfig, preset)
        takes = set(inspect.signature(method).parameters) - {"id", "catalog_path", "z_lens"}
        population = method(population_id, catalog, z_lens, **{key: float(entry.pop(key)) for key in list(entry) if key in takes})
        _unknown(where, entry, {f.name for f in fields(MemberPopulationConfig)} - {"id", "catalog_path", "z_lens"})
        population = replace(population, **{key: (tuple(v) if isinstance(v, list) else v) for key, v in entry.items()})
    else:
        population = _build(MemberPopulationConfig, {"id": population_id, "catalog_path": catalog, "z_lens": z_lens, **entry})
    halos = []
    for catalog_id, halo in independent.items():
        halo = dict(halo or {})
        name = halo.pop("id", None)
        for key, value in halo.items():
            if value != "fixed" and not is_distribution(value):
                raise ValueError(f"{where}.independent.{catalog_id}.{key}: give a prior or `fixed`")
        halos.append(IndependentMemberHaloConfig(population_id=population_id, catalog_id=str(catalog_id), id=name,
                                                 priors={key: None if value == "fixed" else value for key, value in halo.items()}))
    return population, halos


def _scaling(entry: dict):
    entry = dict(entry or {})
    preset = entry.pop("preset", None)
    _unknown("scaling", entry, {f.name for f in fields(ScalingModelConfig)})
    if preset is None:
        return _build(ScalingModelConfig, entry)
    if preset not in SCALING_PRESETS:
        raise ValueError(f"scaling: preset must be one of {sorted(SCALING_PRESETS)}; got {preset!r}")
    return replace(getattr(ScalingModelConfig, preset)(), **{key: (tuple(v) if isinstance(v, list) else v) for key, v in entry.items()})


def load_priors(path: str | Path) -> tuple[LensModelConfig, ScalingModelConfig]:
    """Read a ``priors.yaml``.

    Parameters
    ----------
    path : str or pathlib.Path
        The file.

    Returns
    -------
    model : LensModelConfig
        Reference frame, cosmology, halos, member populations and image catalog.
    scaling : ScalingModelConfig
        The priors of the member scaling relation.

    Raises
    ------
    ValueError
        If a key is unknown or a value malformed.
    """
    data = _read(Path(path), _resolve_priors_paths)
    _unknown(str(path), data, PRIORS_KEYS - {"extends"})
    cosmology = CosmologyConfig(**parse_value(data.get("cosmology") or {}))
    z_lens = float(parse_value(data["z_lens"]))
    from .lens.cosmology import kpc_per_arcsec_from_config
    arcsec = kpc_per_arcsec_from_config(z_lens, {"H0": cosmology.H0, "Om0": cosmology.Om0, "w0": cosmology.w0})
    names = {"arcsec": arcsec}
    scaling = _scaling(parse_value(data.get("scaling"), names))
    softening = physical_center(scaling.softening_length_prior) if scaling.softening_length_prior is not None else 0.0
    halos = tuple(_halo(str(key), parse_value(entry, names), z_lens, softening) for key, entry in (data.get("halos") or {}).items())
    populations, independent = [], []
    for key, entry in (data.get("members") or {}).items():
        population, extra = _population(str(key), parse_value(entry, names), z_lens)
        populations.append(population)
        independent.extend(extra)
    images = data.get("images")
    images = ImageConstraintsConfig(**(images if isinstance(images, dict) else {"catalog_path": images})) if images else None
    model = LensModelConfig(reference=ReferenceFrameConfig(**parse_value(data.get("reference") or {})), cosmology=cosmology,
                            large_halos=halos, member_populations=tuple(populations),
                            independent_member_halos=tuple(independent), image_constraints=images)
    return model, scaling


# ----------------------------------------------------------------------------- run.yaml
def _dotted(assignments) -> dict:
    """``section.field=value`` strings as a nested mapping (values read as YAML)."""
    overlay: dict = {}
    for item in assignments or ():
        if "=" not in item:
            raise ValueError(f"--set takes section.field=value; got {item!r}")
        key, text = item.split("=", 1)
        parts = key.strip().split(".")
        target = overlay
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(text)
    return overlay


def read_run(path: str | Path, profiles=(), overrides=()) -> dict:
    """The mapping of a ``run.yaml`` with its profiles and ``section.field=value`` overrides applied.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``run.yaml``, or the folder that holds it.
    profiles : sequence of str, optional
        Names under ``profiles:`` to apply, in order.
    overrides : sequence of str, optional
        ``section.field=value`` assignments applied last; paths in them are relative to the working directory.

    Returns
    -------
    dict
        The mapping, with ``profiles`` removed.

    Raises
    ------
    ValueError
        If a profile is not declared.
    """
    path = Path(path)
    path = path / RUN_FILE if path.is_dir() else path
    data = _read(path, _resolve_run_paths)
    declared = data.pop("profiles", None) or {}
    for name in profiles:
        if name not in declared:
            raise ValueError(f"{path}: no profile {name!r}; declared: {sorted(declared)}")
        data = _merge(data, declared[name])
    return _merge(data, _dotted(overrides))


def load_run(path: str | Path, profiles=(), overrides=()) -> LensClusterSolverConfig:
    """Build the configuration of a run from its ``run.yaml`` and the ``priors.yaml`` it names.

    The library defaults, then the priors file, then the run file's sections, its profiles and the overrides.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``run.yaml``, or the cluster folder that holds it.
    profiles : sequence of str, optional
        Profiles to apply, in order.
    overrides : sequence of str, optional
        ``section.field=value`` assignments applied last.

    Returns
    -------
    LensClusterSolverConfig
        The configuration (not validated), with ``paths.figure_prefix`` set to ``name``.

    Raises
    ------
    ValueError
        If a key is unknown, a value malformed or a profile not declared.
    TypeError
        If a section names a field its class does not have.
    """
    data = read_run(path, profiles, overrides)
    sections = {f.name for f in fields(LensClusterSolverConfig)} - {"model"}
    _unknown(str(path), data, (RUN_KEYS - {"extends", "profiles"}) | sections)
    model, scaling = load_priors(data.pop("priors"))
    name = data.pop("name", None)
    payload = LensClusterSolverConfig(model=model, scaling=scaling).to_json_dict()
    payload = _merge(payload, _config._jsonable(parse_value(data)))
    if name is not None:
        payload["paths"]["figure_prefix"] = str(name)
    return LensClusterSolverConfig.from_json_dict(payload)


def input_files(path: str | Path, profiles=()) -> list[Path]:
    """The files a run reads: its ``run.yaml``, the ``priors.yaml`` it names and every file they extend."""
    path = Path(path)
    path = path / RUN_FILE if path.is_dir() else path
    chain = []

    def walk(file: Path, resolver):
        chain.append(file)
        data = yaml.safe_load(file.read_text()) or {}
        resolver(data, file.parent)
        if data.get("extends") is not None:
            walk(Path(data["extends"]), resolver)
        return data

    walk(path, _resolve_run_paths)
    priors = read_run(path, profiles)["priors"]
    walk(Path(priors), _resolve_priors_paths)
    return chain
