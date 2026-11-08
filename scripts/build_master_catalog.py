"""Build a cluster's master catalog (catalogs/<cluster>/master/) from its per-paper folders.

Reads ``master/sources.yaml`` (the papers, the base paper, the match radius, the quality flags of each paper and the
reliability scale) and ``master/decisions.yaml`` (hand-made choices), and writes

    images.yaml           every system of every paper, with its tier ``reliability``, and every image (or knot) with
                          its association ``prior``: the trusted and the uncertain images in one file
    members.yaml          the members, each with ``p_member``
    crossmatch.yaml       every paper's systems and images matched to each other, frame offsets, tiers, conflicts

Nothing a paper lists is dropped: a doubtful system or image gets a low prior instead.

Matching.  Positions are matched per (system, knot, image): each paper is first shifted onto the base frame by the
median offset of its matches to the base (measured here), then every pair of papers is matched mutual-nearest within
``match_radius_arcsec``.  Systems linked by a matched position form one group, the same source seen by every paper
that lists it.  A group with a base system enters as the base system (names, knots, positions); a group without one
enters as the systems of the paper with the best redshift (spectroscopic, photometric, model; then sources.yaml
order), renamed with its ``prefix`` and shifted onto the base frame.

Redshifts.  A system takes the base (or source paper) z, replaced by a linked spectroscopic z when its own is not
spectroscopic.  Two spectroscopic z more than ``dz_max`` apart in |dz| / (1 + z) are a conflict until a ``set_z``
decision settles it.  A system whose z is free (Lenstool's z = 0) everywhere keeps z = 0: a fit leaves it out of the
trusted families, and the slot mixture can take its images with their redshift sampled.  Each system with a redshift
gets ``z_err``: 0.01 spectroscopic, 0.05 (1 + z) photometric, 0.1 (1 + z) otherwise.

Reliability.  Each image has a quality q in [0, 1] from its paper's flags: ``quality`` in sources.yaml maps text in
the image notes (e.g. "rank C", "class copper") to a score, first match wins, ``quality_default`` (1) otherwise; an
image several papers list takes the best.  A system's quality is the median of its images'.  With n the number of
independent papers listing the system (``derived_from`` papers count with their source) and the z kind, its tier is

    diamond   spectroscopic z, n >= 2, quality >= high
    gold      spectroscopic z,          quality >= high
    silver    (spectroscopic z or n >= 2), quality >= mid
    bronze    quality >= mid
    copper    otherwise

and its ``reliability`` the tier's value (``reliability:`` in sources.yaml overrides the defaults below).  The
``prior`` of an image (that it belongs to its system) is the system reliability, times the image quality when that is
below high.  Images another paper gives a master system that its source paper does not list are added to it as
candidates, with prior reliability x quality x ``candidate`` (0.5).  Nothing is dropped: in a fit,
``ImageConstraintsConfig.trusted_prior_min`` (``trusted``, 0.8: the diamond and gold images) splits the file into the
trusted families and, with a ``PartitionConfig`` without ``catalog_path``, the uncertain images of the slot mixture.

Members.  ``p_member`` of a member: 1 with a spectroscopic z within ``member_dz`` of ``z_cluster`` (from any paper),
a role bcg, kinematics, or a p_member of 1 in a paper's own list (spectra); ``interloper`` with a spectroscopic z
outside it (kept, not removed); otherwise ``two_papers`` when at least two independent papers list it, else
``one_paper``.  Members only another paper lists enter when that paper gives the base member band (``member_band``
names the band of a Lenstool .cat file); the others are listed in crossmatch.yaml.  A paper may give members only
(no ``images``).

decisions.yaml is a list of entries, each with a ``reason``:

    - {set_z: K14, z: 2.311, reason: ...}                        # a master system
    - {set_tier: B12, tier: copper, reason: ...}                 # a master system
    - {exclude: B12, reason: ...}                                # a master system (kept in crossmatch.yaml)
    - {link: [gledhill2024:7, lagattuta2019:10], reason: ...}    # paper systems that are one source
    - {set_member: B1809, role: bcg, treatment: independent, reason: ...}   # a master member's role, treatment, p_member

Example:
    python scripts/build_master_catalog.py catalogs/m0416/master
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kappaccino.io.image_catalog import write_image_catalog  # noqa: E402
from kappaccino.io.member_catalog import load_member_catalog, write_member_catalog  # noqa: E402
from kappaccino.io.sky import offsets_to_radec, radec_to_offsets_arcsec  # noqa: E402

OFFSET_SEARCH_ARCSEC = 1.0   # radius for the first-pass frame offset, before the offset is applied
TIERS = ("diamond", "gold", "silver", "bronze", "copper")
DEFAULT_RELIABILITY = {"diamond": 1.0, "gold": 0.9, "silver": 0.7, "bronze": 0.4, "copper": 0.15, "high": 0.8, "mid": 0.5,
                       "candidate": 0.5, "trusted": 0.8}
DEFAULT_MEMBER_PRIOR = {"two_papers": 0.9, "one_paper": 0.7, "interloper": 0.02}
Z_RANK = {"spec": 0, "phot": 1, "model": 2, "unknown": 3, "free": 4}


class Paper:
    """One paper folder: its systems as written, and every position flattened to (system, knot, image, ra, dec, notes)."""

    def __init__(self, name: str, spec: dict, root: Path, reference: tuple[float, float]):
        self.name = name
        self.lineage = spec.get("derived_from", name)
        self.prefix = spec.get("prefix", "")
        self.z_kind = spec.get("z_kind", "unknown")
        self.quality_map = list((spec.get("quality") or {}).items())
        self.quality_default = float(spec.get("quality_default", 1.0))
        self.images_path = (root / spec["images"]).resolve() if spec.get("images") else None   # None: a member list only
        self.members_path = (root / spec["members"]).resolve() if spec.get("members") else None
        self.member_band = spec.get("member_band")   # the band of a Lenstool .cat member file, which does not say
        document = yaml.safe_load(self.images_path.read_text(encoding="utf-8")) if self.images_path else {"systems": {}}
        self.defaults = document.get("defaults", {})
        self.systems = OrderedDict((str(k), v) for k, v in document["systems"].items())
        rows = []
        for system, entry in self.systems.items():
            for image, value in entry["images"].items():
                knots = value.get("knots") if isinstance(value, dict) else None
                image_notes = str(value.get("notes", "")) if isinstance(value, dict) else ""
                for knot, position in (knots.items() if knots else [("", value)]):
                    ra, dec = (position["ra"], position["dec"]) if isinstance(position, dict) else position[:2]
                    notes = str(position.get("notes", "")) if isinstance(position, dict) and knot else ""
                    rows.append((system, str(knot), str(image), float(ra), float(dec), "; ".join(filter(None, [image_notes, notes]))))
        self.rows = rows
        self.index = {(r[0], r[1], r[2]): i for i, r in enumerate(rows)}
        self.x, self.y = radec_to_offsets_arcsec(np.array([r[3] for r in rows], dtype=float), np.array([r[4] for r in rows], dtype=float), *reference)
        self.offset = (0.0, 0.0)

    def z_of(self, system: str) -> tuple[float, str]:
        entry = self.systems[system]
        notes = str(entry.get("notes", "")).lower()
        if float(entry["z"]) <= 0.0:
            return float(entry["z"]), "free"   # Lenstool's z = 0: the redshift is a free parameter of the model
        kind = ("spec" if "spectroscopic" in notes else "phot" if "photometric" in notes
                else "model" if "geometric" in notes or "lens-model" in notes else self.z_kind)
        return float(entry["z"]), kind

    def quality(self, i: int) -> float:
        notes = self.rows[i][5]
        for text, score in self.quality_map:
            if text in notes:
                return float(score)
        return self.quality_default

    def label(self, i: int) -> str:
        system, knot, image = self.rows[i][:3]
        return f"{system}{knot}.{image}"

    def shifted(self) -> np.ndarray:
        return np.column_stack([self.x - self.offset[0], self.y - self.offset[1]])


def mutual_nearest(a: np.ndarray, b: np.ndarray, radius: float) -> list[tuple[int, int, float]]:
    if len(a) == 0 or len(b) == 0:
        return []
    da, ia = cKDTree(b).query(a, distance_upper_bound=radius)
    db, ib = cKDTree(a).query(b, distance_upper_bound=radius)
    return [(i, int(j), float(d)) for i, (d, j) in enumerate(zip(da, ia)) if np.isfinite(d) and ib[j] == i]


def measure_offset(paper: Paper, base: Paper, radius: float) -> dict:
    """Median offset of the paper's positions from the base's: a wide first pass, then again within ``radius``."""
    paper.offset = (0.0, 0.0)
    for search in (OFFSET_SEARCH_ARCSEC, radius):
        pairs = mutual_nearest(paper.shifted(), base.shifted(), search)
        if len(pairs) < 5:
            paper.offset = (0.0, 0.0)
            return {"applied_arcsec": [0.0, 0.0], "n_matches": len(pairs), "note": "fewer than 5 matches; no offset applied"}
        i, j = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
        dx = paper.x[i] - base.x[j]; dy = paper.y[i] - base.y[j]
        paper.offset = (float(np.median(dx)), float(np.median(dy)))
    residual = np.hypot(dx - paper.offset[0], dy - paper.offset[1])
    return {"applied_arcsec": [round(paper.offset[0], 4), round(paper.offset[1], 4)], "n_matches": len(pairs),
            "median_residual_arcsec": round(float(np.median(residual)), 4)}


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, k):
        self.parent.setdefault(k, k)
        while self.parent[k] != k:
            self.parent[k] = self.parent[self.parent[k]]
            k = self.parent[k]
        return k

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def tier_of(z_kind: str, n_independent: int, quality: float, scale: dict) -> str:
    if z_kind == "spec" and n_independent >= 2 and quality >= scale["high"]:
        return "diamond"
    if z_kind == "spec" and quality >= scale["high"]:
        return "gold"
    if (z_kind == "spec" or n_independent >= 2) and quality >= scale["mid"]:
        return "silver"
    return "bronze" if quality >= scale["mid"] else "copper"


def build(master_dir: Path) -> None:
    config = yaml.safe_load((master_dir / "sources.yaml").read_text(encoding="utf-8"))
    decisions_path = master_dir / "decisions.yaml"
    decisions = (yaml.safe_load(decisions_path.read_text(encoding="utf-8")) or []) if decisions_path.exists() else []
    radius, dz_max = float(config["match_radius_arcsec"]), float(config["dz_max"])
    scale = {**DEFAULT_RELIABILITY, **(config.get("reliability") or {})}
    names = list(config["papers"])
    base_name = config["base"]
    first = yaml.safe_load((master_dir / config["papers"][base_name]["images"]).read_text(encoding="utf-8"))["systems"]
    first_image = next(iter(next(iter(first.values()))["images"].values()))
    first_position = next(iter(first_image["knots"].values())) if "knots" in first_image else first_image
    reference = tuple(first_position[:2]) if isinstance(first_position, list) else (first_position["ra"], first_position["dec"])
    papers = {n: Paper(n, config["papers"][n], master_dir, reference) for n in names}
    base = papers[base_name]

    frames = {base_name: {"applied_arcsec": [0.0, 0.0], "note": "base frame"}}
    for name in names:
        if name != base_name and papers[name].rows:
            frames[name] = measure_offset(papers[name], base, radius)

    # positions matched across every pair of papers, then systems linked through them
    positions, systems = UnionFind(), UnionFind()
    for name, paper in papers.items():
        for i, row in enumerate(paper.rows):
            positions.find((name, i)); systems.find((name, row[0]))
    for a_i, a in enumerate(names):
        for b in names[a_i + 1:]:
            for i, j, _ in mutual_nearest(papers[a].shifted(), papers[b].shifted(), radius):
                positions.union((a, i), (b, j))
                systems.union((a, papers[a].rows[i][0]), (b, papers[b].rows[j][0]))
    for d in decisions:
        if "link" in d:   # systems the same source although no image matched within the radius
            first, *rest = [tuple(key.split(":", 1)) for key in d["link"]]
            for other in rest:
                for paper, system in (first, other):
                    if paper not in papers or system not in papers[paper].systems:
                        raise SystemExit(f"decisions.yaml: link {paper}:{system} is not a paper system.")
                systems.union(first, other)
    groups: dict = defaultdict(list)
    for name, paper in papers.items():
        for system in paper.systems:
            groups[systems.find((name, system))].append((name, system))
    image_groups: dict = defaultdict(list)
    for name, paper in papers.items():
        for i in range(len(paper.rows)):
            image_groups[positions.find((name, i))].append((name, i))

    def image_quality(paper: str, i: int) -> tuple[float, list[str]]:
        """Best quality over the papers listing this position, and those papers' labels."""
        items = image_groups[positions.find((paper, i))]
        return max(papers[n].quality(k) for n, k in items), sorted(f"{n}:{papers[n].label(k)}" for n, k in items)

    exclude = {d["exclude"]: d["reason"] for d in decisions if "exclude" in d}
    set_z = {d["set_z"]: (float(d["z"]), d["reason"]) for d in decisions if "set_z" in d}
    set_tier = {d["set_tier"]: (d["tier"], d["reason"]) for d in decisions if "set_tier" in d}
    master: OrderedDict = OrderedDict()      # name -> {entry, paper, system, kind, record}
    conflicts, free_z = [], []

    def group_record(members):
        by_paper = defaultdict(list)
        for name, system in members:
            by_paper[name].append(system)
        z = {f"{name}:{system}": list(papers[name].z_of(system)) for name, system in members}
        return {"papers": {n: sorted(by_paper[n]) for n in names if n in by_paper}, "z": z,
                "independent_papers": len({papers[n].lineage for n in by_paper})}

    def choose_z(z_now, kind_now, members, where):
        best, best_kind, note = z_now, kind_now, ""
        specs = [(papers[n].z_of(s)[0], f"{n}:{s}") for n, s in members if papers[n].z_of(s)[1] == "spec"]
        if kind_now != "spec" and specs:
            best, best_kind = specs[0][0], "spec"
            note = f"z from {specs[0][1]} (spectroscopic)"
        for value, who in specs:
            if abs(value - best) / (1 + best) > dz_max:
                conflicts.append({"system": where, "kind": "z", "values": {who: value, "master": best}})
        return best, best_kind, note

    order = sorted(groups.values(), key=lambda m: (not any(n == base_name for n, _ in m), min(names.index(n) for n, _ in m)))
    for members in order:
        rec = group_record(members)
        in_base = [s for n, s in members if n == base_name]
        if in_base:
            source = base_name
        else:   # the paper with the best redshift for this source
            source = min((n for n in names if n in rec["papers"]),
                         key=lambda n: (min(Z_RANK.get(papers[n].z_of(s)[1], 3) for s in rec["papers"][n]), names.index(n)))
        paper = papers[source]
        for system in rec["papers"][source]:
            name = system if source == base_name else f"{paper.prefix}{system}"
            if source != base_name and (name in master or name in base.systems):
                raise SystemExit(f"{source}:{system} renamed {name} collides with an existing system; change its prefix.")
            entry = dict(paper.systems[system]) if source == base_name else shift_system(paper.systems[system], paper.offset, reference)
            z, kind = paper.z_of(system)
            z, kind, note = choose_z(z, kind, members, name)
            if name in set_z:
                z, kind, note = set_z[name][0], "decision", f"z set by decision: {set_z[name][1]}"
                conflicts[:] = [c for c in conflicts if c["system"] != name]
            entry["z"] = z
            entry["notes"] = "; ".join(filter(None, [f"from {source} system {system}" if source != base_name else "",
                                                     str(entry.get("notes", "")), note]))
            master[name] = {"entry": entry, "paper": source, "system": system, "kind": kind, "record": rec}

    # tiers and association priors
    systems_out, crossmatch_systems = OrderedDict(), OrderedDict()
    tier_counts = {t: 0 for t in TIERS}
    where = {}   # (paper, row) of every master position -> (master system, knot)
    for name, item in master.items():
        paper, entry = papers[item["paper"]], item["entry"]
        rows = [(image, knot, paper.index[(item["system"], knot, image)])
                for image, value in entry["images"].items() for knot in ((value.get("knots") if isinstance(value, dict) else None) or {"": None})]
        qualities = {(image, knot): image_quality(paper.name, i)[0] for image, knot, i in rows}
        quality = float(np.median(list(qualities.values())))
        kind = "spec" if item["kind"] in ("spec", "decision") else item["kind"]
        tier = set_tier[name][0] if name in set_tier else tier_of(kind, item["record"]["independent_papers"], quality, scale)
        reliability = float(scale[tier])
        crossmatch_systems[name] = {"from": item["paper"], "tier": tier, "reliability": reliability, "quality": round(quality, 3),
                                    "z_kind": item["kind"], **item["record"]}
        for image, knot, i in rows:
            for n, k in image_groups[positions.find((paper.name, i))]:
                where[(n, k)] = (name, knot)
        if name in exclude:
            continue
        if item["kind"] == "free":
            free_z.append(name)
        tier_counts[tier] += 1
        priors = {key: association_prior(reliability, q, scale) for key, q in qualities.items()}
        entry = with_priors(entry, priors)
        entry["reliability"] = reliability
        if item["kind"] != "free":
            entry["z_err"] = z_error(entry["z"], kind)
        entry["notes"] = "; ".join(filter(None, [f"tier {tier}", str(entry.get("notes", ""))]))
        systems_out[name] = entry

    # images another paper gives a master system that its source paper does not list: low-prior candidates in it
    candidates = []
    for key, items in image_groups.items():
        if any(item in where for item in items):
            continue
        n, i = items[0]
        system, knot, image = papers[n].rows[i][:3]
        claimed = [where[(n, k)] for k, row in enumerate(papers[n].rows) if row[0] == system and (n, k) in where]
        if not claimed:
            continue
        name = max(set(s for s, _ in claimed), key=[s for s, _ in claimed].count)
        if name not in systems_out:
            continue
        knots = [kn for s, kn in claimed if s == name]
        same_knot = [kn for (s, kn), row in zip(claimed, [r for k, r in enumerate(papers[n].rows) if r[0] == system and (n, k) in where])
                     if s == name and row[1] == knot]
        master_knot = max(set(same_knot or knots), key=(same_knot or knots).count)
        q, labels = image_quality(n, i)
        prior = round(crossmatch_systems[name]["reliability"] * q * scale["candidate"], 4)
        ra, dec = shift_point(papers[n].rows[i][3], papers[n].rows[i][4], papers[n].offset, reference)
        label = add_candidate(systems_out[name], master_knot, ra, dec, prior, f"candidate: {' '.join(labels)}")
        candidates.append({"image": f"{name}{master_knot}.{label}", "papers": labels, "prior": prior})

    systems_out = OrderedDict((name, order_keys(entry)) for name, entry in systems_out.items())
    defaults = base.defaults or None
    n_images = sum(len(s["images"]) for s in systems_out.values())
    counts = ", ".join(f"{t} {tier_counts[t]}" for t in TIERS)
    write_image_catalog(master_dir / "images.yaml", systems_out, defaults=defaults, header=(
        f"Master image catalog of {config['cluster']}, built on {date.today().isoformat()} by scripts/build_master_catalog.py "
        f"from {', '.join(names)} (base {base_name}).\n"
        "Do not edit: change sources.yaml or decisions.yaml and rebuild.  crossmatch.yaml gives each system's papers and tier.\n"
        f"{len(systems_out)} systems ({counts}; {len(free_z)} with z = 0, free), {n_images} images, {len(candidates)} of them "
        "candidates from other papers.\n"
        "reliability: the tier's prior.  prior (per image or knot): the prior that it belongs to its system, the reliability "
        f"(times its quality below {scale['high']}).\n"
        f"ImageConstraintsConfig(trusted_prior_min={scale['trusted']}) trusts the diamond and gold images; with a PartitionConfig "
        "the rest enter the slot mixture, each weighted toward its system by its prior."))

    members_record = write_members(master_dir, config, papers, papers[config.get("members_base", base_name)], reference, radius, decisions)
    crossmatch = OrderedDict(
        cluster=config["cluster"], built=date.today().isoformat(), base=base_name, match_radius_arcsec=radius, dz_max=dz_max,
        reliability_scale=scale, frames=frames,
        counts={"master_systems": len(systems_out), **{t: tier_counts[t] for t in TIERS}, "images": n_images,
                "candidate_images": len(candidates), "free_z_systems": len(free_z), "conflicts": len(conflicts)},
        systems=crossmatch_systems, excluded=exclude, conflicts=conflicts, free_z=free_z, candidate_images=candidates,
        members=members_record)
    text = yaml.safe_dump(to_plain(crossmatch), sort_keys=False, width=160, default_flow_style=None)
    (master_dir / "crossmatch.yaml").write_text(
        f"# Built on {date.today().isoformat()} by scripts/build_master_catalog.py; do not edit.\n" + text, encoding="utf-8")
    print(f"{config['cluster']}: {len(systems_out)} systems ({counts}), {n_images} images ({len(candidates)} candidates), "
          f"{len(free_z)} free-z systems, {len(conflicts)} conflicts; members p_member {members_record.get('p_member', {})}")


def shift_point(ra: float, dec: float, offset: tuple[float, float], reference) -> tuple[float, float]:
    x, y = radec_to_offsets_arcsec(np.array([ra]), np.array([dec]), *reference)
    r, d = offsets_to_radec(x - offset[0], y - offset[1], *reference)
    return round(float(r[0]), 8), round(float(d[0]), 8)


def shift_system(system: dict, offset: tuple[float, float], reference) -> dict:
    """A copy of ``system`` with every position moved by -offset (arcsec)."""
    def move(position):
        if isinstance(position, list):
            return [*shift_point(position[0], position[1], offset, reference), *position[2:]]
        out = dict(position)
        out["ra"], out["dec"] = shift_point(position["ra"], position["dec"], offset, reference)
        return out

    out = {k: v for k, v in system.items() if k != "images"}
    out["images"] = OrderedDict(
        (name, {**image, "knots": OrderedDict((k, move(v)) for k, v in image["knots"].items())}
         if isinstance(image, dict) and "knots" in image else move(image))
        for name, image in system["images"].items())
    return out


def association_prior(reliability: float, quality: float, scale: dict) -> float:
    """The prior that an image belongs to its system: the system's reliability, times the image's quality below high."""
    return round(reliability * (1.0 if quality >= scale["high"] else quality), 4)


def z_error(z: float, kind: str) -> float:
    return 0.01 if kind == "spec" else round(0.05 * (1 + z), 3) if kind == "phot" else round(0.1 * (1 + z), 3)


def with_priors(entry: dict, priors: dict) -> dict:
    """``entry`` with a ``prior`` on every image (or knot) whose prior is below 1."""
    images = OrderedDict()
    for name, image in entry["images"].items():
        if isinstance(image, dict) and "knots" in image:
            knots = OrderedDict()
            for k, v in image["knots"].items():
                prior = priors[(name, k)]
                v = dict(zip(("ra", "dec", "sigma"), v)) if isinstance(v, list) else dict(v)
                knots[k] = v if prior >= 1.0 else {**v, "prior": prior}
            images[name] = {**image, "knots": knots}
        else:
            prior = priors[(name, "")]
            image = dict(zip(("ra", "dec", "sigma"), image)) if isinstance(image, list) else dict(image)
            images[name] = image if prior >= 1.0 else {**image, "prior": prior}
    return dict(entry, images=images)


def add_candidate(entry: dict, knot: str, ra: float, dec: float, prior: float, notes: str) -> str:
    """Add a candidate image to ``entry`` under the next free image letter (in ``knot`` for a system with knots)."""
    used = set(entry["images"])
    letter = next(chr(c) for c in range(ord("a"), ord("z") + 1) if chr(c) not in used)
    position = {"ra": ra, "dec": dec, "prior": prior, "notes": notes}
    entry["images"][letter] = {"knots": {knot: position}} if knot else position
    return letter


def order_keys(entry: dict) -> OrderedDict:
    keys = [k for k in ("z", "z_err", "reliability", "knots", "notes") if k in entry]
    return OrderedDict([(k, entry[k]) for k in keys] + [("images", entry["images"])])


def write_members(master_dir: Path, config: dict, papers: dict, base: Paper, reference, radius: float, decisions: list) -> dict:
    """The base members plus the other papers' members in the same band, each with p_member (module docstring)."""
    if base.members_path is None:
        return {}
    if base.members_path.suffix.lower() not in {".yaml", ".yml"}:
        raise SystemExit(f"{base.members_path}: the base member catalog must be YAML (scripts/convert_member_catalog.py).")
    prior = {**DEFAULT_MEMBER_PRIOR, **(config.get("member_prior") or {})}
    z_cluster, member_dz = float(config["z_cluster"]), float(config.get("member_dz", 0.03))
    document = yaml.safe_load(base.members_path.read_text(encoding="utf-8"))
    band = document["band"]
    tables = {n: load_member_catalog(p.members_path, (3, *reference)) for n, p in papers.items() if p.members_path}
    for n, table in tables.items():
        if papers[n].member_band and f"mag_{papers[n].member_band}" not in table:
            table[f"mag_{papers[n].member_band}"] = table["catalog_mag"]
    xy = {n: np.column_stack(radec_to_offsets_arcsec(t["ra"].to_numpy(), t["dec"].to_numpy(), *reference)) - np.array(papers[n].offset)
          for n, t in tables.items()}
    # every galaxy of every paper, grouped by mutual-nearest matches to the base and then to each other
    galaxies = UnionFind()
    names = list(tables)
    for a_i, a in enumerate(names):
        for i in range(len(tables[a])):
            galaxies.find((a, i))
        for b in names[a_i + 1:]:
            for i, j, _ in mutual_nearest(xy[a], xy[b], radius):
                galaxies.union((a, i), (b, j))
    found: dict = defaultdict(list)
    for name in names:
        for i in range(len(tables[name])):
            found[galaxies.find((name, i))].append((name, i))

    def evidence(group):
        """(p_member, spectroscopic z or None, the papers listing it, a note)."""
        listing = sorted({n for n, _ in group})
        spec = [float(tables[n]["catalog_z"].iloc[i]) for n, i in group
                if tables[n]["catalog_z_kind"].iloc[i] == "spec" and np.isfinite(tables[n]["catalog_z"].iloc[i])]
        certain = any(tables[n]["catalog_role"].iloc[i] == "bcg" or len(tables[n]["catalog_kinematics"].iloc[i]) for n, i in group)
        stated = [float(tables[n]["catalog_p_member"].iloc[i]) for n, i in group if np.isfinite(tables[n]["catalog_p_member"].iloc[i])]
        if stated and max(stated) >= 1.0 and not spec:
            return 1.0, None, listing, "a paper states it (spectra)"
        if spec:
            z = spec[0]
            inside = abs(z - z_cluster) <= member_dz * (1 + z_cluster)
            return (1.0 if inside else prior["interloper"]), z, listing, ("spectroscopic member" if inside else "spectroscopic interloper")
        if certain:
            return 1.0, None, listing, "bcg or kinematics"
        n = len({papers[p].lineage for p in listing})
        return (prior["two_papers"] if n >= 2 else prior["one_paper"]), None, listing, f"{n} independent paper{'s' * (n > 1)}"

    members, record = OrderedDict(), {"counts": {n: len(t) for n, t in tables.items()}, "other_band": {}}
    base_ids = [str(v) for v in tables[base.name]["id"]]
    for i, (name, entry) in enumerate(document["members"].items()):
        p, z, listing, why = evidence(found[galaxies.find((base.name, i))])
        entry = dict(entry, p_member=round(p, 4))
        if z is not None and "z" not in entry:
            entry["z"] = {"value": round(z, 5), "kind": "spec"}
        entry["notes"] = "; ".join(filter(None, [str(entry.get("notes", "")), f"in {', '.join(listing)}; {why}"]))
        members[str(name)] = entry
    assert base_ids == list(map(str, document["members"]))
    for key, group in found.items():
        if any(n == base.name for n, _ in group):
            continue
        name, i = group[0]
        row = tables[name].iloc[i]
        if f"mag_{band}" not in tables[name] or not np.isfinite(row.get(f"mag_{band}", np.nan)):
            record["other_band"].setdefault(name, []).append(str(row["id"]))
            continue
        p, z, listing, why = evidence(group)
        ra, dec = shift_point(float(row["ra"]), float(row["dec"]), papers[name].offset, reference)
        entry = OrderedDict(ra=ra, dec=dec, mag={band: float(row[f"mag_{band}"])}, p_member=round(p, 4))
        if z is not None:
            entry["z"] = {"value": round(z, 5), "kind": "spec"}
        entry["notes"] = f"from {name} member {row['id']}; in {', '.join(listing)}; {why}"
        members[f"{papers[name].prefix or name}{row['id']}"] = entry
    for d in decisions:
        if "set_member" in d:
            if d["set_member"] not in members:
                raise SystemExit(f"decisions.yaml: set_member {d['set_member']} is not a master member.")
            fields = {k: v for k, v in d.items() if k in ("role", "treatment", "p_member", "keep")}
            members[d["set_member"]].update(fields)
            members[d["set_member"]]["notes"] += f"; set by decision: {d['reason']}"
            if fields.get("role") == "bcg" or "kinematics" in members[d["set_member"]]:
                members[d["set_member"]]["p_member"] = 1.0
    values = [m["p_member"] for m in members.values()]
    record["p_member"] = {f"{v:g}": values.count(v) for v in sorted(set(values), reverse=True)}
    header = (f"Master member catalog of {config['cluster']}, built on {date.today().isoformat()} by scripts/build_master_catalog.py: "
              f"the {base.name} members plus the other papers' members with a {band} magnitude.\n"
              f"p_member: 1 with a spectroscopic z within {member_dz} (1 + z) of z = {z_cluster}, a bcg role or kinematics; "
              f"{prior['interloper']} for a spectroscopic interloper; {prior['two_papers']} when two independent papers list it, "
              f"else {prior['one_paper']}.  Use with MemberConfig.membership = 'marginalize'.  Do not edit.")
    write_member_catalog(master_dir / "members.yaml", members, band=band, kinematics_sources=document.get("kinematics_sources"), header=header)
    return record


def to_plain(value):
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("master_dir", type=Path, help="catalogs/<cluster>/master, holding sources.yaml")
    build(parser.parse_args(argv).master_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
