"""Build a cluster's master catalog (catalogs/<cluster>/master/) from its per-paper folders.

Reads ``master/sources.yaml`` (the papers, which is the base, the match radius and which papers may add systems) and
``master/decisions.yaml`` (hand-made choices), and writes

    images.yaml        the base paper's systems, plus systems from other papers the policy or a decision adds
    images_gold.yaml   the systems of images.yaml found in at least two independent papers (a paper with
                       ``derived_from: <paper>`` reused that paper's images and does not count separately)
    members.yaml       the members of ``members_base`` (default the base paper), each noted with the papers that list it
    crossmatch.yaml    every paper's systems and images matched to each other, the frame offsets, the conflicts

Positions are matched per (system, knot, image): each paper is first shifted onto the base frame by the median offset
of its matches to the base (measured here, not assumed), then every pair of papers is matched mutual-nearest within
``match_radius_arcsec``.  Systems linked by a matched position form one group; a group holds the systems of every paper
that saw that source (one paper may split it into several systems, e.g. knots counted as separate sources).

Policy.  A group with a base system enters with the base system unchanged: names, knots, positions, z.  If the base z
is not spectroscopic and a linked system's is, the spectroscopic z replaces it.  Two spectroscopic z that disagree by
more than ``dz_max`` in |dz| / (1 + z) are a conflict: the base z stays and the conflict is listed until a decision
settles it.  A base system whose z is free (Lenstool's z = 0) and gets no spectroscopic z from a linked system
is left out and listed under ``free_z`` (a ``set_z`` decision brings it in).  A group without a base system enters when a paper listing it has ``add_systems: spec`` (and gives a
spectroscopic z) or ``add_systems: all``, taking that paper's systems renamed with its ``prefix`` and shifted onto the
base frame; the first such paper in sources.yaml order wins.  Images another paper lists for a base system but the
base does not are candidates in crossmatch.yaml, not added.

decisions.yaml is a list of entries, each with a ``reason``:

    - {set_z: K14, z: 2.311, reason: ...}          # master system name
    - {exclude: B12, reason: ...}                   # a master system
    - {include: diego2024:45, reason: ...}          # a group, by any of its paper systems, added despite the policy

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
from kappaccino.image_catalog import write_image_catalog  # noqa: E402
from kappaccino.member_catalog import load_member_catalog, write_member_catalog  # noqa: E402
from kappaccino.sky import offsets_to_radec, radec_to_offsets_arcsec  # noqa: E402

OFFSET_SEARCH_ARCSEC = 1.0   # radius for the first-pass frame offset, before the offset is applied


class Paper:
    """One paper folder: its systems as written, and every position flattened to (system, knot, image, x, y)."""

    def __init__(self, name: str, spec: dict, root: Path, reference: tuple[float, float]):
        self.name = name
        self.lineage = spec.get("derived_from", name)
        self.prefix = spec.get("prefix", "")
        self.add_systems = spec.get("add_systems", "none")
        self.z_kind = spec.get("z_kind", "unknown")
        self.drop_notes = set(spec.get("drop_images_with_notes", []))
        self.images_path = (root / spec["images"]).resolve()
        self.members_path = (root / spec["members"]).resolve() if spec.get("members") else None
        document = yaml.safe_load(self.images_path.read_text(encoding="utf-8"))
        self.defaults = document.get("defaults", {})
        self.systems = OrderedDict((str(k), v) for k, v in document["systems"].items())
        rows = []
        for system, entry in self.systems.items():
            for image, value in entry["images"].items():
                if isinstance(value, dict) and value.get("notes") in self.drop_notes:
                    continue
                knots = value.get("knots") if isinstance(value, dict) else None
                for knot, position in (knots.items() if knots else [("", value)]):
                    ra, dec = (position["ra"], position["dec"]) if isinstance(position, dict) else position[:2]
                    rows.append((system, str(knot), str(image), float(ra), float(dec)))
        self.rows = rows
        ra = np.array([r[3] for r in rows]); dec = np.array([r[4] for r in rows])
        self.x, self.y = radec_to_offsets_arcsec(ra, dec, *reference)
        self.offset = (0.0, 0.0)

    def z_of(self, system: str) -> tuple[float, str]:
        entry = self.systems[system]
        notes = str(entry.get("notes", "")).lower()
        if float(entry["z"]) <= 0.0:
            return float(entry["z"]), "free"   # Lenstool's z = 0: the redshift is a free parameter of the model
        kind = ("spec" if "spectroscopic" in notes else "phot" if "photometric" in notes
                else "model" if "geometric" in notes or "lens-model" in notes else self.z_kind)
        return float(entry["z"]), kind

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


def build(master_dir: Path) -> None:
    config = yaml.safe_load((master_dir / "sources.yaml").read_text(encoding="utf-8"))
    decisions_path = master_dir / "decisions.yaml"
    decisions = (yaml.safe_load(decisions_path.read_text(encoding="utf-8")) or []) if decisions_path.exists() else []
    radius, dz_max = float(config["match_radius_arcsec"]), float(config["dz_max"])
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
        if name != base_name:
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
    groups: dict = defaultdict(list)
    for name, paper in papers.items():
        for system in paper.systems:
            groups[systems.find((name, system))].append((name, system))
    image_groups: dict = defaultdict(list)
    for name, paper in papers.items():
        for i in range(len(paper.rows)):
            image_groups[positions.find((name, i))].append((name, i))

    include = {d["include"] for d in decisions if "include" in d}
    exclude = {d["exclude"]: d["reason"] for d in decisions if "exclude" in d}
    set_z = {d["set_z"]: (float(d["z"]), d["reason"]) for d in decisions if "set_z" in d}
    master: OrderedDict = OrderedDict()
    record: OrderedDict = OrderedDict()
    conflicts, candidates, left_out, free_z = [], [], [], []

    def group_record(members):
        by_paper = defaultdict(list)
        for name, system in members:
            by_paper[name].append(system)
        z = {f"{name}:{system}": list(papers[name].z_of(system)) for name, system in members}
        return {"papers": {n: sorted(by_paper[n]) for n in names if n in by_paper}, "z": z}

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

    order = sorted(groups.values(), key=lambda m: (not any(n == base_name for n, _ in m),
                                                    min(names.index(n) for n, _ in m)))
    for members in order:
        in_base = [s for n, s in members if n == base_name]
        rec = group_record(members)
        n_papers = len({papers[n].lineage for n in rec["papers"]})
        if in_base:
            for system in in_base:
                entry = dict(base.systems[system])
                z, kind = base.z_of(system)
                z, kind, note = choose_z(z, kind, members, system)
                if kind == "free" and system not in set_z:
                    free_z.append({"system": system, **rec})   # a fit needs a z: a set_z decision brings it in
                    continue
                entry["z"] = z
                if note:
                    entry["notes"] = "; ".join(filter(None, [str(entry.get("notes", "")), note]))
                master[system] = (entry, n_papers, base_name)
                record[system] = rec
            continue
        keys = {f"{n}:{s}" for n, s in members}
        source = None
        for name in names:
            paper_systems = [s for n, s in members if n == name]
            if not paper_systems:
                continue
            kinds = {papers[name].z_of(s)[1] for s in paper_systems}
            if kinds == {"free"}:
                continue   # no redshift to give the system
            allowed = papers[name].add_systems == "all" or (papers[name].add_systems == "spec" and "spec" in kinds)
            if allowed or keys & include:
                source = name
                break
        if source is None:
            left_out.append(rec)
            continue
        paper = papers[source]
        for system in [s for n, s in members if n == source]:
            name = f"{paper.prefix}{system}"
            if name in master or name in base.systems:
                raise SystemExit(f"{source}:{system} renamed {name} collides with an existing system; change its prefix.")
            entry = shift_system(paper.systems[system], paper.offset, reference, paper.drop_notes)
            if not entry["images"]:
                continue
            z, kind = paper.z_of(system)
            z, kind, note = choose_z(z, kind, members, name)
            entry["z"] = z
            entry["notes"] = "; ".join(filter(None, [f"from {source} system {system}", str(entry.get("notes", "")), note]))
            master[name] = (entry, n_papers, source)
            record[name] = rec

    # base systems: images other papers list in the same group that match no base position
    base_groups = {positions.find((base_name, i)) for i in range(len(base.rows))}
    for key, items in image_groups.items():
        if key in base_groups:
            continue
        for name, i in items:
            system = papers[name].rows[i][0]
            group = systems.find((name, system))
            linked = [s for n, s in groups[group] if n == base_name]
            if linked:
                candidates.append({"paper_image": f"{name}:{papers[name].label(i)}", "base_systems": linked,
                                   "also_in": sorted({f"{n}:{papers[n].label(k)}" for n, k in items if n != name})})

    # decisions, then write
    for name, (z, reason) in set_z.items():
        if name not in master:
            raise SystemExit(f"decisions.yaml: set_z {name} is not a master system.")
        entry = master[name][0]
        entry["z"] = z
        entry["notes"] = "; ".join(filter(None, [str(entry.get("notes", "")), f"z set by decision: {reason}"]))
        conflicts[:] = [c for c in conflicts if c["system"] != name]
    for name in exclude:
        if name not in master:
            raise SystemExit(f"decisions.yaml: exclude {name} is not a master system.")
        del master[name]
    systems_out = OrderedDict((name, order_keys(entry)) for name, (entry, _, _) in master.items())
    gold = OrderedDict((name, systems_out[name]) for name, (_, n, _) in master.items() if n >= 2)
    defaults = base.defaults or None
    n_images = sum(len(s["images"]) for s in systems_out.values())
    header = (f"Master image catalog of {config['cluster']}, built on {date.today().isoformat()} by scripts/build_master_catalog.py "
              f"from {', '.join(names)} (base {base_name}).\n"
              "Do not edit: change sources.yaml or decisions.yaml and rebuild.  crossmatch.yaml gives each system's papers.")
    write_image_catalog(master_dir / "images.yaml", systems_out, defaults=defaults,
                        header=header + f"\n{len(systems_out)} systems, {n_images} images.")
    write_image_catalog(master_dir / "images_gold.yaml", gold, defaults=defaults,
                        header=header + f"\nGold: the {len(gold)} systems found in at least two independent papers.")

    members_record = write_members(master_dir, config, papers, papers[config.get("members_base", base_name)], reference, radius)
    crossmatch = OrderedDict(
        cluster=config["cluster"], built=date.today().isoformat(), base=base_name, match_radius_arcsec=radius, dz_max=dz_max,
        frames=frames,
        counts={"master_systems": len(systems_out), "gold_systems": len(gold), "images": n_images,
                "left_out_groups": len(left_out), "free_z_systems": len(free_z), "candidate_images": len(candidates),
                "conflicts": len(conflicts)},
        systems=OrderedDict((name, {"from": master[name][2], **record[name]}) for name in systems_out),
        excluded={name: reason for name, reason in exclude.items()},
        conflicts=conflicts, candidate_images=candidates, left_out=left_out, free_z=free_z, members=members_record)
    text = yaml.safe_dump(to_plain(crossmatch), sort_keys=False, width=160, default_flow_style=None)
    (master_dir / "crossmatch.yaml").write_text(
        f"# Built on {date.today().isoformat()} by scripts/build_master_catalog.py; do not edit.\n" + text, encoding="utf-8")
    print(f"{config['cluster']}: {len(systems_out)} systems ({len(gold)} gold), {n_images} images; "
          f"{len(left_out)} groups left out, {len(free_z)} free-z base systems, {len(candidates)} candidate images, "
          f"{len(conflicts)} conflicts")
    for name, frame in frames.items():
        print(f"  frame {name}: {frame}")


def shift_system(system: dict, offset: tuple[float, float], reference, drop_notes: set) -> dict:
    """A copy of ``system`` with every position moved by -offset (arcsec) and images with dropped notes removed."""
    def move(position):
        if isinstance(position, list):
            ra, dec = shift(position[0], position[1])
            return [ra, dec, *position[2:]]
        out = dict(position)
        out["ra"], out["dec"] = shift(position["ra"], position["dec"])
        return out

    def shift(ra, dec):
        x, y = radec_to_offsets_arcsec(np.array([ra]), np.array([dec]), *reference)
        r, d = offsets_to_radec(x - offset[0], y - offset[1], *reference)
        return round(float(r[0]), 8), round(float(d[0]), 8)

    out = {k: v for k, v in system.items() if k != "images"}
    images = OrderedDict()
    for name, image in system["images"].items():
        if isinstance(image, dict) and image.get("notes") in drop_notes:
            continue
        if isinstance(image, dict) and "knots" in image:
            images[name] = {**image, "knots": OrderedDict((k, move(v)) for k, v in image["knots"].items())}
        else:
            images[name] = move(image)
    used = {k for image in images.values() if isinstance(image, dict) and "knots" in image for k in image["knots"]}
    if "knots" in out:
        out["knots"] = [k for k in out["knots"] if k in used]
    out["images"] = images
    return out


def order_keys(entry: dict) -> OrderedDict:
    keys = [k for k in ("z", "reliability", "knots", "notes") if k in entry]
    return OrderedDict([(k, entry[k]) for k in keys] + [("images", entry["images"])])


def write_members(master_dir: Path, config: dict, papers: dict, base: Paper, reference, radius: float) -> dict:
    """The base members, each noted with the other papers that list a galaxy within ``radius``; the rest listed."""
    if base.members_path is None:
        return {}
    if base.members_path.suffix.lower() not in {".yaml", ".yml"}:
        raise SystemExit(f"{base.members_path}: the base member catalog must be YAML (scripts/convert_member_catalog.py).")
    document = yaml.safe_load(base.members_path.read_text(encoding="utf-8"))
    names = list(document["members"])
    tables = {n: load_member_catalog(p.members_path, (3, *reference)) for n, p in papers.items() if p.members_path}
    xy = {n: np.column_stack(radec_to_offsets_arcsec(t["ra"].to_numpy(), t["dec"].to_numpy(), *reference)) for n, t in tables.items()}
    seen = {name: [base.name] for name in names}
    record = {"counts": {base.name: len(names)}, "only_in": {}}
    for name, table in tables.items():
        if name == base.name:
            continue
        shifted = xy[name] - np.array(papers[name].offset)
        pairs = mutual_nearest(shifted, xy[base.name], radius)
        matched = {i for i, _, _ in pairs}
        for _, j, _ in pairs:
            seen[str(tables[base.name]["id"].iloc[j])].append(name)
        record["counts"][name] = len(table)
        record["only_in"][name] = sorted(str(table["id"].iloc[i]) for i in range(len(table)) if i not in matched)
    members = OrderedDict()
    for name, entry in document["members"].items():
        entry = dict(entry)
        note = f"in {', '.join(seen[str(name)])}"
        entry["notes"] = "; ".join(filter(None, [str(entry.get("notes", "")), note]))
        members[str(name)] = entry
    header = (f"Master member catalog of {config['cluster']}: the {base.name} members ({base.members_path.name}), built on "
              f"{date.today().isoformat()} by scripts/build_master_catalog.py.\nEach member's notes name the papers that list it; "
              "members only other papers list are in crossmatch.yaml.  Do not edit.")
    write_member_catalog(master_dir / "members.yaml", members, band=document["band"],
                         kinematics_sources=document.get("kinematics_sources"), header=header)
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
