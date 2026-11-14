"""Convert the CANUCS MACS J0416 multiple-image catalogue (Rihtarsic et al. 2025) into the YAML image catalog, with knots.

Structure comes from the published labels ``<system><knot>.<image>`` (K6c.2 = system K6, knot c, image 2; systems without a
knot letter have one knot).  Images are kept by class (default gold).  Positions and redshifts are taken from the CANUCS
lenstool input (``...lenstool-multim...``), matched to the catalogue by position: they are the catalogue positions shifted by
a constant offset (about 20 mas), the frame of the CANUCS lens model and of its member catalogue, and the redshifts carry
the spectroscopic precision (0.9397, not 0.940).  The match must be one to one with a constant offset and identical system
number, knot and image, or the script stops.

Image numbers are reconciled across the knots of a system by position: each knot's images are assigned to the images of
the knot with the most images (Hungarian matching, within --max-image-offset arcsec).  A knot whose numbering disagrees
(K6e: 1 and 3 swapped) is relabelled and the published label kept in the entry's notes.  Image n becomes letter n (1 -> a).

Two edits follow the conversion, both listed below so this script alone rebuilds the catalog:
* PARITIES (2026-11-03): 17 images near critical curves where the lens model and the lens+field mean agree, or the knot
  pattern decides (K1: mirrored knots); the rest stay 0 (unknown).
* MERGES (2026-11-12): systems that are one galaxy (z equal within the spectroscopic errors, same knot pattern in every
  image) become one system with knots.  The first name is kept, knots are lettered A, B, ... over the systems in order (a
  system's own knots keep their order), images of the others take the first system's letter by position (one to one,
  within 4"), and every relabelled image notes its previous label.

Example:
    python scripts/convert_canucs_catalog.py catalogs/m0416/canucs2025/images.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kappaccino.io.image_catalog import load_image_catalog, write_image_catalog  # noqa: E402
from kappaccino.io.sky import radec_to_offsets_arcsec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "catalogs/m0416/canucs2025"
CATALOG = ROOT / "source" / "hlsp_canucs_jwst-hst_multi_macs0416-allmultim-cat_multi_v1_model.txt"
LENSTOOL = ROOT / "source" / "hlsp_canucs_jwst-hst_multi_macs0416-lenstool-multim_multi_v1_model.txt"
# Image parities by converted label (system + image letter); every knot of the image has it.
PARITIES = {"K1.a": 1, "K1.b": -1, "K12.a": 1, "K12.b": -1, "K19.a": 1, "K19.b": -1, "K28.a": 1, "K28.b": -1,
            "K33.c": -1, "K33.d": 1, "K33.e": 1, "K33.f": -1, "K73.a": -1, "K73.b": 1, "K77.c": -1, "K82.a": 1, "K82.b": -1}
# One galaxy each: (systems, first name kept; merged z, the spectroscopic value with the most digits).
MERGE_DATE = "2026-11-12"
MERGES = [(("K33", "K77"), 3.2215), (("K10", "K76"), 1.9894), (("K89", "K90", "K91"), 3.0665), (("K36", "K37"), 3.2882),
          (("K69", "K70"), 6.1452), (("K6", "K7"), 1.895), (("K29", "C2"), 3.0773)]
MERGE_MAX_ARCSEC = 4.0
LABEL = re.compile(r"^([A-Z]+[0-9]+)([a-z]?)\.([0-9]+)$")


def read_inputs(classes: set[str]) -> pd.DataFrame:
    cat = pd.read_csv(CATALOG, sep=r"\s+", comment="#", names=["id", "ra_cat", "dec_cat", "sysz", "musez", "nirissz", "nirspecz", "cls"],
                      dtype={"id": str})
    cat = cat[cat.cls.isin(classes)].reset_index(drop=True)
    parts = cat.id.str.extract(LABEL)
    if parts.isna().any(axis=None):
        raise SystemExit(f"labels not of the form <system><knot>.<image>: {cat.id[parts.isna().any(axis=1)].tolist()}")
    cat["system"], cat["knot"], cat["number"] = parts[0], parts[1].str.upper(), parts[2].astype(int)
    rows = [line.split() for line in LENSTOOL.read_text().splitlines() if line.strip() and not line.startswith("#")]
    lt = pd.DataFrame({"lt_id": [r[0] for r in rows], "ra": [float(r[1]) for r in rows], "dec": [float(r[2]) for r in rows],
                       "z": [float(r[6]) for r in rows]})
    # Match every lenstool image to its catalogue row: nearest neighbour, then check a constant offset and the labels.
    x_c, y_c = radec_to_offsets_arcsec(cat.ra_cat, cat.dec_cat, 64.0382, -24.0675)
    x_l, y_l = radec_to_offsets_arcsec(lt.ra, lt.dec, 64.0382, -24.0675)
    j = np.hypot(x_l[:, None] - x_c[None], y_l[:, None] - y_c[None]).argmin(1)
    if len(set(j)) != len(lt) or (classes == {"G"} and len(lt) != len(cat)):
        raise SystemExit("the lenstool catalogue does not match the selected classes one to one")
    shift = np.column_stack([x_l - x_c[j], y_l - y_c[j]])
    if np.max(np.abs(shift - np.median(shift, axis=0))) > 1e-3:
        raise SystemExit(f"the lenstool positions are not a constant shift of the catalogue (spread {np.ptp(shift, axis=0)} arcsec)")
    lt_parts = lt.lt_id.str.extract(r"^[A-Z]+([0-9]+)([a-z]?)\.([0-9]+)$")
    cat_parts = cat.id.str.extract(r"^[A-Z]+([0-9]+)([a-z]?)\.([0-9]+)$").iloc[j].reset_index(drop=True)
    if not (lt_parts == cat_parts).all(axis=None):
        raise SystemExit("matched lenstool and catalogue labels disagree in number, knot or image")
    out = cat.iloc[j].reset_index(drop=True)
    out[["ra", "dec", "z", "lt_id"]] = lt[["ra", "dec", "z", "lt_id"]]
    print(f"matched {len(out)} images; lenstool = catalogue + ({np.median(shift[:, 0]) * 1e3:+.1f}, {np.median(shift[:, 1]) * 1e3:+.1f}) mas "
          f"(x west, y north), spread {np.ptp(shift, axis=0).max() * 1e3:.2f} mas")
    return out.sort_values(["system", "knot", "number"], key=lambda s: s.map(_natural) if s.name == "system" else s).reset_index(drop=True)


def _natural(name: str):
    m = re.fullmatch(r"([A-Z]+)([0-9]+)", name)
    return (m.group(1), int(m.group(2)))


def reconcile(system: pd.DataFrame, max_offset: float) -> tuple[pd.Series, list[str]]:
    """Image number per row, made consistent across the knots of one system by position; and a log of changes."""
    numbers = system.number.copy()
    knots = list(dict.fromkeys(system.knot))
    if len(knots) < 2:
        return numbers, []
    reference = max(knots, key=lambda k: (system.knot == k).sum())
    ref = system[system.knot == reference]
    xr, yr = radec_to_offsets_arcsec(ref.ra, ref.dec, 64.0382, -24.0675)
    log = []
    for k in knots:
        if k == reference:
            continue
        rows = system[system.knot == k]
        x, y = radec_to_offsets_arcsec(rows.ra, rows.dec, 64.0382, -24.0675)
        cost = np.hypot(x[:, None] - xr[None], y[:, None] - yr[None])
        r, c = linear_sum_assignment(cost)
        if cost[r, c].max() > max_offset:
            raise SystemExit(f"system {system.system.iloc[0]} knot {k}: an image is {cost[r, c].max():.1f}\" from every image of knot {reference}")
        assigned = ref.number.to_numpy()[c]
        for (idx, old), new in zip(rows.number.items(), assigned):
            if old != new:
                numbers[idx] = new
                log.append(f"{system.system.iloc[0]}{k.lower()}.{old} -> image {new}")
    return numbers, log


def add_parities(systems: dict) -> None:
    """Set PARITIES on the converted systems (image level, so every knot of an image shares it)."""
    for label, parity in PARITIES.items():
        name, letter = label.split(".")
        image = systems[name]["images"][letter]
        if "knots" in image:
            systems[name]["images"][letter] = OrderedDict([("parity", parity), *image.items()])
        else:
            image["parity"] = parity


def _knots_of(system: dict):
    """{image: {knot or None: position}} and the image-level fields (parity, notes) of each image."""
    knots, fields = {}, {}
    for name, image in system["images"].items():
        fields[name] = {k: image[k] for k in ("parity",) if k in image}
        if "knots" in image:
            knots[name] = dict(image["knots"])
        else:
            knots[name] = {None: {k: v for k, v in image.items() if k != "parity"}}
    return knots, fields


def _centroid(positions: dict) -> np.ndarray:
    x, y = radec_to_offsets_arcsec(np.array([p["ra"] for p in positions.values()]),
                                   np.array([p["dec"] for p in positions.values()]), 64.0382, -24.0675)
    return np.array([x.mean(), y.mean()])


def merge(systems: dict, names: tuple[str, ...], z: float) -> OrderedDict:
    """One system with knots from ``names`` (see MERGES in the module docstring)."""
    first = names[0]
    letters = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    base, _ = _knots_of(systems[first])
    images: dict = {}
    declared, origins = [], []
    for name in names:
        knots, fields = _knots_of(systems[name])
        if name == first:
            letter_of = {image: image for image in knots}
        else:
            ours, theirs = list(knots), list(base)
            cost = np.array([[np.hypot(*(_centroid(knots[a]) - _centroid(base[b]))) for b in theirs] for a in ours])
            r, c = linear_sum_assignment(cost)
            if len(r) != len(ours) or cost[r, c].max() > MERGE_MAX_ARCSEC:
                raise SystemExit(f"merge {names}: images of {name} have no one-to-one partner in {first} within {MERGE_MAX_ARCSEC}\"")
            letter_of = {ours[i]: theirs[j] for i, j in zip(r, c)}
        renamed = {}
        for knot in systems[name].get("knots", [None]):
            renamed[knot] = next(letters)
            declared.append(renamed[knot])
            origins.append(f"{renamed[knot]} = {name}{knot or ''}")
        for image, positions in knots.items():
            target = images.setdefault(letter_of[image], OrderedDict(knots=OrderedDict()))
            parity = fields[image].get("parity", 0)
            if parity and target.get("parity", parity) != parity:
                raise SystemExit(f"merge {names}: {name}.{image} parity {parity} conflicts with {target['parity']}")
            if parity:
                target["parity"] = parity
                target.move_to_end("knots")
            for knot, position in positions.items():
                old, new = f"{name}{knot or ''}.{image}", f"{first}{renamed[knot]}.{letter_of[image]}"
                position = OrderedDict(position)
                if old != new:
                    position["notes"] = "; ".join(filter(None, [position.get("notes"), f"was {old} before the {MERGE_DATE} merge"]))
                target["knots"][renamed[knot]] = position
    notes = "; ".join(f"{n}: {systems[n]['notes']}" for n in names if systems[n].get("notes"))
    return OrderedDict(z=z, notes=f"merged {MERGE_DATE}, knots {', '.join(origins)} (one galaxy: z equal within the "
                                  f"spectroscopic errors, same knot pattern in every image). {notes}",
                       knots=declared, images=OrderedDict(sorted(images.items())))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--classes", nargs="+", default=["G"], help="catalogue classes to keep (G, S, B, Q); default gold")
    parser.add_argument("--max-image-offset", type=float, default=8.0, help="arcsec between matched images of two knots")
    args = parser.parse_args(argv)
    if set(args.classes) != {"G"}:
        raise SystemExit("only the gold class has lenstool positions; other classes need the catalogue frame, not supported here")
    table = read_inputs(set(args.classes))
    systems: dict = OrderedDict()
    changes: list[str] = []
    for name, rows in table.groupby("system", sort=False):
        if rows.knot.nunique() > 1 and (rows.knot == "").any():
            raise SystemExit(f"system {name} mixes knotted and plain labels")
        z = rows.z.unique()
        if len(z) != 1:
            raise SystemExit(f"system {name} has redshifts {z}")
        numbers, log = reconcile(rows, args.max_image_offset)
        changes += log
        r0 = rows.iloc[0]
        spec = [f"{label} {v:g}" for label, v in (("MUSE", r0.musez), ("NIRISS", r0.nirissz), ("NIRSpec", r0.nirspecz)) if v > 0]
        entry = OrderedDict(z=float(z[0]))
        if spec:
            entry["notes"] = "spectroscopic z: " + ", ".join(spec)
        knotted = (rows.knot != "").all()
        if knotted:
            entry["knots"] = list(dict.fromkeys(rows.knot))
        images: dict = OrderedDict()
        for (_, row), number in zip(rows.iterrows(), numbers):
            letter = chr(ord("a") + int(number) - 1)
            position = OrderedDict(ra=float(row.ra), dec=float(row.dec))
            if int(number) != int(row.number):
                position["notes"] = f"catalogue label {row.id}"
            if knotted:
                images.setdefault(letter, OrderedDict(knots=OrderedDict()))["knots"][row.knot] = position
            else:
                images[letter] = position
        entry["images"] = OrderedDict(sorted(images.items()))
        systems[name] = entry
    add_parities(systems)
    merged = {names[0]: merge(systems, names, z) for names, z in MERGES}
    gone = {name for names, _ in MERGES for name in names[1:]}
    systems = OrderedDict((name, merged.get(name, entry)) for name, entry in systems.items() if name not in gone)
    header = (f"CANUCS multiple images of MACS J0416 (Rihtarsic et al. 2025, A&A 696, A15; MAST DOI 10.17909/ph4n-6n76), class "
              f"{' '.join(args.classes)}, converted on {date.today().isoformat()} by scripts/convert_canucs_catalog.py.\n"
              "System, knot and image from the published labels (K6c.2 = system K6, knot C, image b).\n"
              "Positions and z from the CANUCS lenstool input: the catalogue shifted by about 20 mas, the frame of the CANUCS model "
              "and of members.cat.\n"
              + ("Image numbers reconciled across knots by position: " + "; ".join(changes) + "." if changes else "No image renumbering was needed.")
              + f"\nParities of {len(PARITIES)} images (2026-11-03) and systems merged as one galaxy ({MERGE_DATE}): "
              + ", ".join("+".join(names) for names, _ in MERGES) + "; both set in this script (PARITIES, MERGES).")
    write_image_catalog(args.output, systems, header=header)
    loaded = load_image_catalog(args.output)
    n_knotted = sum(1 for s in systems.values() if "knots" in s)
    print(f"wrote {args.output}: {len(systems)} systems ({n_knotted} with knots), {loaded.family_id.nunique()} knot-families, {len(loaded)} images")
    for line in changes:
        print("  renumbered", line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
