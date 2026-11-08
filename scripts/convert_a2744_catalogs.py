"""Convert the published Abell 2744 multiple-image and member tables into the YAML catalogs, one paper at a time,
keeping every row, the published systems and the published redshifts.

Image labels.  Bergamini et al. (2023a, 2023b): ``1.2a`` = system 1, knot 2, image a; ``7a`` = system 7, image a.
Furtak et al. (UNCOVER v2.0): a float, ``1.21`` = system 1, knot 2, image 1 and ``66.1`` = system 66, image 1 (the
DS9 labels of the release show the two forms; a float printed with two decimals ends in 0 only for the second).
Mahler et al. 2018: a float, ``1.2`` = system 1, image 2.  Knot n becomes the nth capital letter (knot 2 -> B), image n
the nth lowercase letter, so family names cannot collide with system names (system 1, knot B -> family 1B).

Redshifts.  A system takes its published spectroscopic z.  Without one: Furtak's photometric z (notes "photometric z"),
Mahler's model z (notes "geometric z (lens-model prediction)"), otherwise z = 0 with notes "free z in the model",
Lenstool's convention, which a fit cannot use as it is.  Quality flags go to the notes.

Example:
    python scripts/convert_a2744_catalogs.py bergamini2023b catalogs/a2744/bergamini2023b/source/J_ApJ_952_84_table2.vot.xml \\
        catalogs/a2744/bergamini2023b/images.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kappaccino.image_catalog import write_image_catalog  # noqa: E402
from kappaccino.member_catalog import write_member_catalog  # noqa: E402

BERGAMINI_ID = re.compile(r"^([A-Z]*\d+)(?:\.(\d+))?([a-zA-Z])$")   # JD1A: system JD1, image a


def value(row, column):
    """The cell as a float, or None when masked or not finite."""
    cell = row[column]
    if np.ma.is_masked(cell):
        return None
    cell = float(cell)
    return cell if np.isfinite(cell) else None


def letter(n: int, upper: bool) -> str:
    return chr((65 if upper else 97) + n - 1)


def add(systems: dict, system: str, knot: str, image: str, position: dict, z: float | None, z_note: str) -> None:
    entry = systems.setdefault(system, OrderedDict(z=set(), z_note=set(), knots=[], images=OrderedDict()))
    if z is not None:
        entry["z"].add(round(z, 5))
    entry["z_note"].add(z_note)
    if knot:
        if knot not in entry["knots"]:
            entry["knots"].append(knot)
        entry["images"].setdefault(image, OrderedDict(knots=OrderedDict()))["knots"][knot] = position
    else:
        entry["images"][image] = position


def finish(systems: dict) -> dict:
    """One z per system (the published one; z = 0 when none), knots declared only where used."""
    out = OrderedDict()
    for name, entry in systems.items():
        values = sorted(entry["z"])
        if len(values) > 1:
            if (values[-1] - values[0]) / (1 + values[0]) > 1e-3:
                raise SystemExit(f"system {name} has more than one redshift: {values}")
            entry["z"] = {float(np.median(values))}   # knots measured separately, within 0.1%
            entry["z_note"].add(f"median of the knot redshifts {', '.join(map(str, values))}")
        if bool(entry["knots"]) and any("knots" not in image for image in entry["images"].values()):
            raise SystemExit(f"system {name} mixes images with and without knots")
        notes = "; ".join(sorted(n for n in entry["z_note"] if n))
        z = entry["z"].pop() if entry["z"] else 0.0
        if not z:
            notes = "free z in the model" + (f"; {notes}" if notes else "")
        system = OrderedDict(z=z)
        if notes:
            system["notes"] = notes
        if entry["knots"]:
            system["knots"] = entry["knots"]
        system["images"] = entry["images"]
        out[name] = system
    return out


def bergamini(table: Table, flags: tuple[str, ...]) -> dict:
    systems: dict = OrderedDict()
    for row in table:
        match = BERGAMINI_ID.fullmatch(str(row["ID"]).strip())
        if match is None:
            raise SystemExit(f"unexpected ID {row['ID']!r}")
        system, knot, image = match.groups()
        z = value(row, "zspec" if "zspec" in table.colnames else "zsp")
        notes = ", ".join(f"{label} {row[column]}" for column, label in flags
                          if column in table.colnames and not np.ma.is_masked(row[column]) and str(row[column]).strip())
        position = OrderedDict(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]))
        if notes:
            position["notes"] = notes
        add(systems, system, letter(int(knot), True) if knot else "", image.lower(), position, z, "spectroscopic z" if z else "")
    return finish(systems)


def furtak(table: Table) -> dict:
    systems: dict = OrderedDict()
    for row in table:
        system, digits = f"{float(row['ID']):.2f}".split(".")
        knot, image = ("", int(digits[0])) if digits.endswith("0") else (letter(int(digits[0]), True), int(digits[1]))
        kind = str(row["Redshift_type"]).strip()
        z = value(row, "z") if kind in ("spec", "phot") else None
        note = {"spec": "spectroscopic z", "phot": "photometric z"}.get(kind, "")
        if z is not None:
            note += f" ({str(row['Redshift_instrument']).strip()}, {str(row['Redshift_reference']).strip()})"
        add(systems, system, knot, letter(image, False), OrderedDict(ra=float(row["RA"]), dec=float(row["Dec"])), z, note)
    return finish(systems)


def mahler(table: Table) -> dict:
    systems: dict = OrderedDict()
    for row in table:
        system, image = f"{float(row['MUL']):.1f}".split(".")
        zspec, zmod = value(row, "zspec"), value(row, "zmod")
        z, note = (zspec, "spectroscopic z") if zspec else (zmod, "geometric z (lens-model prediction)") if zmod else (None, "")
        position = OrderedDict(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]))
        category = str(row["Cat"]).strip()
        position["notes"] = f"confid {row['confid']}" + (f", categories {category}" if category else "")
        add(systems, system, "", letter(int(image), False), position, z, note)
    return finish(systems)


def bergamini_members(table: Table) -> dict:
    members = OrderedDict()
    for row in table:
        kind = {"s": "spec", "p": "phot"}[str(row["Type"]).strip()]
        members[str(row["ID"]).strip()] = OrderedDict(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]),
                                                      mag={"F160W": float(row["mF160W"])})
        if value(row, "zsp") is not None:
            members[str(row["ID"]).strip()]["z"] = {"value": value(row, "zsp"), "kind": kind}
    return members


def furtak_members(table: Table, band: str) -> tuple[dict, list[str]]:
    members, skipped = OrderedDict(), []
    for row in table:
        name = str(row["ID"]).strip()
        mags = OrderedDict((b, [round(value(row, b), 4), round(value(row, f"{b}_err"), 4)]) for b in ("F606W", "F814W", "F070W", "F090W")
                           if value(row, b) is not None and value(row, f"{b}_err") is not None)
        if band not in mags:
            skipped.append(name)
            continue
        member = OrderedDict(ra=float(row["RA"]), dec=float(row["Dec"]), mag=mags)
        z = value(row, "z")
        if z is not None:
            member["z"] = {"value": round(z, 5), "kind": "spec" if str(row["origin"]).strip() == "MUSE" else "phot"}
        label = "" if np.ma.is_masked(row["name"]) else str(row["name"]).strip()
        if label.startswith("BCG") or "BCG" in label:
            member["role"] = "bcg"
        member["notes"] = f"origin {str(row['origin']).strip()}" + (f", {label}" if label else "")
        members[name] = member
    return members, skipped


IMAGE_READERS = {
    "bergamini2023a": lambda t: bergamini(t, (("n_ID", "flag"), ("QF", "QF"), ("QP", "QP"))),
    "bergamini2023b": lambda t: bergamini(t, (("Flag", "flag"), ("q_zspec", "QF"), ("q_pos", "QP"), ("Loc", "location"))),
    "furtak2023": furtak,
    "mahler2018": mahler,
}
REFERENCES = {
    "bergamini2023a": "Bergamini et al. 2023, A&A 670, A60 (VizieR J/A+A/670/A60)",
    "bergamini2023b": "Bergamini et al. 2023, ApJ 952, 84 (VizieR J/ApJ/952/84)",
    "furtak2023": "Furtak et al. 2023, MNRAS 523, 4568; UNCOVER lens model release v2.0 (2024-07-26)",
    "mahler2018": "Mahler et al. 2018, MNRAS 473, 663 (VizieR J/MNRAS/473/663)",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", choices=sorted(IMAGE_READERS))
    parser.add_argument("table", type=Path, help="the published table (VOTable or FITS)")
    parser.add_argument("output", type=Path, help="images.yaml, or members.yaml with --members")
    parser.add_argument("--members", action="store_true", help="convert a member table (bergamini2023a, furtak2023)")
    parser.add_argument("--band", default="F814W", help="scaling-relation band of the Furtak members")
    args = parser.parse_args(argv)
    table = Table.read(args.table)
    stamp = f"converted from {args.table.name} on {date.today().isoformat()} by scripts/convert_a2744_catalogs.py"
    if args.members:
        if args.paper == "bergamini2023a":
            members, band, skipped = bergamini_members(table), "F160W", []
        elif args.paper == "furtak2023":
            band = args.band
            members, skipped = furtak_members(table, band)
        else:
            raise SystemExit(f"no member table for {args.paper}")
        header = (f"{REFERENCES[args.paper]}: members {stamp}.\nEvery row" +
                  (f" except {len(skipped)} without a {band} magnitude (in the source table only)" if skipped else "") + ".")
        write_member_catalog(args.output, members, band=band, header=header)
        print(f"wrote {args.output}: {len(members)} members" + (f", {len(skipped)} skipped (no {band})" if skipped else ""))
        return 0
    systems = IMAGE_READERS[args.paper](table)
    n_free = sum(s["z"] == 0.0 for s in systems.values())
    header = (f"{REFERENCES[args.paper]}: multiple images {stamp}.\nEvery row, published systems; knot n -> nth capital "
              f"letter, image n -> nth lowercase letter.  {n_free} systems without a published z have z = 0 (free in the model).\n"
              "Image notes carry the published quality flags.  Positions J2000 degrees as published; no errors or photometry.")
    write_image_catalog(args.output, systems, header=header)
    n_images = sum(len(s["images"]) for s in systems.values())
    print(f"wrote {args.output}: {len(systems)} systems ({n_free} free z), {n_images} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
