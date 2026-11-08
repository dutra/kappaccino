"""Convert the published Abell 370 multiple-image and member tables into the YAML catalogs, keeping every row, the
published systems and the published redshifts.

gledhill2024: the CANUCS DR1 Lenstool files of Gledhill et al. 2024 on MAST.  ``multim``: labels
``<system><knot>.<image>`` (1b.2 = system 1, knot B, image b; 5.1 = system 5, image a), absolute RA/Dec, one z per
row.  Its header says the redshifts are spectroscopic for systems 1, 2, 3, 4 and 6 and otherwise taken from Diego and
Strait et al., so they are written without a kind.  ``clugal``: id, RA, Dec, placeholder shape, mag (band not stated).

lagattuta2019: Lagattuta et al. 2019, Table 1 (VizieR J/MNRAS/485/3738/table1), labels ``<system>.<image>``.  A
system takes its zspec, else the model zmod (free in their model).  The image notes keep Class (gold, silver, bronze,
copper) and Note (1: not used in their model; 2: counter-image of another system).  The members are the potfile of
their model as released in the MUSE lensing-cluster atlas (Richard et al. 2021), ``#REFERENCE 3`` offsets.

Example:
    python scripts/convert_a370_catalogs.py gledhill2024 catalogs/a370/gledhill2024/source catalogs/a370/gledhill2024
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
from kappaccino.sky import offsets_to_radec  # noqa: E402

CANUCS = "hlsp_canucs_jwst-hst_multi_a370-lenstool-{}_multi_v1_model.txt"
CANUCS_ID = re.compile(r"^(\d+)([a-z]?)\.(\d+)$")


def gledhill_images(folder: Path) -> dict:
    systems: dict = OrderedDict()
    for line in (folder / CANUCS.format("multim")).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        label, ra, dec, _a, _b, _theta, z, _mag = line.split()
        system, knot, image = CANUCS_ID.fullmatch(label).groups()
        entry = systems.setdefault(system, OrderedDict(z=set(), knots=[], images=OrderedDict()))
        entry["z"].add(float(z))
        position = OrderedDict(ra=float(ra), dec=float(dec))
        letter = chr(96 + int(image))
        if knot:
            if knot.upper() not in entry["knots"]:
                entry["knots"].append(knot.upper())
            entry["images"].setdefault(letter, OrderedDict(knots=OrderedDict()))["knots"][knot.upper()] = position
        else:
            entry["images"][letter] = position
    for name, entry in systems.items():
        values = sorted(entry["z"])
        if (values[-1] - values[0]) / (1 + values[0]) > 1e-3:
            raise SystemExit(f"system {name} has more than one redshift: {values}")
        entry["z"] = float(np.median(values))
        if len(values) > 1:   # knots given slightly different redshifts
            entry["notes"] = f"median of the knot redshifts {', '.join(map(str, values))}"
            entry.move_to_end("notes", last=False)
            entry.move_to_end("z", last=False)
        if not entry["knots"]:
            del entry["knots"]
    return systems


def gledhill_members(folder: Path) -> dict:
    members = OrderedDict()
    for line in (folder / CANUCS.format("clugal")).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, ra, dec, _a, _b, _theta, mag, _lum = line.split()
        members[name] = OrderedDict(ra=float(ra), dec=float(dec), mag={"unstated": float(mag)})
    return members


def lagattuta_images(table: Table) -> dict:
    systems: dict = OrderedDict()
    for row in table:
        system, image = f"{float(row['mID']):.1f}".split(".")   # float32 in the VizieR table; images go to .7
        entry = systems.setdefault(system, OrderedDict(zspec=set(), zmod=set(), images=OrderedDict()))
        for column in ("zspec", "zmod"):
            if not np.ma.is_masked(row[column]):
                entry[column].add(round(float(row[column]), 4))
        notes = [f"class {str(row['Class']).strip()}"]
        if not np.ma.is_masked(row["Note"]) and str(row["Note"]).strip():
            notes.append(f"note {str(row['Note']).strip()}")
        entry["images"][chr(96 + int(image))] = OrderedDict(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]), notes=", ".join(notes))
    out: dict = OrderedDict()
    for name, entry in systems.items():
        for column, notes in (("zspec", "spectroscopic z"), ("zmod", "geometric z (lens-model prediction; free in the model)")):
            if entry[column]:
                if len(entry[column]) > 1:
                    raise SystemExit(f"system {name} has more than one {column}: {sorted(entry[column])}")
                out[name] = OrderedDict(z=entry[column].pop(), notes=notes, images=entry["images"])
                break
        else:
            raise SystemExit(f"system {name} has no redshift")
    return out


def lagattuta_members(path: Path) -> dict:
    """Potfile rows ``id x y a b theta mag lum`` (eight columns, no color) after ``#REFERENCE 3 ra0 dec0``."""
    members = OrderedDict()
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "#REFERENCE":
            ra0, dec0 = float(tokens[2]), float(tokens[3])
            continue
        if tokens[0].startswith("#"):   # the commented rows are the two BCGs, modelled as their own potentials
            continue
        name, x, y, a, b, theta, mag, _lum = tokens
        ra, dec = offsets_to_radec(float(x), float(y), ra0, dec0)
        members[name] = OrderedDict(ra=round(float(ra), 7), dec=round(float(dec), 7), mag={"F814W": float(mag)},
                                    shape=[float(a), float(b), float(theta)])
    return members


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", choices=("gledhill2024", "lagattuta2019"))
    parser.add_argument("source", type=Path, help="the paper's source/ folder")
    parser.add_argument("output", type=Path, help="the paper folder (images.yaml and members.yaml are written there)")
    args = parser.parse_args(argv)
    stamp = f"on {date.today().isoformat()} by scripts/convert_a370_catalogs.py"
    if args.paper == "gledhill2024":
        reference = "Gledhill et al. 2024, ApJ 973, 77; CANUCS DR1 Lenstool files on MAST"
        systems = gledhill_images(args.source)
        members, band = gledhill_members(args.source), "unstated"
        member_header = f"{reference}: the {len(members)} galaxies of {CANUCS.format('clugal')}, converted {stamp}.\nThe magnitude band is not stated in the release."
    else:
        reference = "Lagattuta et al. 2019, MNRAS 485, 3738"
        systems = lagattuta_images(Table.read(args.source / "J_MNRAS_485_3738_table1.vot"))
        members, band = lagattuta_members(args.source / "galsortcutI_mosaicBright_noBCGsV3.cat"), "F814W"
        member_header = (f"{reference}: the potfile of their model (galsortcutI_mosaicBright_noBCGsV3.cat, MUSE lensing-cluster atlas, "
                         f"Richard et al. 2021), converted {stamp}.\nBCGs excluded (separate potentials in bestopt.par).  "
                         "The band is not stated; the file name and the selection suggest F814W.")
    write_member_catalog(args.output / "members.yaml", members, band=band, header=member_header)
    header = (f"{reference}: multiple images converted {stamp}.\nEvery row, published systems; image n -> nth lowercase letter, "
              "knot letters capitalized.  See the README for the redshifts.")
    write_image_catalog(args.output / "images.yaml", systems, header=header)
    print(f"wrote {args.output}: {len(systems)} systems, {sum(len(s['images']) for s in systems.values())} images, {len(members)} members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
