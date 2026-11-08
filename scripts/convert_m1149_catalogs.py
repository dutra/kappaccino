"""Convert the published MACS J1149 multiple-image and member tables into the YAML catalogs, keeping every row, the
published systems and the published redshifts.

treu2016: Treu et al. 2016, Tables 3 and 4 (VizieR J/ApJ/817/60).  Table 3: labels ``<system>.<image>`` (2.1 ->
system 2, image a); a system takes the new zspec or the earlier spectroscopic z.  Its system 1 rows (the Refsdal host)
are the knot-1 rows of Table 4, so system 1 comes from Table 4 alone: ``1.<knot>.<image>`` (1.12.3 -> knot K12, image
c), z 1.489 from Table 3, and the four images of SN Refsdal (SN1-SN4) as system SN at the same z.  Image notes keep
Set (Gold, Silver, Rejected, blank: the "all" sample) and Score.  The two rejected rows without an ID are systems
R26 and R27 (their VizieR record numbers), z = 0.

grillo2016: Grillo et al. 2016, Table 2 (VizieR J/ApJ/822/78), images by T16 system in row order (a, b, ...).  The
VizieR RA is wrong (mirrored), so positions are rebuilt from the published offsets from the BCG (offX positive west):
RA = RA_BCG - offX / (3600 cos Dec_BCG), Dec = Dec_BCG + offY / 3600, BCG at 11:49:35.699 +22:23:54.71.  Systems
without z have z = 0 (free in their model).  Members: Table 6, F160W; p_member 1 for the members with spectra (Obs M or
G), roles from Notes (BCG).

canucs: the CANUCS DR1 Lenstool member file (Schuldt et al. 2024 selection), F160W, ``#REFERENCE 0``.

Example:
    python scripts/convert_m1149_catalogs.py treu2016 catalogs/m1149/treu2016/source catalogs/m1149/treu2016
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kappaccino.io.image_catalog import write_image_catalog  # noqa: E402
from kappaccino.io.member_catalog import write_member_catalog  # noqa: E402

BCG = SkyCoord("11:49:35.699 +22:23:54.71", unit=(u.hourangle, u.deg))
REFSDAL_Z = 1.489
CANUCS = "hlsp_canucs_jwst-hst_multi_macs1149-lenstool-clugal_multi_v1_model.txt"


def masked(cell) -> bool:
    return np.ma.is_masked(cell) or (isinstance(cell, (float, np.floating)) and not np.isfinite(cell))


def treu_images(folder: Path) -> dict:
    table3 = Table.read(folder / "J_ApJ_817_60_table3.vot")
    table4 = Table.read(folder / "J_ApJ_817_60_table4.vot")
    systems: dict = OrderedDict()
    host_z = None
    for row in table3:
        if masked(row["ImgS"]):   # a rejected candidate without an ID: its own one-image system
            system, image = f"R{row['recno']}", "1"
        else:
            system, image = f"{float(row['ImgS']):.1f}".split(".")
        z = row["zspec"] if not masked(row["zspec"]) else row["z"]
        if system == "1":
            host_z = host_z or (None if masked(z) else float(z))
            continue
        entry = systems.setdefault(system, OrderedDict(z=set(), images=OrderedDict()))
        if not masked(z):
            entry["z"].add(round(float(z), 4))
        notes = f"set {str(row['Set']).strip() or '-'}, score {row['Score']}"
        entry["images"][chr(96 + int(image))] = OrderedDict(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]), notes=notes)
    host = OrderedDict(z=host_z or REFSDAL_Z, notes="spectroscopic z; the Refsdal host, knots from Table 4", knots=[], images=OrderedDict())
    supernova = OrderedDict(z=host_z or REFSDAL_Z, notes="spectroscopic z; SN Refsdal images S1-S4 (Table 4)", images=OrderedDict())
    for row in table4:
        label = str(row["Knot"]).strip()
        position = OrderedDict(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]))
        if not masked(row["Note"]) and str(row["Note"]).strip():
            position["notes"] = f"note {str(row['Note']).strip()}"
        if label.startswith("SN"):
            supernova["images"][chr(96 + int(label[2:]))] = position
            continue
        _, knot, image = label.split(".")
        knot = f"K{knot}"
        if knot not in host["knots"]:
            host["knots"].append(knot)
        host["images"].setdefault(chr(96 + int(image)), OrderedDict(knots=OrderedDict()))["knots"][knot] = position
    out = OrderedDict([("1", host), ("SN", supernova)])
    for name, entry in systems.items():
        values = sorted(entry["z"])
        notes = "spectroscopic z" if values else "free z in the model"
        z = float(np.median(values)) if values else 0.0
        if len(values) > 1:   # images measured separately: the median within 1%; otherwise the images disagree (12: rejected)
            agree = (values[-1] - values[0]) / (1 + values[0]) <= 0.01
            z = z if agree else values[0]
            notes += f"; image redshifts {', '.join(map(str, values))}" + ("" if agree else " disagree, the first is used")
        out[name] = OrderedDict(z=round(z, 4), notes=notes, images=entry["images"])
    return out


def grillo_images(folder: Path) -> dict:
    systems: dict = OrderedDict()
    cos_dec = math.cos(BCG.dec.radian)
    for row in Table.read(folder / "J_ApJ_822_78_table2.vot"):
        system = str(row["T16"]).strip()
        entry = systems.setdefault(system, OrderedDict(z=0.0 if masked(row["z"]) else round(float(row["z"]), 4), images=OrderedDict()))
        ra = BCG.ra.deg - float(row["offX"]) / (3600.0 * cos_dec)
        dec = BCG.dec.deg + float(row["offY"]) / 3600.0
        entry["images"][chr(97 + len(entry["images"]))] = OrderedDict(ra=round(float(ra), 7), dec=round(float(dec), 7))
    for entry in systems.values():
        entry["notes"] = "spectroscopic z" if entry["z"] else "free z in the model"
        entry.move_to_end("images")
    return systems


def grillo_members(folder: Path) -> dict:
    members = OrderedDict()
    for row in Table.read(folder / "J_ApJ_822_78_table6.vot"):
        coord = SkyCoord(f"{row['RAJ2000']} {row['DEJ2000']}", unit=(u.hourangle, u.deg))
        member = OrderedDict(ra=round(float(coord.ra.deg), 7), dec=round(float(coord.dec.deg), 7), mag={"F160W": [float(row["F160W"]), float(row["e_F160W"])]})
        observed = str(row["Obs"]).strip()
        label = "" if masked(row["Notes"]) else str(row["Notes"]).strip()
        if observed not in ("", "-"):
            member["p_member"] = 1.0
        if label == "BCG":
            member["role"] = "bcg"
        member["notes"] = f"spectra {observed or '-'}" + (f", {label}" if label else "")
        members[str(row["IDph"]).strip()] = member
    return members


def canucs_members(folder: Path) -> dict:
    members = OrderedDict()
    for line in (folder / CANUCS).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, ra, dec, _a, _b, _theta, mag, _lum = line.split()[:8]
        members[name] = OrderedDict(ra=float(ra), dec=float(dec), mag={"F160W": float(mag)})
    return members


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", choices=("treu2016", "grillo2016", "canucs"))
    parser.add_argument("source", type=Path, help="the paper's source/ folder")
    parser.add_argument("output", type=Path, help="the paper folder")
    args = parser.parse_args(argv)
    stamp = f"on {date.today().isoformat()} by scripts/convert_m1149_catalogs.py"
    if args.paper == "canucs":
        members = canucs_members(args.source)
        write_member_catalog(args.output / "members.yaml", members, band="F160W", header=(
            f"CANUCS DR1 Lenstool member file of MACS J1149 ({CANUCS}; the Schuldt et al. 2024 selection), converted {stamp}."))
        print(f"wrote {args.output / 'members.yaml'}: {len(members)} members")
        return 0
    if args.paper == "treu2016":
        systems, reference = treu_images(args.source), "Treu et al. 2016, ApJ 817, 60, Tables 3 and 4 (VizieR J/ApJ/817/60)"
    else:
        systems, reference = grillo_images(args.source), "Grillo et al. 2016, ApJ 822, 78, Table 2 (VizieR J/ApJ/822/78)"
        members = grillo_members(args.source)
        write_member_catalog(args.output / "members.yaml", members, band="F160W", header=(
            f"Grillo et al. 2016, ApJ 822, 78, Table 6, converted {stamp}.\np_member 1: members with spectra (Obs M, G or both)."))
        print(f"wrote {args.output / 'members.yaml'}: {len(members)} members")
    write_image_catalog(args.output / "images.yaml", systems, header=(
        f"{reference}: multiple images converted {stamp}.\nEvery row, published systems; image n -> nth lowercase letter.  "
        "See the README for the redshifts and positions."))
    print(f"wrote {args.output / 'images.yaml'}: {len(systems)} systems, {sum(len(s['images']) for s in systems.values())} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
