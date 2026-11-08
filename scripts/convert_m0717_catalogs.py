"""Convert the published MACS J0717 multiple-image and member tables into the YAML catalogs, keeping every row, the
published systems and the published redshifts.

limousin2016: the CATS v4.1 release on MAST (Limousin et al. 2016).  ``arcs.txt`` is a Lenstool image file without a
#REFERENCE line (absolute RA/Dec), labels ``<system>.<image>`` with numbered images (1.2 -> system 1, image b); z = 0
there is a free redshift, and the best-fit value the release gives for it in ``params.txt`` (``z_m_limit``) is
written with notes "geometric z (lens-model prediction)".  The fixed redshifts are written with notes "fixed z in the
model": the release does not say which are spectroscopic.  The members are the 90 scaled galaxies of ``params.txt``
(the potentials with a ``mag``), offsets from its reference point converted to RA/Dec; the shape is b/a from the
Lenstool ellipticity (a^2 - b^2) / (a^2 + b^2) and its angle_pos.

kawamata2016: Kawamata et al. 2016, Table 8 (VizieR J/ApJ/819/114/table8).  A system takes zspec (given on one of its
images), else the model zmodel, else the photometric prior zprior.

Example:
    python scripts/convert_m0717_catalogs.py limousin2016 catalogs/m0717/limousin2016/source catalogs/m0717/limousin2016
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kappaccino.io.image_catalog import write_image_catalog  # noqa: E402
from kappaccino.io.member_catalog import write_member_catalog  # noqa: E402
from kappaccino.io.sky import offsets_to_radec  # noqa: E402

ARCS = "hlsp_frontier_model_macs0717_cats_v4.1_arcs.txt"
PARAMS = "hlsp_frontier_model_macs0717_cats_v4.1_params.txt"


def image_letter(n: int) -> str:
    return chr(96 + n)


def limousin_images(folder: Path) -> dict:
    params = (folder / PARAMS).read_text(encoding="utf-8")
    model_z = {m.group(1): float(m.group(2)) for m in re.finditer(r"^\s*z_m_limit\s+\d+\s+(\d+)\.0\s+\d+\s+([\d.]+)", params, re.M)}
    systems: dict = OrderedDict()
    for line in (folder / ARCS).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        label, ra, dec, _a, _b, _theta, z, _mag = line.split()
        system, image = label.split(".")
        z = float(z)
        if z > 0:
            entry = systems.setdefault(system, OrderedDict(z=z, notes="fixed z in the model", images=OrderedDict()))
        elif system in model_z:
            entry = systems.setdefault(system, OrderedDict(z=model_z[system], notes="geometric z (lens-model prediction; free in the model)",
                                                           images=OrderedDict()))
        else:
            entry = systems.setdefault(system, OrderedDict(z=0.0, notes="free z in the model", images=OrderedDict()))
        entry["images"][image_letter(int(image))] = OrderedDict(ra=float(ra), dec=float(dec))
    return systems


def limousin_members(folder: Path) -> dict:
    params = (folder / PARAMS).read_text(encoding="utf-8")
    ra0, dec0 = map(float, re.search(r"^\s*reference\s+3\s+([\d.+-]+)\s+([\d.+-]+)", params, re.M).groups())
    members = OrderedDict()
    for block in re.finditer(r"^potentiel\s+(\S+)\n(.*?)^\s*end", params, re.M | re.S):
        values = dict(re.findall(r"^\s*(\w+)\s+([-\d.eE+]+)", block.group(2), re.M))
        if "mag" not in values:
            continue
        ra, dec = offsets_to_radec(float(values["x_centre"]), float(values["y_centre"]), ra0, dec0)
        e = float(values["ellipticite"])
        members[block.group(1)] = OrderedDict(ra=round(float(ra), 7), dec=round(float(dec), 7), mag={"F814W": float(values["mag"])},
                                              shape=[1.0, round(math.sqrt((1 - e) / (1 + e)), 4), float(values["angle_pos"])])
    return members


def kawamata_images(table: Table) -> dict:
    systems: dict = OrderedDict()
    for row in table:
        system, image = str(row["ImgS"]).strip().split(".")
        entry = systems.setdefault(system, OrderedDict(found={}, images=OrderedDict()))
        for column in ("zspec", "zmodel", "zprior"):
            if not np.ma.is_masked(row[column]):
                entry["found"].setdefault(column, set()).add(round(float(row[column]), 5))
        entry["images"][image_letter(int(image))] = OrderedDict(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]),
                                                                notes=f"ref {str(row['Ref']).strip()}")
    out: dict = OrderedDict()
    for name, entry in systems.items():
        for column, notes in (("zspec", "spectroscopic z"), ("zmodel", "geometric z (lens-model prediction)"),
                              ("zprior", "photometric z (prior of the model)")):
            if column in entry["found"]:
                if len(entry["found"][column]) > 1:
                    raise SystemExit(f"system {name} has more than one {column}: {sorted(entry['found'][column])}")
                out[name] = OrderedDict(z=entry["found"][column].pop(), notes=notes, images=entry["images"])
                break
        else:
            out[name] = OrderedDict(z=0.0, notes="free z in the model", images=entry["images"])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paper", choices=("limousin2016", "kawamata2016"))
    parser.add_argument("source", type=Path, help="the paper's source/ folder")
    parser.add_argument("output", type=Path, help="the paper folder (images.yaml, members.yaml are written there)")
    args = parser.parse_args(argv)
    stamp = f"on {date.today().isoformat()} by scripts/convert_m0717_catalogs.py"
    if args.paper == "limousin2016":
        systems = limousin_images(args.source)
        reference = "Limousin et al. 2016, A&A 588, A99; CATS v4.1 release on MAST"
        members = limousin_members(args.source)
        write_member_catalog(args.output / "members.yaml", members, band="F814W", header=(
            f"{reference}: the {len(members)} scaled galaxies of {PARAMS}, converted {stamp}.\n"
            "mag is the Lenstool potfile magnitude; the release does not name the band (its member selection uses F814W)."))
        print(f"wrote {args.output / 'members.yaml'}: {len(members)} members")
    else:
        systems = kawamata_images(Table.read(args.source / "J_ApJ_819_114_table8.vot"))
        reference = "Kawamata et al. 2016, ApJ 819, 114, Table 8 (VizieR J/ApJ/819/114/table8)"
    kinds = [s["notes"].split(" (")[0] for s in systems.values()]
    header = (f"{reference}: multiple images converted {stamp}.\nEvery row, published systems; image n -> nth lowercase letter.  "
              f"z kinds: {', '.join(f'{k} {kinds.count(k)}' for k in sorted(set(kinds)))} (see each system's notes).")
    write_image_catalog(args.output / "images.yaml", systems, header=header)
    print(f"wrote {args.output / 'images.yaml'}: {len(systems)} systems, {sum(len(s['images']) for s in systems.values())} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
