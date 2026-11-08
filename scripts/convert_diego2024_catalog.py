"""Convert the Diego et al. 2024 (PEARLS) MACS J0416 multiple-image table (VizieR J/A+A/690/A114, Table B.1) into the
YAML image catalog, keeping every row and the published labels (source 12, image 12.b -> system 12, image b).

The table gives one redshift per source: the spectroscopic zsp where measured, otherwise the lens-model zgeom.  The
system z is that value and its notes say which.  The rank (A to D, per image) goes to the image notes; no rank cut
is made here.  No positional errors or photometry are published, so none are written.

Example:
    python scripts/convert_diego2024_catalog.py catalogs/m0416/diego2024/images.ecsv catalogs/m0416/diego2024/images.yaml
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kappaccino.io.image_catalog import write_image_catalog  # noqa: E402


def convert(table: Table) -> dict:
    systems: dict = OrderedDict()
    for row in table:
        source, image = str(row["Arc"]).split(".")
        system = systems.setdefault(source, OrderedDict(zsp=set(), zgeom=set(), images=OrderedDict()))
        for column in ("zsp", "zgeom"):
            if not np.ma.is_masked(row[column]):
                system[column].add(round(float(row[column]), 5))  # float32 in the VizieR table
        system["images"][image] = OrderedDict(ra=float(row["RAJ2000"]), dec=float(row["DEJ2000"]), notes=f"rank {row['Rank']}")
    out: dict = OrderedDict()
    for name, system in systems.items():
        for column in ("zsp", "zgeom"):
            if len(system[column]) > 1:
                raise SystemExit(f"source {name} has two values of {column}: {sorted(system[column])}")
        if system["zsp"]:
            z, notes = system["zsp"].pop(), "spectroscopic z"
        elif system["zgeom"]:
            z, notes = system["zgeom"].pop(), "geometric z (lens-model prediction)"
        else:
            raise SystemExit(f"source {name} has no redshift")
        out[name] = OrderedDict(z=z, notes=notes, images=system["images"])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table", type=Path, help="Table B.1 as ECSV (images.ecsv)")
    parser.add_argument("output", type=Path, help="YAML catalog to write (.yaml)")
    args = parser.parse_args(argv)
    systems = convert(Table.read(args.table))
    n_spec = sum(s["notes"] == "spectroscopic z" for s in systems.values())
    header = (f"Diego et al. 2024, A&A 690, A114 (PEARLS), Table B.1, converted from {args.table} on {date.today().isoformat()} "
              "by scripts/convert_diego2024_catalog.py.\n"
              f"Every row, published labels.  z: {n_spec} systems spectroscopic, the rest the lens-model zgeom (see each system's notes).\n"
              "Image notes give the published rank (A best to D).  Positions J2000 degrees as published; no errors or photometry.")
    write_image_catalog(args.output, systems, header=header)
    print(f"wrote {args.output}: {len(systems)} systems ({n_spec} spectroscopic), {sum(len(s['images']) for s in systems.values())} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
