"""Degrade a simulated cluster's image catalog for the uncertain-image test: keep a few families trusted and hide the rest.

The trusted families stay in a YAML image catalog with their redshifts.  Every image of the other families goes to a
CSV of uncertain images (PartitionConfig.catalog_path) with no family label and a noisy photometric redshift, and with
``--counterimages K`` one image of each of K trusted families with at least three images is moved there too (a new
counterimage of a known family).  The FF-SIMS catalogs give images of one source identical colors with 0.01 mag
errors, so every image (trusted and uncertain) gets realistic magnitude errors and is scattered by them.

Writes to the output directory:

trusted.yaml
    The trusted families (``--images`` of the ``ignore`` and ``groups`` fits).
uncertain.csv
    The uncertain images (``--set partition.catalog_path`` of the ``groups`` fit).
truth.csv
    The true family, redshift and role (``hidden`` or ``counterimage``) of every uncertain image, for scoring.
degrade.json
    The settings, the trusted families and the run commands of the two fits.

Example (Hera, 5 trusted families, seed 0):

    python scripts/degrade_image_catalog.py HERA --trusted 5 --seed 0 --counterimages 2
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kappaccino.io.image_catalog import write_image_catalog  # noqa: E402

CATALOGS = {"HERA": "catalogs/hera/images.yaml", "ARES": "catalogs/ares/images.yaml"}
FOLDERS = {"HERA": "catalogs/hera", "ARES": "catalogs/ares"}     # their run.yaml and priors.yaml
CAMPAIGN = "uncertain_images"
Z_MIN, Z_MAX = 0.1, 7.0


def mag_error(mag: np.ndarray, err_at_26: float, floor: float) -> np.ndarray:
    """Background-limited errors: ``err_at_26`` at magnitude 26, a factor 10^0.4 per magnitude, at least ``floor``."""
    return np.maximum(float(floor), float(err_at_26) * 10.0 ** (0.4 * (np.asarray(mag, dtype=float) - 26.0)))


def degrade_bands(bands: dict, rng: np.random.Generator, err_at_26: float, floor: float) -> dict:
    """The band entries with magnitudes scattered by the new errors (detection flags kept)."""
    out = copy.deepcopy(bands)
    for entry in out.values():
        error = float(mag_error(entry["mag"], err_at_26, floor))
        entry["mag"] = float(entry["mag"] + rng.normal(0.0, error))
        entry["err"] = error
    return out


def photometric_redshift(z: float, rng: np.random.Generator, sigma: float, outlier_fraction: float) -> tuple[float, float]:
    """A photometric redshift and its 1-sigma error: z + N(0, sigma (1 + z)), or uniform on 0.5-6 for an outlier."""
    if rng.random() < outlier_fraction:
        value = rng.uniform(0.5, 6.0)
    else:
        value = z + rng.normal(0.0, sigma * (1.0 + z))
    value = float(np.clip(value, Z_MIN, Z_MAX))
    return value, float(sigma * (1.0 + value))


def choose_trusted(systems: dict, n: int, ids: list[str] | None, rng: np.random.Generator) -> list[str]:
    names = list(systems)
    if ids:
        unknown = sorted(set(ids) - set(names))
        if unknown:
            raise SystemExit(f"--trusted-ids: no systems {unknown}")
        return list(ids)
    if not 1 <= n < len(names):
        raise SystemExit(f"--trusted must be between 1 and {len(names) - 1}")
    return [str(s) for s in rng.choice(names, size=n, replace=False)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cluster", choices=sorted(CATALOGS), type=str.upper)
    parser.add_argument("--trusted", type=int, default=5, help="number of trusted families, chosen at random (default 5)")
    parser.add_argument("--trusted-ids", help="comma-separated trusted systems instead of a random choice")
    parser.add_argument("--counterimages", type=int, default=0,
                        help="trusted families (with >= 3 images) that each lose one image to the uncertain set (default 0)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--photz-sigma", type=float, default=0.05, help="photometric-redshift scatter per (1 + z) (default 0.05)")
    parser.add_argument("--photz-outliers", type=float, default=0.05, help="fraction of catastrophic photometric redshifts (default 0.05)")
    parser.add_argument("--mag-err-26", type=float, default=0.1, help="magnitude error at magnitude 26 (default 0.1)")
    parser.add_argument("--mag-err-floor", type=float, default=0.03, help="smallest magnitude error (default 0.03)")
    parser.add_argument("--case", help="name of the case (default n<trusted>_c<counterimages>_s<seed>)")
    parser.add_argument("--output-root", default=f"results/{CAMPAIGN}", help=f"default results/{CAMPAIGN}")
    args = parser.parse_args(argv)

    source = Path(CATALOGS[args.cluster])
    document = yaml.safe_load(source.read_text())
    systems = {str(k): v for k, v in document["systems"].items()}
    rng = np.random.default_rng(args.seed)
    ids = [s.strip() for s in args.trusted_ids.split(",")] if args.trusted_ids else None
    trusted = choose_trusted(systems, args.trusted, ids, rng)
    case = args.case or f"n{len(trusted)}_c{args.counterimages}_s{args.seed}"
    cluster_key = args.cluster.lower()
    output = Path(args.output_root) / cluster_key / "inputs" / case
    output.mkdir(parents=True, exist_ok=True)

    eligible = [s for s in trusted if len(systems[s]["images"]) >= 3]
    if args.counterimages > len(eligible):
        raise SystemExit(f"--counterimages {args.counterimages}: only {len(eligible)} trusted families have >= 3 images")
    donors = [str(s) for s in rng.choice(eligible, size=args.counterimages, replace=False)] if args.counterimages else []
    moved = {s: str(rng.choice(list(systems[s]["images"]))) for s in donors}

    kept, rows, truth = {}, [], []
    for name, system in systems.items():
        images = {}
        for image_id, entry in system["images"].items():
            entry = copy.deepcopy(entry)
            if "bands" in entry:
                entry["bands"] = degrade_bands(entry["bands"], rng, args.mag_err_26, args.mag_err_floor)
            label = f"{name}.{image_id}"
            if name in trusted and moved.get(name) != str(image_id):
                images[image_id] = entry
                continue
            z_phot, z_sigma = photometric_redshift(float(system["z"]), rng, args.photz_sigma, args.photz_outliers)
            row = {"image_label": label, "ra": entry["ra"], "dec": entry["dec"], "zphot_best": z_phot, "catalog_z_sigma": z_sigma}
            for band, value in entry.get("bands", {}).items():
                row[f"mag_{band}"], row[f"mag_err_{band}"] = value["mag"], value["err"]
            rows.append(row)
            truth.append({"image_label": label, "family": name, "z_true": float(system["z"]),
                          "role": "counterimage" if name in trusted else "hidden"})
        if name in trusted:
            kept[name] = {**{k: v for k, v in system.items() if k != "images"}, "images": images}

    # Shuffle the rows so the table order does not give the families away.
    order = rng.permutation(len(rows))
    pd.DataFrame([rows[i] for i in order]).to_csv(output / "uncertain.csv", index=False)
    pd.DataFrame([truth[i] for i in order]).to_csv(output / "truth.csv", index=False)
    write_image_catalog(output / "trusted.yaml", kept, defaults=document.get("defaults"),
                        header=f"Degraded from {source} by scripts/degrade_image_catalog.py: trusted families {', '.join(trusted)}; "
                               f"magnitudes rescattered (seed {args.seed}).")

    base = ["kappaccino", FOLDERS[args.cluster], "--campaign", CAMPAIGN, "--images", str(output / "trusted.yaml")]
    commands = {"ignore": base + ["--run-label", f"{case}_ignore"],
                "groups": base + ["--run-label", f"{case}_groups", "--set", f"partition.catalog_path={output / 'uncertain.csv'}"]}
    settings = {"cluster": args.cluster, "case": case, "source_catalog": str(source), "trusted": trusted,
                "counterimage_images": [f"{s}.{i}" for s, i in moved.items()], "seed": args.seed,
                "photz_sigma": args.photz_sigma, "photz_outliers": args.photz_outliers,
                "mag_err_26": args.mag_err_26, "mag_err_floor": args.mag_err_floor,
                "n_uncertain": len(rows), "n_hidden_families": len(systems) - len(trusted),
                "commands": {k: shlex.join(v) for k, v in commands.items()},
                "full_command": shlex.join(["kappaccino", FOLDERS[args.cluster], "--campaign", CAMPAIGN, "--run-label", "full"])}
    (output / "degrade.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{output}: {len(trusted)} trusted families ({', '.join(trusted)}), {len(rows)} uncertain images "
          f"from {settings['n_hidden_families']} hidden families and {len(moved)} counterimages")
    for name, command in settings["commands"].items():
        print(f"  {name}: {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
