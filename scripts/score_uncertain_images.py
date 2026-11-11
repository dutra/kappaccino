"""Score the uncertain-image test on a simulated cluster: the recovered partition and the lens against the truth.

For every case under ``<root>/inputs`` (written by scripts/degrade_image_catalog.py) this reads the fits
``<root>/<case>_ignore`` (trusted families only) and ``<root>/<case>_groups`` (trusted families plus the uncertain
images), and the reference fit ``<root>/full`` (every family trusted), and writes to ``<root>/scores``:

partition_summary.csv
    Per case, from the groups fit's partition (``uncertain_images.h5``): the adjusted Rand index of the sampled partitions of the
    uncertain images against the true families (mean, 16th and 84th percentiles over draws); the pair precision and
    recall of "same source" (from the pair probabilities); the fraction of hidden families recovered exactly (all their
    images in one group with no other image, averaged over draws); the mean probability that a hidden image is
    called unrelated or put in a trusted family; and the mean probability that a moved counterimage joins its own family.
partition_families.csv
    Per case and hidden family: images, redshift, probability of exact recovery, mean probability of unrelated.
lens_summary.csv
    Per case and fit (full, ignore, groups), over two regions of the truth-recovery grid, ``circle`` (within
    ``--radius-pad`` arcsec beyond the outermost image of the full catalog) and ``near`` (within ``--near`` arcsec of
    any image of the full catalog, hidden ones included): the median and scatter of the fractional convergence error
    (kappa_model - kappa_true) / kappa_true, the rms of the pull (kappa_model - kappa_true) / half the 16-84 width,
    the fraction of pixels whose truth lies inside the 16-84 band, and the same error for the shear modulus.
aperture_profiles.csv
    Per case and fit, the enclosed-mass ratio profile (truth_recovery_m2d_aperture_profile.csv).

Example:

    python scripts/score_uncertain_images.py results/uncertain_images/hera
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kappaccino.io.hdf5 import load_arrays  # noqa: E402
from kappaccino.io.image_catalog import load_image_catalog  # noqa: E402
from kappaccino.io.sky import radec_to_offsets_arcsec  # noqa: E402

FITS = ("full", "ignore", "groups")
PARTITION_FILE = "uncertain_images.h5"     # kappaccino.partition.groups.OUTPUT_FILE


def run_dir(root: Path, case: str, fit: str) -> Path:
    return root / ("full" if fit == "full" else f"{case}_{fit}") / "final"


# ------------------------------------------------------------------ partition
def partition_scores(partition_file: Path, truth: pd.DataFrame, trusted: list[str]) -> tuple[dict, pd.DataFrame]:
    data = load_arrays(partition_file)
    options = [str(o) for o in data["options"]]
    labels = [str(s) for s in data["image_labels"]]
    truth = truth.set_index("image_label").loc[labels]
    family = truth["family"].astype(str).to_numpy()
    hidden = (truth["role"] == "hidden").to_numpy()
    assignment = np.asarray(data["assignment"])                          # (draw, image) option index
    probability = np.asarray(data["probability"])                        # (image, option)
    pair = np.asarray(data["pair_probability"])
    n_trusted = len(options) - 1 - sum(o.startswith("group") for o in options)
    new_source = lambda k: k >= 1 + n_trusted

    def predicted(draw):
        return [f"unrelated{i}" if k == 0 else options[k] for i, k in enumerate(draw)]
    ari = np.array([adjusted_rand_score(family, predicted(draw)) for draw in assignment])

    i, j = np.triu_indices(len(labels), k=1)
    same = family[i] == family[j]
    p = pair[i, j]
    precision = float(p[same].sum() / p.sum()) if p.sum() > 0 else np.nan
    recall = float(p[same].mean()) if same.any() else np.nan

    rows = []
    for name in sorted(set(family[hidden]), key=lambda s: (len(s), s)):
        members = np.flatnonzero(family == name)
        exact = []
        for draw in assignment:
            k = draw[members[0]]
            exact.append(bool(new_source(k) and np.all(draw[members] == k) and np.sum(draw == k) == len(members)))
        rows.append({"family": name, "n_images": len(members), "z_true": float(truth["z_true"].iloc[members[0]]),
                     "p_exact": float(np.mean(exact)), "p_unrelated": float(probability[members, 0].mean())})
    families = pd.DataFrame(rows)

    trusted_columns = [options.index(t) for t in trusted if t in options]
    counter = np.flatnonzero(~hidden)
    own = [probability[c, options.index(family[c])] for c in counter if family[c] in options]
    summary = {"n_uncertain": len(labels), "n_hidden_families": len(families), "n_counterimages": int(len(counter)),
               "n_groups": sum(o.startswith("group") for o in options),
               "ari_mean": float(ari.mean()), "ari_q16": float(np.quantile(ari, 0.16)), "ari_q84": float(np.quantile(ari, 0.84)),
               "pair_precision": precision, "pair_recall": recall,
               "family_exact_recovery": float(families["p_exact"].mean()) if len(families) else np.nan,
               "hidden_p_unrelated": float(probability[hidden, 0].mean()),
               "hidden_p_trusted": float(probability[np.ix_(hidden, trusted_columns)].sum(axis=1).mean()) if trusted_columns else 0.0,
               "counterimage_p_own_family": float(np.mean(own)) if own else np.nan}
    return summary, families


# ------------------------------------------------------------------ lens
def offsets(path: Path, reference: dict) -> tuple[np.ndarray, np.ndarray]:
    header = fits.getheader(path)
    ny, nx = header["NAXIS2"], header["NAXIS1"]
    yy, xx = np.mgrid[:ny, :nx]
    ra, dec = WCS(header).pixel_to_world_values(xx, yy)
    ra = np.where(ra > 180.0, ra - 360.0, ra)
    return radec_to_offsets_arcsec(ra, dec, float(reference["ra0_deg"]), float(reference["dec0_deg"]))


def lens_scores(final: Path, image_xy: np.ndarray, pad: float, near: float) -> tuple[dict, pd.DataFrame | None]:
    fits_dir = final / "fits"
    if not (fits_dir / "truth_recovery_kappa_model_median.fits").exists():
        return {}, None
    reference = json.loads((final / "artifacts" / "config.json").read_text())["model"]["reference"]
    read = lambda name: fits.getdata(fits_dir / f"truth_recovery_{name}.fits").astype(float)
    x, y = offsets(fits_dir / "truth_recovery_kappa_true.fits", reference)
    profile_path = final / "tables" / "truth_recovery_m2d_aperture_profile.csv"
    profile = pd.read_csv(profile_path) if profile_path.exists() else None
    center = (profile[["center_x_arcsec", "center_y_arcsec"]].iloc[0].to_numpy(float) if profile is not None
              else image_xy.mean(axis=0))
    radius = np.hypot(*(image_xy - center).T).max() + pad
    nearest = cKDTree(image_xy).query(np.column_stack([x.ravel(), y.ravel()]))[0].reshape(x.shape)
    regions = {"circle": np.hypot(x - center[0], y - center[1]) <= radius, "near": nearest <= near}

    truth, median = read("kappa_true"), read("kappa_model_median")
    if (fits_dir / "truth_recovery_kappa_model_q16.fits").exists():
        q16, q84 = read("kappa_model_q16"), read("kappa_model_q84")
    else:                                                    # a best fit (scripts/uncertain_images_ladder.py): no band
        q16 = q84 = np.full_like(median, np.nan)
    has_gamma = (fits_dir / "truth_recovery_gamma1_true.fits").exists()
    if has_gamma:
        gamma_true = np.hypot(read("gamma1_true"), read("gamma2_true"))
        gamma_model = np.hypot(read("gamma1_model_median"), read("gamma2_model_median"))
    out = {"circle_radius_arcsec": float(radius), "near_arcsec": float(near)}
    for region, mask in regions.items():
        ok = mask & np.isfinite(truth) & np.isfinite(median) & (truth > 0)
        fractional = (median[ok] - truth[ok]) / truth[ok]
        width = 0.5 * (q84[ok] - q16[ok])
        pull = (median[ok] - truth[ok]) / np.where(width > 0, width, np.nan)
        out.update({f"{region}_pixels": int(ok.sum()),
                    f"{region}_kappa_frac_bias": float(np.median(fractional)),
                    f"{region}_kappa_frac_scatter": float(0.5 * np.subtract(*np.quantile(fractional, [0.84, 0.16]))),
                    f"{region}_kappa_frac_abs_median": float(np.median(np.abs(fractional))),
                    f"{region}_kappa_pull_rms": float(np.sqrt(np.nanmean(pull ** 2))) if np.isfinite(pull).any() else np.nan,
                    f"{region}_kappa_coverage_68": (float(np.mean((truth[ok] >= q16[ok]) & (truth[ok] <= q84[ok])))
                                                    if np.isfinite(q16).any() else np.nan)})
        if has_gamma:
            good = mask & np.isfinite(gamma_true) & np.isfinite(gamma_model) & (gamma_true > 0)
            frac = (gamma_model[good] - gamma_true[good]) / gamma_true[good]
            out.update({f"{region}_gamma_frac_bias": float(np.median(frac)),
                        f"{region}_gamma_frac_abs_median": float(np.median(np.abs(frac)))})
    return out, profile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="results/uncertain_images/<cluster>")
    parser.add_argument("--radius-pad", type=float, default=5.0, help="arcsec beyond the outermost image scored (default 5)")
    parser.add_argument("--near", type=float, default=10.0, help="the 'near' region: pixels within this many arcsec of an image (default 10)")
    args = parser.parse_args(argv)
    root = args.root
    cases = sorted(p.parent for p in (root / "inputs").glob("*/degrade.json"))
    if not cases:
        raise SystemExit(f"{root / 'inputs'}: no cases (run scripts/degrade_image_catalog.py first)")
    out = root / "scores"
    out.mkdir(exist_ok=True)

    partition, families, lens, profiles = [], [], [], []
    full_catalog = None
    for case_dir in cases:
        settings = json.loads((case_dir / "degrade.json").read_text())
        case = settings["case"]
        meta = {"case": case, "n_trusted": len(settings["trusted"]), "seed": settings["seed"],
                "n_counterimages": len(settings["counterimage_images"])}
        if full_catalog is None:
            table = load_image_catalog(ROOT / settings["source_catalog"])
            full_catalog = np.column_stack(radec_to_offsets_arcsec(table["ra"].to_numpy(float), table["dec"].to_numpy(float), 0.0, 0.0))
        partition_file = run_dir(root, case, "groups") / "artifacts" / PARTITION_FILE
        if partition_file.exists():
            summary, per_family = partition_scores(partition_file, pd.read_csv(case_dir / "truth.csv", dtype={"family": str}),
                                                   [str(t) for t in settings["trusted"]])
            partition.append({**meta, **summary})
            families.append(per_family.assign(**meta))
        else:
            print(f"[score] {case}: no groups fit yet ({partition_file})")
        for fit in FITS:
            final = run_dir(root, case, fit)
            scores, profile = lens_scores(final, full_catalog, args.radius_pad, args.near)
            if not scores:
                print(f"[score] {case}: no truth recovery for the {fit} fit ({final})")
                continue
            lens.append({**meta, "fit": fit, **scores})
            if profile is not None:
                profiles.append(profile.assign(**meta, fit=fit))

    for name, frames in (("partition_summary", partition), ("lens_summary", lens)):
        if frames:
            pd.DataFrame(frames).to_csv(out / f"{name}.csv", index=False)
    for name, frames in (("partition_families", families), ("aperture_profiles", profiles)):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    if partition:
        print(pd.DataFrame(partition)[["case", "ari_mean", "pair_precision", "pair_recall", "family_exact_recovery",
                                       "hidden_p_unrelated", "counterimage_p_own_family"]].to_string(index=False))
    if lens:
        print(pd.DataFrame(lens)[["case", "fit"] + [f"{r}_{m}" for r in ("near", "circle") for m in
                                  ("kappa_frac_bias", "kappa_frac_abs_median", "kappa_pull_rms", "kappa_coverage_68")]].to_string(index=False))
    print(f"[score] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
