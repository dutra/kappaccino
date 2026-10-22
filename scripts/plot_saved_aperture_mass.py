"""Regenerate a saved aperture profile in solar masses without rerunning inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import pandas as pd
from astropy import constants as const, units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lenscluster import plotting
from lenscluster.output_paths import output_paths


def render(stage: Path, paper_output: Path | None = None) -> Path:
    paths = output_paths(stage)
    with h5py.File(paths.artifacts / "plot_bundle.h5", "r") as handle:
        args = json.loads(handle["runtime_args_json"][()])
        meta = json.loads(handle["state/build_state_meta_json"][()])
    config = meta["cosmo_config"]
    if config["class"] != "FlatLambdaCDM" or meta.get("fit_cosmology_flat_wcdm", False):
        raise ValueError("This renderer requires a fixed saved FlatLambdaCDM cosmology")
    cosmo = FlatLambdaCDM(H0=config["H0"], Om0=config["Om0"])
    zl, zs = float(meta["z_lens"]), float(args["caustic_source_redshift"])
    dl = cosmo.angular_diameter_distance(zl)
    ds = cosmo.angular_diameter_distance(zs)
    dls = cosmo.angular_diameter_distance_z1z2(zl, zs)
    sigma_crit = (const.c**2 / (4 * np.pi * const.G) * ds / (dl * dls)).to(u.Msun / u.Mpc**2)
    header = fits.getheader(paths.fits / "truth_recovery_kappa_model_median.fits")
    pixel_area_deg2 = float(proj_plane_pixel_area(WCS(header).celestial))
    pixel_solid_angle = pixel_area_deg2 * (np.pi / 180)**2
    factor = float((sigma_crit * dl**2).to_value(u.Msun) * pixel_solid_angle)
    profile = pd.read_csv(paths.tables / "truth_recovery_m2d_aperture_profile.csv")
    center = profile.iloc[0].to_dict()
    images = pd.read_csv(paths.tables / "image_fit_quality.csv")
    radii = plotting._truth_recovery_image_aperture_radii(images, center)
    plotting._plot_truth_recovery_m2d_aperture_ratio(
        stage, profile, center, image_radii_arcsec=radii,
        mass_per_kappa_pixel_msun=factor,
    )
    converted = profile.copy()
    for source, target in [("kappa_true_sum", "mass_true_msun"),
                           ("kappa_model_sum", "mass_model_msun"),
                           ("kappa_model_sum_q16", "mass_model_q16_msun"),
                           ("kappa_model_sum_q84", "mass_model_q84_msun")]:
        converted[target] = profile[source] * factor
    converted.to_csv(paths.tables / "truth_recovery_m2d_aperture_profile_msun.csv", index=False)
    provenance = dict(z_lens=zl, z_source=zs, cosmology=config,
                      pixel_area_arcsec2=pixel_area_deg2 * 3600**2,
                      mass_per_kappa_pixel_msun=factor)
    (paths.tables / "truth_recovery_m2d_mass_conversion.json").write_text(json.dumps(provenance, indent=2) + "\n")
    output = plotting._plot_path(stage, "truth_recovery_m2d_aperture_ratio.pdf")
    if paper_output is not None:
        shutil.copy2(output, paper_output)
    print(json.dumps(provenance, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("--paper-output", type=Path)
    options = parser.parse_args()
    print(render(options.stage, options.paper_output))
