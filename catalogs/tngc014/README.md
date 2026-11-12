# tngc014: mock catalog from TNG-Cluster

Simulated cluster. TNG-Cluster zoom 014 (halo 2571291) at snapshot 73, z = 0.3802, M200c = 1.69e15 Msun, projected
along the simulation x axis (the yz view). This is the rank-one Einstein-radius analog of Abell 370 in
`ggsl_env/lenscluster/analog_matches.py` (Cian's axis 0, theta_E = 39.6" at z_s = 6; these maps give 40.9").

Built on 2026-11-11 by `geometric_telescope/scripts/build_tngc014_mock.py`, checked by
`geometric_telescope/scripts/qa_tngc014_mock.py` (figure in `geometric_telescope/results/tngc014_mock/`).

## Files

| file | content |
|---|---|
| `large/images.yaml` | observed catalog: 69 systems, 299 images, 362 positions with knots |
| `large/members.yaml` | 246 members (245 + BCG), band F125W |
| `large/priors.yaml`, `large/run.yaml` | the fit: one cluster halo, shear, the BCG off the relation, neutral member priors; `kappaccino catalogs/tngc014/large [--profile fast]` |
| `data/tngc014/{kappa,gammax,gammay}_z9.fits` | the maps at z_s = 9 for the truth comparison (not tracked) |
| `source/images_truth.yaml` | same systems with exact positions, every image and true redshifts |
| `source/images_truth.csv` | one row per (system, knot, image): true and observed position, source position, z, mu, parity, kappa, shear, class, tied satellite |
| `source/members_truth.csv` | Subfind masses, velocity dispersion, Vmax and radii of each member |
| `data/tngc014/tngc014_snap73_yz_lens_maps.h5` | kappa, gamma1, gamma2, alpha_x, alpha_y, potential for D_ds/D_s = 1, and mu at z_s = 2, 4, 9 (not tracked) |

## Frame and cosmology

Positions are offsets about (0, 0) in degrees, as for Hera and Ares. With `reference 3 0 0` the kappaccino offsets
are the map coordinates: x = -ra 3600, y = dec 3600 arcsec. The BCG is at (0, 0). Distance ratios use flat LCDM with
H0 = 70 and Om0 = 0.3. The maps are 400" wide with 0.0977" pixels and scale with D_ds/D_s.

## Lens plane

The maps are the particle maps of `geometric_telescope` (`bulk/maps/zoom014_yz_zs9_fiducial.h5`), a single lens plane
with the FoF particles of the original zoom only. There is no outer diffuse mass and no line-of-sight structure. The
force softening is 1.48 kpc, 0.28" at this redshift, so pairs closer than 0.3" are not catalogued.

Image positions solve theta - (D_ds/D_s) alpha(theta) = beta on a cubic spline of the deflection maps. Every truth
image maps back to its source within 0.05 mas. Parities and magnifications come from the Jacobian of the same spline.
The shear maps differ from the deflection derivatives by about 1e-2, so magnifications above 100 are uncertain by a
factor of about two.

## Sources and classes

Sources were seeded in the image plane and kept by class quota, so the sample is not a uniform source-plane
population. Redshifts follow eight bins between z = 1 and 8 with weights 0.10, 0.17, 0.20, 0.17, 0.12, 0.12, 0.08,
0.04. The class of each system is in its `notes`.

| class | systems | definition (observable images) |
|---|---|---|
| wide | 22 | closest pair at least 8" apart |
| intermediate | 14 | closest pair 3" to 8" apart |
| cusp | 8 | three images of alternating parity within a few arcsec |
| fold | 15 | opposite-parity pair 0.3" to 3" apart, third image at least twice as far; five per bin 0.3-0.7", 0.7-1.5", 1.5-3" |
| subhalo | 10 | at least one image tied to a satellite, named S1 to S10 |

Twelve systems carry two or three knots, placed 0.03" to 0.15" from the source centre. Knot separations within an
image run from 0.08" to 3.7".

The observed catalog adds Gaussian noise of 0.03" per axis, removes images within 3" of the BCG centre or with
|mu| < 0.25, and gives 12% of the systems a photometric redshift with `z_err` = 0.06 (1 + z). Magnitudes are a
source magnitude drawn between 26 and 29 minus 2.5 log |mu|. No magnitude limit is applied.

## Images tied to satellites

For each satellite with bound mass above 3e10 Msun we remove its bound particles, refit the source position and
solve the images again. An image is tied to the satellite when it lies within four point-mass Einstein radii of it
(clipped to 1.5" to 6") and either vanishes or moves by more than 0.5". Systems with a tied image are named S<n>.
The system `notes` list the satellite and its images, and each tied image has its own `notes` with the shift.
Satellite names are Subfind IDs, the same as in `large/members.yaml`. A satellite fainter than the member limit is flagged
in the note. The satellite deflection uses a 0.16" Gaussian smoothing, not the adaptive kernel of the maps, so the
tag is a classification and not a measurement.

## Members

Subfind galaxies within 120" and 2 R200c along the line of sight, brighter than 24. F125W is the rest-frame SDSS z
absolute magnitude plus the distance modulus minus 2.5 log(1 + z), without dust. Shapes are stellar second moments
within twice the stellar half-mass radius. The BCG magnitude (13.75) counts every star bound to the central subhalo,
so it includes the intracluster light. Satellites below the limit stay in the maps.

A dark-matter-only version of the same lens is in `../tngc014_dm`.

## Catalog sizes

| folder | systems | members | run name |
|---|---:|---:|---|
| `small/` | 20 | 50 | `tngc014_small` |
| `medium/` | 40 | 100 | `tngc014_medium` |
| `large/` | 69 | 246 | `tngc014` |

Small is a subset of medium, which is a subset of large. Medium retains every small entry and adds
systems by class and coverage, then members by their deflection at the selected images. Entries are copied unchanged.
The truth lens is identical for all three sizes; `source/` remains shared.

Use `kappaccino catalogs/tngc014/medium --profile fast` for the medium fit.
The small and medium run files inherit from `large/`; there is no run file at the parent level.
Rebuild medium with `scripts/build_tngc014_medium.py` using the existing lenstronomy interpreter.
The original external mock and small builders predate this layout and must be pointed to `large/` before reuse.
