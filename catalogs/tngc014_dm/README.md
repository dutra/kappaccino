# tngc014_dm: dark-matter-only mock from TNG-Cluster

The same cluster and projection as `../tngc014` (TNG-Cluster zoom 014, halo 2571291, snapshot 73, z = 0.3802, yz view),
with a lens made of its dark matter only: particle types 1 and 2 of the hydrodynamical run. Gas, stars and black holes
are left out and the mass is not rescaled. This is not a dark-matter-only simulation: the dark matter has already
responded to the baryons.

Built on 2026-11-11 by `geometric_telescope/scripts/build_tngc014_mock.py --dm-only`, checked by
`geometric_telescope/scripts/qa_tngc014_mock.py tngc014_dm` (figure in `geometric_telescope/results/tngc014_dm_mock/`).
Everything not listed here follows `../tngc014/README.md`: frame, cosmology, source redshifts, classes, noise, knots
and the rule that ties images to satellites.

## What changes

| | tngc014 | tngc014_dm |
|---|---|---|
| theta_E at z_s = 2, 4, 6, 9 | 34.2", 39.0", 40.8", 42.1" | 28.7", 32.7", 34.3", 35.4" |
| mass within 2", 5", 10", 40" | 1 | 0.49, 0.72, 0.81, 0.87 of the full lens |
| systems, families with knots | 69, 84 | 69, 83 |
| observed images, positions with knots | 299, 362 | 277, 327 |
| truth images within 3" of the centre | 86, median magnification 0.0003 | 6, median magnification 0.7 |
| images with magnification above 100 | 39 | 40 |

The dark matter is 85% of the projected mass in the 400" field. Without the stars the centre is shallow, so most
systems have no demagnified central image. The satellites are the same Subfind objects with their dark matter only:
between m = 17 and 21 that is 60% of the mass within twice the stellar half-mass radius.

The sources are drawn again on this lens, so system names do not match those of `../tngc014`. The classes have the
same quotas: 22 wide, 14 intermediate, 8 cusp, 15 fold and 10 subhalo (S1 to S10, 41 tied images, 11 satellites).
Every truth image maps back to its source within 0.003 mas. Here the shear maps agree with the deflection derivatives
to 1e-3.

## Files

| file | content |
|---|---|
| `large/images.yaml` | observed catalog |
| `large/members.yaml` | the 246 members of `../tngc014/large/members.yaml`, unchanged: the galaxies mark the subhalos, their light is not in the lens |
| `large/priors.yaml`, `large/run.yaml` | the fit: standalone copies of the files of `../tngc014` with the same priors and settings, this folder's catalogs and truth maps |
| `source/images_truth.yaml`, `source/images_truth.csv`, `source/members_truth.csv` | truth, as in `../tngc014/source`; the member table lists the full Subfind masses, not the dark-matter masses |
| `data/tngc014_dm/tngc014_dm_snap73_yz_lens_maps.h5` | kappa, gamma1, gamma2, alpha_x, alpha_y, potential for D_ds/D_s = 1, and mu at z_s = 2, 4, 9 (not tracked) |
| `data/tngc014_dm/{kappa,gammax,gammay}_z9.fits` | the maps at z_s = 9 for the truth comparison (not tracked) |

## Priors

`large/priors.yaml` uses the priors of `../tngc014` unchanged, so the two fits differ only in the lens. The dark matter
alone gives theta_E = 34" against 41" at z_s = 6, 49% of the mass within 2" of the centre and 60% of the mass within
twice the stellar half-mass radius of the members (40-73% between m = 17 and 21); all of these lie inside the shared
priors.

## Catalog sizes

| folder | systems | members | run name |
|---|---:|---:|---|
| `small/` | 20 | 50 | `tngc014_dm_small` |
| `medium/` | 40 | 100 | `tngc014_dm_medium` |
| `large/` | 69 | 246 | `tngc014_dm` |

Small is a subset of medium, which is a subset of large. Medium retains every small entry and adds
systems by class and coverage, then members by their deflection at the selected images. Entries are copied unchanged.
The truth lens is identical for all three sizes; `source/` remains shared.

Use `kappaccino catalogs/tngc014_dm/medium --profile fast` for the medium fit.
The small and medium run files inherit from `large/`; there is no run file at the parent level.
Rebuild medium with `scripts/build_tngc014_medium.py` using the existing lenstronomy interpreter.
The original external mock and small builders predate this layout and must be pointed to `large/` before reuse.
