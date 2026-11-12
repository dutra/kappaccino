# tngc014_dm/small: the important images and members

A reduced version of `../large/images.yaml` and `../large/members.yaml` for quick fits, built by
`geometric_telescope/scripts/build_tngc014_small.py`. The entries are copied unchanged, so names, positions, noise
and notes are those of the full catalog. The lens is the same: the members left out are still in the truth maps.

| | full | small |
|---|---|---|
| systems | 69 | 20 |
| observed images (positions with knots) | 277 (327) | 86 (101) |
| members | 246 | 50 |

## Systems

Spectroscopic redshifts only, 6 wide, 4 intermediate, 3 cusp, 4 fold, 3 subhalo. Within a class the first system
is the one with the most images, and each next one is the farthest from those taken in image centroid, redshift and,
for folds, pair separation. Each subhalo system brings a satellite not yet covered.

| system | class | z | images | knots | closest pair ["] | widest pair ["] | tied satellites |
|---|---|---|---|---|---|---|---|
| 2 | wide | 1.565 | 3 |  | 11.74 | 24.1 |  |
| 5 | wide | 2.126 | 3 |  | 22.69 | 49.7 |  |
| 6 | wide | 2.439 | 5 |  | 14.68 | 65.8 |  |
| 17 | wide | 4.338 | 3 |  | 26.85 | 65.3 |  |
| 20 | wide | 5.815 | 5 |  | 10.15 | 71.3 |  |
| 22 | wide | 7.365 | 3 |  | 26.25 | 56.8 |  |
| 24 | intermediate | 1.551 | 5 |  | 5.65 | 56.1 |  |
| 25 | intermediate | 1.618 | 3 |  | 3.74 | 10.4 |  |
| 34 | intermediate | 3.897 | 3 |  | 3.19 | 49.0 |  |
| 36 | intermediate | 4.441 | 3 |  | 4.09 | 53.7 |  |
| 37 | cusp | 1.693 | 5 |  | 0.64 | 46.8 |  |
| 39 | cusp | 3.080 | 5 |  | 0.63 | 62.2 |  |
| 43 | cusp | 4.434 | 5 |  | 0.88 | 28.7 |  |
| 47 | fold | 1.786 | 3 | 2 | 0.37 | 50.5 |  |
| 52 | fold | 2.200 | 5 |  | 2.55 | 59.2 |  |
| 57 | fold | 4.034 | 3 | 2 | 0.34 | 41.6 |  |
| 59 | fold | 4.402 | 3 |  | 0.82 | 54.8 |  |
| S2 | subhalo | 2.323 | 9 | 2 | 0.74 | 62.4 | 1554469, 1554487 |
| S7 | subhalo | 3.060 | 5 |  | 0.52 | 16.5 | 1554439, 1554489, 1554499, 1554713 |
| S10 | subhalo | 5.823 | 7 |  | 0.56 | 21.5 | 1554717, 1554777 |

## Members

The BCG (1), the satellites tied to the selected S systems (8) and the
41 members with the largest point-mass deflection at any selected image: M / (pi Sigma_cr D_d^2 d)
for D_ds/D_s = 1, with M the Subfind bound mass in the particle types of the lens and d the distance to the image.
The smallest value kept is 0.05" and the largest left out is 0.05".
The ranking uses the truth masses, not the light.

`selection.csv` lists every selected system and member with these numbers. `priors.yaml` and `run.yaml` extend the
files of the large folder and change only the two catalogs and the run name (`tngc014_dm_small`).
