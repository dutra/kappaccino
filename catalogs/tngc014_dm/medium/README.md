# tngc014_dm/medium: the important images and members

A reduced version of `../large/images.yaml` and `../large/members.yaml` for quick fits, built by
`scripts/build_tngc014_medium.py`. The entries are copied unchanged, so names, positions, noise
and notes are those of the full catalog. The lens is the same: the members left out are still in the truth maps.

| | large | medium |
|---|---|---|
| systems | 69 | 40 |
| observed images (positions with knots) | 277 (327) | 164 (196) |
| members | 246 | 100 |

## Systems

Spectroscopic redshifts only, 12 wide, 8 intermediate, 6 cusp, 8 fold, 6 subhalo. All 20 small systems are retained. Each added system is the farthest from those taken in image centroid, redshift and,
for folds, pair separation. Added subhalo systems maximize the number of newly covered satellites, then tied images.

| system | class | z | images | knots | closest pair ["] | widest pair ["] | tied satellites |
|---|---|---|---|---|---|---|---|
| 1 | wide | 1.377 | 3 |  | 8.41 | 50.0 |  |
| 2 | wide | 1.565 | 3 |  | 11.74 | 24.1 |  |
| 5 | wide | 2.126 | 3 |  | 22.69 | 49.7 |  |
| 6 | wide | 2.439 | 5 |  | 14.68 | 65.8 |  |
| 9 | wide | 3.147 | 3 |  | 34.16 | 64.5 |  |
| 11 | wide | 3.735 | 3 |  | 19.20 | 69.5 |  |
| 15 | wide | 4.197 | 4 |  | 36.80 | 72.0 |  |
| 17 | wide | 4.338 | 3 |  | 26.85 | 65.3 |  |
| 19 | wide | 5.006 | 4 |  | 37.29 | 73.4 |  |
| 20 | wide | 5.815 | 5 |  | 10.15 | 71.3 |  |
| 21 | wide | 6.775 | 3 |  | 23.82 | 67.9 |  |
| 22 | wide | 7.365 | 3 |  | 26.25 | 56.8 |  |
| 23 | intermediate | 1.446 | 3 |  | 7.18 | 45.6 |  |
| 24 | intermediate | 1.551 | 5 |  | 5.65 | 56.1 |  |
| 25 | intermediate | 1.618 | 3 |  | 3.74 | 10.4 |  |
| 30 | intermediate | 2.430 | 3 |  | 5.44 | 40.6 |  |
| 32 | intermediate | 2.862 | 3 | 3 | 4.49 | 60.8 |  |
| 33 | intermediate | 3.328 | 3 | 2 | 3.56 | 63.7 |  |
| 34 | intermediate | 3.897 | 3 |  | 3.19 | 49.0 |  |
| 36 | intermediate | 4.441 | 3 |  | 4.09 | 53.7 |  |
| 37 | cusp | 1.693 | 5 |  | 0.64 | 46.8 |  |
| 38 | cusp | 2.870 | 5 |  | 0.46 | 37.7 |  |
| 39 | cusp | 3.080 | 5 |  | 0.63 | 62.2 |  |
| 41 | cusp | 3.898 | 5 |  | 0.36 | 22.0 |  |
| 42 | cusp | 4.373 | 5 |  | 0.77 | 47.8 |  |
| 43 | cusp | 4.434 | 5 |  | 0.88 | 28.7 |  |
| 47 | fold | 1.786 | 3 | 2 | 0.37 | 50.5 |  |
| 50 | fold | 2.037 | 3 |  | 0.75 | 52.2 |  |
| 51 | fold | 2.144 | 3 |  | 1.72 | 56.5 |  |
| 52 | fold | 2.200 | 5 |  | 2.55 | 59.2 |  |
| 55 | fold | 3.477 | 5 |  | 1.00 | 65.1 |  |
| 57 | fold | 4.034 | 3 | 2 | 0.34 | 41.6 |  |
| 58 | fold | 4.064 | 3 | 2 | 1.32 | 63.0 |  |
| 59 | fold | 4.402 | 3 |  | 0.82 | 54.8 |  |
| S2 | subhalo | 2.323 | 9 | 2 | 0.74 | 62.4 | 1554469, 1554487 |
| S5 | subhalo | 2.783 | 7 |  | 0.67 | 44.5 | 1554464 |
| S6 | subhalo | 2.826 | 5 | 2 | 0.44 | 63.7 | 1554422 |
| S7 | subhalo | 3.060 | 5 |  | 0.52 | 16.5 | 1554439, 1554489, 1554499, 1554713 |
| S8 | subhalo | 3.854 | 5 |  | 0.88 | 6.2 | 1554414 |
| S10 | subhalo | 5.823 | 7 |  | 0.56 | 21.5 | 1554717, 1554777 |

## Members

All 50 small members are retained. The selection includes the BCG (1), satellites tied to the selected S systems (11),
other retained small members (38), and the
50 members with the largest point-mass deflection at any selected image: M / (pi Sigma_cr D_d^2 d)
for D_ds/D_s = 1, with M the Subfind bound mass in the particle types of the lens and d the distance to the image.
The smallest value kept is 0.02" and the largest left out is 0.02".
The ranking uses the truth masses, not the light.

`selection.csv` lists every selected system and member with these numbers. `priors.yaml` and `run.yaml` extend the
files of the large folder and change only the two catalogs and the run name (`tngc014_dm_medium`).
