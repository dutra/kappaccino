# tngc014/medium: the important images and members

A reduced version of `../large/images.yaml` and `../large/members.yaml` for quick fits, built by
`scripts/build_tngc014_medium.py`. The entries are copied unchanged, so names, positions, noise
and notes are those of the full catalog. The lens is the same: the members left out are still in the truth maps.

| | large | medium |
|---|---|---|
| systems | 69 | 40 |
| observed images (positions with knots) | 299 (362) | 174 (204) |
| members | 246 | 100 |

## Systems

Spectroscopic redshifts only, 12 wide, 8 intermediate, 6 cusp, 8 fold, 6 subhalo. All 20 small systems are retained. Each added system is the farthest from those taken in image centroid, redshift and,
for folds, pair separation. Added subhalo systems maximize the number of newly covered satellites, then tied images.

| system | class | z | images | knots | closest pair ["] | widest pair ["] | tied satellites |
|---|---|---|---|---|---|---|---|
| 1 | wide | 1.167 | 3 |  | 25.55 | 49.8 |  |
| 2 | wide | 1.175 | 4 |  | 10.50 | 56.9 |  |
| 4 | wide | 1.515 | 4 |  | 38.23 | 68.3 |  |
| 8 | wide | 2.081 | 3 |  | 9.05 | 64.0 |  |
| 13 | wide | 2.588 | 4 |  | 12.27 | 73.0 |  |
| 14 | wide | 2.607 | 4 |  | 26.19 | 77.2 |  |
| 15 | wide | 2.649 | 4 |  | 13.05 | 78.1 |  |
| 16 | wide | 2.783 | 3 |  | 8.63 | 54.9 |  |
| 18 | wide | 3.288 | 4 |  | 10.12 | 78.8 |  |
| 20 | wide | 3.482 | 4 |  | 41.44 | 83.0 |  |
| 21 | wide | 4.322 | 3 |  | 11.37 | 75.5 |  |
| 22 | wide | 7.350 | 4 |  | 49.83 | 89.8 |  |
| 23 | intermediate | 1.154 | 3 |  | 4.19 | 26.3 |  |
| 27 | intermediate | 1.536 | 3 |  | 7.56 | 18.9 |  |
| 29 | intermediate | 1.591 | 4 |  | 4.93 | 64.7 |  |
| 31 | intermediate | 1.767 | 3 |  | 5.02 | 56.8 |  |
| 32 | intermediate | 2.111 | 3 |  | 7.98 | 54.1 |  |
| 33 | intermediate | 2.535 | 5 |  | 3.50 | 74.5 |  |
| 35 | intermediate | 3.897 | 3 |  | 4.44 | 58.6 |  |
| 36 | intermediate | 4.848 | 3 |  | 3.83 | 72.6 |  |
| 37 | cusp | 2.031 | 3 |  | 0.32 | 1.0 |  |
| 39 | cusp | 3.909 | 6 |  | 0.43 | 68.2 |  |
| 40 | cusp | 4.085 | 7 |  | 0.53 | 80.8 |  |
| 42 | cusp | 4.339 | 5 |  | 0.50 | 37.4 |  |
| 43 | cusp | 5.816 | 3 |  | 1.09 | 2.4 |  |
| 44 | cusp | 6.681 | 7 |  | 0.41 | 85.4 |  |
| 45 | fold | 1.017 | 4 | 2 | 0.78 | 47.5 |  |
| 46 | fold | 1.128 | 3 |  | 0.29 | 45.4 |  |
| 51 | fold | 2.039 | 3 |  | 2.86 | 54.5 |  |
| 52 | fold | 2.130 | 3 | 2 | 0.38 | 60.0 |  |
| 53 | fold | 2.223 | 3 | 2 | 0.70 | 53.5 |  |
| 55 | fold | 2.564 | 4 |  | 2.93 | 68.5 |  |
| 57 | fold | 3.194 | 5 |  | 1.09 | 65.5 |  |
| 59 | fold | 3.755 | 3 | 3 | 1.90 | 28.6 |  |
| S1 | subhalo | 1.063 | 6 |  | 0.82 | 9.9 | 1554439, 1554499, 1554606 |
| S2 | subhalo | 1.938 | 6 |  | 0.87 | 71.3 | 1554469, 1554487 |
| S4 | subhalo | 2.102 | 7 | 3 | 0.89 | 30.1 | 1554614, 1554717, 1554777 |
| S5 | subhalo | 2.408 | 9 |  | 0.63 | 73.4 | 1554455, 1554484 |
| S6 | subhalo | 2.437 | 8 |  | 1.01 | 65.5 | 1554464, 1554513 |
| S9 | subhalo | 4.382 | 8 |  | 0.30 | 82.6 | 1554422, 1554490, 1554515, 1554551 |

## Members

All 50 small members are retained. The selection includes the BCG (1), satellites tied to the selected S systems (16),
other retained small members (33), and the
50 members with the largest point-mass deflection at any selected image: M / (pi Sigma_cr D_d^2 d)
for D_ds/D_s = 1, with M the Subfind bound mass in the particle types of the lens and d the distance to the image.
The smallest value kept is 0.03" and the largest left out is 0.03".
The ranking uses the truth masses, not the light.

`selection.csv` lists every selected system and member with these numbers. `priors.yaml` and `run.yaml` extend the
files of the large folder and change only the two catalogs and the run name (`tngc014_medium`).
