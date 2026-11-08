# Furtak et al. 2023: UNCOVER lens model of Abell 2744, release v2.0

Furtak et al. 2023, MNRAS 523, 4568 (2023MNRAS.523.4568F); release v2.0 of 2024-07-26 (Furtak & Zitrin; Price et al. in
prep.), from the public Google Drive folder linked on jwst-uncover.github.io/DR4.html
(https://drive.google.com/drive/folders/1Lc7uveJr_ZNIcvAQYVfAeRF8Qr68D_5p), downloaded 2026-11-08. Only the tables, the
region file and the README; the maps are not kept (catalogs/README.md).

| file | sha256 |
|---|---|
| `source/A2744_multiple-images_v2.0.fits` (186 rows) | 0d2649b3991129df0b775965caadf10109714d32cd804dcc76afa17c9c789119 |
| `source/A2744_cluster-members_v2.0.fits` (552 rows) | dfb635f972f635bc7fb39502d239fbf957a1799be6bc070b15b13e00ba6978a2 |
| `source/A2744_multiple-images_v2.0.reg` | 433b10529aae79060cc268e1e46d1aaa7822d48320e46c8fb302254d82754dba |
| `source/README_v2.0.txt` | c903666fa0034d502da8ce86203aee340622284e1d793a21a42019fbe1f5d99c |

- `images.yaml`: `convert_a2744_catalogs.py furtak2023 source/A2744_multiple-images_v2.0.fits images.yaml`: 35 systems,
  every row. The IDs are floats: `1.21` = system 1, knot B, image a; `66.1` = system 66, image a (the DS9 labels show
  the two forms). Systems 8 and 22 have slightly different knot redshifts; the median is used and noted.
- Redshifts: 172 rows spectroscopic (MUSE, NIRSpec, NIRCam grism; the reference is in each system's notes), systems 66
  and 77 photometric, 71 and 82 none (z = 0).
- The release README says 141 images of 48 sources; the table has 186 rows. Unresolved; ask the authors before using
  every row as a constraint.
- `members.yaml`: `convert_a2744_catalogs.py furtak2023 source/A2744_cluster-members_v2.0.fits members.yaml --members`:
  the 419 members with F814W (band F814W, with F606W, F070W, F090W where measured). The 133 MegaScience (JWST-only)
  members have no F814W and stay in the FITS only. Ellipticity and theta are not converted (their convention is not
  stated). Members whose name contains BCG have role bcg.
- Frame: the core positions equal Bergamini 2023a's; (-0.03", -0.07") from Bergamini 2023b.
