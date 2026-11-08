# Limousin et al. 2016: MACS J0717, CATS v4.1

Limousin et al. 2016, A&A 588, A99 (2016A&A...588A..99L), as released by the CATS team on MAST
(https://archive.stsci.edu/pub/hlsp/frontier/macs0717/models/cats/v4.1/), downloaded 2026-11-08. Only the image list,
the best-fit parameter file and the readme; the maps are not kept.

| file | sha256 |
|---|---|
| `source/hlsp_frontier_model_macs0717_cats_v4.1_arcs.txt` | abc9f95f89467160a59d87e7252eea2667c97ff292d25d67880ce8b1565ad73e |
| `source/hlsp_frontier_model_macs0717_cats_v4.1_params.txt` | 7ea3af8dbb3e6dd9917f9155a17a24903d0337ddd6fb63ff17522ca262ed22e4 |
| `source/hlsp_frontier_model_macs0717_cats_v4.1_readme.txt` | c6da772ed4f91d93be2ec9aa49f5f551efdfd5fe44541ec428a087d6f968dd54 |

- `images.yaml`: `python scripts/convert_m0717_catalogs.py limousin2016 source .`: 51 systems, 132 images (the set the
  model uses; the paper lists 61 systems, 165 images, the rest only in its tables). `1.2` -> system 1, image b.
- Redshifts: 10 systems fixed in the model (notes "fixed z in the model"); the other 41 free, written with the best-fit
  z of `params.txt` (notes "geometric z"). The fixed values are not all spectroscopic: Kawamata et al. 2016 give no
  spectroscopic z for 6 (2.393 here) or for the images of 68 (5.51 here), and 12 is 1.699 here, 1.710 there.
- `members.yaml`: the 90 scaled galaxies of `params.txt` (offsets from its reference 109.398200, 37.745778 converted
  to RA/Dec; b/a from the Lenstool ellipticity). The band of `mag` is not stated; the member selection is in F814W, so
  it is labeled F814W. O5, optimized on its own, sits on member 518; potential 6 is a z = 0.15 foreground galaxy.
