# Gledhill et al. 2024: Abell 370, CANUCS

Gledhill et al. 2024, ApJ 973, 77 (2024ApJ...973...77G, doi:10.3847/1538-4357/ad684a): the CANUCS DR1 Lenstool files on
MAST (https://archive.stsci.edu/hlsps/canucs/dr1/a370/model/ and .../a370/clu/), downloaded 2026-11-08. Only the image
and member lists, the two parameter files and the readme; the maps are not kept.

| file (`source/hlsp_canucs_jwst-hst_multi_a370-lenstool-*_multi_v1_model.txt`) | sha256 |
|---|---|
| multim | d72c3d98e675e5bc00cfbdd9b84d1b8528b22e36311924fea072293af23ef9e2 |
| clugal | 95ee806eb5a4b97a138d3387c1d2e96bc3f1dfa0d66ee6784858e9d34b14d85d |
| param (Lenstool input) | aacc2dadd442645d2222c23ee2c3f9f6a76fddaa73996a26a6e88c0691a6bf3d |
| bestparam (best fit) | 3c1eea91755ea532e424b1b143e39b35a9d89beae2958d3b042876af50443580 |
| readme | a73bfe3b7d23c7605e16717781d94736d003e7b5fc5d08c0ef80a0358367edc0 |

- `images.yaml`: `python scripts/convert_a370_catalogs.py gledhill2024 source .`: 24 systems, 115 positions.
  `1b.2` = system 1, knot B, image b. Not every knot is in every image. System 24's knots have z 4.9153 and 4.916; the
  median is used. CANUCS 7 includes Lagattuta et al. 2019 system 10 (its 7.6 is their 10.2).
- Redshifts: the readme says all 24 systems are spectroscopic (with new values for three, e.g. 8 and 11), but the file
  header says spectroscopic z for systems 1, 2, 3, 4, 6 and Diego / Strait et al. redshifts otherwise, so no kind is
  written; the master takes Lagattuta's MUSE z where it has one.
- `members.yaml`: the 277 galaxies of clugal (placeholder shapes). The band of mag is not stated (written as
  `unstated`); the potfile in param uses mag0 20.701. The BCGs are separate potentials in param, not members.
- Image numbering within a system differs from Lagattuta et al. 2019 (CANUCS 1a.1 is their 1.2).
