# Lagattuta et al. 2019: Abell 370, MUSE

Lagattuta et al. 2019, MNRAS 485, 3738 (2019MNRAS.485.3738L). Downloaded 2026-11-08.

- `source/J_MNRAS_485_3738_table1.vot` (138 images, 45 systems) and `J_MNRAS_485_3738_footnotes.tsv`, from the CfA VizieR
  mirror (https://vizier.cfa.harvard.edu/viz-bin/votable?-source=J/MNRAS/485/3738/table1&-out.all&-out.max=unlimited; the
  CDS server did not respond). sha256 df69b969e4b605785be00cca89bcd52dda2e758ac23316d2d60f6b96777bcbeb (table1),
  388a6553dcbdceea89f22c4ad9ae8093a5bd99de4180583fa2903d597b0f1909 (footnotes).
- The Lenstool files of their model from the MUSE lensing-cluster atlas (Richard et al. 2021,
  https://cral-perso.univ-lyon1.fr/labo/perso/johan.richard/MUSE_data_release/models/A370/): `galsortcutI_mosaicBright_noBCGsV3.cat`
  (the potfile, sha256 5ba1a2cf21aae0e9dfdd9ae413e049afd63a8d3e8e6763acfad56e51de2345d9), `bestopt.par` (sha256
  31d3ad730101e7237d42a5457d1b1beb32140e7dc37e5b1c1a69380a6f07a1ff) and `mul_a370_specMosaic_Copperv3.cat` (the
  images as Lenstool input, z = 0 free; sha256 01160235a9b9d3d4c4726d1c939fd6810bf8d962dad2291660fcb24c10dd85e7).
- `images.yaml`: `python scripts/convert_a370_catalogs.py lagattuta2019 source .`: 45 systems, every row. `12.3` =
  system 12, image c. z: zspec, else the model zmod (systems 8, 11, 41, 43, 44, 45). Image notes keep Class (gold,
  silver, bronze, copper) and Note (1: not used in their model, 10.1 and 16.2; 2: counter-image of another system).
- `members.yaml`: the 120 members of the potfile (`#REFERENCE 3` offsets converted to RA/Dec; the two BCG rows are
  commented out there, being separate potentials), with their shapes. The band is not stated; the file name and the
  selection suggest F814W, so it is labeled F814W. They match the CANUCS members to 0.04" (103 of 120 within 0.5").
- The MUSE redshift catalogs (their Table C1, the atlas `A370_v1.0.fits`, Lagattuta et al. 2022) are not kept: no member
  flag, and not image or member tables.
