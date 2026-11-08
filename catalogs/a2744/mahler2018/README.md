# Mahler et al. 2018: Abell 2744, HST Frontier Fields + MUSE

Mahler et al. 2018, MNRAS 473, 663 (2018MNRAS.473..663M), Table A1. VizieR J/MNRAS/473/663.

- `source/J_MNRAS_473_663_tablea1.vot.xml`: downloaded 2026-11-08 from the CfA VizieR mirror (https://vizier.cfa.harvard.edu/viz-bin/votable?-source=<table>&-out.all&-out.max=unlimited); 185 rows;
  sha256 cb0632092c70f8d7f7e5ddf504f0f859d022ffd1b5f719c5fde3c9159c5d9414. The 514-row MUSE redshift catalog of the
  paper is not kept (no member flag; not an image or member catalog).
- `images.yaml`: `convert_a2744_catalogs.py mahler2018 source/J_MNRAS_473_663_tablea1.vot.xml images.yaml`: 59 systems.
  MUL is a float, `1.2` = system 1, image b; knots are not distinguished. z: the spectroscopic z, else the model zmod
  (notes "geometric z"). Image notes keep confid and Cat (the gold/silver/bronze/copper runs the image is in).
- Frame: (-0.02", -0.10") from Bergamini 2023b.
