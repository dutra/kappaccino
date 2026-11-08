# Bergamini et al. 2023a: Abell 2744 core, HST + MUSE

Bergamini et al. 2023, A&A 670, A60 (2023A&A...670A..60B, arXiv:2207.09416). VizieR J/A+A/670/A60.

- `source/J_AA_670_A60_tablea1.vot.xml`: the 90 multiple images, all spectroscopic (z 1.688 to 5.726), VizieR's
  "corrected version"; sha256 6f1d7bc666ce8cdb4a1b85c66d3dd04d7c2e0182b5492c0080e8edf9333a1c61.
- `source/J_AA_670_A60_tableb1.vot.xml`: the 225 core members (202 spectroscopic, 23 photometric), F160W;
  sha256 5729a4b37f69d68342af40be5b087392c807488094fd1afb8c081bc0e6cab6eb.
- Both downloaded 2026-11-08 from the CfA VizieR mirror (https://vizier.cfa.harvard.edu/viz-bin/votable?-source=<table>&-out.all&-out.max=unlimited); the CDS server did not respond.
- `images.yaml`: `convert_a2744_catalogs.py bergamini2023a source/J_AA_670_A60_tablea1.vot.xml images.yaml`: 16 systems,
  90 positions. Image notes keep the n_ID flag, QF and QP.
- `members.yaml`: `convert_a2744_catalogs.py bergamini2023a source/J_AA_670_A60_tableb1.vot.xml members.yaml --members`:
  225 members, band F160W, z with kind spec or phot. No BCG role or shapes are published in the table.
- Frame: HST, (-0.037", -0.076") from Bergamini 2023b. The same images are the `d` rows of Bergamini 2023b.
