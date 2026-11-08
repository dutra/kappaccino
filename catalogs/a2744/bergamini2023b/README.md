# Bergamini et al. 2023b: GLASS-JWST, Abell 2744

Bergamini et al. 2023, ApJ 952, 84 (2023ApJ...952...84B), Table 2: the multiple images of the core and the northern
(G1-G2, G3) clumps. VizieR J/ApJ/952/84.

- `source/J_ApJ_952_84_table2.vot.xml`: downloaded 2026-11-08 from the CfA VizieR mirror (https://vizier.cfa.harvard.edu/viz-bin/votable?-source=<table>&-out.all&-out.max=unlimited with
  <table> = J/ApJ/952/84/table2; the CDS server did not respond), 149 rows,
  sha256 7818ddd537c085afc42cc3e643b14d825eba685bfb24f2babc39cdb81aac2d47.
- `images.yaml`: `python scripts/convert_a2744_catalogs.py bergamini2023b source/J_ApJ_952_84_table2.vot.xml images.yaml`:
  33 systems, every row. `1.2a` = system 1, knot B, image a; `JD1A` = system JD1, image a.
- Image notes keep Flag, QF (q_zspec: 3 secure, 2 likely, 1 tentative, 9 single line), QP (q_pos: 1 compact HST/JWST
  counterpart to 3 MUSE-only or diffuse) and the location. The meaning of Flag (d, s, \*) is in the CDS ReadMe, which we
  could not fetch; the 90 `d` rows are the images of Bergamini et al. 2023a.
- 10 systems have no spectroscopic z (7, 28, 53, A200, B200, C200, 301, 601, 602, 700): z = 0, free in their model.
  The master catalog takes UNCOVER spectroscopic redshifts for 8 of them.
- The 177 members of this model are not public; `../bergamini2023a/members.yaml` is the closest released list.
- Frame: JWST-registered; the base frame of `../master`. Bergamini 2023a and Furtak are offset by about (-0.04", -0.07").
