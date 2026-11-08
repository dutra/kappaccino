# Treu et al. 2016: MACS J1149, the SN Refsdal comparison

Treu et al. 2016, ApJ 817, 60 (2016ApJ...817...60T): the multiple images agreed on by the modeling teams for the SN
Refsdal predictions. Tables 3 and 4 downloaded 2026-11-08 from the CfA VizieR mirror (https://vizier.cfa.harvard.edu/viz-bin/votable?-source=<table>&-out.all&-out.max=unlimited, <table> =
J/ApJ/817/60/table3 and table4; the CDS server did not respond).

| file | sha256 |
|---|---|
| `source/J_ApJ_817_60_table3.vot` (97 images) | 788e871a7fcbda0d939c691696160c367bf62adae32472bdef18119db28a52a2 |
| `source/J_ApJ_817_60_table4.vot` (99 knots and SN images) | 00fa63ef2835a1b4dc7531450f5d07a545b7b872603fcb56702524e7075a4968 |

- `images.yaml`: `python scripts/convert_m1149_catalogs.py treu2016 source .`: 36 systems, every row.
  - System 1 (the Refsdal host, z 1.489) from Table 4: `1.12.3` = knot K12, image c, 30 knots; Table 3's system 1
    rows are its knot 1 and are not repeated. SN Refsdal's images S1-S4 are system SN (images a-d).
  - Other systems from Table 3: `2.1` = system 2, image a. Image notes keep Set (Gold, Silver, Rejected, blank: the
    "all" sample) and Score (1 secure, 2 possible, more unreliable).
  - Redshifts: the new zspec, else the earlier spectroscopic z; none: z = 0. System 13's two image redshifts (1.23,
    1.25) give their median; system 12's three (0.93, 1.02, 1.12) disagree, which is why it is Rejected, and the first
    is used. The two Rejected rows without an ID are systems R26 and R27.
