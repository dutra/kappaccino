# Grillo et al. 2016: MACS J1149, CLASH-VLT + MUSE

Grillo et al. 2016, ApJ 822, 78 (2016ApJ...822...78G). Tables 2 and 6 downloaded 2026-11-08 from the CfA VizieR
mirror (https://vizier.cfa.harvard.edu/viz-bin/votable?-source=<table>&-out.all&-out.max=unlimited, <table> = J/ApJ/822/78/table2 and table6). Tables 1 (the MUSE redshifts) and 3 (the host knots, which
Treu et al. 2016 give with image numbers) are not kept.

| file | sha256 |
|---|---|
| `source/J_ApJ_822_78_table2.vot` (26 images) | d920e2731b46f084c5778be092bd0d0c4bdc8a6ef1e17ac4d06e2a2bb372d05f |
| `source/J_ApJ_822_78_table6.vot` (300 members) | 659f699b91782eaec069a30433b07180112640712e8cc1f2e49e0d0b053d0be6 |

- **The VizieR `_RA` of Table 2 is wrong** (mirrored in RA; 5" off). Positions are rebuilt from the published offsets
  from the BCG (offX positive west, BCG at 11:49:35.699 +22:23:54.71); they then match Treu et al. to 0.03" (0.11"
  at most), all 26 with the same system.
- `images.yaml`: `python scripts/convert_m1149_catalogs.py grillo2016 source .`: 10 systems, T16 numbering, images
  in row order. Systems 6, 7, 8 have no z (free in their model): z = 0.
- `members.yaml`: the 300 members of Table 6, F160W with errors. p_member 1 for the 164 with spectra (Obs M, G or
  both); the BCG (166) has role bcg; Notes (G1, G2, Ref) kept.
