# CANUCS DR1: MACS J1149 members

The CANUCS DR1 Lenstool member file of MACS J1149 on MAST
(https://archive.stsci.edu/hlsps/canucs/dr1/macs1149/clu/ and .../model/ for the readme), downloaded 2026-11-08. The
readme says the members follow Schuldt et al. 2024 and that the 30 gold image systems come with v2, so there is no
image catalog yet.

| file | sha256 |
|---|---|
| `source/hlsp_canucs_jwst-hst_multi_macs1149-lenstool-clugal_multi_v1_model.txt` (306 rows) | 5ab16f6f2da6070f19ea913c52e0ec65264a6a561e04a0e23209fc9ecb4b243f |
| `source/hlsp_canucs_jwst-hst_multi_macs1149-lenstool-readme_multi_v1_model.txt` | aef1cfffaf1865a0f8f2927f4df6fe2b3d44ed5e211aa7dc3f57a97d4651cee1 |

- `members.yaml`: `python scripts/convert_m1149_catalogs.py canucs source .`: 306 members, F160W (placeholder shapes).
  IDs follow Grillo et al. 2016 (296 match it within 0.2"); 301-310 are Schuldt's additions. The BCG (166) and G1
  (150) are not in it, being separate halos.
