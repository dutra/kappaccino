# Catalogs

The image and member catalogs the pipeline reads, one folder per cluster and per published source:

```
catalogs/<cluster>/<source>/
    images.yaml          multiple images (format lenscluster-images/1, src/kappaccino/image_catalog.py)
    members.yaml|.cat    cluster members (src/kappaccino/member_catalog.py)
    README.*             the source's own notes
    source/              the published files the YAML was converted from (Lenstool catalogs, par files, MAST tables)
    history/             earlier versions of the YAML kept for the record (before the parity edit, before the light shapes)
```

| folder | run.xsh entry | images | members | from |
|---|---|---|---|---|
| `hera/` | HERA | images.yaml | members.yaml | FF-SIMS Hera: `source/hera_obs_arcs.cat`, `hera_cluster_members_potfile.cat`, `hera_lenscluster.par`, the 7-band truth magnitudes |
| `ares/` | ARES | images.yaml | members.yaml | FF-SIMS Ares, the same files |
| `m0416/canucs2025/` | M0416_CANUCS2025 (default) | images.yaml (knots, parities) | members.yaml (DR1 light shapes, Granata 2026 kinematics) | Rihtarsic et al. 2025 MAST release: `source/hlsp_canucs_*lenstool-*` and `allmultim`; `image_f200w_photometry.*` is our F200W photometry of the images |
| `m0416/bergamini2023/` | M0416_BERGAMINI2023 | images.yaml | members.cat | Bergamini et al. 2023 (2022 release): `source/obs_arcs.cat`, the par files; members.cat was `CM_cat_MACSJ0416.cat` |
| `m0416/diego2024/` | (comparison only) | images.yaml (every rank), images.ecsv / .fits | | Diego et al. 2024 PEARLS Table B.1, `raw/` and the earlier BUFFALO/Flashlights table; images.yaml by `scripts/convert_diego2024_catalog.py` |
| `m0416/master/` | (not an entry yet) | images.yaml | members.yaml | built from the three folders above, see below |
| `as1063/bergamini2019/` | AS1063 | images.yaml | members.cat | Bergamini et al. 2019 RXC J2248: `source/obs_arcs.dat`, the par files |
| `as1063/diego2026/` | AS1063_JWST | images.yaml | members.yaml (Beauchesne 2025) | Diego et al. 2026 Table 1 (`raw/`), `image_mapping.json`, `validation.json` |
| `as1063/caminha2016/` | (not an entry) | images.yaml | members.yaml (no BCG) | Caminha et al. 2016, model v2.0: `source/A1063_obs_arcs_v11.dat`, `members_95_160_nonan_cuts.cat`, par, region files; systems 50 and 51 have z = 0 (free in Lenstool) |
| `as1063/master/` | (not an entry yet) | images.yaml | members.yaml | built from the three folders above, see below |
| `a2744/bergamini2023b/` | (not an entry yet) | images.yaml (knots; 10 systems free z) | | Bergamini et al. 2023, ApJ 952, 84 (GLASS-JWST) Table 2, VizieR |
| `a2744/bergamini2023a/` | (not an entry) | images.yaml | members.yaml (225, F160W) | Bergamini et al. 2023, A&A 670, A60 Tables A1, B1, VizieR |
| `a2744/furtak2023/` | (not an entry) | images.yaml | members.yaml (419 with F814W) | UNCOVER lens model release v2.0 (Furtak et al. 2023) |
| `a2744/mahler2018/` | (not an entry) | images.yaml | | Mahler et al. 2018, MNRAS 473, 663 Table A1, VizieR |
| `a2744/master/` | (not an entry yet) | images.yaml | members.yaml | built from the four folders above, base bergamini2023b |
| `m0717/limousin2016/` | (not an entry) | images.yaml (41 systems with model z) | members.yaml (90) | CATS v4.1 release of Limousin et al. 2016, MAST |
| `m0717/kawamata2016/` | (not an entry) | images.yaml | | Kawamata et al. 2016, ApJ 819, 114 Table 8, VizieR |
| `m0717/master/` | (not an entry yet) | images.yaml | members.yaml | built from the two folders above, base limousin2016 |
| `a370/gledhill2024/` | (not an entry) | images.yaml (knots) | members.yaml (277, band unstated) | CANUCS DR1 Lenstool files of Gledhill et al. 2024, MAST |
| `a370/lagattuta2019/` | (not an entry) | images.yaml (classes) | members.yaml (120, with shapes) | Lagattuta et al. 2019 Table 1 (VizieR) and their potfile (MUSE atlas) |
| `a370/master/` | (not an entry yet) | images.yaml | members.yaml | built from the two folders above, base gledhill2024 |
| `m1149/treu2016/` | (not an entry) | images.yaml (Refsdal host knots, SN) | | Treu et al. 2016 Tables 3, 4 (VizieR) |
| `m1149/grillo2016/` | (not an entry) | images.yaml | members.yaml (300, spectra flagged) | Grillo et al. 2016 Tables 2, 6 (VizieR; RA rebuilt from offsets) |
| `m1149/canucs/` | (not an entry) | | members.yaml (306) | CANUCS DR1 member file (no image catalog until v2) |
| `m1149/master/` | (not an entry yet) | images.yaml | members.yaml | built from the three folders above, base treu2016 |

Hera and Ares positions are offsets about (0, 0) in degrees (the par files' `reference 3 0 0`), not sky coordinates.
Positions of the real clusters are ICRS RA/Dec; the CANUCS catalog is in the frame of the CANUCS lenstool input,
20 mas from the DR1 photometry, and the Bergamini HST catalogs are 0.2" from the JWST frame
(.agents/m0416-catalog-astrometric-frames.md).

Not here, because of size: `bayes.dat` of the Lenstool runs, the FF-SIMS truth maps and simulated images
(`data/imaging/<cluster>/ffsims/`, scripts/download_imaging.py), the Caminha lensing maps, and the CANUCS DR1 photometry catalog
(`data/CANUCS/M0416_Rihtarsic2025/canucs_dr1/`, also under `/data/sync/GalaxyClusterImaging/RedshiftCatalogs/M0416/`
on Linux). Imaging lives under `data/imaging/<cluster>/<source>` (scripts/download_imaging.py).

Converters: `scripts/convert_lenstool_catalog.py` (Lenstool arcs to images.yaml), `scripts/convert_member_catalog.py`
(potfile to members.yaml, with kinematics), `scripts/convert_canucs_catalog.py`, `scripts/convert_as1063_jwst_catalog.py`,
`scripts/add_member_shapes.py` (DR1 light shapes into members.yaml), `scripts/group_bergamini_knots.py`.

## Adding a published catalog

Catalogs only: multiple images, cluster members and their redshifts, as small text tables. Not lens-model products
(mass, magnification and deflection maps, `bayes.dat`, chains) or imaging. A par file a few kB in size may go in `source/`
when the converter needs its cosmology or reference point.

One folder per paper, `catalogs/<cluster>/<author><year>/`, with the cluster named as in `download_catalogs.py`
(`a2744`, `a370`, `m0416`, `m0717`, `m1149`, `as1063`). The folder is faithful to the paper:

- `source/` holds the published tables as downloaded, unedited.
- `images.yaml` and `members.yaml` are converted from `source/` by a script, keeping the paper's systems, names,
  redshifts and positions. No corrections, renames or rejections here.
- `README.md` gives the bibcode, the URL and date of the download, the converter and its arguments, and the astrometric
  frame (the offset from the cluster's reference frame, when known).

## Master catalogs

A cluster with more than one paper has a `master/` folder, our own catalog built from the paper folders by
`python scripts/build_master_catalog.py catalogs/<cluster>/master`:

```
catalogs/<cluster>/master/
    sources.yaml       input: the papers, the base paper, the match radius, each paper's quality flags, z_cluster
    decisions.yaml     input: hand-made choices (set_z, set_tier, exclude, link, set_member), each with a reason
    images.yaml        output: every system of every paper, each image (or knot) with its association prior
    members.yaml       output: the members, each with p_member
    crossmatch.yaml    output: frame offsets, each system's ids in every paper, tier and quality, conflicts, candidates
```

**Nothing is dropped.** Every system any paper lists is in `images.yaml`, every image with the prior that it belongs to
its system; a doubtful system or image gets a low prior instead of being left out. The one file serves both kinds of
fit: `ImageConstraintsConfig(trusted_prior_min=0.8)` trusts the diamond and gold images, and a `PartitionConfig()`
(no `catalog_path`) puts every other image into the slot mixture, weighted toward its system by its prior.

- **Tiers.** Each system is ranked by the number of independent papers listing it (`derived_from` papers count with
  their source), whether its z is spectroscopic, and its quality, the median over its images of the papers' own flags
  mapped to 0-1 by `quality` in `sources.yaml` (Diego ranks, Bergamini QF/QP, Lagattuta classes, Mahler categories):

  | tier | rule | reliability |
  |---|---|---|
  | diamond | spectroscopic z, two or more independent papers, quality >= 0.8 | 1.0 |
  | gold | spectroscopic z, quality >= 0.8 | 0.9 |
  | silver | spectroscopic z or two independent papers, quality >= 0.5 | 0.7 |
  | bronze | quality >= 0.5 | 0.4 |
  | copper | otherwise (copper class, rank C/D, ...) | 0.15 |

  The system's `reliability` is its tier's; an image's `prior` is the reliability, times its own quality when that is
  below 0.8. Images another paper gives a system that its source paper does not list are added to it as candidates at
  half that prior. The values are priors we chose, set in `reliability:` of `sources.yaml` (defaults in the script).
  `docs/catalogs.md` (Master catalogs: tiers and priors) defines them in full, with each paper's quality flags.
- **Redshifts.** A non-spectroscopic z is replaced by a linked spectroscopic one; two spectroscopic redshifts more than
  `dz_max` apart in |dz|/(1+z) are listed under `conflicts` until a `set_z` decision settles them. A system whose z is
  free everywhere (Lenstool's z = 0) keeps z = 0: a fit leaves it out of the trusted families and the slot mixture
  samples its redshift. `z_err`: 0.01 spectroscopic, 0.05 (1+z) photometric, 0.1 (1+z) model.
- **Members.** `p_member` is 1 with a spectroscopic z within `member_dz` (0.03) (1+z) of `z_cluster` from any paper, a
  bcg role, kinematics, or a p_member of 1 stated in a paper's own list; 0.02 for a spectroscopic interloper (kept,
  not removed); 0.9 when two independent papers list the galaxy; else 0.7. Members only another paper lists are added
  when that paper gives the base band (`member_band` names it for a Lenstool .cat file). Use with
  `MemberConfig(membership="marginalize")`.
- **Matching.** Each paper is shifted onto the base paper's frame by the median offset of its positions from the base's,
  measured by the script and recorded in `crossmatch.yaml` (Bergamini 2023 to CANUCS: 0.23"), then every pair of
  papers is matched mutual-nearest within `match_radius_arcsec`, per (system, knot, image). Systems linked through
  matched images form one group, so renamed, split and merged systems line up (CANUCS K6 and K7 are one Bergamini
  system, `05`, and two Diego sources). A group with a base system enters as the base system; one without, as the
  systems of the paper with the best redshift, renamed with its `prefix`. A `link` decision joins systems that are one
  source when no image matched within the radius (A370 7, 25).
- `members_base` names the paper whose members the master starts from, when the base paper has none (A2744).

Related: `as1063/diego2026/image_mapping.json` (Diego labels to our systems and knots) and
`scripts/match_published_images.py` (published M0416 images compared with the images a fit predicts).
`scripts/build_hff_master_catalog.py` is different: it finds candidate families in the BUFFALO photometry and writes
to `results/`, not here.

## Open issues

- **M0416 K8 redshift (revisit).** MUSE gives z = 1.953, CANUCS NIRISS 2.035 to 2.043 (CANUCS adopts 2.044). The ratio
  of the two, 1.031, matches [OIII] 5007 / Hbeta, which points to a line confusion in the grism, so `m0416/master` uses
  the MUSE value by a `set_z` decision. That choice is provisional: check the MUSE quality flag, the NIRISS spectrum
  (Halpha) or a fit with K8's z free, then update `m0416/master/decisions.yaml`. The evidence is in the comment there.
- **M0717 redshifts.** Only 8 of the 73 master systems have spectroscopic redshifts (diamond); the rest carry model
  redshifts (the CATS best fit, Kawamata's zmodel) and rank silver or bronze. The fixed CATS redshifts of systems 6 and
  68 have no spectroscopic support in Kawamata et al. 2016 (see `m0717/limousin2016/README.md`); check them against
  Limousin et al. 2016, Table A.1.
- **M1149 independence.** Grillo et al. 2016 use the Treu et al. 2016 community list, so no M1149 system is confirmed
  by two independent papers (no diamond tier; 9 gold). The curated Schuldt et al. 2024 list (106 images) is not public
  in machine-readable form and the CANUCS image catalog is due with its v2; add either when available.
