# Catalogs

A fit reads two catalogs: the **multiple images** (positions of each lensed source, with its redshift) and the
**cluster members** (positions and magnitudes of the galaxies on the scaling relation). Both are YAML files under
`catalogs/<cluster>/<source>/`, one folder per published source:

```text
catalogs/<cluster>/<source>/
    images.yaml          multiple images (format lenscluster-images/1)
    members.yaml|.cat    cluster members
    README.*             the source's own notes
    source/              the published files the YAML was converted from (Lenstool .par and catalogs)
    history/             earlier versions of the YAML, kept for the record
```

## Available catalogs

| folder | images | members | from |
|---|---|---|---|
| `hera/` | images.yaml | members.yaml | FF-SIMS Hera, with 7-band truth magnitudes |
| `ares/` | images.yaml | members.yaml | FF-SIMS Ares |
| `m0416/canucs2025/` | images.yaml (knots, parities) | members.yaml (DR1 light shapes, Granata 2026 kinematics) | Rihtarsic et al. 2025 |
| `m0416/bergamini2023/` | images.yaml | members.cat | Bergamini et al. 2023 |
| `m0416/diego2024/` | images.yaml, images.ecsv / .fits | | Diego et al. 2024 (PEARLS) |
| `m0416/master/` | images.yaml | members.yaml | our merge of the three M0416 catalogs |
| `as1063/bergamini2019/` | images.yaml | members.cat | Bergamini et al. 2019 |
| `as1063/diego2026/` | images.yaml | members.yaml (Beauchesne 2025) | Diego et al. 2026 |
| `as1063/caminha2016/` | images.yaml | members.yaml | Caminha et al. 2016 |
| `as1063/master/` | images.yaml | members.yaml | our merge of the three AS1063 catalogs |
| `a2744/bergamini2023b/` | images.yaml | | Bergamini et al. 2023 (ApJ 952) |
| `a2744/bergamini2023a/` | images.yaml | members.yaml | Bergamini et al. 2023 (A&A 670) |
| `a2744/furtak2023/` | images.yaml | members.yaml | Furtak et al. 2023 (UNCOVER v2.0) |
| `a2744/mahler2018/` | images.yaml | | Mahler et al. 2018 |
| `a2744/master/` | images.yaml | members.yaml | our merge of the four A2744 catalogs |
| `m0717/limousin2016/` | images.yaml | members.yaml | Limousin et al. 2016 (CATS v4.1) |
| `m0717/kawamata2016/` | images.yaml | | Kawamata et al. 2016 |
| `m0717/master/` | images.yaml | members.yaml | our merge of the two M0717 catalogs |
| `a370/gledhill2024/` | images.yaml | members.yaml | Gledhill et al. 2024 (CANUCS) |
| `a370/lagattuta2019/` | images.yaml | members.yaml | Lagattuta et al. 2019 |
| `a370/master/` | images.yaml | members.yaml | our merge of the two A370 catalogs |
| `m1149/treu2016/` | images.yaml | | Treu et al. 2016 |
| `m1149/grillo2016/` | images.yaml | members.yaml | Grillo et al. 2016 |
| `m1149/canucs/` | | members.yaml | CANUCS DR1 |
| `m1149/master/` | images.yaml | members.yaml | our merge of the M1149 catalogs |

Positions of the real clusters are ICRS RA/Dec in degrees. Hera and Ares positions are offsets about (0, 0) in degrees
(the Lenstool `reference 3 0 0` convention), not sky coordinates. `catalogs/README.md` records the astrometric frames
of each source, and how the `master/` catalogs are built (`scripts/build_master_catalog.py`).

## Multiple images

A **system** is one background source at one redshift. Each image of a system has either one position or a set of
named **knots** (clumps of the same galaxy seen in every image). The knot names are declared once per system.

```yaml
format: lenscluster-images/1
defaults: {shape: [0.3734, 0.3734, 90.0], mag: 25.0}
systems:
  "1":
    z: 3.238
    images:
      a: {ra: 64.0490842, dec: -24.0628619}
      b: {ra: 64.0469592, dec: -24.0607969, sigma: 0.03, parity: -1}
  "D1":
    z: 0.9397
    knots: [A, B]
    images:
      a:
        parity: -1
        knots:
          A: [64.0365743, -24.0670439]
          B: [64.0366543, -24.0672539, 0.02]
      b:
        knots:
          A: {ra: 64.0369043, dec: -24.0674839, bands: {F150W: [25.8, 0.06]}}
```

System
: `z` (required; 0 when unknown, as in Lenstool), `z_err`, `reliability` (0 to 1, default 1), `knots` (list of
  names), `images`, `notes`.

Image
: `ra`/`dec` (degrees) or `knots`; `parity` (-1, 0 or +1; 0 means unknown); `sigma` (astrometric error in arcsec, also
  the default for its knots); `notes`.

Position (an image without knots, or one knot)
: `ra`, `dec`, `sigma`, `shape` (`[a, b, theta]`), `mag`, `mag_err`, `bands` (`{band: [mag, err]}` or
  `{band: {mag, err, detected, use}}`), `prior` (0 to 1, default 1: the prior that the image belongs to its system;
  a knotted image's `prior` is the default for its knots), `notes`. A knot may also be written `[ra, dec]` or `[ra, dec, sigma]`.

`defaults` sets `sigma`, `shape`, `mag`, `mag_err`, `reliability` and `prior` for every entry that omits them. An astrometric
error given nowhere takes `ImageConstraintsConfig.sigma_inst_arcsec`.

The solver sees one **family** per (system, knot). Its id is the system name followed by the knot name (`D1` + `A` =
`D1A`), and each image label is `<family>.<image>`, for example `D1A.b`.

:::{note}
System and knot names use capital letters and digits; image names use lowercase letters. YAML reads an unquoted `021`
as a number and `y`/`n` as booleans, so quote names that could be read that way.
`image_catalog.write_image_catalog` quotes them for you.
:::

## Cluster members

```yaml
band: F160W
kinematics_sources:
  G26: {reference: "Granata et al. 2026, A&A, Appendix C", quantity: sigma_ap, aperture_arcsec: 0.8}
members:
  "2526":
    ra: 64.03816791
    dec: -24.06750808
    mag: {F160W: 17.0166}
    role: bcg
    kinematics: [{source: G26, sigma: 275.6, err: 1.5, sn: 164.6}]
  "4112": {ra: 64.0352, dec: -24.0735, mag: {F160W: 17.95}}
  "5120": {ra: 64.0391, dec: -24.0702, mag: {F160W: 22.31}, z: {value: 0.41, kind: phot}, p_member: 0.7}
```

Top level
: `band` (required: the magnitude the scaling relation uses), `members` (required), `kinematics_sources`, `notes`.

Member
: `ra`/`dec` (required); `mag` (`{band: mag}` or `{band: [mag, err]}`, the `band` entry required); `shape`
  (`[a, b, theta]`, default round); `z` (a number, or `{value, kind}` with kind `spec` or `phot`); `role` (`member` or
  `bcg`); `treatment`; `keep`; `p_member`; `kinematics`; `notes`.

`treatment`
: `relation` (default): the member follows the scaling relation. `independent`: the member gets its own dPIE, added as
  an independent member halo with default priors unless the configuration declares it. `exclude`: kept for the record,
  not modeled.

`keep`
: `true` protects the member from the `MemberConfig.min_deflection_arcsec` impact cut. Members with kinematics are
  always kept.

`p_member`
: The prior probability, in (0, 1], that the galaxy belongs to the cluster, for example $P(\mathrm{member} \mid
  \mathrm{colors})$ from a color classifier. Used only with `MemberConfig(membership="marginalize")`. Without it, a
  member with a spectroscopic `z`, kinematics or role `bcg` gets 1 and the others
  `MemberConfig.membership_default_probability`. A known non-member is `treatment: exclude`, not `p_member: 0`. See
  {doc}`likelihoods` (Uncertain membership).

`kinematics`
: A list of measurements `{source, sigma, err, sn, notes}` with `sigma` and `err` in km/s. A kinematics source gives
  its `reference`, a `quantity` (`sigma_ap` within a circular aperture of `aperture_arcsec`; `sigma_0` within
  $R_e/8$; `sigma_e` within $R_e$), and optionally `aperture_arcsec`. Measurements enter the likelihood only for the
  sources listed in `MemberPopulationConfig.kinematics_sources`, for example `("G26",)`.

Member names are strings, so quote numeric names. A name repeated in `members` is an error.

Lenstool potfiles (`.cat`, columns `id x y a b theta mag lum color` after a `#REFERENCE` line) are read too.
`#REFERENCE 0` means `x`, `y` are RA and Dec in degrees; `#REFERENCE 3` means arcsecond offsets from the model's
reference point.

## Converting published catalogs

| script | converts |
|---|---|
| `scripts/convert_lenstool_catalog.py` | a Lenstool multiple-image catalog to `images.yaml` |
| `scripts/convert_member_catalog.py` | a potfile to `members.yaml`, with kinematics |
| `scripts/convert_canucs_catalog.py` | the CANUCS MACS J0416 release |
| `scripts/convert_as1063_jwst_catalog.py` | the Diego et al. 2026 Abell S1063 table |
| `scripts/add_member_shapes.py` | light shapes from a photometry catalog into `members.yaml` |
| `scripts/group_bergamini_knots.py` | Bergamini images into knotted systems |

Large inputs are not tracked: the Lenstool `bayes.dat` files, the FF-SIMS truth maps and images, and the CANUCS DR1
photometry catalog. See {doc}`scripts` for downloading imaging.

## Master catalogs: tiers and priors

A cluster with several published catalogs has a master catalog, `catalogs/<cluster>/master/`, built by
`scripts/build_master_catalog.py` (layout and matching in `catalogs/README.md`). Nothing a paper lists is dropped:
every system enters `images.yaml` with a **tier** and a `reliability`, every image (or knot) with a `prior`, and every
member enters `members.yaml` with a `p_member`. The values below are priors we chose, not measured rates; they are set
in `sources.yaml` (`reliability:`, `member_prior:`, `member_dz`) with these defaults.

### Image quality

Each image has a quality $q$ between 0 and 1 from its paper's own flag. `quality` in the paper's entry of
`sources.yaml` maps text in the image notes to a score, the first match winning; an image without a matching flag has
$q = 1$. An image several papers list takes the best of their scores. A system's quality $Q$ is the median of its
images' $q$.

| cluster | paper | flag (image notes) | q |
|---|---|---|---|
| M0416 | Diego et al. 2024 | rank B / C / D | 0.7 / 0.4 / 0.2 |
| AS1063 | Diego et al. 2026 | rank B | 0.7 |
| A2744 | Bergamini et al. 2023a, b | QF 1 (tentative z) / QP 3 (MUSE only, diffuse) / QF 9 (single line) / QF 2 (likely z) | 0.5 / 0.6 / 0.7 / 0.85 |
| A2744 | Mahler et al. 2018 | in the copper run only / bronze and copper / silver and below | 0.3 / 0.5 / 0.7 |
| A370 | Lagattuta et al. 2019 | not used in their model (Note 1), copper class, no class / bronze / silver | 0.3 / 0.5 / 0.7 |
| M1149 | Treu et al. 2016 | Rejected / "all" sample only / Silver | 0.1 / 0.4 / 0.7 |

The other papers give no per-image flag ($q = 1$).

### Tiers

With $n$ the number of independent papers that list the system (a paper with `derived_from` reused another's images
and counts with it) and the kind of its redshift (spectroscopic, photometric, model, or free), the first rule that
holds sets the tier:

| tier | rule | reliability $R$ |
|---|---|---|
| diamond | spectroscopic z, $n \ge 2$, $Q \ge 0.8$ | 1.0 |
| gold | spectroscopic z, $Q \ge 0.8$ | 0.9 |
| silver | (spectroscopic z or $n \ge 2$) and $Q \ge 0.5$ | 0.7 |
| bronze | $Q \ge 0.5$ | 0.4 |
| copper | otherwise | 0.15 |

A `set_tier` decision overrides the rule for one system.

### Image priors

The `prior` of an image is the probability that it belongs to its system:

$$p_i = R \times \begin{cases} 1 & q_i \ge 0.8 \\ q_i & q_i < 0.8 \end{cases}$$

so a well-flagged image of a diamond system has $p = 1$, a rank-B image of a spectroscopic system $0.7 \times 0.7$.
An image another paper gives a system that the system's own paper does not list is added to it as a **candidate**,
with $p = 0.5\,R\,q$. A system whose redshift is free in every paper (Lenstool's z = 0) keeps z = 0; its images keep
their priors.

Each system with a redshift also gets `z_err`: 0.01 for a spectroscopic z, $0.05\,(1+z)$ photometric, $0.1\,(1+z)$ for
a model (geometric) z.

### Member probabilities

The first rule that holds sets `p_member`:

| evidence | p_member |
|---|---|
| a spectroscopic z within $0.03\,(1 + z_{cl})$ of the cluster redshift (`z_cluster`), from any paper | 1 |
| a spectroscopic z outside that window (an interloper, kept) | 0.02 |
| role bcg, measured kinematics, or $p_{member} = 1$ stated in a paper's own list (Grillo et al. 2016: spectra) | 1 |
| listed by two or more independent papers (photometric or color selection) | 0.9 |
| listed by one paper | 0.7 |

The master keeps the members of the base paper (or `members_base`) and adds the members only another paper lists
when that paper gives the same band. A `set_member` decision sets one member's role, treatment or `p_member`
(a role bcg then gives 1).

### Using them in a fit

- `ImageConstraintsConfig(trusted_prior_min=0.8)` trusts the images with $p \ge 0.8$: the diamond and gold systems
  without their poorly flagged images. An image below it leaves its family, as do families without a positive
  redshift or left with a single image.
- With a `PartitionConfig` whose `catalog_path` is None, the untrusted images are the uncertain images of the slot
  mixture. An image whose system is trusted puts its prior on that family's slot; the images of an untrusted system
  (two or more of them) share a new slot seeded on them, and each puts its prior on it; the remaining $1 - p$ is
  shared among unrelated and the other slots as for an image without a claim. Free redshifts are sampled.
- `MemberConfig(membership="marginalize")` sums each member's membership over its `p_member`
  (`membership_max_enumerated` members exactly, the rest by their expected mass and its variance).
- The default `trusted_prior_min=0.0` trusts every image with a positive redshift, as for a catalog without priors.
