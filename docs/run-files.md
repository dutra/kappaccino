# Priors and run files

A fit of a catalogued cluster is declared in two YAML files in its catalog folder, next to `images.yaml` and the
member catalog:

- `priors.yaml`: what is fitted. The reference point, cosmology, lens redshift, image catalog, cluster-scale halos,
  member populations and the priors of the member scaling relation.
- `run.yaml`: how it is fitted. The likelihood, search, sampler, member classification, reported products and
  figures, with named profiles layered on top.

{func}`kappaccino.run_files.load_run` reads both into a {class}`~kappaccino.config.LensClusterSolverConfig`. The
`kappaccino` command (or `python -m kappaccino`, {mod}`kappaccino.cli`) runs a folder from the repository root:

```bash
kappaccino catalogs/hera --dry-run
kappaccino catalogs/m0416/canucs2025 --profile fast --profile jwst
kappaccino catalogs/as1063/bergamini2019 --set sampling.warmup=1000 --set runtime.skip_plots=true
```

Each run goes to `results/<name>/<description>-<hash>`, for example `results/hera/w500s250t7_allblocks-3f9a1c`. The
description gives the NUTS warmup, samples and tree depth, the GP blocks (or the likelihood when it is not the GRF),
the image catalog when `--images` replaces it, `uncertain` with uncertain images, and the profiles. The hash is six
characters of a hash of the whole configuration except its paths and WBIC settings, the part the pipeline checks
when it resumes. The same command therefore resumes its run on any day, and any change, a prior set with `--set`
included, starts a new folder. `--run-label NAME` names the folder by hand, and `--results DIR` moves the tree
(`DIR/<name>/...`). The files the run read and the command are copied to the run's `inputs/` folder.
`--images YAML` fits another image catalog in place of the one `priors.yaml` names (a degraded catalog, say).
Images of unknown family are a `partition` section: `--set partition.catalog_path=uncertain.csv`, or
`partition.catalog_path=null` for the untrusted images of the image catalog, with
`partition.redshift_grid=[z_min,z_max,points]` for their redshift quadrature.

The common fields have named options, applied after the profiles and before `--set`: `--chains`, `--seed`,
`--starts`, `--warm-start RUN_DIR`, `--warmup`, `--samples`, `--tree-depth`, `--target-accept`, `--draws` (both
reporting draw counts) and `--skip-plots`. `kappaccino --help` lists them with the field each sets.

`--member-profile power-law` fits the members as cored power laws, the dPIE with a free 3D slope (one sampled slope
per population), and `--member-slope G` holds that slope at G. In `priors.yaml` the same is `profile_type: 102` on a
member population, with `slope` or `slope_prior`.

| folder | name | profiles |
|---|---|---|
| `catalogs/hera` | hera | fast, cosmography |
| `catalogs/ares` | ares | fast, cosmography |
| `catalogs/as1063/bergamini2019` | as1063 | fast, jwst, cosmography |
| `catalogs/as1063/diego2026` | as1063_jwst | fast, hst, cosmography |
| `catalogs/m0416/bergamini2023` | m0416_bergamini2023 | fast, jwst, cosmography |
| `catalogs/m0416/canucs2025` | m0416_canucs2025 | fast, jwst, cosmography |

## Values

A number is fixed. A string of arithmetic is evaluated, with `inf` and, in `priors.yaml`, `arcsec` (the kpc per arcsec
at the lens redshift): `2.3 / 0.72`, `13.5871 * arcsec`. A string that calls a prior function is a prior on the
physical value:

| prior | sampled in |
|---|---|
| `uniform(low, high)` | the value |
| `loguniform(low, high)` | ln of the value |
| `normal(loc, scale)`, `normal(loc, scale) in [low, high]` | the value |
| `student_t(df, loc, scale)` | the value |
| `lognormal(median, width)`, `lognormal(mean=m, std=s)`, either `in [low, high]` | ln of the value |

`width` is the standard deviation of ln of the value; `mean` and `std` are the moments of the untruncated
log-normal. `inf` leaves a side of `in [low, high]` open. Any other string stays a string. A prior's commas and
brackets end a YAML flow mapping (`{...}`), so give a parameter with a prior its own line.

Paths are relative to the file that names them. A file may start with `extends: <file>`: its mappings merge key by key
into that file's, and every other value replaces the base's. `catalogs/m0416/canucs2025/run.yaml` extends the Bergamini
run and changes only its name, priors and candidate catalog. Unknown keys raise.

## priors.yaml

```yaml
reference: {ra0_deg: 64.03821668520617, dec0_deg: -24.06750122335985}
cosmology: {H0: 70.0, Om0: 0.3, Ode0: 0.7}
z_lens: 0.396
images: images.yaml

halos:
  "1":
    x_centre: normal(0.2, 1.5) in [-5.8, 6.2]
    y_centre: normal(0.0, 1.5) in [-6.0, 6.0]
    shape: {ellipticity: [0.2, 0.94], angle: [100.0, 180.0], soft_cap: 0.9}
    core_radius_kpc: loguniform(10.0, 100.0)
    cut_radius_kpc: 2000.0 * arcsec
    v_disp:
      start: 650.0
      prior: lognormal(mean=650.0, std=150.0) in [400.0, 1000.0]
  shear: {profile: shear, preset: free, gamma_upper: 0.3}

members:
  potfile_1:
    catalog_path: members.yaml
    preset: neutral
    prior_mag0: 18.66
    independent:
      "2526":
        id: bcg_2526
        x_centre: fixed
        v_disp: lognormal(mean=250.0, std=50.0) in [100.0, 400.0]

scaling:
  preset: neutral
  alpha_sigma_prior: normal(0.25, 0.05) in [0.1, 0.4]
```

`reference` and `cosmology` take the fields of {class}`~kappaccino.config.ReferenceFrameConfig` and
{class}`~kappaccino.config.CosmologyConfig`. `z_lens` is the redshift of every halo and population.

`halos` are keyed by id. `profile` is `dpie` (default), `nfw`, `shear` or `sheet`. `preset` builds the component with a
constructor of its class, {meth}`~kappaccino.config.DPIEHaloConfig.cluster_scale` or
{meth}`~kappaccino.config.ShearHaloConfig.free`, from the keys it takes; a `cluster_scale` halo takes its softening
from the centre of `scaling.softening_length_prior`. Every other key is a parameter of the component, named as in its
class. A number fixes it. A prior samples it from the prior's centre; `start` and `prior` sample it from `start`.
`e1`, `e2`, `cut_excess_kpc`, `gamma1` and `gamma2` take priors only. `shape` samples the shape uniformly in the
(e1, e2) box of {func}`~kappaccino.priors.ellipticity_box` over the `ellipticity` (Lenstool `ellipticite`) and
`angle` ranges, starting from the middle of the box, below `soft_cap` when one is given, unless `ellipticite` and
`angle_pos` are set.

`members` are keyed by population id: `catalog_path`, an optional `preset` (`bergamini2019` or `neutral`, see
{class}`~kappaccino.config.MemberPopulationConfig`) with the keys it takes, and any other field of the population.
`independent` takes members off the relation, keyed by catalog id; each entry's priors replace the defaults of
{class}`~kappaccino.config.IndependentMemberHaloConfig`, and `fixed` holds a parameter at its start.

`scaling` takes an optional `preset` (`bergamini2019` or `neutral`) and the fields of
{class}`~kappaccino.config.ScalingModelConfig`.

(cosmography)=
### Cosmography

A prior on `Om0` or `w0` under `cosmology` samples that parameter of the flat wCDM cosmology with the lens:

```yaml
cosmology:
  H0: 70.0
  Om0:
    start: 0.3
    prior: uniform(0.05, 0.95)
  w0:
    start: -1.0
    prior: uniform(-2.0, -0.2)
```

The multiple images measure the ratios of $D_{ls}/D_s$ between families at different source redshifts, which depend
on $\Omega_m$ and $w$ and not on $H_0$, so `H0` stays a number. `Ode0` must be absent or null (the cosmology is flat).
At every likelihood evaluation the fit computes $D_{ls}/D_s$ of each source redshift, the kpc per arcsec at the lens
and the NFW critical density from the sampled values; the distances are a 64-point Gauss–Legendre integral that agrees
with astropy to $10^{-11}$ (`tests/test_cosmology.py`). The parameters are `cosmology.Om0` and `cosmology.w0`, in the
block of the cluster-scale halos: the search draws them from their priors and optimizes them with the halos.

Each cluster folder has a `priors_cosmography.yaml` that extends its `priors.yaml` with these priors, and a
`cosmography` profile that selects it: `kappaccino catalogs/hera --profile cosmography` (Hera was simulated with
$\Omega_m = 0.24$, Ares with 0.272, both with $w = -1$).

Three things stay at the start values: the redshift weights of the GRF and of the image-plane GP (they change by
less than 2.5% over these priors), the member classification, and the quantities reported in physical units (masses,
kpc). A family with a photometric redshift sits at its catalog redshift, so cosmography should use spectroscopic
families. Uncertain images (`partition`) are not supported with a sampled cosmology.

## run.yaml

```yaml
name: hera
priors: priors.yaml
likelihood:
  gp_blocks: all
  lambda_b_prior: lognormal(20.0, 1.0) in [0.5, 80.0]
sampling: {warmup: 500, samples: 250, max_tree_depth: 7, target_accept: 0.8}
reporting: {draws: 64, image_draws: 64, extra_images: true}
profiles:
  fast:
    likelihood: {gp_blocks: system, gp_knee_project_modes: []}
    reporting: {draws: 8, image_draws: 8, extra_images: false}
```

`name` sets the figure prefix and the output folder, and `priors` names the priors file. Every other top-level key is a
section of {class}`~kappaccino.config.LensClusterSolverConfig` with its field names; `scaling` here changes what
`priors.yaml` set. `profiles` are overlays of the same sections, applied in the order given on the command line, and
`--set section.field=value` overrides come last. The defaults of every field are the library defaults.

The fast profile of each cluster is exploratory: far-field members down to 2 cut radii, classified at the current
model with 3x headroom on the cut radius, system GP blocks, member shapes fixed at the catalog, tree depth 8, 8
reporting draws and no extra-image search.
