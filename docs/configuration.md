# Configuration

Every fit is described by one frozen dataclass, {class}`~kappaccino.config.LensClusterSolverConfig`. It groups the
lens model, the likelihood, the search and the sampler, the run paths and the figure options. Build it, call
`validate()`, and pass it to {func}`kappaccino.pipeline.run`. Build it in Python, as on this page, or read the lens
model from a Lenstool parameter file (see {ref}`lenstool-par`). Change a field of a frozen config with
`dataclasses.replace`:

```python
from dataclasses import replace

config = replace(config, sampling=replace(config.sampling, warmup=500, samples=250))
```

| field | class | what it holds |
|---|---|---|
| `model` | `LensModelConfig` | reference frame, cosmology, halos, member populations, image catalog |
| `likelihood` | `LikelihoodConfig` | likelihood mode and its error-scale priors (see {doc}`likelihoods`) |
| `members` | `MemberConfig` | exact versus far-field member classification, uncertain membership |
| `scaling` | `ScalingModelConfig` | the member scaling relation and per-member deviations |
| `search` | `SearchConfig` | the init stage's multi-start MAP search |
| `sampling` | `SamplingConfig` | NUTS |
| `runtime` | `RuntimeConfig` | seed, chains, CPU threads, resume, plotting on/off |
| `paths` | `RunPathsConfig` | output directory, run name, figure prefix |
| `image_diagnostics` | `ImageDiagnosticsConfig` | the exact-image finder and posterior image diagnostics |
| `truth` | `TruthRecoveryConfig` | truth maps of simulated clusters |
| `image_catalog` | `ImageCatalogCutoutConfig` | imaging, RGB display and cutout figures |
| `partition` | `PartitionConfig` or None | uncertain images assigned to sources with the lens (see {doc}`likelihoods`) |

## Lens model

`LensModelConfig` collects the components:

```python
LensModelConfig(
    reference=ReferenceFrameConfig(reference=3, ra0_deg=342.18321, dec0_deg=-44.530878),
    cosmology=CosmologyConfig(H0=70.0, Om0=0.3, Ode0=0.7),
    large_halos=(...),                 # DPIEHaloConfig, NFWHaloConfig, ShearHaloConfig, MassSheetConfig
    member_populations=(...),          # MemberPopulationConfig
    independent_member_halos=(...),    # IndependentMemberHaloConfig
    image_constraints=ImageConstraintsConfig(catalog_path="catalogs/as1063/bergamini2019/images.yaml"),
)
```

Positions are arcsecond offsets from the reference point `(ra0_deg, dec0_deg)`, with the Lenstool `reference 3`
convention.

### Large-scale components

`DPIEHaloConfig`
: A dual pseudo-isothermal elliptical halo: center, `ellipticite` (Lenstool convention,
  $(a^2 - b^2) / (a^2 + b^2)$), `angle_pos`, `core_radius_kpc`, `cut_radius_kpc`, `v_disp` (km/s) and `z_lens`.
  `DPIEHaloConfig.cluster_scale(id, x, y, z_lens, v_disp=...)` gives the standard cluster-halo priors: center uniform in
  a 5" box around the light, shape uniform in the (e1, e2) box of ellipticities up to 0.8, core log-uniform on
  0.5–100 kpc, cut radius about 1500 ± 300 kpc on 500–3000 kpc, and a 35% wide dispersion prior (both log-normal).
  `ellipticity_soft_cap` replaces the hard upper ellipticity bound with a smooth wall, which NUTS handles better.

`NFWHaloConfig`
: An elliptical NFW halo with `log10_m200` (log10 $M_\odot$, no little *h*) and `scale_radius_kpc`.

`ShearHaloConfig`
: External shear. `ShearHaloConfig.free(id, z_lens)` samples (gamma1, gamma2) uniformly in the box of amplitudes up to
  0.3 at any angle.

`MassSheetConfig`
: A uniform convergence `kappa` (for a source at infinite redshift; each family sees $\kappa\, D_{ls}/D_s$). Several
  source redshifts constrain it; one cannot, because of the mass-sheet degeneracy.

The profiles the solver knows are registered in `kappaccino/profiles.py`.

### Priors

Priors are `numpyro.distributions` objects of the physical value of each sampled quantity, as in jaxsedfit. Each
component has a `priors` dict from sampled field to distribution; a field without an entry (or with `None`) is fixed at
its configured value.

```python
import numpyro.distributions as dist
from kappaccino.priors import ellipticity_box, lognormal_from_moments, truncated_lognormal

priors = {
    "x_centre": dist.Uniform(-2.0, 4.0),                      # arcsec
    "y_centre": dist.TruncatedNormal(0.0, 1.5, low=-6.0, high=6.0),
    **ellipticity_box(0.8, lower=0.4, angle_deg=(-42.0, -36.0)),   # "e1", "e2": uniform on their box
    "core_radius_kpc": dist.LogUniform(5.0, 100.0),
    "cut_excess_kpc": lognormal_from_moments(1490.0, 300.0, low=490.0, high=2990.0),
    "v_disp": lognormal_from_moments(1200.0, 200.0, low=1000.0),
}
```

A distribution is sampled in its own base coordinate: one transformed by `exp` (`dist.LogNormal`, `dist.LogUniform`,
and the helpers `truncated_lognormal` and `lognormal_from_moments`) as the log of the value, which suits radii,
dispersions and error scales; any other (`dist.Uniform`, `dist.Normal`, `dist.TruncatedNormal`, `dist.StudentT`) as the
value itself. NumPyro cannot truncate a positive distribution, so `truncated_lognormal(median, log_sigma, low=, high=)`
gives the truncated log-normal.

The sampled fields are the ones the model has: the shape as its Cartesian pair (`e1`, `e2`; `ellipticity_box` turns
Lenstool ellipticity and angle limits into a box), the shear as (`gamma1`, `gamma2`; `shear_box`), and the dPIE cut
radius as `cut_excess_kpc`, its excess above the effective core $\sqrt{r_\mathrm{core}^2 + s^2}$
(`cut_excess_from_moments` gives the prior of a cut radius of a given mean and width). Error scales and GP
hyperparameters take log-normal priors in the same way, e.g. `LikelihoodConfig(gp_knee_prior=truncated_lognormal(6.0, 0.6,
low=2.5, high=8.0))`.

Priors that start JAX (any `exp`-transformed one) need `configure_cpu_runtime` first (see {doc}`quickstart`).

### Member population

Members follow scaling relations in luminosity $L$ relative to a pivot $L^*$ (magnitude `mag0`):

$$
\sigma_i = \sigma^* (L_i/L^*)^{\alpha}, \qquad r_{\mathrm{cut},i} = r_{\mathrm{cut}}^* (L_i/L^*)^{\beta},
$$

and each member deviates from the relation by a sampled offset.

```python
MemberPopulationConfig.bergamini2019("potfile_1", "catalogs/hera/members.yaml", z_lens, prior_mag0=19.82)
```

`bergamini2019` uses the MUSE-calibrated relation of Bergamini et al. (2019): $\sigma^*$ about 200 ± 70 km/s on
40–500 and $r^*_{\mathrm{cut}}$ about 12 ± 10 kpc on 2–60 (`sigma_prior`, `cutkpc_prior`: log-normals with those
moments), given at `prior_mag0`, with the pivot `mag0="auto"` chosen from image leverage. `MemberPopulationConfig.neutral` uses priors that do not come from any cluster's lens model; pair
it with `ScalingModelConfig.neutral()`.

`kinematics_sources` adds measured stellar velocity dispersions to the likelihood, one log-normal term per member,
with `kinematics_aperture_factor` converting the aperture dispersion to the dPIE dispersion and
`kinematics_systematic` added in quadrature to the measurement error.

`IndependentMemberHaloConfig(population_id, catalog_id)` takes a member (usually a BCG) off the relation and gives it
its own dPIE. Members with `treatment: independent` in the catalog are added automatically.

### Scaling relation and deviations

`ScalingModelConfig` sets the slope priors (`alpha_sigma_prior`, `beta_radius_prior`: truncated normals, 0.27 ± 0.05
and 0.65 ± 0.15 by default), the member core prior (`corekpc_prior`) and how members leave the relation:

`member_deviation`
: `"sigma_rcut"` (default): independent dispersion and cut-radius deviations. `"sigma"`: dispersion only. `"mass"`:
  dispersion and radius together along the relation. `"mass_rcut"`: a mass deviation plus an independent radius
  deviation.

`member_velocity_scatter_prior`, `member_radius_scatter_prior`
: Log-normal priors on the scatter about the relation (medians 0.10 and 0.3 in ln).

`member_deviation_tail`
: `"student_t"` (default, `member_deviation_dof=4`) lets a few members sit far from the relation without widening the
  scatter for all; `"normal"` is the Gaussian.

`member_shape_scatter`
: `None` fixes each member's ellipticity at its catalog light shape; a number samples e1, e2 of every exact member
  around the light shape with that prior width.

`fixed_relation_fields`
: Hold any of `sigma`, `cutkpc`, `alpha_sigma`, `beta_radius` at their nominal values, for tests against a published
  relation.

`softening_length_prior`
: A softening length added in quadrature to the cores to match the resolution of an N-body simulation, e.g.
  `dist.LogNormal(np.log(3.19), 0.15)` for Hera; `None` (default) for none.

### Exact and far-field members

Members close to an image are evaluated with the exact elliptical profile ("exact"); members far from every image use
the circular far-field series of the dPIE ("far-field"), which is much cheaper. `MemberConfig` sets the split:

`far_field_cut_radii`
: A far-field member stays at least this many of its sampled cut radii from every image (default 3). During sampling
  the boundary is a smooth log-barrier, not a hard wall.

`cut_radius_inflation`, `classification_prior_quantile`
: Members are classified with an inflated cut radius, so the sampled $r^*_{\mathrm{cut}}$ can grow before a far-field
  member reaches the guard.

`min_deflection_arcsec`
: Drop members whose deflection at every image is below this value at the initial model. `None` (default) keeps every
  member.

`centering`
: Which members sample their deviation directly rather than as a unit offset: `"none"`, `"auto"` (where the data
  constrain the deviation more than the prior does) or `"exact"`. The production fits use `"auto"`.

The posterior audit checks the far-field approximation after sampling (see {doc}`pipeline`).

### Uncertain membership

By default every catalog member counts fully. `membership="marginalize"` gives each member a prior membership
probability and sums the likelihood over membership (see {doc}`likelihoods`, Uncertain membership):

```python
import numpyro.distributions as dist

config = replace(config, members=replace(
    config.members,
    membership="marginalize",
    membership_default_probability=0.9,       # members without p_member or spectroscopic confirmation
    membership_max_enumerated=4,              # exact members summed exactly: 2^4 likelihood evaluations
    membership_calibration_prior=dist.Normal(0.0, 1.0),   # optional log-odds shift of every p_i
))
```

`membership`
: `"fixed"` (default; the catalog `p_member` is ignored) or `"marginalize"`.

`membership_default_probability`
: The prior of a member without `p_member` in the catalog and without spectroscopic confirmation (a `z` of kind
  `spec`, kinematics, or role `bcg`). Default 1, so only members with a catalog `p_member` below 1 are uncertain.

`membership_max_enumerated`
: The most uncertain exact members whose indicators are summed exactly, taken in the order of the member
  classification's score (the ranking `top_exact` uses). The rest enter with their expected mass and a deflection
  covariance. Default 4.

`membership_calibration_prior`
: Prior of a shared shift $\delta$ of the log-odds, $\mathrm{logit}\,p_i' = \mathrm{logit}\,p_i + \delta$, sampled as
  `membership.logit_shift`. `None` (default) keeps the catalog probabilities.

## Search and sampling

`SearchConfig` controls the init stage. The main fields are `n_starts` (prior draws for the large-scale search; 256 in
the production fits), `halo_method` and `member_method` (`"lbfgs"`, `"optax-lbfgs"`, `"adam"` or `"svi"`), `member_starts`, and
`warm_start` (a previous run directory whose saved start replaces the search). See {doc}`pipeline` for how the stages
use them.

`SamplingConfig` controls NUTS: `warmup`, `samples`, `max_tree_depth`, `target_accept`, and the mass matrix.
`mass_matrix="hessian"` (default) seeds a dense block for the global parameters from the Hessian at the start;
`dense_offset_members` chooses which member offsets join that block (`"centered"` by default). `wbic=True` adds a
second NUTS run on the tempered posterior that gives WBIC for model comparison. `wbic_warmup` and `wbic_samples` set
the length of that run (see {doc}`model-comparison`).

`RuntimeConfig` sets `seed`, `chains`, `nuts_chain_method` (`"parallel"`, `"sequential"` or `"vectorized"`),
`cpu_threads_per_chain`, `resume` and `skip_plots`.

## Run paths

```python
RunPathsConfig(output_dir="results/demo/hera", run_name="v1", figure_prefix="hera")
```

The run directory is `<output_dir>/<run_name>/`, and figure file names start with `figure_prefix`. `run_name` must be a
single directory name.

## Recommended settings

The library defaults are conservative. The production fits of the supported clusters change these:

```python
from dataclasses import replace
from kappaccino.config import MemberConfig, SamplingConfig, SearchConfig

config = replace(
    config,
    likelihood=grf_likelihood,                              # see the Likelihoods page
    search=replace(config.search, n_starts=256),
    sampling=SamplingConfig(warmup=250, samples=100, max_tree_depth=8, target_accept=0.8),
    members=replace(config.members, centering="auto"),
)
```

| setting | library default | production fits | quick look |
|---|---|---|---|
| likelihood | `smoothed` | GRF + image-plane GP, `gp_blocks="all"` | GRF + image-plane GP, `gp_blocks="system"` |
| `search.n_starts` | 32 | 256 | 32–256 |
| NUTS warmup + samples | 250 + 200 | 250 + 100 per chain | 150 + 100 |
| `target_accept` | 0.85 | 0.8 | 0.8 |
| `members.centering` | `none` | `auto` | `auto` |
| `members.far_field_cut_radii` | 3 | 3 | 2 (check the audit) |
| `scaling.member_shape_scatter` | `None` (fixed) | `None`, or 0.1 to sample shapes | `None` |

Common changes on an existing configuration:

| to | change |
|---|---|
| add member kinematics | `MemberPopulationConfig(kinematics_sources=("G26",))` |
| sample member shapes | `ScalingModelConfig(member_shape_scatter=0.1)` |
| hold relation parameters fixed | `ScalingModelConfig(fixed_relation_fields=("sigma", "cutkpc"))` |
| drop a large-scale component | rebuild `LensModelConfig.large_halos` without it |
| start from a previous run | `SearchConfig(warm_start="results/demo/v1")` |
| skip figures and tables | `RuntimeConfig(skip_plots=True)` |

(lenstool-par)=
## From a Lenstool `.par` file

{func}`~kappaccino.io.lenstool_par.config_from_lenstool_par` reads a Lenstool parameter file and returns a
`LensClusterSolverConfig` with library defaults outside the lens model:

```python
from kappaccino.io.lenstool_par import config_from_lenstool_par

config = config_from_lenstool_par(
    "model.par",
    image_catalog="images.yaml",       # YAML image catalog (default: the multfile, if it is YAML)
    member_catalog="members.cat",      # default: the potfile's filein, next to the .par file
    output_dir="results",              # run directory: output_dir/run_name
    run_name=None,                     # default: the file name without .par
)
```

What it reads:

| `.par` block | becomes |
|---|---|
| `runmode` `reference` | `ReferenceFrameConfig` |
| `cosmology` / `cosmologie` (`H0`, `omega`, `lambda`, `wX`) | `CosmologyConfig` |
| `potentiel` with `profil 81` | `DPIEHaloConfig`; `core_radius`/`cut_radius` in arcsec converted to kpc at `z_lens`, `*_kpc` radii kept |
| `potentiel` with `profil 14` | `ShearHaloConfig` |
| `limit` | the halo's `priors` |
| `potfile` (`mag0`, `core`/`corekpc`, `sigma`, `cut`/`cutkpc`, `type`, `z_lens`) | `MemberPopulationConfig` |
| `potfile` `vdslope`, `slope` | relation exponents $\alpha = 1/\mathrm{vdslope}$, $\beta = 2/\mathrm{slope}$ |
| `image` `sigposArcsec` | not read: the measurement error of every image is the sampled $\sigma_\mathrm{meas}$ |

Limits become numpyro priors by their Lenstool flag (0 fixed, 1 uniform `lower upper step`, 3 normal `mean std`,
9 truncated normal `mean std lower [upper]`), in the coordinate the solver samples:

- positions as given: `dist.Uniform`, `dist.Normal`, `dist.TruncatedNormal`;
- radii and dispersions in log: a uniform limit becomes `dist.LogUniform`, a (truncated) normal one the log-normal
  with that mean and width;
- the cut radius as its excess above the core (`cut_excess_kpc`);
- `ellipticite` with `angle_pos` (and the shear's `gamma` with `angle_pos`), which need uniform limits together, as the
  uniform (e1, e2) (or (gamma1, gamma2)) box of those limits.

A field without a limit stays fixed, as in Lenstool. A starting value outside its limit is moved to the nearest bound.

Things to know:

- The solver reads multiple images from YAML catalogs only. Convert the `multfile` once with
  `scripts/convert_lenstool_catalog.py`.
- Only dPIE (`profil 81`) and shear (`profil 14`) potentials are read. Add NFW halos or a mass sheet in Python.
- One `potfile` is read. Configure several member populations in Python.
- What the `.par` file fixes stays fixed: relation slopes given with flag 0, and the member core
  (`ScalingModelConfig.fit_corekpc=False`). Free them on the returned configuration if you want them sampled:

```python
config = replace(config, scaling=replace(config.scaling, fixed_relation_fields=(), fit_corekpc=True))
```

- The `.par` file does not set the likelihood, search or sampler. Apply the recommended settings above.
- Reading the file starts JAX (for the distance calculation), so call
  {func}`~kappaccino.cpu_runtime.configure_cpu_runtime` before it.
