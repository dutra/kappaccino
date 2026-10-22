# Lenscluster Workflow

## Current cluster launcher

`run.xsh` fits Ares, Hera, or AS1063 using `smoothed-local-image-plane` in all
three stages. It samples family source positions, integrates image astrometric
displacements, and includes the conditional, smoothed absolute Jacobian
determinant in the positional likelihood. The launcher supports positions only;
`--magnitude-likelihood on` is rejected. Other likelihood modes remain available
through the Python configuration API.

Validate the configuration and catalogs without creating run outputs:

```bash
/home/dutra/.conda/envs/lenstronomy/bin/python -m xonsh run.xsh HERA --dry-run
```

Launch a fit with a distinct result label:

```bash
/home/dutra/.conda/envs/lenstronomy/bin/python -m xonsh run.xsh HERA \
  --campaign oct21_smoothed_local --run-label position-fit
```

CPU setup follows `build_runtime()` in `run.xsh`. The launcher uses
`RuntimeConfig(chains=cores, nuts_chain_method="parallel", cpu_threads_per_chain=2)`.
Changing `chains` also changes the JAX CPU-device count and the shared CPU budget:
two, four, or six parallel chains request four, eight, or twelve allowed CPUs.
The budget is capped by the process's existing affinity mask, preferring distinct
physical cores before SMT siblings. JAX shares these CPUs across chains; chains
are not individually pinned. BLAS/OpenMP and NumExpr use one thread to avoid
nested parallelism. This setup applies to all stages and is inherited by the
fresh stage-2 process. The resolved CPU plan appears in `--dry-run`, startup
logging, and saved runtime arguments.

Set `cpu_threads_per_chain=None` to retain the existing CPU affinity and library
thread settings. Python callers enabling this option must call
`configure_cpu_runtime(runtime.chains, runtime.nuts_chain_method, runtime.cpu_threads_per_chain)`
from `lenscluster.cpu_runtime` before any JAX backend initialization, including
JAX cosmology calculations. The launcher handles this ordering automatically.
The runner rejects ineffective late changes and does not alter the chain count
when the available CPU allocation is smaller than the requested budget.

`scripts/benchmark_cpu_threads.py` compares CPU budgets in separate processes
against saved smoothed-model artifacts. It checks complete short NUTS trajectories
against a supplied reference and separates compilation from synchronized execution
timing. These measurements do not establish posterior convergence or ESS per second.

The production `nuts.run` timer waits for every chain's final state, samples,
and collected diagnostics. Its elapsed time and saved `nuts_runtime` include
compilation, warmup, and sampling; summary printing and posterior reporting
are timed separately. New runs also save `nuts_runtime_seconds`,
`nuts_runtime_synchronized=True`, and `nuts_runtime_scope` in the sampler's
initialization diagnostics. Earlier saved runtimes are not corrected retroactively.

Stage 0 uses exact member profiles with catalog shapes, sampled sources, and
provisionally absorbed member scatter. It provides the model used to assign
four member classes. Stage 1 uses those fixed classes for SVI; stage 2 uses the
same likelihood and classes for SVI followed by NUTS. Both sampled classes have
one velocity-dispersion offset per member; their core and cut radii remain on
the scaling relations. Absorbed members contribute positional covariance,
determinant variance, and their cross-covariance. The stage-0 absorbed population
does not yet satisfy the final covariance budget and is an initializer only.

The default source prior is a Gaussian centered on the coordinate origin with
100 arcsec standard deviation per coordinate. It remains fixed across stages.
Sources use likelihood-whitened coordinates with the required prior and
change-of-variables correction. The source scatter has prior
`LogNormal(log(0.02 arcsec), 0.7)`; the shared logarithmic velocity-dispersion
scatter has prior `LogNormal(log(0.02), 1.0)`. These are configured priors, not
measurements from galaxy kinematics. NUTS adapts a dense block for lens parameters
and hyperparameters, with diagonal source and member-offset coordinates.

Member scaling uses `scaling_relation_mode="velocity-radius-exponents"` across
all engines. It samples `alpha_sigma` and `beta_radius` directly. Both core and
cut radii scale as luminosity to `beta_radius`; velocity dispersion scales as
luminosity to `alpha_sigma`. The mass-to-light tilt is derived for every joint
draw as `gamma_ml = 2 * alpha_sigma + beta_radius - 1`, with no additional prior.
Configure the radius-exponent prior through:

```python
ScalingModelConfig(
    potfile_beta_radius_prior_mean=0.5,
    potfile_beta_radius_prior_std=0.2,
    potfile_beta_radius_prior_lower=0.0,
    potfile_beta_radius_prior_upper=float("inf"),
)
```

This is a normal distribution truncated below at zero, not a lognormal prior.
The launcher retains its cluster-specific normalization and alpha priors.
Historical alpha/gamma configurations and artifacts cannot be loaded or resumed;
use a fresh run label. Existing result files are left untouched. Synthetic mock
configurations likewise specify `subhalo_beta_radius` and its uniform truth-draw
half-width; gamma is recorded only as a derived truth quantity.

Astrometric errors default to 0.5 arcsec per coordinate and have a 0.5 arcsec
floor. Optional `--image-metadata PATH` reads a CSV keyed by `image_label`, with
`sigma_obs_arcsec` and/or `parity` columns. Parity is `-1`, `+1`, or `0` for
unknown. Missing rows or blank cells use the default error and unknown parity;
invalid supplied values, duplicate labels, and unknown labels are rejected.
Only independently established parities should be supplied. Unknown parity
uses both determinant signs, and no model assigns a parity flag.

Member classification limits summed template deflection errors to
`0.1 sigma_src` and finite changes in `log h` to 0.1. Individual scatter effects
use the paper's `sqrt(0.1)` threshold, with finite changes in `log h` used at
every image. The total absorbed deflection variance must be below
`sigma_src**2 / 4`. Tier membership stays fixed during sampling; the posterior
audit checks its approximation budgets again. Exact profiles are used for
posterior lens-equation checks and lensing maps.

In `run.xsh`, set `MemberTierConfig(top_sampled=K, top_exact=L)` to select
fixed numbers of population members by their worst normalized error impact
over the reference models and images. Sampling ranks scatter effects; exact
evaluation ranks far-field approximation errors. Each option defaults to
`None`, which retains automatic classification for that axis. Zero selects
none, and counts above the catalog size select all. Independent halos are
excluded. Explicit counts are never increased to satisfy budgets: failures
are printed and saved. Members with unusable far-field templates must fit
within `top_exact`, or selection stops. Without enough uncertainty draws,
explicit rankings use the valid initialization reference, while automatic
axes retain their conservative fallback. Saved budgets include the ranking
scores and identities; changing either limit requires a new run label.

After stages 1 and 2, the terminal also compares frozen counts with the smallest
impact-ranked prefixes that satisfy the budgets at the posterior median and
up to 16 deterministic saved draws, using actual posterior scatters. Counts,
identities, reference indices, and measured budgets are saved under
`budget_required_members` in each stage's `member_tier_audit.json`. This does
not change membership or claim a minimum over arbitrary member subsets. The
separate audit of frozen membership still checks every saved draw.

Sampling evaluates exact dPIE members through batched JAX kernels using
JAXtronomy's factored dPIE expression and normalization, and
approximate members through JAX tensor contractions at the current parameters.
The exact kernel computes the Hessian's spatial derivatives analytically;
fixed catalog geometry remains in its own batch. Parameter gradients and
Hessians are still differentiated by JAX.
Only geometric terms are cached. The sampling kernel contracts the total
spatial derivative directly, avoiding a per-member derivative output while
retaining member deflections and Jacobians for the scatter covariance.
Smoothed mode dispatches directly to its own likelihood, skips the legacy
brightness ranking and source-metric cache construction, and rejects legacy
warm-up or surrogate modes. Exact image diagnostics use its sampled physical
sources. Shared JAXtronomy lens kernels and reporting utilities remain in use.
The two sampling rejection guards use a fixed reference with a validated
finite likelihood and gradient. Proposals rejected by the lens or scatter
checks evaluate the physics at that reference before selecting the original
rejection value, keeping those rejected likelihood gradients zero. The
reference is established from each evaluator's latent initialization values;
it does not move the source prior or require refreshes. An invalid reference
stops initialization with an explicit error.

Unknown image parity uses the closed-form two-sided Gaussian absolute
determinant moment, with bounded intermediate ratios and analytic derivatives
to retain small tail gradients. Independently supplied parity keeps the
branch-restricted moment. All-unknown catalogs compile without the one-sided
tail quadrature. Zero-variance limits and determinant-degeneracy flags are
preserved; no variance floor is introduced.

Both stage audits also print the saved best-fit far-field error relative to
exact dPIE. The table reports maximum net and summed member deflection errors,
the finite change in `log h` used by the budget, and the net Jacobian error.
Violations of the configured deflection and `log h` criteria appear in red;
nonfinite errors are failures. Jacobian error is informational because no
separate tolerance is defined. Per-image values are saved in
`best_fit_template_errors` within `member_tier_audit.json`.

Classification uses the initialization reference and 16 deterministic joint SVI
draws. Fewer than 16 distinct finite draws trigger all-exact/all-sampled classes.
Selection caps source scatter at its prior median and floors member scatter at
its prior median. The saved budgets describe that finite reference ensemble,
not the entire prior. Audits check the posterior median and every saved draw,
save failures without changing the classes, and report absorbed magnification
variance separately with a critical-curve flag.

After stage 1, a diagnostic reclassification reports frozen versus proposed
class counts, entries, exits, and transition counts in the terminal. Stage 2
keeps the original membership. The stage-1 artifacts save the proposed classes
in `member_tier_proposal.json`, their reference budgets in
`member_tier_proposal_budgets.json`, and every proposed identity change in
`member_tier_comparison.json`. This check does not enable surrogate refreshes.

Set `posterior_image_diagnostic_draws=0` and `posterior_truth_recovery_draws=0`
to evaluate only the selected best-fit model for these diagnostics. Zero truth
draws override both `truth_grid_mode="median"` and `"posterior"`; the saved
summary identifies `best_fit` mode and reports no posterior spread.

The likelihood uses first-order astrometric and absorbed-member perturbations,
and its per-image marginals neglect residual correlations between images caused
by absorbed members. It conditions on the catalog and does not score absent
images or selection. Numerical tests and short sampler checks do not establish
posterior convergence, parameter-recovery coverage, or scientific calibration.
See [the review notes](paper/smoothed_likelihood_implementation_notes.md) for the
normalization convention requiring manuscript review.

Run manifests fingerprint catalogs and image metadata and record the new
algorithm revision. Incompatible results require a different run label.

Bounded execution checks are available independently of the cluster launcher:

```bash
PYTHONPATH=src /home/dutra/.conda/envs/lenstronomy/bin/python \
  scripts/validate_smoothed_workflow.py --output /tmp/smoothed_synthetic_check
PYTHONPATH=src /home/dutra/.conda/envs/lenstronomy/bin/python \
  scripts/validate_smoothed_likelihood.py --output-dir /tmp/smoothed_hera_check
```

The first runs the three-stage workflow on a synthetic fixture, including the
fresh-process final stage. The second uses two Hera families and eight catalog
members for synchronized timing and short SVI/NUTS checks. Both save reports
that separate execution evidence from scientific validation.

## Other inference modes

The repository also retains the earlier source-plane/image-plane likelihoods
and the one-shot `evidence-ns` path. The following descriptions and command
examples document those earlier interfaces; the current cluster launcher uses
the typed configuration and invocation above.

Use `--fit-method svi` for a fast variational result, `--fit-method svi+nuts` for SVI initialization followed by posterior sampling, or `--fit-method mchmc` / `--fit-method mclmc` for direct BlackJAX microcanonical sampling in the model's latent parameter space. In the current sequential workflow, `--fit-method`, `--warmup`, `--samples`, and `--max-tree-depth` configure the sampled production stages, while `--svi-steps` and `--refresh-every` configure the initializer, backprojected-centroid fit, and optional free-source forward fit. Nested sampling is reserved for the one-shot evidence workflow: use `--fit-mode evidence-ns` with an explicit `--evidence-source-prior-sigma-arcsec`; `--fit-method`, `--svi-steps`, `--warmup`, and `--samples` are ignored in that mode.

For `mchmc` and `mclmc`, `--warmup` controls BlackJAX tuning/adaptation and `--samples` controls production transitions per chain. Prefer `mchmc` when asymptotic Metropolis correction is required. Unadjusted `mclmc` can be faster, but each saved draw is one integration transition and the run should be checked by increasing samples or reducing the tuned step size in a sensitivity run.

## Run

Fast NUTS smoke test:

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name joint_nuts_smoke \
  --fit-method svi+nuts \
  --svi-steps 1000 \
  --warmup 100 \
  --samples 50 \
  --chains 1 \
  --sampling-engine refreshing_surrogate_flat \
  --active-scaling-galaxies 32 \
  --target-accept 0.75 \
  --max-tree-depth 5 \
  --z-bin-efficiency-tol 0.01 \
  --profile-variant original
```

Longer joint run:

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name joint_workflow \
  --fit-method svi+nuts svi+nuts svi \
  --image-plane-mode forward-metric-image-plane \
  --image-plane-newton-steps 0 \
  --warmup 1000 1000 0 \
  --svi-steps 2000 2000 500 \
  --samples 250 250 100 \
  --chains 4 \
  --sampling-engine refreshing_surrogate_flat \
  --active-scaling-galaxies -1 \
  --refresh-param-drift-frac 0.08 \
  --target-accept 0.9 \
  --max-tree-depth 8 8 6 \
  --z-bin-efficiency-tol 0.01 \
  --profile-variant original \
  --plot-caustics
```

Sequential production methods are configured separately for stages 1 and 2.
Use `fit_method=("svi", "svi+nuts")` for an SVI stage 1 followed by SVI and NUTS
in stage 2. Stage 0 always provides the initializer.

One-shot evidence nested sampling example:

```bash
python -m cluster_solver \
  --par-path data/M0416_Bergamini22/Bergamini22_MACS0416.par \
  --output-dir plots/m0416_original \
  --run-name joint_workflow_evidence_ns \
  --fit-mode evidence-ns \
  --evidence-source-prior-sigma-arcsec 20.0 \
  --ns-num-live-points 2000 \
  --ns-max-samples 200000 \
  --ns-dlogz 1e-3 \
  --sampling-engine refreshing_surrogate_flat \
  --active-scaling-galaxies -1
```

For mock runs with known truth values, pass a truth JSON directly to the solver:

```bash
python -m cluster_solver \
  --par-path data/clustersim/input.par \
  --truth data/clustersim/truth.json \
  --output-dir plots/clustersim \
  --run-name joint_workflow
```

When `--truth` is provided, recovery validation PDFs are written under the run's
`validation/` directory after solver artifacts are saved.

## Mock-Cluster Validation

Mock validation is configured from Python dataclasses, matching the main
`lenscluster.config` API. There is no mock-validation CLI. The runner generates
the mock inputs, converts them directly to a `LensModelConfig`, compiles a
`LensClusterSolverConfig` with `compile_run_plan`, runs `LensClusterRunner`, and
writes recovery diagnostics from the final compiled solver stage.

A minimal single-BCG validation run looks like:

```python
from lenscluster.config import (
    ImageDiagnosticsConfig,
    LensClusterSolverConfig,
    RuntimeConfig,
    StageScheduleConfig,
    TruthRecoveryConfig,
)
from lenscluster.mock_validation import (
    MockValidationConfig,
    MockValidationPathsConfig,
    MockValidationRuntimeConfig,
    MockValidationSolverConfig,
    SingleBCGMockConfig,
    run_single_bcg_validation,
)

config = MockValidationConfig(
    mock=SingleBCGMockConfig(
        n_primary_families=20,
        n_subhalo_families=0,
        n_subhalos=50,
        min_images_per_family=3,
        primary_source_redshifts=(1.5, 2.0, 3.0),
        subhalo_source_redshifts=(1.5, 2.0, 3.0),
        pos_sigma_arcsec=0.15,
    ),
    paths=MockValidationPathsConfig(
        output_dir="validation_runs",
        campaign_name="covariance_test",
        run_name="single_bcg_recovery",
        variant_name="anisotropic",
    ),
    runtime=MockValidationRuntimeConfig(realizations=1, seed=12345),
    solver=MockValidationSolverConfig(
        template=LensClusterSolverConfig(
            runtime=RuntimeConfig(skip_plots=True),
            image_diagnostics=ImageDiagnosticsConfig(
                posterior_image_diagnostic_draws=8,
                posterior_image_diagnostic_mode="exact",
            ),
            truth=TruthRecoveryConfig(
                posterior_truth_recovery_draws=128,
                caustic_plot_grid_scale_arcsec=0.2,
            ),
            schedule=StageScheduleConfig(
                fit_method=("svi+nuts",),
                svi_steps=(2000, 2000),
                refresh_every=(250, 250),
                warmup=(300,),
                samples=(500,),
                sampling_refresh_runs=(1,),
                max_tree_depth=(10,),
            ),
        ),
        run_name="fit",
    ),
)

outputs = run_single_bcg_validation(config)
```

The default mock samples primary source families inside the largest tangential
caustic, accepts sources only when they satisfy the configured image-count
bounds, offsets the BCG slightly from the cluster halo, and records mock truth
for the halo, BCG, sources, and optional subhalo population. `max_images_per_family=None`
keeps the upper multiplicity unbounded.

When subhalos are enabled, the mock draws a Natarajan-style count-matched
Schechter cluster-member luminosity function. It samples a parent population,
maps luminosity to the dPIE mass proxy through the configured scaling relation,
applies the faint member limit, and selects the requested number of members.
The generated member catalog is used only when subhalos are requested.

Validation outputs are written to:

```text
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/
```

The default run name and seed produce:

```text
validation_runs/covariance_test/single_bcg_recovery/seed_00012345/anisotropic/
```

Solver outputs are written under the realization directory using the current
compiled stage names:

```text
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/solver/fit/stage0_fast_initializer/
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/solver/fit/stage1_backprojected_centroid_fit/
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/solver/fit/stage2_free_source_forward_fit/  # when stage2_forward_mode != "none"
```

Mock inputs and pre-fit diagnostics are shared by seed:

```text
validation_runs/<campaign-name>/<run-name>/seed_<seed>/mock/
validation_runs/<campaign-name>/<run-name>/seed_<seed>/prefit/
```

Mock-truth recovery PDFs are additionally written at the variant directory level:

- `parameter_recovery.pdf`
- `mass_profile_recovery.pdf`
- `surface_density_recovery.pdf`
- `critical_caustic_recovery.pdf`
- `magnification_recovery.pdf`
- `absolute_magnification_recovery.pdf`
- `image_recovery.pdf`
- `image_residual_histogram.pdf`
- `source_recovery.pdf`
- `subhalo_shmf.pdf`
- `subhalo_recovery_shmf.pdf`
- `subhalo_recovery_radial.pdf`
- `validation_summary.pdf`
- `corner.pdf`
- `potfile_corner.pdf` when potfile scaling parameters are present

The mass-profile validation figures decompose the recovered deflection profile
and annular projected surface density into total, halo, BCG, subhalos, and
BCG+subhalos. Strong-lensing image positions mostly constrain the total
deflection field, so these component-level plots help expose degeneracies.

The posterior artifacts used to make these PDFs are saved under the final
solver stage, for example:

```text
validation_runs/<campaign-name>/<run-name>/seed_<seed>/<variant-name>/solver/fit/stage2_free_source_forward_fit/artifacts/plot_bundle.h5
```

All figures are saved as PDFs. Standard solver diagnostic tables remain under
each stage's `tables/` directory. Set `RuntimeConfig(skip_plots=True)` on the
solver template to suppress standard solver plots; mock-truth recovery PDFs are
still generated.

## Model

The sampled model is assembled from the priors in the Lenstool `.par` file and
any potfile scaling priors. A quantity is free only if it has a decoded prior;
otherwise it remains fixed at the input value.

Free large-scale parameters come from supported potential profiles with priors
in the `.par` file. Supported profiles are dPIE (`81`) and shear (`14`), with
fields such as `x_centre`, `y_centre`, `ellipticite`, `angle_pos`,
`core_radius_kpc`, `cut_radius_kpc`, `v_disp`, and `gamma`.

Member-galaxy subhalos are dPIE components generated from member catalogs. The
scaling hyperparameters are `sigma`, `cutkpc`, `corekpc`, `alpha_sigma`, and
`beta_radius`. Their mean scaling relation is:

```text
sigma_i = sigma_ref * L_i^alpha_sigma
core_i  = core_ref  * L_i^beta_radius
cut_i   = cut_ref   * L_i^beta_radius
```

where `L_i` is the catalog luminosity ratio relative to `mag0`. The derived
mass-to-light tilt is `gamma_ml = 2 * alpha_sigma + beta_radius - 1`.

With `--scaling-scatter`, the model also samples positive scatter
hyperparameters per potfile for the requested fields: `sigma_log_scatter`,
`core_log_scatter`, and/or `cut_log_scatter`. These are marginalized with a
linearized source-plane covariance approximation rather than sampled as
per-member latent offsets. The approximation perturbs each enabled scaling
field around the current model, estimates the first-order source-plane
sensitivity, and adds that variance to the source-plane likelihood.

```text
Cov_extra ~= J_scatter diag(scatter^2) J_scatter^T
```

Positive quantities such as scaling `sigma` and `cutkpc` are sampled in latent
log space and converted back to physical units when building the lens model and
writing outputs.

The source-plane likelihood includes a global positive intrinsic scatter
parameter, `source.sigma_int`, sampled in log space and added isotropically to
the propagated positional covariance.

## Likelihood

For source-plane inference, the lens model ray-shoots each observed image to
`beta_i`. The Gaussian inlier density is

```text
log L_i = -0.5 * [(beta_i - beta_f)^T C_i^-1 (beta_i - beta_f)
                  + log det C_i + 2 log(2 pi)]
```

The existing reliability-weighted outlier mixture and optional Student-t loss
are retained. Reliability does not enter the precision-weighted centroid.
Single-image families remain excluded from the positional likelihood.

The opt-in `source_plane_covariance_mode="full-jacobian"` propagates the observational uncertainty through the full
local Jacobian, `C_i = sigma_obs_i^2 A_i A_i^T + sigma_src^2 I + diag(Vx_i, Vy_i)`.
Family source positions are profiled with precision weights `W_i = C_i^-1`;
residuals and log determinants use the same covariance. Full-flat evaluation
uses current Jacobians; the refreshing surrogate retains its cached Jacobians.
Warm-up retains zero member variance. Source scatter must be positive (the
existing prior spans 0.001–2 arcsec).

The default `source_plane_covariance_mode="magnification"` retains the legacy
determinant-scaled covariance and coordinatewise centroids; `"unit"` also retains
its existing behavior. Only `"full-jacobian"` requires positive source scatter and
zero covariance/gain floors. An unspecified `source_plane_covariance_floor`
resolves to zero for this new source-plane mode and 1e-6 otherwise.

For the legacy source-plane mode, set
`source_plane_covariance_mode="full-jacobian"` in `LikelihoodConfig` and use
`sample_likelihood_mode="source"`. The current launcher instead selects the
smoothed local image-plane likelihood described above.

The sequential workflow contains stage 0 (initializer), stage 1, and optional
stage 2. Set `WorkflowConfig.stage1_likelihood` to `"local-jacobian"`,
`"critical-arc"`, or `"critical-arc-anisotropic"` to choose the stage-1 image-plane
approximation, or `"source"` for source-plane fitting.

Set `WorkflowConfig.stage2_forward_mode` to `"linearized"`, `"critical-arc"`,
`"critical-arc-anisotropic"`, or `"isotropic-latent-displacement"` to enable
stage 2 with sampled family source positions. Set it to `"none"` to finish
at stage 1. The current launcher uses `smoothed-local-image-plane` in every stage.
The anisotropic critical-arc likelihood remains available in stages 1 and 2.

For sequential resumes, `resume="all"` reuses or finalizes completed stages
and may refresh their outputs. `resume="fast"` loads the preceding stages'
artifacts and processes only the final enabled stage. A run directory with
an incompatible manifest requires a different run label; manifests and
saved results are not migrated.

Optional CAB arc-morphology constraints are supplied by `image.arcfile` as an
independent arc catalog, not as image annotations. Rows have the form
`arc_id coord_1 coord_2 z_arc tangent_angle_rad curvature_arcsec_inv
sigma_tangent_angle_rad sigma_curvature_arcsec_inv [reliability]`; `arc_id`
must be unique, and `z_arc` is retained only for provenance diagnostics
(`z_arc >= 0` or `-1`). CAB uses the arc anchor, tangent, and curvature only,
so it never links an arc to an image family. CAB rows use a smoothed local
tangent frame, so constraints automatically fade to an axial-uniform outlier
branch when the local stretch direction or branch choice is ill-defined.

If `--fit-cosmology-flat-wcdm` is enabled, every executed sequential fitting
stage samples the flat wCDM cosmology parameters.

The explicit-beta image-plane modes also include a smooth
observed-image presence penalty by default. For each observed image, the local
image-plane residual is converted to a soft presence probability with
`--image-presence-match-radius-arcsec` and
`--image-presence-temperature-arcsec`; each family then receives a smooth
penalty when the reliability-weighted number of present observed images falls
below the catalog count. The effective default
`--image-presence-penalty-weight` is `2.0` for eligible forward image-plane modes and `0.0`
for evidence, non-image-plane likelihoods, and critical-arc mixture likelihoods;
pass a positive value to enable it. In `critical-arc-mixture-image-plane`, the
presence probability reuses the critical-arc branch probability and counts an
image as present when it is close in the noncritical direction, even if displaced
along the arc. This is a differentiable local surrogate for missing
observed-image anchors, not an exact predicted-image multiplicity count from the
full image solver.

Nearby source redshifts are grouped by fractional lensing-efficiency
`D_ls / D_s` tolerance instead of raw redshift. The default
`--z-bin-efficiency-tol 0.01` keeps each effective source plane within about
1% in lensing strength, which automatically makes bins finer near the cluster
and coarser at high source redshift.

`tiered_members` is the engine for `smoothed-local-image-plane`, used by all
three stages in `run.xsh`. It evaluates the current likelihood with four fixed
member classes: exact/sample, exact/absorb, approximate/sample, and
approximate/absorb. Approximate members use the far-field profiles during
sampling; `top_sampled` and `top_exact` independently control the selected counts.
Set both to `"all"` to sample every population member's velocity offset and use
exact profiles for every member. Either axis can use `"all"` independently;
`None` retains automatic classification. Independent halos keep their standalone
models and are excluded from population counts.
Stage 0 uses exact member profiles and provisional absorbed scatter before
classification. Reported maps, magnifications, and image residuals use exact
profiles. No surrogate refresh is used. The `full_flat` spelling resolves to
`tiered_members` for the smoothed likelihood. This engine alias does not permit
loading historical alpha/gamma runs; artifacts must use the current sampled
beta parameterization. Other likelihood modes retain `full_flat`.

The smoothed engine uses a standalone `TieredMembersEvaluator`. It shares packed
lens physics and exact reporting with the legacy evaluator, while its constructor,
image data, source transformations, and likelihood path are separate. It does not
initialize surrogate models, legacy member selection, or legacy covariance caches.

For those older likelihoods, `full_flat` evaluates both lensing predictions and
first-order member-scatter covariance at the current parameters inside JAX. The covariance and its parameter
derivatives are computed directly; scaling-cache refresh requests are no-ops.
`refreshing_surrogate_flat` retains its reference-based deflection and scatter
caches. Direct covariance evaluation preserves the existing scatter model but
costs more, particularly when computing likelihood gradients.

When `--svi-steps` is larger than `--refresh-every`, SVI is run in blocks for either
engine. Each block's median is checked for physical validity before it becomes
the next starting point; an invalid median retains the last valid center.
Applicable surrogate caches are refreshed at that accepted center and held fixed
during NUTS. Direct member-scatter covariance remains parameter-dependent
during both SVI and NUTS.

For `--fit-mode evidence-ns`, the solver skips the sequential stages and builds
one joint evidence target with analytically marginalized source positions under
the configured Gaussian source prior by default. For correctness checks, pass
`--evidence-likelihood-mode linearized-forward-beta-image-plane` to sample one
source position per family from the same evidence source prior and evaluate the
linearized image-plane residual likelihood. After the full NS run, the saved posterior
table is always 4096 posterior draws produced by `NestedSampler.get_samples(...)`.
Those resampled draws are used for log-probability postprocessing, artifact
writing, plotting, and best-fit selection. This resampling step does not change
the evidence run or the `--ns-max-samples` termination limit. `run_summary.json`
records `ns_posterior_samples=4096` and `ns_posterior_resampling` alongside the
JAXNS evidence diagnostics `ns_log_z_mean` and `ns_log_z_uncert`.

## Outputs

In the `results-v1` layout, every stage saves its figures under
`stages/stageN/figures/`, including the final stage. Final-stage tables and FITS
products are written at the run root. Earlier stages retain their own products.

For a run root `results/<campaign>/<cluster>/<run-label>`, stage outputs are:

- `stages/stage0/` for the initializer.
- `stages/stage1/` for the first fit.
- `stages/stage2/` when `stage2_forward_mode` is enabled.
- `tables/sequential_summary.json` and `summary.txt` for the run summary.

With stage 2 enabled, supply three entries in `svi_steps` and `refresh_every`
for stages 0–2, and two production entries in `fit_method`, `warmup`, `samples`,
`sampling_refresh_runs`, and `max_tree_depth` for stages 1–2. Without stage 2,
supply two initializer entries and one production entry.

Each completed stage writes:

- `artifacts/plot_bundle.h5`
- `artifacts/svi/plot_bundle.h5` for every stage that runs SVI, saved immediately
  after SVI and retained when NUTS follows. This standalone bundle contains the
  guide draws in reported physical units, the guide center, diagnostics, and
  the SVI model's parameter definitions. Load it with the same artifact reader,
  passing `artifacts/svi` instead of `artifacts`. Match parameters by name when
  comparing with the final bundle because member selection can change the model
  between SVI and NUTS. Existing runs are not backfilled. SVI `log_prob` uses
  sampling-parameter coordinates; NUTS potential energies include unconstraining
  Jacobians, so these stored values are not directly comparable.
- `tables/run_summary.json`
- `tables/potential_summary.csv`
- `tables/family_diagnostics.csv`
- `tables/subhalo_properties.csv`
- diagnostic PDFs in the stage directory, including `image_residual_histogram.pdf`;
  potfile subhalo runs also include `subhalo_mass_function.pdf` and
  `subhalo_radial_distribution.pdf`;
  with `--plot-caustics`, this also includes `absolute_magnification.pdf`

The current `run.xsh` constructs a typed configuration directly.

The implementation uses a standard `src/lenscluster/` package layout. The
repository-root `cluster_solver.py` remains as a compatibility shim, so existing
`python -m cluster_solver` commands continue to work.

## HFF Pagul21 Catalogs

Generated HFF master/member/image-family catalogs now default to
`results/hff_master_catalogs`, while raw catalog inputs remain under `data/`:

```bash
python scripts/build_hff_master_catalog.py all
python scripts/build_hff_master_catalog.py plots
```

The `all` command builds the catalogs and plot products. Use `plots` when you
only need to regenerate figures from existing CSV outputs.

Rendered HFF `lenscluster.cluster_solver` parameter folders likewise default to
`results/hff_lenscluster_pars`:

```bash
python scripts/build_hff_lenscluster_pars.py render
```

The Pagul21 HFF Zenodo record can be read and split into cluster-specific
tables with:

```bash
python -m lenscluster.hff_pagul21 --summary-only
python -m lenscluster.hff_pagul21
```

This writes prepared directories under `data/HFF_Pagul21/prepared/<cluster>/`
with photometric catalogs, magnification summaries, an `obs_arcs.cat`, and a
bootstrap `.par` file. By default the bootstrap `.par` includes a Pagul21
photometry-selected BCG candidate plus a `cluster_members_potfile.cat` for
likely cluster members; use `--no-pagul21-members` to omit these. For example:

```bash
python -m lenscluster.cluster_solver \
  --par-path data/HFF_Pagul21/prepared/a2744/a2744_bootstrap.par \
  --fit-mode joint \
  --fit-method svi
```

The upstream `ZSPEC` values in the Pagul21 files are unset. To enrich the
prepared catalogs with coordinate-matched SIMBAD spectroscopic redshifts, run:

```bash
python -m lenscluster.hff_pagul21 --query-simbad-specz --simbad-match-arcsec 1.0
```

This leaves the raw Zenodo files unchanged, writes `simbad_sources.csv` and
match tables into each prepared cluster directory, and uses a matched SIMBAD
redshift in `obs_arcs.cat` only when it is behind the cluster lens and
consistent with the catalog redshift. Add `--no-use-simbad-specz` to write the
match diagnostics without changing the generated arc redshifts.

To also assign heuristic candidate family labels from close SIMBAD spec-z, HST
color agreement, and a strong-lensing-scale angular span, add:

```bash
python -m lenscluster.hff_pagul21 \
  --query-simbad-specz \
  --assign-families
```

This writes `candidate_families.csv` and `candidate_family_members.csv`, then
uses those labels in `obs_arcs.cat`. These are candidate labels for inspection,
not a replacement for a vetted multiple-image catalog. The default maximum
within-family span is 120 arcsec and can be changed with
`--family-max-separation-arcsec`. Candidate-family rows also get a fixed
`family_reliability` value from their redshift offset, color offset, and
angular distance from the family centroid. The source-plane likelihood uses
that value in a reliability-weighted per-image mixture, with a broad outlier
term controlled by `--source-plane-outlier-sigma-arcsec`.

These generated `.par` files are parser/integration bootstrap inputs only. The
Pagul21 catalogs do not include multiple-image family memberships, so each
generated image is a single-image pseudo-family and does not by itself provide
a science-ready strong-lensing constraint. Replace `obs_arcs.cat` with real HFF
multiple-image family labels before using these files for an actual mass fit.

### Cartesian dPIE backend

New runs use `DPIE_NIE_E` from the local lenstronomy and JAXtronomy forks.
It preserves the existing dPIE normalization and radii while evaluating directly
in `e1,e2`, including exact circularity. Spatial Hessians are analytic.
Saved `DPIE_NIE` states resolve to this backend when loaded; saved files are not
rewritten. Both original library profiles remain available for compatibility.
See [validation and timing](docs/benchmarks/cartesian_profiles_20261010.md).

The experimental stage-2 `isotropic-latent-displacement` mode uses explicit
family sources, current full Jacobians, and full direct member-scatter covariance.
It requires `full_flat` and an explicit positive
`latent_displacement_sigma_beta_arcsec`; this is a separate legacy mode from
the current launcher.
See [the mathematical contract and limitations](docs/isotropic_latent_displacement.md).
