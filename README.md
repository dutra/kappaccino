# Lenscluster

Bayesian strong-lensing cluster models with JAX and NumPyro. The model is a sum of
dPIE halos, an optional external shear, and a population of cluster-member dPIE
subhalos that follow luminosity scaling relations with per-member velocity-dispersion
deviations. The likelihood is the smoothed local image-plane likelihood: every observed
image is ray-traced to the source plane, the source-plane residual is compared with a
Gaussian whose covariance combines the intrinsic source scatter and the propagated
astrometric error, and the conditional expectation of the absolute Jacobian determinant
normalizes the density. Members far from every image are evaluated with the circular
far-field series of the dPIE; members close to an image use the exact profile.

## Running a cluster

```bash
/home/dutra/.conda/envs/lenstronomy/bin/xonsh run.xsh HERA --dry-run
/home/dutra/.conda/envs/lenstronomy/bin/xonsh run.xsh HERA --campaign oct22 --run-label positions
```

`run.xsh` holds the model definitions for `ARES`, `HERA` and `AS1063` and builds a
`LensClusterSolverConfig`. The dry run builds the model, reports the provisional member
classification and the search and sampling sizes, and writes nothing. CPU devices follow
`RuntimeConfig(chains, nuts_chain_method, cpu_threads_per_chain)`; the launcher configures
them before JAX initializes.

Every run is one stage, implemented in `lenscluster.pipeline.run`:

1. Build the model state from the typed configuration.
2. Provisional member classification with prior-inflated cut radii at the initial model.
3. Multi-start MAP: L-BFGS (zoom line search) from prior draws of the lens parameters, with
   source offsets and member deviations at zero and scatter hyperparameters at their prior
   medians, run under `vmap`; converged points are clustered into modes.
4. Final classification at the best mode with inflated MAP cut radii.
5. NUTS from the best mode(s), with a dense mass matrix seeded by the Hessian of the global
   block and unit mass for the source and member latent sites.
6. Posterior audit of the far-field approximation, then tables and figures.

A completed run resumes from its artifacts; the saved configuration must match.

## Outputs

For a run root `results/<campaign>/<cluster>/<run-label>`:

- `artifacts/`: `config.json` and `config.pkl`, `state.json`, `classification.json`,
  `search.npz` and `modes.json`, `posterior.npz`, `audit.json`, `summary.json`.
- `tables/`: `posterior_summary.csv`, `member_classification.csv`,
  `member_scaling_relation.csv`, `image_fit_quality.csv`, `image_count_recovery.csv`,
  `model_magnification.csv`, `posterior_image_residual_draws.csv`, `run_summary.json`,
  and for simulated clusters `truth_recovery_m2d_aperture_profile.csv`.
- `figures/`: `image_residual_histogram.pdf`, `image_recovery.pdf`, `corner_globals.pdf`,
  `scaling_relation.pdf`, `critical_curves.pdf`, `image_catalog_family_cluster_mass.pdf`,
  and for simulated clusters `truth_recovery_residuals.pdf` and
  `truth_recovery_m2d_aperture_ratio.pdf`.
- `fits/`: convergence, shear, det A and magnification grids of the posterior median and
  16/84 percentiles on the truth WCS, and the overview convergence map.
- `logs/run.log`.

Figure names carry the cluster prefix from `RunPathsConfig.figure_prefix`.

## Mock validation

```bash
/home/dutra/.conda/envs/lenstronomy/bin/xonsh run_mock.xsh --campaign oct22_mock --preset smoke --realizations 2
/home/dutra/.conda/envs/lenstronomy/bin/xonsh run_mock.xsh --campaign oct22_mock --preset production --seed 12340 --realizations 40
```

Each realization generates a single-BCG mock cluster with a Schechter population of
subhalos (`lenscluster.mock_validation.generation`), runs the single stage on it, and
writes `seed_*/recovery/recovery.json` with parameter, magnification, image-residual,
subhalo-mass and aperture-mass comparisons against the truth. `--aggregate-only` (or
`python -m lenscluster.mock_validation.aggregate_plots <campaign dir>`) renders the five
paper figures under `aggregate_plots/`: parameter recovery, subhalo mass function,
magnification recovery, aperture-mass components of the representative seed, and the
image-residual distribution.

## Package layout

- `config.py`: typed configuration dataclasses and validation.
- `inputs.py`, `lenstool_parser.py`: catalogs and Lenstool-style potential definitions.
- `state.py`, `model.py`: parameter specifications, packed lens state, transforms.
- `evaluator.py`, `farfield.py`, `smoothed_likelihood.py`, `jax_lensing.py`: the likelihood.
- `classification.py`, `audit.py`: exact versus far-field members and the posterior audit.
- `search.py`, `sampling.py`: multi-start MAP and NUTS.
- `pipeline.py`, `artifacts.py`, `outputs.py`: the stage, its files and its products.
- `diagnostics.py`, `exact_images.py`: exact lens-equation solutions and recovery tables.
- `truth_recovery.py`, `plots.py`, `mass_figure.py`, `cutouts.py`, `rgb.py`: figures.
- `mock_validation/`: mock generator, campaign runner and aggregate figures.
- `image_tools/`: photometry helpers for the FF-sims truth magnitudes.

## Data preparation scripts

- `scripts/build_ff_sims_lenscluster_inputs.py`: FF-sims (Ares, Hera) catalogs.
- `scripts/build_hff_master_catalog.py`, `scripts/build_hff_lenscluster_pars.py`: HFF inputs.
- `scripts/make_ff_sims_truth_magnitudes.py`, `scripts/measure_ff_sims_image_magnitudes.py`.
- `scripts/make_dpie_figure.py`, `scripts/make_covariance_figure.py`: paper figures.
- `download_catalogs.py`: catalog and image downloads.

## Tests

```bash
/home/dutra/.conda/envs/lenstronomy/bin/python -m pytest tests
```

`tests/test_pipeline.py` runs the whole stage on a synthetic cluster.
