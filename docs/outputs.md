# Outputs

This page lists the files a run writes; {doc}`posterior` defines each posterior parameter and derived quantity.

A run writes everything under its run root, `<output_dir>/<run_name>/` from
{class}`~kappaccino.config.RunPathsConfig`. Figure names start with `RunPathsConfig.figure_prefix`, for example
`m0416_critical_curves.pdf`.

## Directory layout

`init/artifacts/`
: `config.json`, `init_signature.json`, `state.json`, `classification.json`, `search.h5` and `modes.json`,
  `start.json`, `summary.json`, `model_comparison.json` (BIC, see {doc}`model-comparison`).

`init/tables/`
: `classification_report.txt`, `shallow_fold_pairs.json`.

`final/artifacts/`
: `config.json`, `posterior.h5`, `audit.json`, `summary.json`, and with uncertain images `uncertain_images.h5` (see
  {doc}`likelihoods`), `model_comparison.json` (BIC and, with `SamplingConfig.wbic`, WBIC) and with WBIC
  `wbic_draws.h5` (see {doc}`model-comparison`).

Arrays are saved as HDF5 (`.h5`, one dataset per array; read them with `kappaccino.io.hdf5.load_arrays` or `h5py`) and
everything else as JSON or CSV. `config.json` restores the run configuration exactly
(`kappaccino.io.artifacts.load_config`). Nothing is pickled.

Every reported product (figures, tables and results macros) is computed over one set of reporting draws:
`reporting.draws` random posterior draws over all chains, chosen with `runtime.seed` (64 in a full run, 8 in the
fast profile). Each draw is evaluated once and shared by every product. In runs with a shared GRF, each draw carries one
conditional realization of the field: an unconditional draw of the field corrected to agree with the images (Matheron's
rule), with a seed fixed by `runtime.seed` and the draw's posterior row. The median over the draws is close to the mean
field, and the 16–84% range holds the uncertainty of the lens parameters and of the field together. Maps are
per-pixel medians over the draws with their 16th and 84th percentiles, curves are zero contours of median maps with
16–84% bands, and numbers are medians with 16–84% ranges. No product uses the single best sample.

The image solver follows the same rule: where a draw's lens puts the images, their residuals from the catalog, the RMS
and recovered counts, the images the lens adds to a family, and the magnifications at the images are all solved in the
draw's own lens, parametric plus its realization. The RMS of a draw is therefore that of one posterior sample of the
mass distribution, and its range over the draws includes the uncertainty of the field.

This holds for runs with `gp_blocks: all`. A run with `gp_blocks: system` (the fast profile) gave each system its own
field, and the products rebuild one shared field from its draws. That field is not a posterior sample of the lens, so
the residuals, RMS, position contours and map uncertainties of such a run are not trustworthy and likely inflated (on
an M0416 fast run the RMS came out twice what the fitted image-plane scatter allows). The output pass logs a warning
and writes it to `run_summary.json` under `warnings`.

The lens position of an image is not where its catalog entry is expected, because the likelihood lets the two differ by
the image-plane GP and the measurement error. Each draw's lens position therefore also has a predicted observed
position (`x_predicted_arcsec`, `y_predicted_arcsec`): the lens position plus one draw of that scatter, taken from
its prior with the draw's own `sigma_gp`, `ell_gp` and `sigma_meas` and not conditioned on the catalog. The position
contours of the observed images are the 68.3% and 95.4% regions of these positions, so about 68% of the catalog
positions should fall inside their 68.3% contours. `position_coverage.csv` has that check per image (`credible_level`,
`inside_68`, `inside_95`), and the results macros `PositionCoverageOneSigmaPercent` and
`PositionCoverageTwoSigmaPercent` the fractions. The contours of extra images, which have no catalog position, are of
their true positions. The tables also hold a true position per draw (`x_true_arcsec`,
`y_true_arcsec`): the lens position plus the GP displacement drawn given the catalog positions. For a catalog image it
is the catalog position without its measurement error, so its spread is at most `sigma_meas`; where the data do not
constrain `sigma_meas` (no images closer than `ell_gp`) that spread is the prior's.

`final/tables/`
: `posterior_summary.csv`, `member_classification.csv`, `member_scaling_relation.csv`, `image_fit_quality.csv`,
  `image_count_recovery.csv`, `model_magnification.csv`, `posterior_image_residual_draws.csv`,
  `posterior_extra_images.csv`, `position_coverage.csv`,
  `chain_convergence.csv`, `run_summary.json` and `run_summary.txt`, and for simulated
  clusters `truth_recovery_m2d_aperture_profile.csv`. With uncertain membership (`MemberConfig.membership`),
  `member_membership.csv`: one row per member with `treatment` (`certain`, `enumerated` or `marginalized`),
  `p_catalog`, `p_prior` (after any calibration shift, averaged over the draws), and `p_posterior` with
  `p_posterior_std` over the draws. A marginalized member's posterior equals its prior. This table is written even with
  `runtime.skip_plots`. With uncertain images, `uncertain_images.csv` and `uncertain_pairs.csv`.

`final/figures/`
: The figures listed below and `<prefix>_results.tex`.

`final/fits/`
: Convergence, shear, $\det A$ and magnification grids on the truth WCS: the per-pixel median and 16th and 84th
  percentiles over the reporting draws, each draw's GRF conditioned at that draw. Also the overview convergence maps,
  per-pixel medians over the same draws.

`init/logs/init.log`, `final/logs/final.log`
: The stage logs.

## First checks

`final/tables/run_summary.txt`
: Images recovered out of observed, rms and median image-plane residuals, split R-hat, ESS, divergences and run times.

`<prefix>_chain_traces.pdf`
: Traces of the sampler diagnostics, all global parameters and the twelve other parameters with the highest split
  R-hat. `chain_convergence.csv` holds split R-hat and ESS for every parameter.

`final/artifacts/audit.json`
: Whether the far-field approximation held over the posterior.

## Standard figures

| figure | content |
|---|---|
| `image_residual_histogram.pdf` | image-plane residuals |
| `image_recovery.pdf` | observed and model images per family |
| `corner_globals.pdf` | posterior of the global parameters |
| `scaling_relation.pdf` | member scaling relation and deviations |
| `critical_curves.pdf` | critical curves and caustics |
| `image_catalog_family_cluster_mass.pdf` | cluster mass overview on the imaging, with the image catalog |
| `member_deviations.pdf`, `member_constraints.pdf` | per-member deviations and how well the data constrain them |
| `whitened_image_residuals.pdf`, `far_field_accuracy.pdf` | residuals in units of their covariance; far-field error |
| `<population>_scaling_correlations.pdf` | correlations among the relation parameters |

Corner plot axes keep log10 coordinates for log-sampled parameters; the summaries above the histograms report physical
values with asymmetric uncertainties.

## GP and GRF diagnostics

GP runs write, with plotting on:

- `gp_kernel_sky.pdf`: positions and links colored by total displacement correlation;
- `gp_shared_vs_white.pdf`: relative-displacement noise and a joint uncertainty ellipse;
- `gp_scale_posteriors.pdf`: shared and white amplitude posteriors (mas).

Shared-field GRF runs (`gp_blocks="all"`) also write `total_grf_dimensionless_power.pdf` and
`gp_correction_correlation.pdf` (per-pixel medians over the reporting draws, and the median and 16–84% of the
knee spectrum), with the maps in
`final/tables/conditional_field_maps.*`. For these runs the truth maps, magnification recovery, aperture profiles,
critical curves and caustics include each posterior draw's realization of the GRF. System- and family-block runs do
not define one cluster-wide field and skip these figures.

## Simulated clusters

With truth maps (`TruthRecoveryConfig`, set for Hera and Ares), the run also writes:

- `truth_recovery_residuals.pdf` and `mass_profile.pdf` (enclosed projected mass of model and truth, and their ratio);
- `kappa_mass_residuals.pdf`: model over truth for the enclosed mass and the azimuthally averaged convergence, with the
  published CATS and GLAFIC ratios when `meneghetti2017_ratios.csv` sits beside the truth map;
- `magnification_recovery.pdf`: truth against model magnification at the observed images and over the image-position
  convex hull at the truth-map source redshift (9). The orange error bars are spatial 16–84 percentiles, not posterior
  uncertainties;
- `cumulative_source_area.pdf`: source-plane area magnified above μ at the truth-map source redshift,
  A(>μ) = Σ A_pix/|μ| over the truth-map pixels with |μ| > μ, for the truth, the parametric model and the parametric
  + GRF model (each draw's realization of the field), with the comoving volume of the shell z ± 0.5 on the right axis and the
  ratio to the truth below. Lines and bands are the median and 16–84% range over the reporting draws. Thresholds below
  μ = 3 depend on the size of the map and are shaded. Numbers in `cumulative_source_area.csv`;
- `magnification_scatter.csv` (table only): scatter and bias of the model magnification about the truth at the map
  pixels inside the image hull whose true magnification is within 10% of 3 or 10, over all reporting draws, for the
  parametric model and the parametric + GRF model (after Raney et al. 2020). `scripts/make_scatter_summary.py` draws
  these for several clusters as `scatter_summary.pdf`;
- `subhalo_mass_concentration.pdf`: one row of four panels for the members on the true map and the parametric + GRF
  model map (evaluated on the truth-map pixels). Mass is the zeta statistic (Fahlman et al. 1994) in 10 kpc, the mass
  inside 10 kpc above the mean of the 10–20 kpc annulus; concentration is κ̄(<R_e)/κ̄(R_e–2R_e) − 1. Panels: number of
  subhalos per 0.2 dex of mass, model against true mass, concentration against mass, model against true concentration.
  The two brightest members and members whose 16–84% posterior range of mass or concentration exceeds twice the median
  are left out. Tables `member_zeta_masses.csv` (every member, with the parametric-only model too), `zeta_mass_function.csv`,
  `zeta_masses.json` and the cache `zeta_masses_cache.h5` (reused when the draws match; `regenerate_zeta_masses.py
  --recompute` ignores it); `aperture_mass_histogram.json` keeps the 2R_e aperture-mass ratios of Meneghetti et al. (2017), each member's model
  mass the median over the same posterior draws as the truth maps.

## Imaging figures

When imaging is configured and present under `data/imaging/`:

- `small_lambda_covariance_cutouts.pdf`: RGB cutouts, linear covariance ellipses and second-order lens-map contours for
  the twelve images with the smallest Jacobian eigenvalues;
- `fold_pairs_source_caustics.pdf`: model-identified fold pairs (mutual-nearest opposite-parity images within
  `image_diagnostics.fold_pair_max_separation_arcsec`), with posterior critical-curve bands and source panels;
- for M0416 and AS1063, JWST versions with an `_jwst` suffix, including contours of the predicted observed image positions (68.3% solid,
  95.4% dashed).

Missing imaging is reported and skipped.

## Reading results in Python

```python
import json
import pandas as pd
from kappaccino.io.hdf5 import load_arrays

run = "results/demo/hera"
summary = pd.read_csv(f"{run}/final/tables/posterior_summary.csv")
posterior = load_arrays(f"{run}/final/artifacts/posterior.h5")
fit = json.load(open(f"{run}/final/tables/run_summary.json"))
```

## Manuscript macros

Every final run writes `final/figures/<prefix>_results.tex`, and mock aggregates write
`aggregate_plots/mock_results.tex`. Include one with `\input{...}` and use commands such as `\HeraNImages`,
`\HeraNRecovered`, `\HeraRMS`, `\HeraMedianResidual` and `\HeraRuntime`. Residuals are in arcseconds; add the units
in the text. Names contain letters only: M0416 uses `MJZeroFourOneSix`, AS1063 uses `ASOneZeroSixThree`.

`\HeraRMS`, `\HeraMedianResidual`, `\HeraLensOnlyRMS` and `\HeraLensOnlyMedianResidual` are posterior medians with
16–84% ranges over the saved draws (`posterior_image_residual_draws.csv`, and `..._lens_only.csv` for the parametric
lens without the GRF), written for math mode, e.g. `$\HeraRMS''$`. The image counts (`\HeraNRecovered`,
`\HeraNMissed`, `\HeraNExtraImages`, `\HeraRecoveredPercent`) are medians over the reporting draws, with
`...Interval` macros for the 16–84% range. In the image tables an image is recovered when at least half the draws
recover it; its model position, residual and magnification are medians over the draws that do. Extra images are
those found in at least half the draws.

Simulated clusters with a truth map also get the truth recovery within the image footprint: `\HeraFootprintMassRatio`
and `\HeraFootprintKappaRatio` (mean model/true ratio of the enclosed mass and of the azimuthally averaged
convergence, uniform in log R from `\HeraFootprintInnerArcsec` to `\HeraFootprintOuterArcsec`, the outermost
image cut at the edge of the published figure), `\HeraFootprintMassMeanAbsPercent`, `\HeraFootprintKappaMeanAbsPercent`, `\HeraFootprintMassRMSPercent` and
`\HeraFootprintKappaRMSPercent` (mean |model/true − 1| and rms of model/true − 1 over the same radii, in per cent; unlike the mean ratio it
does not let high and low stretches cancel, and it weights the large excursions), and the same with a model name appended (`...RatioCATS`,
`...RatioGLAFIC`, `...RatioJohnsonSharon`, `...RatioZitrinLTMgauss`, `...RatioZitrinNFW`) for the digitized
parametric reconstructions of Meneghetti et al. (2017). Our numbers are medians over the saved posterior draws
(`final/tables/truth_recovery_{m2d_aperture,kappa_annulus}_ratio_draws.csv`, `\HeraFootprintNDraws` of them, the
reporting draws), and each has an `...Interval` macro for math mode with the 16-84 per cent
range, e.g. `$\HeraFootprintMassRMSPercentInterval\%$`, and `...Minus` and `...Plus` macros with the lower and upper error alone.

Simulated clusters with shear truth also get the magnification against the truth:
`\HeraMagnificationScatterMuThreePercent`, `\HeraMagnificationScatterMuTenPercent` and
`\HeraMagnificationBiasMu...Percent` (from `magnification_scatter.csv`, in per cent), and `\HeraSourceAreaRatioMuThree`,
`...MuTen` and `...MuThirty` (median source-plane area magnified above that threshold over the true area, from
`cumulative_source_area.csv`). These use the lens of each draw, parametric plus its realization of the GRF; the same
names with `LensOnly` after the cluster prefix (`\HeraLensOnlyMagnificationScatterMuTenPercent`) are the parametric
lens alone. They are "not available" until the run has those tables.

Regenerate from saved artifacts, without inference or plotting:

```bash
PYTHONPATH=src python -m kappaccino.reporting.results_tex path/to/run/final
```

To regenerate individual figures without rerunning inference, see {doc}`scripts`.
