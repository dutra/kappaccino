# Posterior output reference

This page defines every posterior parameter and derived quantity of a kappaccino fit: its name, unit or coordinate,
and meaning. Which parameters exist depends on the configuration. A field without a prior is fixed and has no column,
and the error-model parameters depend on the likelihood. A parameter that is present but unconstrained by the data
returns its prior; it is not a measurement.

## Access and array conventions

```python
result = run(config)                               # or run(config) again to reload a finished run
posterior, specs = result.posterior, result.state.parameter_specs
names = [spec.name for spec in specs]

v_disp = posterior.physical[:, names.index("halo.v_disp")]          # (draw,) in km/s
p16, median, p84 = np.percentile(v_disp, [16, 50, 84])
```

From the files of a finished run, without rebuilding the model:

```python
import json
from kappaccino.io.hdf5 import load_arrays

root = "results/demo/hera"
saved = load_arrays(f"{root}/final/artifacts/posterior.h5")
names = [spec["name"] for spec in json.load(open(f"{root}/init/artifacts/state.json"))["parameter_specs"]]
physical = saved["physical"]                                          # (draw, parameter), same order as names
```

`final/tables/posterior_summary.csv` holds the same quantities as one row per parameter.

Every array is in parameter order, the order of `state.parameter_specs` and of `names` above. Draws are concatenated
chain by chain; `grouped_samples` keeps the chain axis.

| `posterior.h5` dataset (`PosteriorResults` attribute) | shape | content |
|---|---|---|
| `physical` | (draw, parameter) | Physical values (units below). Sources are physical source positions. |
| `samples` | (draw, parameter) | Latent values: the coordinate NUTS samples (see [Coordinates](#coordinates-and-units)). |
| `grouped_samples` | (chain, draw, parameter) | `samples` by chain. |
| `log_prob` | (draw,) | Log posterior density in NumPyro's unconstrained coordinates (negative potential energy), Jacobians of bounded supports included; for comparing draws, not models. |
| `median_fit` | (parameter,) | Coordinate-wise median of `physical`; not a draw, so not necessarily a consistent model. |
| `maximum_likelihood_fit` | (parameter,) | The draw with the highest `log_prob` (despite the name, the highest posterior density, not a maximum-likelihood fit). The `best_fit` column of `posterior_summary.csv`. |
| `accept_prob`, `diverging`, `num_steps`, `energy` | (draw,) | NUTS acceptance probability, divergence flag, leapfrog steps and Hamiltonian energy of each transition. |
| `step_size` | (chain,) | Step size after warmup. |
| `warmup_steps`, `sample_steps`, `num_chains` | scalar | Sampler settings. |

`posterior_summary.csv` columns: `parameter`, `site` (NumPyro site), `family` (parameter group, below), `field`;
`median`, `p16`, `p84`, `mean`, `std` of `physical`; `latent_median`; `best_fit`; `r_hat` (split R-hat) and `n_eff`
(effective sample size). `chain_convergence.csv` repeats `split_rhat` and `ess` for every parameter.

## Coordinates and units

- **Sky positions** are offsets in arcsec from the reference point $(\alpha_0, \delta_0)$
  (`ReferenceFrameConfig`, Lenstool `reference 3`): $x = (\alpha_0 - \alpha)\cos\delta_0$, positive toward the
  west, and $y = \delta - \delta_0$, positive toward the north.
- **Radii** are in kpc at the lens redshift; **velocity dispersions** in km/s; **masses** in $M_\odot$ (no little $h$).
- **Logarithms**: `log10_*` columns are base 10. The member deviations and their scatters are in **natural-log**
  units ($\delta \ln \sigma$, $\delta \ln r$).
- **Latent values** (`samples`): a parameter whose prior is transformed by `exp` (`dist.LogNormal`, `dist.LogUniform`,
  `truncated_lognormal`, `lognormal_from_moments`) is sampled as the natural log of its physical value; any other
  parameter is sampled as its value. `physical = exp(samples)` for the first kind and `physical = samples` otherwise,
  except for the sources (see [Sources](#sources)).

## Large-scale components

Names are `<id>.<field>`, with `<id>` the component's `id` (`DPIEHaloConfig.id` and the others); family `large`.
"Sampled as" is the coordinate of `samples` under the default priors (`DPIEHaloConfig.cluster_scale`,
`ShearHaloConfig.free`, the `.par` reader); a custom prior sets its own (see [Coordinates](#coordinates-and-units)).

| field | unit (`physical`) | sampled as | definition |
|---|---|---|---|
| `x_centre`, `y_centre` | arcsec | arcsec | Centre of the component, in the offset frame above. |
| `e1`, `e2` | dimensionless | dimensionless | Ellipticity $e(\cos 2\phi, \sin 2\phi)$ with $e = (1 - q)/(1 + q)$, $q$ the axis ratio of the convergence and $\phi$ the position angle (`angle_pos`, degrees). The Lenstool ellipticity is $(1 - q^2)/(1 + q^2)$. |
| `core_radius_kpc` | kpc | ln(kpc) | dPIE core radius $r_\mathrm{core}$. |
| `cut_excess_kpc` | kpc | ln(kpc) | dPIE cut radius above the effective core: $r_\mathrm{cut} = r_\mathrm{core,eff} + $ `cut_excess_kpc`, with $r_\mathrm{core,eff} = \sqrt{r_\mathrm{core}^2 + s^2}$ ($s$ the softening in kpc, zero by default). |
| `v_disp` | km s$^{-1}$ | ln(km s$^{-1}$) | dPIE velocity dispersion $\sigma$ (Lenstool's $\sigma_\mathrm{LT}$). The total mass is $M = \tfrac{3}{2}\pi \sigma^2 r_\mathrm{cut}/G$ in $M_\odot$. |
| `log10_m200` | log10($M_\odot$) | log10($M_\odot$) | NFW mass within $r_{200}$ (200 times the critical density at the lens redshift; no little $h$). |
| `scale_radius_kpc` | kpc | as its prior | NFW scale radius. |
| `gamma1`, `gamma2` | dimensionless | dimensionless | External shear $\gamma(\cos 2\phi, \sin 2\phi)$, $\phi$ in degrees. |
| `kappa` | dimensionless | as its prior | Mass sheet: the convergence for a source at infinite redshift; a family at $z_s$ sees $\kappa\,D_{ls}/D_s$. |

An independent member halo (`IndependentMemberHaloConfig`) has the dPIE fields, with the same units, under its `id`
(default `independent_member_<population>_<catalog id>`).

## Member population

### Scaling relation

Names `<population>.<field>`, family `scaling`. The relation is given at the population's pivot magnitude
`mag0` (with `mag0="auto"`, the pivot the run chose, `state.potfiles[i]["mag0"]`; `prior_mag0` is only where the
priors were written):

$$
\sigma_i = \sigma^\star \left(\frac{L_i}{L^\star}\right)^{\alpha} e^{\delta\ln\sigma_i}, \qquad
r_{\mathrm{cut},i} = r_\mathrm{cut}^\star \left(\frac{L_i}{L^\star}\right)^{\beta} e^{\delta\ln r_i}, \qquad
r_{\mathrm{core},i} = r_\mathrm{core}^\star \left(\frac{L_i}{L^\star}\right)^{\beta} e^{\delta\ln r_i},
$$

with $L_i / L^\star = 10^{-0.4\,(m_i - \mathrm{mag0})}$ (magnitudes in the population's band, AB or as in the catalog).

| field | unit (`physical`) | sampled as | definition |
|---|---|---|---|
| `sigma` | km s$^{-1}$ | ln(km s$^{-1}$) | $\sigma^\star$, the dispersion of a member at the pivot. |
| `cutkpc` | kpc | ln(kpc) | $r_\mathrm{cut}^\star$, the cut radius at the pivot. |
| `corekpc` | kpc | ln(kpc) | $r_\mathrm{core}^\star$, the core radius at the pivot (when `ScalingModelConfig.fit_corekpc`). |
| `alpha_sigma` | dimensionless | dimensionless | $\alpha$, the slope of $\ln\sigma$ in $\ln L$. |
| `beta_radius` | dimensionless | dimensionless | $\beta$, the slope of $\ln r$ in $\ln L$. A constant mass-to-light ratio has $2\alpha + \beta = 1$; mock recovery reports $\gamma_{M/L} = 2\alpha + \beta - 1$ (dimensionless) as `gamma_ml`. |

A field in `ScalingModelConfig.fixed_relation_fields` is held at its nominal value and has no column.

### Deviations from the relation

| name | unit (`physical`) | sampled as | definition |
|---|---|---|---|
| `<population>.velocity_log_scatter` | dimensionless (standard deviation of $\ln\sigma$) | ln of the value | Scatter $s_\sigma$ of the dispersion deviations; a member typically sits within a factor $e^{\pm s_\sigma}$ of the relation. Family `member_velocity_scatter`. |
| `<population>.radius_log_scatter` | dimensionless (standard deviation of $\ln r$) | ln of the value | Scatter $s_r$ of the radius deviations (`member_deviation` `"sigma_rcut"` or `"mass_rcut"`). Family `member_velocity_scatter`. |
| `<population>.<catalog id>.velocity_offset` | dimensionless: unit offset $z$, or $\delta\ln\sigma$ if centered | same as `physical` | The member's dispersion deviation, see below. Family `member_scatter_offset`. |
| `<population>.<catalog id>.radius_offset` | dimensionless: unit offset $z$, or $\delta\ln r$ if centered | same as `physical` | The member's radius deviation, see below. Family `member_scatter_offset`. |
| `<population>.<catalog id>.shape_e1_offset`, `shape_e2_offset` | dimensionless (unit offset $u$) | same as `physical` | The member's ellipticity from its light shape: $e = e_\mathrm{light} + s\,u$, $s$ = `member_shape_scatter` (dimensionless). Family `member_shape_offset`. |

An offset column holds a **unit offset** $z$ for most members, with deviation $\delta \ln \sigma = s_\sigma z$
(likewise $\delta\ln r = s_r z$), so the member's dispersion is $e^{s_\sigma z}$ times the relation's. For members
sampled **centered** (chosen at init, listed in `init/artifacts/classification.json` as `centered_members` and in
`member_scaling_relation.csv` as `centered`), it holds the deviation $\delta\ln\sigma$ (or $\delta\ln r$) itself. The
prior on $z$ is a Student-$t$ with `member_deviation_dof` degrees of freedom (default) or a normal. With
`member_deviation="mass"` one deviation $\delta$ moves the dispersion and the radii together along the relation:
$\delta\ln\sigma = \delta\,\alpha/(2\alpha+\beta)$ and $\delta\ln r = \delta\,\beta/(2\alpha+\beta)$, so that
$\delta\ln M = \delta$. Far-field members' radius deviations do not act on the model when
`radius_deviation_members="exact"` (the default), so their columns return the prior.

The physical properties of each member are in `member_scaling_relation.csv`, each as `_p16`, `_median`, `_p84` over
the draws: `v_disp_*` (km s$^{-1}$), `v_disp_scaling_*` (km s$^{-1}$, the relation's value without the deviation),
`core_radius_kpc_*` and `cut_radius_kpc_*` (kpc), `log10_mass_msun_*` (log10 $M_\odot$, the dPIE total mass), the
offsets (as above); `catalog_mag` (mag), `luminosity_ratio` ($L_i/L^\star$, dimensionless), `x_centre`, `y_centre`
(arcsec), `member_class` (`exact` or `far-field`) and `centered`. `member_diagnostics.csv` adds the deviations against
their prior and population widths, the member's environment and its far-field error (arcsec).

## Error model

Names `images.<field>`, family `error_scale`. Every scale is sampled as ln of its value (default priors).

| field | likelihood | unit (`physical`) | sampled as | definition |
|---|---|---|---|---|
| `sigma_def` | `smoothed` | arcsec (source plane) | ln(arcsec) | Isotropic deflection error per image. |
| `sigma_meas` | every mode | arcsec (image plane) | ln(arcsec) | Independent measurement scatter of every image, the only measurement error. |
| `sigma_sel` | `smoothed` | arcsec (source plane) | ln(arcsec) | Feature-selection scatter, weighted by the image's stretching. |
| `lambda_sel` | `smoothed` | dimensionless (Jacobian eigenvalue) | ln of the value | Stretching scale of that weight: the selection variance is halved where an absolute eigenvalue of the lensing Jacobian equals it. |
| `sigma_shared` | `gp_image_plane` | arcsec (image plane) | ln(arcsec) | Amplitude of the correlated image-plane displacement (the image-plane GP). |
| `gp_ell` | `gp_image_plane` | arcsec | ln(arcsec) | Correlation length of that GP (absent when fixed). |
| `gp_sigma_alpha` | knee kernel | arcsec (deflection) | ln(arcsec) | RMS deflection error per component of the unmodeled-mass field, for a source at `gp_reference_redshift`. |
| `gp_knee` | knee kernel | arcsec | ln(arcsec) | Knee wavelength $\lambda_t = 2\pi/k_t$ of the convergence spectrum (absent when fixed or for a power law). |
| `gp_knee_gamma` | knee kernel | dimensionless | dimensionless | High-$k$ slope $\gamma$ of the convergence spectrum (when sampled). |
| `gp_sigma_alpha2` | knee kernel, two-field spectra | arcsec (deflection) | ln(arcsec) | The second field's RMS deflection error. |
| `gp_knee2` | knee kernel, two-field spectra | arcsec | ln(arcsec) | The second field's knee wavelength. |
| `gp_knee_beta_low` | `broken_power_law` | dimensionless | dimensionless | Low-$k$ slope of the convergence spectrum. |

## Sources

| name | unit (`physical`) | sampled as | definition |
|---|---|---|---|
| `source.<family>.beta_x`, `source.<family>.beta_y` | arcsec (source plane, offset frame) | dimensionless whitened offset | The family's source position. Family `source_position`. |

The source given the lens is marginalized analytically, and the sampled unit offsets map onto its conditional
distribution (`smoothed_likelihood.transport_sources`); `physical` holds the resulting positions. Use `physical` for
source positions; `samples` for these columns is not a position.

## Softening

| name | unit (`physical`) | sampled as | definition |
|---|---|---|---|
| `softening_length.softening_length_kpc` | kpc | ln(kpc) (a log-normal prior) | Resolution softening added in quadrature to every core, when `ScalingModelConfig.softening_length_prior` is set. Family `softening_length`. |

## Derived outputs

These are model evaluations at posterior draws, not new measurements.

| file | content |
|---|---|
| `image_fit_quality.csv` | Per observed image: observed position and model position (arcsec) at the best fit and its 16th, 50th and 84th percentiles over draws, image-plane residual (arcsec), source redshift `z_source`, `image_recovery_status` (`recovered` or `not_recovered`), the family's produced, missing and extra image counts, signed magnification (dimensionless; negative for odd parity) at the observed and at the model position. |
| `posterior_image_residual_draws.csv` | Image-plane residual (arcsec) and signed magnification (dimensionless) per draw and image. |
| `model_magnification.csv` | Signed magnification (dimensionless) per image, at the observed and the model position (arcsec), best fit and percentiles. |
| `image_count_recovery.csv` | Images produced, recovered, missing and extra per family (counts). |
| `fits/` | Simulated clusters, on the truth map's grid: convergence $\kappa$, shear $\gamma$, $\det A$ and magnification $\mu$ maps (all dimensionless, for a source at `TruthRecoveryConfig.caustic_source_redshift`, default $z_s = 9$): posterior median and 16th and 84th percentiles. |
| `conditional_field_maps.*` | Shared-field GRF runs, at the best draw and `reference_redshift`: convergence maps (dimensionless) of the parametric model (`parametric`), the field's conditional mean (`grf`) and their sum (`total`) on the `x`, `y` grid (arcsec); the mean deflection (`mean`, arcsec); and the field's spectrum (`P_kappa_arcsec2` in arcsec$^2$ at `k_arcsec_inverse` in arcsec$^{-1}$, `delta2_kappa` dimensionless). |
| `truth_recovery_m2d_aperture_profile.csv` | Simulated clusters: projected mass within apertures (radius in arcsec) around the truth's peak, as summed convergence of model and truth and their ratio `m2d_ratio` (dimensionless) with its 16th–84th percentiles over draws. |

The derived maps and magnifications of shared-field GRF runs include the conditional mean of the unmodeled-mass field
at each draw; system- and family-block runs do not define one cluster-wide field. See {doc}`outputs` for every file and
figure.
