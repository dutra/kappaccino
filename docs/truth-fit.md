# Fitting the truth

For a simulated cluster with a known mass distribution, {mod}`kappaccino.diagnostics.truth_fit` fits the lens model
directly to the **true deflection field**, with no image data. The FF-SIMS clusters Hera and Ares provide such truths.

```text
xonsh run.xsh HERA --truth-fit
xonsh run.xsh ARES --truth-fit --truth-fit-compare results/<campaign>/<image-fit run>
```

## Why fit the truth directly

An image fit can miss the true mass distribution for two different reasons.

1. **The model family is too limited.** No choice of parameters reproduces the cluster, however good the data. Examples
   are halos with the wrong profile shape, members with the wrong profiles, or substructure the model has no component
   for.
2. **The images do not constrain the model well enough.** A better choice of parameters exists, but the multiple
   images do not carry enough information to find that choice.

The two causes need different remedies. A limited model needs a more flexible model. Weak constraints need more data,
or better use of the data. An image fit alone cannot tell the two causes apart.

The truth fit separates the two causes. The truth fit hands the model the best possible data, the true deflection
everywhere in the strong-lensing region, and asks how close the model can get. The residual of the truth fit is the
model's limit. Any extra error in an image fit then comes from the constraints. With an image fit of the same model,
the split is explicit.

$$\alpha_{\rm image\ fit} - \alpha_{\rm true} = (\alpha_{\rm truth\ fit} - \alpha_{\rm true}) + (\alpha_{\rm image\ fit} - \alpha_{\rm truth\ fit})$$

The first term is the model's limit. The second term is what the images did not pin down.

The truth fit also tests the Gaussian random field (GRF) of the image likelihood. The GRF stands in for the mass the
parametric model misses. The truth-fit residual **is** that missed mass, measured directly. So the truth-fit residual
shows whether the GRF prior has the right amplitude ($\sigma_\alpha$), the right correlation scale (the knee
$\lambda_t$) and the right spectral slope ($\gamma$).

## Why the deflection, and not the convergence or the potential

The deflection $\alpha$, the convergence $\kappa$ and the lensing potential $\psi$ carry the same information in
principle, since $\alpha = \nabla\psi$ and $\nabla^2\psi = 2\kappa$. The three choices differ in which scales a fit
on a finite region emphasizes.

- **The convergence** weighs small scales heavily. A fit on $\kappa$ would chase member-scale structure, which the
  image positions barely feel.
- **The potential** weighs large scales heavily, as $\kappa / k^2$. A fit on $\psi$ would be dominated by the mass sheet,
  the outskirts and the map edges. FF-SIMS also provides no potential map, and a potential computed from $\kappa$ on a
  finite map depends on the boundary.
- **The deflection** sits in between, and the image positions respond to the deflection directly, through the lens
  equation $\beta = \theta - \alpha(\theta)$. So the deflection weighs scales the way the data do.

The fit therefore uses the deflection, and reports the convergence residual as a diagnostic.

## The likelihood

The objective is a least-squares sum written as a Gaussian log-likelihood.

$$\ln L(\theta) = -\frac{1}{2\sigma^2} \sum_{p=1}^{N} \left|\alpha_{\rm model}(x_p; \theta) - \alpha_{\rm true}(x_p) - c\right|^2$$

- **The points** $x_p$ form a square grid with a 1" step, inside a circle around the multiple images. The circle's
  radius is 1.5 times the distance of the farthest image from the image centroid. Hera has $N$ = 11,612 points.
- **The deflections** $\alpha_{\rm model}$ and $\alpha_{\rm true}$ have two components each (west and north), for a source
  at the truth's redshift.
- **The constant** $c$ is the mean residual deflection, recomputed and subtracted at every step. A uniform deflection is
  invisible to the images, because the source positions absorb a uniform shift. The image likelihood removes the same
  mode through the translation projection of the GRF.
- **The weight** $\sigma$ is 0.1".

The fit adds the model's priors and minimizes $-(\ln L + \ln p(\theta))$, with the L-BFGS optimizer of the init search.
So the result is a maximum-a-posteriori fit. The priors act as a weak regularizer for directions the truth leaves
undetermined.

### What the objective is not

The objective has the form of a likelihood but is **not a statistical likelihood**, for three reasons.

1. **The truth has no noise.** The weight $\sigma$ is not a measurement error. The value of $\sigma$ only sets how
   strongly the data outweigh the priors.
2. **The grid points are not independent.** The deflection is smooth, so neighboring points carry nearly the same
   information. A $\chi^2$ built from the objective has no statistical meaning, and the curvature at the best fit gives
   no meaningful parameter uncertainties.
3. **The data dominate the soft priors.** With about 10,000 points, the data term outweighs any normal or log-normal
   prior by orders of magnitude. Only hard limits, such as a uniform box on a halo center, can stop the fit. A
   parameter that ends on the edge of a box therefore signals a prior limit, not the model's limit.

So the truth fit answers a geometric question. Which parameters bring the model's deflection closest to the true
deflection, in the root-mean-square sense over the region? The truth fit gives a best fit and residuals, not a
posterior.

### A statistical alternative

A statistically meaningful version would treat the residual as the GRF itself. The residual deflection would follow a
Gaussian field with the knee spectrum, and the GRF parameters ($\sigma_\alpha$, $\lambda_t$, $\gamma$) would be fitted
together with the halo parameters. In Fourier space the GRF covariance is diagonal, so that likelihood is cheap to
compute. The fitted GRF parameters would then come out of the fit directly, instead of from a separate fit to the
residual spectrum. That version depends on the assumed spectral shape, and the shape is part of what the truth fit
tests, so the least-squares version comes first.

## The variants

`--truth-fit-variants` chooses the variants. The default is `relation`.

- **`relation`** keeps every member on the scaling relation (all member offsets held at zero). The cluster-scale halos,
  the shear, the relation's parameters and the softening are free. This variant tests the relation and the halos.
- **`free`** frees the member offsets the image fit can constrain. The members are classified at the `relation` fit by
  the pipeline's own rule (`pipeline.classify`). The exact members, near the images, get free offsets, and the far-field
  members stay on the relation, as `SamplingConfig(far_field_offsets="fixed")` keeps them. The `free` variant starts
  from the `relation` fit.

## What the fit reports

The fit writes `<run>/truth_fit/summary.json` and `parameters.csv` (the start, each variant, and the compared image
fit). Each variant also writes `summary.json`, `image_plane_floor.csv`, `residual_maps.npz`, `residual_maps.pdf` and
`correlation.pdf`.

- **The deflection residual** over the region and at the images (root mean square and maximum), with a map, and the
  **convergence residual** at $z_s = 9$. The convergence residual shows what the model misses. Typical causes are
  member profiles, substructure without a model component, or the shape of the cluster halos.
- **The image-plane floor.** Each observed image goes to the source plane through the fitted model. The offset of each
  image's source from the family's mean source, mapped back by the model's inverse magnification, is the image-plane
  error the best possible model would leave. The mapping is linearized and diverges at critical curves, so the summary
  gives the median and the 84th percentile, not a root mean square. The same calculation through the true deflection
  gives the catalog's own scatter (`truth_catalog_floor`), the baseline no model can beat.
- **The GRF check.** The longitudinal and transverse correlation functions of the residual deflection, compared with
  the knee prior at the prior's median scale. The residual's $\sigma_\alpha$ is compared with the prior median.
- **The comparison** (with `--truth-fit-compare RUN_DIR`, an image fit of the same model). The run's best posterior draw,
  or the init start when no posterior exists, gets the same metrics, plus the error split above.

## Reading the residual

The residual mixes three kinds of error, and the GRF is meant for only one kind.

- **Large-scale misfit of the cluster halos.** A halo with the wrong center or profile shape leaves broad residual
  patches. A GRF prior fitted to such a residual gets a long correlation scale and a high amplitude, which describes the
  halo misfit rather than small-scale structure. A large-scale misfit calls for more halo flexibility, not a stronger
  GRF.
- **Member profile errors.** Members whose profiles are too concentrated, or too diffuse, leave small spots at member
  positions. Those spots add small-scale power that comes from the model, not from missing mass.
- **Missing substructure.** Mass without a model component, such as dark subhalos or line-of-sight structure. This is
  the error the GRF is designed to absorb.

The image likelihood also projects out a uniform convergence and a uniform shear, besides the translation, because the
sources and the external shear absorb those modes. The truth fit removes only the translation, so the truth-fit
residual can hold extra large-scale power the GRF never sees. Comparisons with the GRF prior should project the same
modes out of the residual, and should look at small scales separately from large scales.

## The truth maps

`TruthRecoveryConfig.alpha_true_fits` names the deflection file. FF-SIMS gives two 2048 x 2048 images. Extension 0 is
the deflection toward east and extension 1 the deflection toward north, in arcsec, for a source at the header's `ZS`
(2 for Hera, 1 for Ares). The images share the pixel grid of the convergence map, whose WCS places the pixels. The fit
uses $\alpha_{\rm west} = -$ extension 0 and $\alpha_{\rm north}$ = extension 1, and evaluates the model at the same
`ZS`. Two checks confirm the convention. The catalog images of each family meet in the source plane to 0.15" (median),
where any other orientation gives 11 to 33". Half the divergence of the deflection reproduces the convergence map to
0.6%.

## Checks

A self-test makes a truth from the Hera model itself, at known parameters, and starts the fit from shifted halos and
shear. The fit recovers the halo centers to 0.02", the logarithms of the dispersions to 0.006 and the shear to
0.0004, and leaves a 0.022" residual. That residual comes from interpolating the map at member cores. So a residual on
a real truth belongs to the model, not to the fitting machinery.
