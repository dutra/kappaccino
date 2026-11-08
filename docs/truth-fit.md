# Fitting the truth

For a simulated cluster whose true mass is known (the FF-SIMS clusters Hera and Ares), {mod}`kappaccino.diagnostics.truth_fit`
fits the lens model directly to the **true deflection field**, with no image data. The result is the best the model
can do: what is left over is the part of the cluster the model cannot represent, whatever the constraints.

```text
xonsh run.xsh HERA --truth-fit
xonsh run.xsh ARES --truth-fit --truth-fit-compare results/<campaign>/<image-fit run>
```

## The fit

The model is the cluster's run entry unchanged: the same halos, members, scaling relation, shear, softening and
priors as the image fit. It is fitted by least squares to the true deflection on a 1" grid inside a circle around the
multiple images (1.5 times the distance of the farthest image from their centroid):

$$\ln L(\theta) = -\frac{1}{2\sigma^2} \sum_p \left|\alpha_{\rm model}(x_p; \theta) - \alpha_{\rm true}(x_p) - c\right|^2,$$

with the constant deflection $c$ profiled out (the images cannot see it: the sources absorb it, as the image
likelihood's translation projection does). The truth has no noise, so $\sigma$ (0.1") only weighs the data against the
priors, which stay as a weak regularizer. The deflection is the quantity the images respond to; the convergence is
reported as well. L-BFGS on the same objective as the init search, two variants:

- `relation`: the members on their scaling relation; tests the relation and the cluster-scale halos;
- `free`: the member offsets free, as in the image fit; tests the model's full flexibility.

A self-test (a truth made by the model itself, the fit started from shifted halos and shear) recovers the parameters
and leaves a 0.02" rms residual, from interpolating the map at member cores.

## What it reports

`<run>/truth_fit/summary.json`, `parameters.csv` (start, each variant, and the compared image fit) and, per variant,
`summary.json`, `image_plane_floor.csv`, `residual_maps.npz`, `residual_maps.pdf` and `correlation.pdf`:

- **Deflection residual** over the region and at the images (rms, max), and its map; the **convergence residual**
  at $z_s = 9$, which shows what is missing: member halos, substructure, or the shape of the cluster halos.
- **Image-plane floor.** The observed images are taken to the source plane through the fitted model; each image's
  offset from its family's mean source, mapped back by the model's inverse magnification (linearized), is the
  image-plane error the best possible model would leave. The same through the true deflection is the catalog's own
  scatter (`truth_catalog_floor`), the baseline no model can beat.
- **The GRF check.** The residual deflection's longitudinal and transverse correlation functions against the knee
  prior's at its median scale, and its rms against the prior median of $\sigma_\alpha$: is the unmodeled mass as large,
  and on the scales, that the GRF prior assumes?
- **With `--truth-fit-compare RUN_DIR`** (an image fit of the same model: its best posterior draw, or its init start),
  the same metrics for it, and the split of its deflection error: image fit - truth = (oracle - truth), the model's
  limit, plus (image fit - oracle), what the images did not constrain.

## The truth maps

`TruthRecoveryConfig.alpha_true_fits` names the deflection file. FF-SIMS gives two 2048 x 2048 images, the deflection
in arcsec toward east (extension 0) and north (extension 1) for a source at the header's `ZS` (Hera 2, Ares 1), on the
pixel grid of the convergence map, whose WCS places them. With $\alpha_{\rm west} = -$ extension 0, the catalog images
of each family meet in the source plane to 0.15" (median), and half the divergence reproduces the convergence map to
0.6%; the model is evaluated at the same `ZS`.
