# Uncertain images

Candidate multiple images whose family is unknown can be fit together with the lens. Each uncertain image may be a new
image of a trusted family, an image of a new source it shares with other uncertain images, or unrelated to every other
image. NUTS samples the lens with that choice summed out per image, and the assignment probabilities and joint draws of
lens, sources and families come from the posterior. The trusted families of the image catalog are kept as given.

```python
from dataclasses import replace
from lenscluster.config import PartitionConfig

config = replace(config, partition=PartitionConfig(catalog_path="uncertain_images.csv",
                                                   population_catalog_path="field_photometry.csv"))
```

The init stage fits the trusted families as usual; the uncertain images enter at the final stage.

## Input

`catalog_path` is a CSV table with one row per uncertain image:

| column | required | meaning |
|---|---|---|
| `image_label` | yes | name used in the outputs |
| `ra`, `dec` | yes | position in degrees (ICRS), converted to the run's reference frame |
| `sigma_arcsec` | no | astrometric error (default `sigma_inst_arcsec`, 0.1) |
| `parity` | no | -1, +1, or 0 when unknown |
| `zspec_best`, `catalog_z`, `zphot_best` | no | redshift; the uncertainty from `catalog_z_sigma`, `pagul_zpdf_low`/`pagul_zpdf_high`, or 0.5 |
| `mag_<band>`, `mag_err_<band>` | no | photometry; colors are differences of adjacent bands in `bands` |

`population_catalog_path` is a CSV photometric catalog of the field with a redshift column and the same `mag_<band>`
columns. A Gaussian mixture fit to it in (redshift, colors) is the prior of a source: how common its redshift and colors
are. Without it, the mixture is fit on the trusted and uncertain images, which is crude; give a field catalog for real
data.

## The model

### Slots

A **slot** is a source an uncertain image may come from: a source position $\beta_k$, a redshift $z_k$ and intrinsic
colors $\theta_k$. The trusted families are slots whose source the position likelihood already samples, at their
catalog redshift. New slots are extra sources whose position, redshift and colors are sampled. Label 0 means the image
is the only image of its own source.

For image $i$ and slot $k$,

$$
p(D_i \mid k) = \mathcal N\!\big(\beta_k \,\big|\, \hat\beta_i(z_k), C_i(z_k)\big)\, h_i
\;\times\; \mathcal N(\hat z_i \mid z_k, \sigma_i^2)
\;\times\; p(c_i \mid \theta_k),
$$

where the first factor is the per-image position term of the trusted images: $\hat\beta_i(z) = x_i - \alpha(x_i; z)$,
the source-plane covariance $C_i$ of the configured likelihood, and the smoothed $|\det A_i|$. The colors scatter
about the source's with a same-source scatter $s$ (fit on the trusted families unless set) and, for a fraction of
images, a broad outlier scatter. For a trusted slot the color term is the predictive of the family's own members. For
label 0 the source is integrated over its priors in closed form (the source position analytically, the redshift on a
grid).

Given the lens and the slot parameters the images are independent, so each image's label sums out:

$$
\log L_\text{slots} = \sum_i \log\Big[\pi_0\, E_i + \sum_k \pi_k\, p(D_i \mid k)\Big].
$$

This term is added to the position likelihood of the trusted images. The weights $\pi$ are fixed: $\pi_0$ =
`unrelated_fraction`, the rest shared equally by the slots.

### Why this is a sum over partitions

Expanding the product over images of the sums over slots gives one term per assignment of images to slots.
Integrating each slot's parameters over its prior turns the images assigned to it into the evidence $E_F$ of that
family, so

$$
\int \prod_i \sum_k \pi_k\, p(D_i \mid \beta_k, z_k, \theta_k)\; p(\beta, z, \theta)\; d\beta\, dz\, d\theta
= \sum_{Z} p(Z) \prod_{F \in Z} E_F,
$$

the sum over every partition $Z$ of the images with the prior $p(Z)$ the weights induce. The two forms are the same
model. Integrating the sources first couples the images of a family and leaves a sum over all partitions, whose number
grows as the Bell number of the number of images; summing the labels first, with the sources kept as parameters, costs
one term per image and slot, and NUTS integrates the sources.

### Seeding the new slots

A slot whose source is far from every image contributes nothing and feels no gradient, so new slots are placed before
sampling: at the start lens, two uncertain images are linked when the evidence of the pair as one new source beats each
image's best alternative (unrelated, or its best trusted family) by more than `slot_gain_threshold`, and one slot is
placed on each linked group, at its evidence-weighted source and redshift. The seeding is deterministic, so a resumed
run rebuilds the same slots. Groupings the seeding misses can still form if an image's alternative slot is close, but
the sampler does not move a slot across the source plane to a distant group.

### Costs and approximations

Each gradient evaluates the lens once more at the uncertain images (every member with its exact kernel): the
deflection is proportional to the lensing efficiency except for the external shear, so one evaluation serves every slot
redshift. The rest is a few small matrix operations per image and slot.

Uncertain images get the diagonal blocks of the configured covariance: under `gp_image_plane` they are not correlated
with other images, and with the knee kernel their deflection floor carries the GRF amplitudes weighted by the lensing
efficiency at the slot's redshift. The `knee_background` spectrum is not supported.

The position term linearizes the lens equation at each image, as the trusted images' does. The model assigns detected
images only: it does not predict missing counterimages or penalize a family whose predicted bright images are not
detected, and two detections of one image (knots, a shredded arc) can both attach to a source. Check the most
probable families with the exact-image solver.

## Settings

`source_prior_sigma_arcsec`
: Width of the source-position prior of a new or unrelated source. A source costs about $\log(2\pi\sigma^2)$ in
  evidence, so this sets how strongly a lone image is preferred over joining a source. Defaults to
  `likelihood.source_position_prior_sigma_arcsec` (100 arcsec); a value near the size of the source-plane region
  behind the field is more physical.

`unrelated_fraction`
: Prior probability that an uncertain image is unrelated (default 0.5).

`color_scatter`, `outlier_fraction`, `outlier_scatter`
: The same-source color scatter (fit on the trusted families when None) and the outlier mixture.

`redshift_grid`
: The quadrature of an unrelated image's evidence (default: the catalog redshifts $\pm 2\sigma$, 12 points).

`slot_gain_threshold`
: The seeding threshold in log evidence (default 0).

## Outputs

`final/artifacts/slots.npz`
: `options` (`unrelated`, the trusted family ids, `slot<k>`), `image_labels`, per-draw `responsibilities` (draw,
  image, option) and sampled `assignment`, the averaged `probability` (image, option), `pair_probability` (image,
  image: the probability two images share a source), and the new slots' `slot_z` and `slot_beta` per draw.

`final/tables/uncertain_images.csv`
: Per image, the probability of every option and the most probable one.

`final/tables/uncertain_pairs.csv`
: Image pairs that share a source with probability above 0.01.

The new slots' parameters (`slot.<k>.u_x`, `u_y`, `v_log_dz`, `w_<color>`) are unit offsets in the posterior samples;
`slots.npz` holds their physical values.
