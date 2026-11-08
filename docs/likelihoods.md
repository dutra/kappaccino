# Likelihoods

The data are the observed image positions. Every observed image $\theta_i$ is ray-traced to its source plane,
$b_i = \theta_i - \alpha(\theta_i)$, and the likelihood compares the images of each family with one source position
$\beta$. The likelihoods differ in how they model the residuals, that is, which errors they assume besides the
catalog astrometry.

Choose a likelihood with `LikelihoodConfig.likelihood_mode` and `gp_kernel`:

| name | `likelihood_mode` | `gp_kernel` | model |
|---|---|---|---|
| GRF (production) | `gp_image_plane` | `knee` | Gaussian random field of unmodeled mass + image-plane GP |
| image-plane GP | `gp_image_plane` | `image_plane` | image-plane GP within each family |
| smoothed (library default) | `smoothed` | | per-image source-plane floors |

In all three, the source positions are sampled as whitened offsets from the conditional source centroid, and the
conditional expectation of the absolute Jacobian determinant normalizes the density.

## GRF of unmodeled mass (`grf_gp`)

The likelihood of the production fits. The residual of image $i$ (family redshift weight $w_i$, magnification matrix $A_i$) is

$$
F_i = b_i - \beta_{k(i)} = w_i\, \delta\alpha(\theta_i) + A_i\, \big(d(\theta_i) + e_i\big),
$$

with three terms:

$\delta\alpha$
: The deflection error of mass the parametric model lacks: a curl-free Gaussian random field on the lens potential
  with the convergence power spectrum $P_\kappa(k) = P_0 / [1 + (k/k_t)^\gamma]$ between fixed wavelength cutoffs. Its
  amplitude $\sigma_\alpha$ is the rms deflection error per component for a source at the reference redshift
  (`gp_reference_redshift`, 2). The knee wavelength $\lambda_t = 2\pi/k_t$ sets the scale of the unmodeled mass.

$d$
: A correlated image-plane displacement with an isotropic Gaussian kernel of length $\ell$ and amplitude
  $\sigma_{\mathrm{shared}}$ (for example, measurement systematics shared by nearby images).

$e_i$
: The independent position error, $\sigma_{\mathrm{inst},i}^2 + \sigma_{\mathrm{white}}^2$.

The covariance of the stacked residuals of a block of images is

$$
C = W K_\alpha W + B\,[\sigma_{\mathrm{sh}}^2 K_{\mathrm{img}} + \mathrm{diag}(\sigma_{\mathrm{inst},i}^2 + \sigma_w^2)]\,B^T + \sigma_{\mathrm{fl}}^2 I,
$$

where $K_\alpha$ holds the longitudinal and transverse correlations of the deflection field (Hankel transforms of
$P_\kappa$, tabulated) and $\sigma_{\mathrm{fl}}$ (`gp_floor_arcsec`, 1.5 mas) is a small smoothness floor.

### Covariance blocks

`gp_blocks` sets which images share one field:

`all`
: One shared mass field over every image (default for full runs). The field-wide translation, convergence and shear
  are projected out (`gp_knee_project_modes`), because the sources and the external shear absorb them.

`system`
: An independent field per system (all knots of one galaxy), with no projection. About five times cheaper per
  gradient, and useful for quick looks. Its posterior differs from the shared field's, so use it for quick tests.

`family`
: An independent field per family.

The init search runs on system blocks either way; the chain-start step and NUTS use the configured blocks.

### Hyperparameters

The sampled hyperparameters are $\sigma_\alpha$, the knee $\lambda_t$, the image-plane $\ell$,
$\sigma_{\mathrm{shared}}$ and $\sigma_{\mathrm{white}}$; the slope $\gamma$ is fixed at 4 unless
`gp_knee_gamma_prior` (a `dist.Uniform` or a two-sided `dist.TruncatedNormal`, whose bounds are also those of the
tables) frees it. The production fits use these settings:

```python
from kappaccino.config import LikelihoodConfig
from kappaccino.priors import truncated_lognormal

grf_likelihood = LikelihoodConfig(
    likelihood_mode="gp_image_plane",
    gp_kernel="knee",
    gp_blocks="all",                                            # "system" for quick looks
    gp_knee_project_modes=("translation", "convergence", "shear"),  # only with gp_blocks="all"
    sigma_sel_prior=None,
    gp_knee_gamma=4.0,
    gp_knee_gamma_prior=None,                                   # dist.TruncatedNormal(4.0, 0.5, low=2.0, high=6.0) samples it
    gp_knee_scales_arcsec=(0.2, 100.0),
    gp_knee_prior=truncated_lognormal(6.0, 0.6, low=2.5, high=8.0),
    gp_knee_start_arcsec=6.0,
    gp_sigma_alpha_prior=truncated_lognormal(0.08, 0.5, low=1e-4, high=0.5),
    gp_sigma_alpha_start_arcsec=0.08,
    gp_knee_ell_prior=truncated_lognormal(1.0, 0.4, low=0.4, high=3.0),
    gp_knee_ell_start_arcsec=1.0,
)
config = replace(config, likelihood=grf_likelihood)
```

`gp_knee_fixed_arcsec` holds the knee at one value instead of sampling it.

The knee prior keeps the field at the galaxy scale: left free, the knee moved to 14–25" and absorbed the halo-scale
misfit. A slope of 2 corresponds to truncated isothermal lumps, 4 to NFW-like cusps, and 6 to cored lumps.

## Image-plane GP (`gp_image_plane`)

The image-plane GP alone, per family, with no source-plane floor. Within a family the image-plane displacements are
correlated:

$$
S = \sigma_{\mathrm{shared}}^2 (K \otimes I) + (\sigma_{\mathrm{inst}}^2 + \sigma_{\mathrm{white}}^2) I, \qquad
K_{ij} = \exp\!\left(-\frac{|\theta_i - \theta_j|^2}{2\ell^2}\right),
$$

and the stacked source-plane residuals have covariance $C = B S B^T + \sigma_{\mathrm{fl}}^2 I$ with
$B = \mathrm{blockdiag}(A_i)$. One Cholesky factor of $C$ per family gives the Gaussian term and the conditional
source position.

The sampled scales are $\sigma_{\mathrm{shared}}$ (log-normal, median 0.15", truncated at 1"), $\sigma_{\mathrm{white}}$
and $\ell$ (`gp_correlation_length_prior`, log-normal, median 7", truncated at 30"). Set
`gp_correlation_length_prior=None` to fix $\ell$ at `gp_correlation_length_arcsec`.

## Smoothed local likelihood (`smoothed`)

The library default. Each image's source-plane residual is compared with a Gaussian of covariance

$$
C_i = \sigma_{\mathrm{meas}}^2 A_i A_i^T + \sigma_{\mathrm{def}}^2 I + \sigma_{\mathrm{sel}}^2 R_i, \qquad
R_i = (I + A_i A_i^T / \lambda_{\mathrm{sel}}^2)^{-1},
$$

the propagated measurement error with $\sigma_{\mathrm{meas}}^2 = \sigma_{\mathrm{inst}}^2 + \sigma_{\mathrm{extra}}^2$,
an isotropic residual-deflection scatter $\sigma_{\mathrm{def}}$, and a feature-selection term with stretching weight
$R_i$. $\sigma_{\mathrm{extra}}$, $\sigma_{\mathrm{def}}$, $\sigma_{\mathrm{sel}}$ and $\lambda_{\mathrm{sel}}$ are
sampled. Set `sigma_sel_prior=None` to drop the selection term (as the production fits do) or `sigma_extra_prior=None` to drop
$\sigma_{\mathrm{extra}}$.

## Astrometric errors

$\sigma_{\mathrm{inst}}$ comes from each image's `sigma` in the catalog, otherwise from
`ImageConstraintsConfig.sigma_inst_arcsec`. `LikelihoodConfig.sigma_inst_arcsec` overrides it for every image.
The production fits set `ImageConstraintsConfig(sigma_inst_arcsec=0.001)`, a 1 mas floor, so that the white scatter $\sigma_{\mathrm{white}}$ is inferred rather than assumed; the
"chi-square (catalog errors)" line of the run summary is not meaningful at that value.

## Uncertain membership

Member catalogs are built by classification. Spectroscopic members (a redshift within a few thousand km/s of the
cluster) are close to certain. Fainter members come from their colors: a red-sequence cut or a classifier trained on
the spectroscopic members (Grillo et al. 2015 and later CLASH-VLT and MUSE catalogs), with a purity of about
90–95% at the faint end. A published catalog applies a threshold and lists every galaxy above it as a member. With
`MemberConfig(membership="marginalize")` each member instead carries a prior probability $p_i$ of belonging to the
cluster, and the images update it.

### The likelihood

Each member has an indicator $z_i \in \{0, 1\}$ with prior $\mathrm{Bernoulli}(p_i)$: with $z_i = 1$ it is on the
scaling relation, with $z_i = 0$ it has no mass. The likelihood of the image positions $\theta$ at lens parameters
$\Theta$ sums over the indicators,

$$
\mathcal{L}(\Theta) = \sum_{z \in \{0,1\}^N} \prod_i p_i^{z_i}(1-p_i)^{1-z_i}\; p(\theta \mid \Theta, z),
$$

where $p(\theta \mid \Theta, z)$ is the likelihood of the sections above with member $i$'s mass included only when
$z_i = 1$. So $p_i$ is the prior and the images give the posterior,
$P(z_i = 1 \mid \theta) \propto p_i\, p(\theta \mid z_i = 1)$. A member with $p_i = 1$ drops out of the sum and is
modeled as before. The sum has $2^N$ terms and does not factor over members, because every image feels every member,
so it is split by how much a member can matter.

Enumerated members
: The uncertain exact members nearest the images, at most `membership_max_enumerated` (default 4). These can decide
  an image's position, so their $2^K$ configurations are summed exactly (the target is a `logsumexp` over the
  configurations). NUTS samples only the continuous parameters, and each draw gives the configuration weights, whose
  average over the draws is the posterior membership probability.

Marginalized members
: Every other uncertain member. A member's deflection is linear in its mass, so the total deflection is
  $\alpha = \alpha_{\mathrm{rest}} + \sum_g z_g \alpha_g$. Averaging over the $z_g$ gives the mean deflection
  $\alpha_{\mathrm{rest}} + \sum_g p_g \alpha_g$, used as the model, and the covariance

  $$
  C_{\mathrm{mem}}^{(jk)} = \sum_g p_g (1 - p_g)\, \alpha_g(\theta_j)\, \alpha_g(\theta_k)^T,
  $$

  added to the source-plane residual covariance. It correlates images $j$ and $k$, because one missing member moves
  all of its nearby images together. Under the GRF likelihood it adds to $W K_\alpha W$ after the low-order
  projection, so the shared field's blocks gain the cross terms. Under the image-plane GP it adds to each family's
  covariance, and under the smoothed likelihood to each image's $C_i$. The diagonal blocks also enter the smoothed
  determinant. This is the Gaussian, first-order view of the indicators: their membership is not updated (posterior
  equals prior), but the uncertainty they add to the mass model propagates.

### Prior probabilities

$p_i$ is the catalog's `p_member` (see {doc}`catalogs`) when given. Otherwise it is 1 for a spectroscopically
confirmed member (a `z` of kind `spec`, kinematics, or role `bcg`) and `membership_default_probability` for the rest.
The prior is the output of the color classification, made beforehand; the lens model does not refit the photometry.
Galaxies that a published threshold rejected can be added with a low `p_member`, so the images can bring back one
that matters.

Classifier probabilities are often over- or under-confident. With `membership_calibration_prior` (for example
`dist.Normal(0.0, 1.0)`), one shared shift of the log-odds is sampled,
$\mathrm{logit}\,p_i' = \mathrm{logit}\,p_i + \delta$, for every uncertain member. The enumerated members inform
$\delta$ directly, and the marginalized ones through their covariance.

### Cost and limits

- The likelihood costs $2^K$ evaluations for $K$ enumerated members (16 at the default).
  `membership_max_enumerated=0` marginalizes every uncertain member at no extra evaluations.
- A non-member is modeled as no mass. A real interloper is a galaxy at another redshift: in front of the cluster it
  still lenses, off the scaling relation. Removing it is a good approximation for faint interlopers; a bright
  foreground galaxy is better modeled as its own halo (`treatment: independent`).
- The images separate "not a member" from "a low-mass member" only when the difference in mass is large, because
  each member already has a deviation from the scaling relation.
- Maps, image predictions and diagnostics use the expected mass $p_i$ of every uncertain member.
- After sampling, `final/tables/member_membership.csv` lists the prior and posterior of every member (see
  {doc}`outputs`).

### Validation

`scripts/validate_membership_injection.py` tests whether the images recognize a wrong member. At the posterior
draws of a finished run, it gives one galaxy at a time a prior of 0.5 and computes its posterior membership. The
galaxy is either an interloper (a fake galaxy added near the images, not part of the true lens) or a true member that
the classifier doubts. On the Hera tutorial run (GRF likelihood, 65 images, 340 members, 100 draws):

| galaxy | magnitude (F160W) | nearest image (arcsec) | largest deflection at an image (arcsec) | posterior membership |
|---|---|---|---|---|
| interloper | 19.0 | 1–4 | 0.20–0.24 | 0.07–0.17 |
| interloper | 19.0 | 7 | 0.19 | 0.36 ± 0.26 |
| interloper | 20.5 | 1–7 | 0.05–0.08 | 0.43–0.49 |
| interloper | 22.0 | 1–8 | 0.01–0.03 | 0.48–0.50 |
| true member (`1`) | 14.4 | 0.8 | 2.7 | 1.00 |
| true member (`24`) | 19.0 | 3.6 | 0.21 | 0.64 ± 0.21 |
| true member (`79`) | 20.5 | 0.6 | 0.07 | 0.60 ± 0.12 |
| true members (`174`, `205`, `308`) | 21.4–22.9 | 0.5–3.7 | 0.006–0.025 | 0.50–0.52 |

The posterior moves in the right direction in every case, and how far it moves follows the galaxy's deflection at
the images rather than its distance. Above about 0.2" of deflection the images decide. Below about 0.05" (fainter
than about 21 mag on Hera) the posterior stays at the prior: the GRF and the member scatter absorb such a small
perturbation. That is why the marginalized members can keep their prior. The draws were fit without the
interlopers, so this tests the likelihood at the fitted lens, not a joint refit in which the lens could partly adjust
to a wrong member.

## Other terms

Member kinematics
: With `MemberPopulationConfig.kinematics_sources`, the measured dispersions add one log-normal term per member (see
  {doc}`configuration`).

Far-field barrier
: A smooth log-barrier keeps far-field members outside `far_field_cut_radii` of every image during sampling.

`source-anisotropic`, `source-isotropic`
: Source-plane likelihoods kept for diagnostics only.
