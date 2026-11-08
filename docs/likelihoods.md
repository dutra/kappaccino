# Likelihoods

The data are the observed positions $\hat\theta_i$ of multiple images, grouped into families: the images of one source
point. This page derives the position likelihood from the lens equation, then assembles the full posterior with every
other term (membership, kinematics, priors). The error models that follow (GRF, image-plane GP, smoothed) differ only in
the covariance they assign to the errors in that derivation.

## The position likelihood from probability theory

### One image

A source at $\beta$ has an image wherever the lens equation holds,

$$
\beta = \theta - \alpha(\theta) - \eta(\theta),
$$

with $\alpha$ the deflection of the parametric lens (halos, shear, members) and $\eta$ the deflection error of mass
the model lacks. Write $\beta(\theta) = \theta - \alpha(\theta) - \eta(\theta)$ for the lens map. Near one image,
the map is one-to-one, so the density of the image position given the source follows from a change of variables:

$$
p(\theta \mid \beta) = \delta^2\!\big(\beta(\theta) - \beta\big)\, \lvert \det A(\theta) \rvert,
\qquad A = \frac{\partial \beta}{\partial \theta}.
$$

The factor $\lvert\det A\rvert$ is what turns a point in the source plane into a probability per unit area in the
image plane, where the data live. Leaving it out (a source-plane $\chi^2$) favors lenses that demagnify every image,
because a residual shrinks along with the image.

The image is observed with a position error $e$, $\hat\theta = \theta + e$. Marginalizing over the true position
$\theta = \hat\theta - e$ and over both errors,

$$
p(\hat\theta \mid \beta) = \iint p(e)\, p(\eta)\; \delta^2\!\big(\beta(\hat\theta - e) - \beta\big)\,
\lvert \det A(\hat\theta - e) \rvert \; de \, d\eta .
$$

Linearize about the observed position, where everything is evaluated: the ray-traced position
$b = \hat\theta - \alpha(\hat\theta)$, the magnification matrix $A = A(\hat\theta)$, its determinant
$D = \det A$ and the gradient $g = \nabla_\theta \det A$. To first order in the errors,

$$
\beta(\hat\theta - e) \approx b - A e - \eta, \qquad \det A(\hat\theta - e) \approx D - g \cdot e .
$$

Define the source-plane residual $F = b - \beta$ and the two error quantities

$$
\tilde F = A e + \eta, \qquad X = D - g\cdot e .
$$

The delta function sets $\tilde F = F$, so the integral is the density of $\tilde F$ at $F$ times the expected
$\lvert X \rvert$ given that value:

$$
p(\hat\theta \mid \beta) = p_{\tilde F}(F)\; \mathbb{E}\big[\,\lvert X\rvert \;\big|\; \tilde F = F\big].
$$

With Gaussian errors, $e \sim N(0, \sigma_{\mathrm{meas}}^2 I)$ and $\eta \sim N(0, \Sigma_\eta)$, $\tilde F$ and
$X$ are jointly Gaussian:

$$
C = \mathrm{Cov}(\tilde F) = \sigma_{\mathrm{meas}}^2 A A^T + \Sigma_\eta, \qquad
c = \mathrm{Cov}(\tilde F, X) = -\sigma_{\mathrm{meas}}^2 A g .
$$

The deflection error enters $\tilde F$ only, not the determinant. Conditioning gives $X \mid F \sim N(\mu_X, v_X)$
with $\mu_X = D + c^T C^{-1} F$ and $v_X = \sigma_{\mathrm{meas}}^2 |g|^2 - c^T C^{-1} c$, and the image term is

$$
\boxed{\;p(\hat\theta \mid \beta, \Theta) = N(F;\, 0,\, C)\; h, \qquad
h = \mathbb{E}\big[\lvert X\rvert\, \mathbb{1}(\pi X > 0) \,\big|\, F\big]\;}
$$

$h$, the "smoothed determinant", has a closed form for a Gaussian $X$. The indicator applies the catalog parity
$\pi = \pm 1$ of the image (both signs when it is unknown): an image of the wrong parity has no probability. Where the
measurement error is negligible against the distance to a critical curve, $h \to \lvert D \rvert$. Near a fold, $D$
crosses zero within the error, and $h$ stays finite and smooth where $\lvert D\rvert$ would vanish.

### A family of images

The images of one source are conditionally independent given $\beta$ and the lens when their errors are. Then the
family likelihood is the product of the image terms, and that is the smoothed likelihood below. When the errors
are correlated between images (a shared field of unmodeled mass, or displacements shared by nearby images), the
residuals of a family or block of images are stacked into one vector $F$ with a full covariance $C$. The Gaussian
term becomes $N(F;\, 0,\, C)$ with one Cholesky factor per block. Each image's $h$ is computed from its own diagonal
block $C_{ii}$, which keeps the determinant factor per image (an approximation: the cross-correlations enter the
Gaussian term but not $h$).

The redshift of the family enters through the lensing efficiency $D_{ls}/D_s$, which scales the parametric
deflection. It also scales the deflection error, by the weight $w_i$ relative to a reference redshift (GRF).

### The source position

Each family has one unknown source position with a broad Gaussian prior, $\beta \sim N(\mu_\beta, s_\beta^2 I)$
(`LikelihoodConfig.source_position_prior_*`). It is sampled, not integrated out analytically, because $h$ depends on
$\beta$ through $F$. To remove the strong correlation between $\beta$ and the lens, it is reparametrized: the
weighted mean of the ray-traced images, $\bar\beta = (\sum C_i^{-1})^{-1} \sum C_i^{-1} b_i$ (per block under
correlated errors), has covariance $L L^T = (\sum C_i^{-1})^{-1}$. NUTS samples unit offsets $u$ with
$\beta = \bar\beta + L u$, and the target carries the change-of-variables correction

$$
\log N(\beta;\, \mu_\beta, s_\beta^2 I) + \log\lvert\det L\rvert - \log N(u;\, 0, I),
$$

so the density over $(\Theta, \beta)$ is unchanged. The init search drops the volume factor
$\log\lvert\det L\rvert$ (`search_log_likelihood`): the marginal over the lens is the same, but the maximum is in the
physical source coordinates, which avoids a spurious mode at lenses that demagnify every image.

## The full posterior

With lens parameters $\Theta$ (halos, shear, sheet, scaling relation, member deviations), error-model parameters
$\phi$ (the scales of $e$ and $\eta$), source positions $\beta_f$ and membership indicators $z$, the sampled density is

$$
p(\Theta, \phi, \beta \mid \mathcal{D}) \;\propto\;
\underbrace{p(\Theta)\, p(\phi)\, \prod_f p(\beta_f)}_{\text{priors}}
\;\underbrace{\sum_z p(z) \prod_f p(\hat\theta_f \mid \beta_f, \Theta, \phi, z)}_{\text{positions, membership summed}}
\;\underbrace{\prod_m p(\hat\sigma_m \mid \Theta)}_{\text{kinematics}}
\;\underbrace{B(\Theta)}_{\text{soft constraints}} .
$$

Each factor:

Priors $p(\Theta)$, $p(\phi)$
: The halo, shear and scaling-relation priors ({doc}`configuration`), the hierarchical prior of the member deviations
  (each member's offset from the scaling relation, Student-t or normal in units of a sampled population scatter),
  and the error-scale priors of the chosen error model.

Positions $p(\hat\theta_f \mid \beta_f, \Theta, \phi, z)$
: The family likelihood derived above. The error model fixes the covariance $C$, that is, $\Sigma_\eta$ and any
  correlated displacement (the sections below).

Membership $\sum_z p(z)$
: With uncertain membership, each member's mass is multiplied by an indicator $z_i \sim \mathrm{Bernoulli}(p_i)$ and
  the position likelihood is summed over the indicators. Uncertain members near the images are summed exactly; for
  the rest, the indicators become one more Gaussian deflection error in $\Sigma_\eta$
  ([Uncertain membership](#uncertain-membership)). With every $p_i = 1$ the sum has one term.

Kinematics $p(\hat\sigma_m \mid \Theta)$
: Measured stellar velocity dispersions are independent data given the lens:
  $\ln \hat\sigma_m \sim N\big(\ln(c\, \sigma_m(\Theta)),\, e_m^2\big)$, with $\sigma_m$ the member's model dispersion
  (scaling relation times its deviation), $c$ the aperture factor and $e_m$ the measurement and systematic errors in
  $\ln\sigma$ (`MemberPopulationConfig.kinematics_sources`).

Soft constraints $B(\Theta)$
: Not data. They are terms that keep the model where its approximations hold, so they act as prior factors. The
  far-field guard, $-s\,\mathrm{softplus}(-x/w)$ per far-field member with $x = \ln(R/(k\, r_{\mathrm{cut}}))$ and $R$
  its distance to the nearest image, keeps every image outside $k$ cut radii (`far_field_cut_radii`) of a member
  evaluated with the far-field series (`far_field_barrier_strength` $s$, `far_field_barrier_width` $w$). The ellipticity caps,
  $-\mathrm{softplus}((\lvert\epsilon\rvert - \epsilon_{\max})/w)$, keep halo ellipticities below a smooth limit.

What the likelihood does not include: the number of images. Each observed image is matched to the prediction near
it, so an image the model predicts but the catalog lacks costs nothing here. The posterior image diagnostics check
image counts after the fit ({doc}`outputs`).

## Error models

The error models set the covariance of the position errors $e$ and the deflection error $\eta$. Choose one with
`LikelihoodConfig.likelihood_mode` and `gp_kernel`:

| name | `likelihood_mode` | `gp_kernel` | $\eta$ | correlated image-plane displacement |
|---|---|---|---|---|
| GRF (production) | `gp_image_plane` | `knee` | Gaussian random field over all images | image-plane GP |
| image-plane GP | `gp_image_plane` | `image_plane` | none (small floor) | image-plane GP within each family |
| smoothed (library default) | `smoothed` | | independent per image | none |

### GRF of unmodeled mass (`grf_gp`)

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

#### Covariance blocks

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

#### Hyperparameters

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

### Image-plane GP (`gp_image_plane`)

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

### Smoothed local likelihood (`smoothed`)

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

### Astrometric errors

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
scaling relation, with $z_i = 0$ it has no mass. The indicators are latent variables like the true image positions
in the derivation above, and are marginalized the same way, by the sum rule:

$$
p(\hat\theta \mid \Theta) = \sum_{z \in \{0,1\}^N} \prod_i p_i^{z_i}(1-p_i)^{1-z_i}\; p(\hat\theta \mid \Theta, z),
$$

where $p(\hat\theta \mid \Theta, z)$ is the position likelihood with member $i$'s mass included only when
$z_i = 1$. So $p_i$ is the prior and the images give the posterior,
$P(z_i = 1 \mid \theta) \propto p_i\, p(\theta \mid z_i = 1)$. A member with $p_i = 1$ drops out of the sum and is
modeled as before. The sum has $2^N$ terms and does not factor over members, because every image feels every member,
so it is split by how much a member can matter.

Enumerated members
: Uncertain members that the member classification already made exact (see {doc}`configuration`, Exact and
  far-field members), at most `membership_max_enumerated` (default 4). When there are more, the ones with the
  highest classification score are enumerated, as for `top_exact`. The score is the larger of the truncation term
  and the whitened far-field error over `template_tolerance`, so it measures how much the member matters at the
  images relative to their errors. These members can decide an image's position, so their $2^K$ configurations are summed exactly (the target is a `logsumexp` over the
  configurations). NUTS samples only the continuous parameters, and each draw gives the configuration weights, whose
  average over the draws is the posterior membership probability.

Marginalized members
: Every other uncertain member. A member's deflection is linear in its mass, so the total deflection is
  $\alpha = \alpha_{\mathrm{rest}} + \sum_g z_g \alpha_g$. Split it into its mean and a fluctuation,
  $\alpha = \big(\alpha_{\mathrm{rest}} + \sum_g p_g \alpha_g\big) + \sum_g (z_g - p_g)\, \alpha_g$. The mean is the
  model deflection. The fluctuation has zero mean, and approximated as Gaussian it is one more deflection error: a
  term in $\Sigma_\eta$ of [One image](#one-image), with covariance

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

## Uncertain images

Candidate images whose family is unknown can be fit with the lens (`PartitionConfig`, `config.partition`). Each
uncertain image may be a new image of a trusted family, an image of a new source it shares with other uncertain images,
or unrelated to every other image. The trusted families of the image catalog are kept as given, and the choice for each
uncertain image is summed out of the likelihood, so NUTS samples the lens with the assignment marginalized.

```python
from kappaccino.config import PartitionConfig

config = replace(config, partition=PartitionConfig(catalog_path="uncertain_images.csv",
                                                   population_catalog_path="field_photometry.csv"))
```

The init stage fits the trusted families as usual; the uncertain images enter at the final stage.

### Source slots

A slot is a source an uncertain image may come from, with source position $\beta_k$, redshift $z_k$ and intrinsic
colors $c_k$. The trusted families are slots whose source is the one sampled above, at their catalog redshift. New
slots are extra sources whose position, redshift and colors are sampled. For image $i$ and slot $k$,

$$
p(D_i \mid k) = \mathcal N\big(b_i(z_k) \mid \beta_k,\, C_i(z_k)\big)\, h_i
\;\mathcal N(\hat z_i \mid z_k, \sigma_{z,i}^2)\; p(\mathbf c_i \mid \mathbf c_k),
$$

where the first factor is the per-image term of the trusted images, at the slot's redshift: the source-plane Gaussian
with the covariance $C_i$ of the configured likelihood and the smoothed $|\det A_i|$. The colors scatter about the
source's with a same-source scatter (fit on the trusted families unless `color_scatter` is set) and, for a fraction
`outlier_fraction`, a broad outlier scatter; for a trusted slot the color term is the predictive of the family's own
members. Given the lens and the slot parameters the images are independent, so each image's choice sums out:

$$
\ln L_{\mathrm{slots}} = \sum_i \ln\Big[\pi_0\, E_i + \sum_k \pi_k\, p(D_i \mid k)\Big],
$$

added to the likelihood of the trusted images. $E_i$ is the evidence of the image as the only image of its own
source: the same terms integrated over the source position (analytically) and redshift (on a grid). The weights are
fixed: $\pi_0$ = `unrelated_fraction` (0.5), the rest shared equally by the slots. The redshift and colors of a new or
unrelated source follow a Gaussian mixture fit to `population_catalog_path` in (redshift, colors), and its position the
Gaussian prior of width `source_prior_sigma_arcsec` (default: that of the trusted sources). That width sets the cost of
a source, about $\ln(2\pi\sigma^2)$ in evidence, and so how readily a lone image joins one.

Expanding the product over images of the sums over slots gives one term per assignment of images to slots, and
integrating each slot's parameters over its prior turns the images assigned to it into the evidence $E_F$ of that
family:

$$
\int \prod_i \sum_k \pi_k\, p(D_i \mid \beta_k, z_k, \mathbf c_k)\; p(\beta, z, \mathbf c)\; d\beta\, dz\, d\mathbf c
= \sum_{Z} p(Z) \prod_{F \in Z} E_F,
$$

the sum over every partition $Z$ of the images, with the prior $p(Z)$ the weights induce. Summing out the labels with
the sources kept as parameters costs one term per image and slot; NUTS integrates the sources. Each posterior draw gives
the probability of every option per image, from which the outputs draw joint assignments.

### Seeding and limits

A slot far from every image feels no gradient, so new slots are placed before sampling: at the init start, two
uncertain images are linked when the evidence of the pair as one new source beats each image's best alternative
(unrelated, or its best trusted family) by more than `slot_gain_threshold`, and one slot is placed on each linked
group. Groupings the seeding misses are not reached during sampling.

The deflection is proportional to the lensing efficiency except for the external shear, so one evaluation of the lens
at each uncertain image serves every slot redshift. Uncertain images get the diagonal blocks of the configured
covariance: under `gp_image_plane` they are not correlated with other images, and with the knee kernel their
deflection floor carries the GRF amplitudes weighted by $w$ at the slot's redshift (`knee_background` is not
supported). With uncertain membership, each enumerated member configuration evaluates the uncertain images with its own
lens inside the membership sum; marginalized members enter through their expected mass, without their deflection
covariance at the uncertain images. The model assigns detected images only: it does not penalize a family whose predicted bright images are
undetected, and two detections of one image (knots, a shredded arc) can attach to one source.

### Input and outputs

`catalog_path` is a CSV table with `image_label`, `ra`, `dec` (degrees) and optionally `sigma_arcsec`, `parity`,
redshifts (`zspec_best`, `catalog_z` or `zphot_best`, with `catalog_z_sigma` or `pagul_zpdf_low`/`pagul_zpdf_high`) and
`mag_<band>`, `mag_err_<band>`; colors are differences of adjacent bands in `bands`. `population_catalog_path` is a CSV
photometric catalog of the field with a redshift column and the same bands; without it the mixture is fit on the
trusted and uncertain images, which is crude.

The final stage writes `final/artifacts/slots.npz` (per-draw option probabilities and sampled assignments, their
averages, the probability that two images share a source, and the new slots' redshifts and positions) and the tables
`uncertain_images.csv` and `uncertain_pairs.csv`. The new slots' sampled parameters (`slot.<k>.*`) are unit offsets.

## Diagnostic likelihoods

`source-anisotropic` and `source-isotropic` keep only the Gaussian source-plane term, without $h$ (the second also
without the Jacobian in $C$). They are cheap and biased toward demagnifying lenses (see [One image](#one-image)), and
are kept for diagnostics.
