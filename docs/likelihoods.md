# Likelihoods

The data of a strong-lensing cluster model are the observed positions of multiple images. This page derives the
probability of those positions given a lens model, the likelihood that kappaccino samples with NUTS. We start from the
lens equation for a single image ({ref}`§2 <sec-position-likelihood>`) and then extend the result to cluster members
that may not belong to the cluster
({ref}`§3 <sec-membership>`) and to images whose family is unknown ({ref}`§4 <sec-uncertain-images>`). Section
{ref}`5 <sec-full-posterior>` assembles the full posterior with the remaining terms. The error models of
{ref}`§6 <sec-error-models>` differ only in the covariance that each model assigns to the errors of §2.

Every step uses three rules of probability. A change of variables carries a density from the source plane to the image
plane. The sum rule removes a latent quantity, such as a true image position, a cluster membership or a family
assignment. Conditional independence factors the terms that do not interact.

(sec-notation)=
## 1. Notation

A family is the set of images of one background source. We call an image trusted when the catalog assigns the image to
a family, and uncertain when the family is unknown. Table 1 lists the symbols used throughout. We write
$N(x;\, \mu, C)$ for the normal density with mean $\mu$ and covariance $C$.

| symbol | meaning |
|---|---|
| $\hat\theta_i$, $\theta_i$ | observed and true position of image $i$ |
| $\beta_F$, $z_F$ | source position and redshift of family $F$ |
| $\Theta$ | lens parameters (halos, external shear, mass sheet, member scaling relation, member deviations) |
| $\phi$ | error scales of the likelihood |
| $\alpha(\theta)$ | deflection of the parametric lens |
| $\eta(\theta)$ | deflection error, from mass the parametric lens lacks |
| $A = \partial\beta/\partial\theta$ | inverse magnification matrix, with determinant $D = \det A$ |
| $e_i$ | measurement error of the position of image $i$ |
| $p_i$, $z_i$ | prior membership probability and membership indicator of cluster member $i$ |
| $D_j$, $\ell_j$ | data and source label of uncertain image $j$ |

(sec-position-likelihood)=
## 2. The position likelihood

We first assume that the member catalog and the image families are known. The only unknowns are then the lens
parameters, the error scales and one source position per family. Sections 3 and 4 relax these two assumptions.

(sec-one-image)=
### 2.1. A single image

A source at $\beta$ has an image at every $\theta$ that satisfies the lens equation,

$$
\beta = \theta - \alpha(\theta) - \eta(\theta) .
$$ (eq-lens-equation)

Near one image the lens map $\beta(\theta) = \theta - \alpha(\theta) - \eta(\theta)$ is one-to-one. The density of the
image position given the source then follows from a change of variables,

$$
p(\theta \mid \beta) = \delta^2\!\big(\beta(\theta) - \beta\big)\, \lvert \det A(\theta) \rvert .
$$ (eq-change-of-variables)

The factor $\lvert \det A \rvert$ converts a point in the source plane into a probability per unit area in the image
plane, where the data are measured. A source-plane $\chi^2$ omits this factor. Such a $\chi^2$ favors lenses that
demagnify every image, because the source-plane residual of an image shrinks with the image.

We observe the image with a measurement error $e$, so that $\hat\theta = \theta + e$. Marginalizing over the true
position $\theta = \hat\theta - e$ and over both errors gives

$$
p(\hat\theta \mid \beta) = \iint p(e)\, p(\eta)\; \delta^2\!\big(\beta(\hat\theta - e) - \beta\big)\,
\lvert \det A(\hat\theta - e) \rvert \; de \, d\eta .
$$ (eq-image-integral)

We linearize the integrand about the observed position. Let $b = \hat\theta - \alpha(\hat\theta)$ be the ray-traced
position, $A = A(\hat\theta)$ the inverse magnification matrix, $D = \det A$ the determinant and
$g = \nabla_\theta \det A$ the gradient of the determinant, all evaluated at $\hat\theta$. To first order in the errors,

$$
\beta(\hat\theta - e) \approx b - A e - \eta, \qquad \det A(\hat\theta - e) \approx D - g \cdot e .
$$ (eq-linearized)

We define the source-plane residual $F = b - \beta$ and two error variables,

$$
\tilde F = A e + \eta, \qquad X = D - g \cdot e .
$$ (eq-error-variables)

The delta function in Equation {eq}`eq-image-integral` sets $\tilde F = F$. The integral is therefore the density of
$\tilde F$ at $F$ times the conditional mean of $\lvert X \rvert$,

$$
p(\hat\theta \mid \beta) = p_{\tilde F}(F)\; \mathbb{E}\big[\,\lvert X \rvert \;\big|\; \tilde F = F\big] .
$$ (eq-image-factorized)

For Gaussian errors, $e \sim N(0, \sigma_{\mathrm{meas}}^2 I)$ and $\eta \sim N(0, \Sigma_\eta)$, the variables
$\tilde F$ and $X$ are jointly Gaussian, with

$$
C \equiv \mathrm{Cov}(\tilde F) = \sigma_{\mathrm{meas}}^2 A A^T + \Sigma_\eta, \qquad
c \equiv \mathrm{Cov}(\tilde F, X) = -\sigma_{\mathrm{meas}}^2 A g .
$$ (eq-image-covariance)

The deflection error enters $\tilde F$ but not the determinant. Conditioning on $F$ gives $X \mid F \sim N(\mu_X, v_X)$,
with $\mu_X = D + c^T C^{-1} F$ and $v_X = \sigma_{\mathrm{meas}}^2 \lvert g \rvert^2 - c^T C^{-1} c$. The likelihood
of a single image is then

$$
p(\hat\theta \mid \beta, \Theta) = N(F;\, 0,\, C)\; h, \qquad
h = \mathbb{E}\big[\lvert X \rvert\, \mathbb{1}(\pi X > 0) \,\big|\, F\big] .
$$ (eq-image-likelihood)

Here, $h$ is the smoothed determinant, which has a closed form for Gaussian $X$. The indicator applies the catalog
parity $\pi = \pm 1$ of the image, with both signs allowed when the parity is unknown, so that an image of the wrong
parity has zero probability. When the distance to the nearest critical curve is large compared with the measurement
error, $h \to \lvert D \rvert$. Near a fold, $D$ changes sign within the error, and $h$ remains finite and smooth where
$\lvert D \rvert$ would vanish.

(sec-family)=
### 2.2. A family of images

When the errors of different images are independent, the images of one source are conditionally independent given
$\beta$ and the lens. The family likelihood is then the product of the single-image terms of Equation
{eq}`eq-image-likelihood`, which is the smoothed likelihood of {ref}`§6.3 <sec-smoothed>`. When the errors are
correlated between images, for example through a shared field of unmodeled mass or a displacement shared by nearby
images, we stack the residuals of a family or block of images into one vector $F$ with a full covariance $C$. The
Gaussian term becomes $N(F;\, 0,\, C)$, evaluated with one Cholesky factor per block. The smoothed determinant of each
image is computed from the diagonal block $C_{ii}$ of that image. This is an approximation, since the cross-correlations
enter the Gaussian term but not $h$.

The family redshift enters through the lensing efficiency $D_{ls}/D_s$, which scales the parametric deflection. The
efficiency also scales the deflection error, through a weight $w_i$ relative to a reference redshift
({ref}`§6.1 <sec-grf>`).

(sec-source-position)=
### 2.3. Source positions

Each family has one unknown source position with a broad Gaussian prior, $\beta \sim N(\mu_\beta, s_\beta^2 I)$, set by
`LikelihoodConfig.source_position_prior_*`. We sample $\beta$ rather than integrate over $\beta$ analytically, because
$h$ depends on $\beta$ through $F$. The source position is strongly correlated with the lens parameters, so we
reparameterize. The weighted mean of the ray-traced images,
$\bar\beta = (\sum_i C_i^{-1})^{-1} \sum_i C_i^{-1} b_i$ (per block under correlated errors), has covariance
$L L^T = (\sum_i C_i^{-1})^{-1}$. NUTS (Hoffman & Gelman 2014) samples unit offsets $u$ with $\beta = \bar\beta + L u$,
and the target includes the change-of-variables correction

$$
\ln N(\beta;\, \mu_\beta, s_\beta^2 I) + \ln \lvert \det L \rvert - \ln N(u;\, 0, I),
$$ (eq-source-correction)

so that the density over $(\Theta, \beta)$ is unchanged. The initial search (`search_log_likelihood`) drops the volume
term $\ln \lvert \det L \rvert$. The marginal density of the lens is the same, but the maximum is found in the physical
source coordinates, which avoids a spurious mode at lenses that demagnify every image.

(sec-membership)=
## 3. Uncertain membership

Section 2 treats the member catalog as exact. In practice, member catalogs are built by classification. Spectroscopic
members, with a redshift within a few thousand km/s of the
cluster, are close to certain. Fainter members are selected by color, with a red-sequence cut or a classifier trained
on the spectroscopic members (Grillo et al. 2015 and later CLASH-VLT and MUSE catalogs), and reach a purity of about
90–95% at the faint end. A published catalog applies a threshold and lists every galaxy above the threshold as a
member. A galaxy just below the threshold is dropped, and an interloper just above the threshold is modeled with full
mass. With `MemberConfig(membership="marginalize")`, every member instead carries a prior probability $p_i$ of
belonging to the cluster, and the images update that probability. This section derives the resulting likelihood.

(sec-uncertain-members)=
### 3.1. Membership indicators

We give each member $i$ an indicator $z_i$, equal to 1 when the member belongs to the cluster (at full mass, on the
scaling relation) and 0 when the member has no mass. The prior on the indicators is

$$
p(z) = \prod_i p_i^{z_i} (1 - p_i)^{1 - z_i} .
$$ (eq-membership-prior)

A confirmed member has $p_i = 1$, so confirmed and uncertain members follow the same equations. The deflection of a
member is proportional to the mass of the member, so the deflection of the lens is linear in the indicators,

$$
\alpha(\theta;\, \Theta, z) = \alpha_{\mathrm{halo}}(\theta;\, \Theta) + \sum_i z_i\, \alpha_i(\theta;\, \Theta) .
$$ (eq-membership-deflection)

Here, $\alpha_{\mathrm{halo}}$ is the deflection of the halos, shear and mass sheet, and $\alpha_i$ is the deflection of
member $i$ at full mass. The indicators are unknown, so the sum rule gives

$$
p(\hat\theta \mid \Theta) = \sum_{z} p(z)\; p(\hat\theta \mid \Theta, z),
$$ (eq-membership-sum)

where each term is the position likelihood of §2 for the lens with deflection $\alpha(\theta;\, \Theta, z)$. A member
with $p_i = 1$ has zero prior weight on $z_i = 0$, so only the configurations that include the member contribute.

(sec-membership-marginal)=
### 3.2. Marginalizing the indicators

The sum in Equation {eq}`eq-membership-sum` has $2^N$ terms for $N$ members, and the terms do not factor over members
because every image feels every member. We therefore split the members into two sets. The enumerated members
$\mathcal E$ are the $K$ uncertain members that matter most at the images ({ref}`§3.5 <sec-membership-settings>`).
The presence of an enumerated member can change where an image lands, so the $2^K$ configurations $z_{\mathcal E}$
are summed exactly. The marginalized members $\mathcal M$ are all other members, confirmed or not. Each marginalized
member enters every configuration at the expected mass, $p_g$ times the full mass, and the remaining uncertainty becomes
an additional deflection error. The likelihood of the images is then a weighted sum of $2^K$ position likelihoods,

$$
p(\hat\theta \mid \Theta) = \sum_{z_{\mathcal E} \in \{0,1\}^K} p(z_{\mathcal E})\;
\prod_f p\big(\hat\theta_f \mid \beta_f, \Theta;\ \bar\alpha(z_{\mathcal E}),\ \Sigma_\eta + C_{\mathrm{mem}}\big),
$$ (eq-membership-likelihood)

with the configuration prior, the model deflection and the membership covariance

$$
\begin{aligned}
p(z_{\mathcal E}) &= \prod_{i \in \mathcal E} p_i^{z_i}(1 - p_i)^{1 - z_i}, \\
\bar\alpha(\theta;\, z_{\mathcal E}) &= \alpha_{\mathrm{halo}}(\theta) + \sum_{i \in \mathcal E} z_i\, \alpha_i(\theta)
  + \sum_{g \in \mathcal M} p_g\, \alpha_g(\theta), \\
C_{\mathrm{mem}}^{(jk)} &= \sum_{g \in \mathcal M} p_g (1 - p_g)\, \alpha_g(\theta_j)\, \alpha_g(\theta_k)^T .
\end{aligned}
$$ (eq-membership-terms)

The membership covariance $C_{\mathrm{mem}}$ is added to the covariance $\Sigma_\eta$ of the deflection error. The
covariance couples images $j$ and $k$, because a missing member shifts all nearby images together. A confirmed member
($p_g = 1$) enters $\bar\alpha$ at full mass and contributes $p_g(1 - p_g) = 0$ to $C_{\mathrm{mem}}$, so the member is
a fixed part of the lens. A member with $p_g = 0.95$ enters at nearly full mass and adds a small covariance, and a
member with $p_g = 0.5$ adds the largest covariance. When every $p_i = 1$, no member is enumerated, the sum has a single
term, $C_{\mathrm{mem}} = 0$ and Equation {eq}`eq-membership-likelihood` reduces to the likelihood of §2.

The sum is evaluated in logarithms as a `logsumexp` over the configurations, in one vectorized pass. The sources and
all continuous parameters are shared by the configurations, and NUTS never samples the indicators. Table 2 summarizes
how each kind of member enters the likelihood.

| member | in the model deflection $\bar\alpha$ | in $C_{\mathrm{mem}}$ | posterior membership |
|---|---|---|---|
| $p_i = 1$ | full mass | nothing | 1 |
| $p_i < 1$, enumerated | full mass or zero, summed over both | nothing | updated by the images |
| $p_i < 1$, marginalized | $p_i$ times the full mass | $p_i(1 - p_i)\,\alpha_i \alpha_i^T$ | equal to the prior $p_i$ |

(sec-membership-gaussian)=
### 3.3. The Gaussian approximation

For a marginalized member $g$, we write $z_g = p_g + (z_g - p_g)$. The first term gives the expected deflection
$p_g\, \alpha_g$, which enters $\bar\alpha$. The second term has zero mean and variance $p_g(1 - p_g)$, and the second
terms of different members are independent. Summed over many members, the second terms are approximately Gaussian with
covariance $C_{\mathrm{mem}}$. The sum over the marginalized indicators thus reduces to the Gaussian integral over the
deflection error already done in §2.1, with $C_{\mathrm{mem}}$ added to $\Sigma_\eta$. The step is exact for a
confirmed member, whose second term vanishes. For an uncertain member the approximation holds when the member moves an
image by much less than the position error of the image. The members that matter most are enumerated, so the
approximation applies only to the members that matter least.

(sec-membership-posterior)=
### 3.4. Posterior membership and calibration

At a posterior draw, configuration $z_{\mathcal E}$ has weight
$w(z_{\mathcal E}) \propto p(z_{\mathcal E})\, p(\hat\theta \mid \Theta, z_{\mathcal E})$. The posterior membership of
enumerated member $i$ is the total weight of the configurations with $z_i = 1$, averaged over the draws. The
marginalized members are not tracked, so the posterior membership of each equals the prior. The approximation therefore
propagates the mass uncertainty from these members without updating the membership of these members.

Classifier probabilities are often over- or under-confident. With `membership_calibration_prior` (for example
`dist.Normal(0.0, 1.0)`), we sample one shared shift $\delta$ of the log-odds of every member with $p_i < 1$,

$$
\mathrm{logit}\, p_i' = \mathrm{logit}\, p_i + \delta .
$$ (eq-membership-calibration)

Members with $p_i = 1$ remain at 1. The shift changes $p(z_{\mathcal E})$, $\bar\alpha$ and $C_{\mathrm{mem}}$ and has
a prior factor $p(\delta)$. The enumerated members constrain $\delta$ directly, and the marginalized members constrain
$\delta$ through the covariance.

(sec-membership-settings)=
### 3.5. Implementation

The prior $p_i$ is the catalog column `p_member` ({doc}`catalogs`) when given. Otherwise $p_i = 1$ for a
spectroscopically confirmed member (a `z` of kind `spec`, kinematics, or role `bcg`), and
`membership_default_probability` for the remaining members. The prior is the output of the color classification, made
beforehand, and the lens model does not refit the photometry. Galaxies rejected by a published threshold can be added
with a low `p_member`, so that the images can bring back a galaxy that matters.

The enumerated members are the uncertain members that the member classification already made exact
({doc}`configuration`,
Exact and far-field members), up to `membership_max_enumerated` (default 4). When more qualify, the members with the
highest classification score are enumerated, as for `top_exact`. The score is the larger of the truncation term and
the whitened far-field error over `template_tolerance`, and so measures how much a member matters at the images
relative to the position errors. Under the GRF likelihood, $C_{\mathrm{mem}}$ is added to $W K_\alpha W$ after the
low-order projection, so the blocks of the shared field gain the cross terms. Under the image-plane GP,
$C_{\mathrm{mem}}$
is added to the covariance of each family, and under the smoothed likelihood to the covariance $C_i$ of each image.

The likelihood costs $2^K$ evaluations, or 16 at the default. Setting `membership_max_enumerated=0` marginalizes every
uncertain member at no extra cost.

The model has three limitations. First, a non-member is modeled as having no mass. A real interloper lies at another
redshift and, in front of the cluster, still lenses off the scaling relation. Removing an interloper is a good
approximation for a faint galaxy, while a bright foreground galaxy is better modeled as a separate halo
(`treatment: independent`). Second, the images separate a non-member from a low-mass member only when the difference
in mass is large, because each member already has a deviation from the scaling relation. Third, maps, image predictions
and diagnostics use the prior expected mass $p_i$ of every uncertain member, not the posterior membership. After
sampling, `final/tables/member_membership.csv` lists the
prior and posterior membership of every member ({doc}`outputs`).

(sec-membership-validation)=
### 3.6. Validation

The script `scripts/validate_membership_injection.py` tests whether the images recognize a wrong member. At the
posterior draws of a finished run, the script gives one galaxy at a time a prior of 0.5 and computes the posterior
membership. The galaxy is either an interloper, a fake galaxy added near the images that is not part of the true lens,
or a true member that the classifier doubts. Table 3 lists the results on the Hera tutorial run (GRF likelihood, 65
images, 340 members, 100 draws).

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

The posterior moves in the right direction in every case. The size of the shift follows the deflection of the galaxy
at the images rather than the distance to the images. Above about 0.2″ of deflection the images decide the membership.
Below about 0.05″, fainter than about 21 mag on Hera, the posterior stays at the prior, because the GRF and the member
scatter absorb such a small perturbation. This justifies keeping the prior for the marginalized members. The draws were
fit without the interlopers, so the test probes the likelihood at the fitted lens. A joint refit, in which the lens can
partly adjust to a wrong member, is not tested.

(sec-uncertain-images)=
## 4. Uncertain images

Section 3 relaxes the member catalog. We now relax the image catalog. Some candidate images have no known family. An
uncertain image may be a new image of a trusted family, an image of a new source shared with other uncertain images,
or unrelated to every other image. We keep the trusted families as given in the catalog and sum the family assignment
of the uncertain images out of the likelihood, with the sources of the new families integrated analytically, so that
NUTS samples the lens alone. The fit is configured with `PartitionConfig`,

```python
from kappaccino.config import PartitionConfig

config = replace(config, partition=PartitionConfig(catalog_path="uncertain_images.csv",
                                                   population_catalog_path="field_photometry.csv"))
```

The initial stage fits the trusted families alone, and the uncertain images enter at the final stage. Each uncertain
image $j = 1, \dots, U$ carries data $D_j = (\hat\theta_j, \hat z_j, \hat c_j)$, namely a position, a catalog redshift
with an error $\sigma_{z,j}$ (or none), and colors.

(sec-candidate-sources)=
### 4.1. Candidate sources

An uncertain image has one of three possible origins. The image may be a new counterimage of a trusted family $F$, in
which case the source is the sampled source $\beta_F$ at the family redshift $z_F$. The image may instead belong to a
new source that no trusted family has, with a source position, a redshift and intrinsic colors
$\psi = (\beta, z, c)$. Finally, the image may be the only image of an unrelated source.

A partition $Z$ of the uncertain images records the origins. Each block of $Z$ is either a group $G$ of two or more
images that share one new source, or a single image alone. An image alone is unrelated or a new image of a trusted
family. The prior of the partition follows from a mixture over sources. Image $j$ is unrelated with weight
$\pi_{j0}$, joins trusted family $F$ with weight $\pi_{jF}$, and joins one of $S$ new sources with weight $\pi_{jk}$.
By default, a fraction `PartitionConfig.unrelated_fraction` (0.5) goes to $\pi_{j0}$, and the trusted families and new
sources share the remainder equally. When the catalog claims the image for a family, that family receives the
association prior of the catalog instead (`PartitionConfig.association_weight`). A group then has the weight

$$
w_G = \sum_{k=1}^{S} \prod_{j \in G} \pi_{jk}
$$ (eq-group-weight)

of falling on one new source. The number $S$ of new sources is fixed before sampling
({ref}`§4.7 <sec-uncertain-settings>`).

(sec-population)=
### 4.2. The prior of a new source

A new source is a field galaxy behind the cluster, so the prior of its parameters is

$$
p(\psi) = N(\beta;\, \mu_\beta, s_\beta^2 I)\; q(z, c) .
$$ (eq-source-prior)

Here, $q$ is the density of the field-galaxy population in redshift and colors. The colors $c$ are the $d$ differences
of adjacent bands ($m_1 - m_2$, $m_2 - m_3$, and so on) in `PartitionConfig.bands`. We model $q$ as a Gaussian mixture
in the $1 + d$ dimensions of $(z, c)$,

$$
q(z, c) = \sum_{m=1}^{M} w_m\, N\big((z, c);\, \mu_m, \Sigma_m\big), \qquad \sum_{m=1}^{M} w_m = 1,
$$ (eq-population-mixture)

with weights $w_m$, means $\mu_m$ and full $(1 + d) \times (1 + d)$ covariances $\Sigma_m$, so that the mixture captures
how the colors change with redshift. The number of components $M$ is `PartitionConfig.population_components` (default
6), reduced when the catalog has few rows. We fit the mixture once, before sampling, by maximum likelihood with the EM
algorithm (Dempster, Laird & Rubin 1977), to the redshifts and `mag_<band>` colors of the galaxies in
`PartitionConfig.population_catalog_path`. The mixture parameters stay fixed during sampling. The redshift marginal
$\int q(z, c)\, dc$ is the redshift prior of a new source. Because a new source pays the prior volume of $q$, grouping
uncertain images is favored only when the images agree in redshift and color better than two random field galaxies
would by chance.

The mixture falls back to simpler forms when data are missing. Without a population catalog, the mixture is fit to the
trusted and uncertain images, which is crude. Without usable colors, $q$ is a mixture in $z$ alone with at most three
components. Without redshifts as well, $q$ is the single Gaussian $N(z;\, 3, 2^2)$.

(sec-source-image)=
### 4.3. The likelihood of one image

If image $j$ comes from a source $\beta$ at redshift $z$ with colors $c$, the position term is the single-image
likelihood of Equation {eq}`eq-image-likelihood` at that redshift. The redshift and colors of the image are independent
data given the source, so

$$
p(D_j \mid \beta, z, c) =
N\big(F_j(z);\, 0,\, C_j(z)\big)\, h_j(z)\;
p(\hat z_j \mid z)\;
p(\hat c_j \mid c), \qquad F_j(z) = b_j(z) - \beta .
$$ (eq-source-image)

The three factors are the position, redshift and color terms. The lensing efficiency $D_{ls}/D_s$ scales the deflection
and the GRF weight $w$, so one evaluation of the lens at $\hat\theta_j$ serves every source redshift. The catalog
redshift is a catastrophic outlier with probability $f_z$ (`PartitionConfig.photoz_outlier_fraction`, default 0.05),

$$
p(\hat z_j \mid z) = (1 - f_z)\, N(\hat z_j;\, z,\, \sigma_{z,j}^2) + \frac{f_z}{z_{\max} - z_{\min}},
$$ (eq-photoz-outlier)

uniform over $z_{\min} = 0.5$ to $z_{\max} = 6$, so that one wrong photometric redshift does not split a family. The
observed colors scatter about the source colors with a same-source scatter, plus a broad outlier component. For a
trusted family the source colors are not sampled, and the color term is the predictive distribution given the other
images of the family.

(sec-group-evidence)=
### 4.4. The evidence of a group

If the images of a group $G$ (one image, for an unrelated image) share a new source, the source is drawn from the prior,
and the term is the evidence of the group,

$$
E_G = \int \prod_{j \in G} p(D_j \mid \beta, z, c)\; N(\beta;\, \mu_\beta, s_\beta^2 I)\; q(z, c)\; d\beta\, dz\, dc .
$$ (eq-group-evidence)

The position terms are Gaussian in $\beta$, so the integral over $\beta$ is analytic (with $h_j$ evaluated at the
conditional mean of $\beta$). The integral over colors is analytic against the mixture, and the integral over redshift
is a quadrature on a grid. A group of several images fixes the lensing efficiency to a few percent, about the ratio of
its position error to its deflection, so the posterior of $z$ given the group is narrow. The grid is therefore spaced
uniformly in the lensing efficiency, with 96 points by default (`PartitionConfig.redshift_grid`). With a coarse grid,
the quadrature misses the peak, and the evidence of a true family is underestimated by up to tens in the log.

(sec-partition-sum)=
### 4.5. Summing over the partitions

Given the lens and the trusted sources, the images of different blocks are independent, and integrating the source of
each group gives its evidence. Each uncertain image receives the diagonal block of the configured covariance, so no
error term correlates two uncertain images. The likelihood of the uncertain images is the sum over the partitions,

$$
p(D_U \mid \Theta, \phi, \beta) = \sum_{Z} \prod_{G \in Z} w_G\, E_G \prod_{j \text{ alone in } Z} a_j,
\qquad a_j = \pi_{j0}\, E_j + \sum_F \pi_{jF}\, p(D_j \mid \beta_F, z_F) .
$$ (eq-partition-sum)

With no uncertain images the sum is 1. The number of partitions is a Bell number, so the sum runs over a library of
groups. A group enters the library when its images are mutually compatible at a lens: every pair, and the group
as a whole, has a weighted evidence $w_G E_G$ within a margin of the product of its $a_j$ (§4.7). Two images are
connected when a library group holds both. The sum factorizes over the connected components, and the partitions of
each component (the sets of disjoint library groups, with the other images alone) are enumerated once, before
sampling. Equation {eq}`eq-partition-sum` is then a fixed log-sum-exp over those partitions, exact for the library. The
sources of the new groups are integrated out, so the uncertain images add no parameters.

(sec-assignment)=
### 4.6. Posterior assignments

At a posterior draw the probability of a partition is its term in Equation {eq}`eq-partition-sum` over the sum of its
component. The probability that image $j$ is in group $G$ is the summed probability of the partitions that use $G$.
Given that image $j$ is alone, its origin is unrelated or trusted family $F$ with probabilities proportional to the
terms of $a_j$. The average over the draws is the assignment probability. The probability that images $j$ and $k$
share a source is the probability that one group holds both, plus the probability that both are alone in the same
trusted family. Drawing one partition per component and one origin per lone image at every posterior draw gives joint
draws of the lens and the partition.

(sec-uncertain-settings)=
### 4.7. Implementation

The library is built at the initial lens. A pair of images is linked when its weighted evidence as one source is within
`library_pair_margin` (default 3) of the two images alone in the log, and every clique of linked images, up to
`library_max_group` images (default 8), enters when its weighted evidence is within `library_group_margin` (default 5)
of its images alone. The images that the catalog claims for one untrusted family always form a library group. The lens
is then refit to its maximum a posteriori with the library summed, and the library is rebuilt at every lens so far, for
`library_refits` rounds (default 2). The chains start at the last refit. A group that is not compatible at any of those
lenses never enters the sum, so a family whose images disagree with every library lens is split. A component with more
than 20,000 partitions loses its least supported groups until it fits.

The number $S$ of new sources in Equation {eq}`eq-group-weight` is the number of groups that the catalog claims, plus
the number of groups linked at the initial lens. Two images are linked there when the evidence of the pair as one new
source exceeds the best alternative of each image (unrelated, or the best trusted family).

The source position prior has width `source_prior_sigma_arcsec`, which defaults to the width for the trusted sources.
The width sets the cost of a new source, about $\ln(2\pi s_\beta^2)$ in log evidence, and therefore how readily a lone
image joins a group. The same-source color scatter is fit on the trusted families unless `color_scatter` is set, and a
fraction `outlier_fraction` of the color terms uses a broad outlier scatter.

The deflection is proportional to the lensing efficiency except for the external shear, so one evaluation of the lens at
each uncertain image serves every source redshift. Under `gp_image_plane` the uncertain images are not correlated with
other images. With the knee kernel the deflection floor of an uncertain image carries the GRF amplitude weighted by $w$
at the source redshift (`knee_background` is not supported). With uncertain membership, each enumerated configuration
evaluates the uncertain images with the lens of that configuration, inside the membership sum. The marginalized
members enter through the expected mass, without the membership covariance at the uncertain images. The model assigns
detected images only. A family whose predicted bright images are undetected is not penalized, and two detections of one
image (knots, or a shredded arc) can attach to one source.

The file `catalog_path` is a CSV table with `image_label`, `ra` and `dec` (degrees), and optionally `parity`, redshifts
(`zspec_best`, `catalog_z` or `zphot_best`, with `catalog_z_sigma` or `pagul_zpdf_low` and `pagul_zpdf_high`) and
`mag_<band>` with `mag_err_<band>`. The table carries no positional errors, since the uncertain images take the sampled
error model of the trusted images. The file `population_catalog_path` is a CSV photometric catalog of the field with a
redshift column and the same bands. The final stage writes `final/artifacts/uncertain_images.h5`, which holds the
library groups, the per-draw assignment probabilities and sampled assignments, the averages of both, and the
probability that two images share a source. The final stage also writes the tables `uncertain_images.csv` and
`uncertain_pairs.csv`.

(sec-full-posterior)=
## 5. The full posterior

The sampled parameters are the lens $\Theta$, the error scales $\phi$ of $e$ and $\eta$, the trusted sources
$\beta = \{\beta_F\}$ and the membership calibration $\delta$. The data are the
trusted images $\hat\theta_F$, the uncertain images $D_j$ and the measured member velocity dispersions $\hat\sigma_m$.
Sections 2–4 each derived one part of the likelihood. We now assemble the parts in four steps, each adding one term
to the previous step.

First, the trusted images alone give the product of family likelihoods (§2.2),

$$
L_1(\Theta, \phi, \beta) = \prod_F p(\hat\theta_F \mid \beta_F, \Theta, \phi) .
$$ (eq-full-trusted)

The product runs over families, or over covariance blocks when the error model correlates families. A shared GRF makes
all trusted images one block. $L_1$ is the complete likelihood of a fit with no uncertain images and no uncertain
members.

Second, we add the uncertain images. Given the lens and the sources, the uncertain images are independent of the
trusted images, so the likelihood of
Equation {eq}`eq-partition-sum` multiplies $L_1$,

$$
L_2(\Theta, \phi, \beta) = L_1(\Theta, \phi, \beta)\, p(D_U \mid \Theta, \phi, \beta) .
$$ (eq-full-uncertain)

The two factors are coupled in the posterior. Both depend on the lens, and an uncertain image assigned to trusted
family $F$ is evaluated at the source $\beta_F$. A new counterimage therefore constrains $\beta_F$ and the lens as a
trusted image would, weighted by the assignment probability.

Third, we add the uncertain members. Each membership configuration is a different lens, and every image, trusted or
uncertain, sees the same lens. The members therefore do not add a factor. The members change the lens inside $L_2$,
and the configurations are summed
around $L_2$,

$$
L_3(\Theta, \phi, \beta, \delta) = \sum_{z_{\mathcal E}} p(z_{\mathcal E} \mid \delta)\;
L_2(\Theta, \phi, \beta;\, z_{\mathcal E}) .
$$ (eq-full-members)

Here, $L_2(\,\cdot\,;\, z_{\mathcal E})$ denotes $L_2$ evaluated with the deflection $\bar\alpha(z_{\mathcal E})$ and
the covariance $C + C_{\mathrm{mem}}$ of Equation {eq}`eq-membership-terms`. The marginalized members are inside
$L_2$, through the expected mass in $\bar\alpha$ and the fluctuation in $C_{\mathrm{mem}}$. The sum cannot be split into
separate sums for the trusted and the uncertain images, since the two sets of images would then see different lenses.

Finally, we add the data and terms that do not involve the images. The member dispersions depend only on the lens,
and the soft constraints act as prior factors, so both multiply $L_3$
outside the membership sum. With the priors, the sampled density is

$$
\begin{aligned}
p(\Theta, \phi, \beta, \delta \mid \mathcal{D}) \;\propto\;
& p(\Theta)\, p(\phi)\, p(\delta) \prod_F p(\beta_F) \\
& \times \sum_{z_{\mathcal E}} p(z_{\mathcal E} \mid \delta)
\Big[\prod_F p(\hat\theta_F \mid \beta_F, \Theta, \phi, z_{\mathcal E})
p(D_U \mid \Theta, \phi, \beta, z_{\mathcal E})\Big] \\
& \times \prod_m p(\hat\sigma_m \mid \Theta)\; B(\Theta) .
\end{aligned}
$$ (eq-full-posterior)

The first line holds the priors. The prior $p(\Theta)$ includes the halo, shear and scaling-relation priors
({doc}`configuration`) and the hierarchical prior of the member deviations, a Student-t or normal distribution of the
offset of each member from the scaling relation in units of a sampled population scatter. The
prior $p(\phi)$ holds the error-scale priors of the chosen error model. The second line is the membership sum of
Equation {eq}`eq-full-members` around the trusted and uncertain images. Under each configuration, the trusted images
use the family likelihood with the covariance $C$ of the error model plus $C_{\mathrm{mem}}$, and the uncertain images
use the partition sum with the lens of that configuration.

The third line holds the kinematics and the soft constraints. The measured stellar velocity dispersions are independent
data given the lens (Bergamini et al. 2019), with $\ln \hat\sigma_m \sim N\big(\ln(c\, \sigma_m(\Theta)),\,
e_m^2\big)$. Here, $\sigma_m$ is
the model dispersion of the member (the scaling relation times the deviation of the member), $c$ is the aperture factor,
and $e_m$ combines the measurement and systematic errors in $\ln\sigma$ (`MemberPopulationConfig.kinematics_sources`).
The soft constraints $B(\Theta)$ are not data. These terms keep the model where the approximations hold. The far-field
guard adds $-s\,\mathrm{softplus}(-x/w)$ for each far-field member, with $x = \ln(R/(k\, r_{\mathrm{cut}}))$ and $R$ the
distance from the member to the nearest image. The guard keeps every image outside $k$ cut radii
(`far_field_cut_radii`) of a member evaluated with the far-field series, with strength $s$
(`far_field_barrier_strength`) and width $w$ (`far_field_barrier_width`). The ellipticity caps,
$-\mathrm{softplus}((\lvert\epsilon\rvert - \epsilon_{\max})/w)$, keep the halo ellipticities below a smooth limit.

Every latent quantity is removed inside the likelihood, so NUTS samples only continuous parameters. Table 4 lists how
each latent quantity is removed.

| latent quantity | removed by | location in the likelihood |
|---|---|---|
| true image positions, deflection error $\eta$ | Gaussian integral (§2.1) | $N(F;\, 0,\, C)\, h$ of every image |
| marginalized member indicators $z_{\mathcal M}$ | Gaussian approximation (§3.3) | mean in $\bar\alpha$, fluctuation in $C_{\mathrm{mem}}$ |
| enumerated member indicators $z_{\mathcal E}$ | exact sum over $2^K$ terms (§3.2) | outer sum of Equation {eq}`eq-full-members` |
| partition of the uncertain images $Z$, new sources $\psi$ | exact sum over library partitions, analytic source integral (§4.4–4.5) | $p(D_U \mid \Theta, \phi, \beta)$ in Equation {eq}`eq-full-uncertain` |

In logarithms, the target that NUTS adds to the priors is

$$
\begin{aligned}
\ln L = \operatorname*{logsumexp}_{z_{\mathcal E}}\Big[&\ln p(z_{\mathcal E} \mid \delta)
+ \sum_{\text{blocks}} \ln N(F;\, 0,\, C) + \sum_i \ln h_i
+ \ln p(D_U \mid \Theta, \phi, \beta)\Big] \\
&+ \sum_m \ln p(\hat\sigma_m \mid \Theta) + \ln B(\Theta),
\end{aligned}
$$ (eq-log-likelihood)

plus the change-of-variables corrections of the sampled source offsets (Equation {eq}`eq-source-correction`).
The method `Evaluator.log_likelihood` evaluates the bracket once per configuration, vectorized over the $2^K$
configurations, and takes the `logsumexp`. Each step drops out when unused. With every $p_i = 1$ the outer sum has one
term, and with no uncertain images the partition sum is 1, which leaves $L_1$.

The likelihood does not include the number of images. Each observed image is matched to the prediction near the
observed position, so an image that the model predicts but the catalog lacks has no cost. The posterior image
diagnostics check the image counts after the fit ({doc}`outputs`).

(sec-error-models)=
## 6. Error models

The derivation of §2 leaves the covariance of the measurement errors $e$ and the deflection error $\eta$ open. An error
model fixes that covariance, and nothing else in the likelihood changes between models. The model is chosen with
`LikelihoodConfig.likelihood_mode` and `gp_kernel` (Table 5).

| name | `likelihood_mode` | `gp_kernel` | deflection error $\eta$ | correlated image-plane displacement |
|---|---|---|---|---|
| GRF (production) | `gp_image_plane` | `knee` | Gaussian random field over all images | image-plane GP |
| image-plane GP | `gp_image_plane` | `image_plane` | none (small floor) | image-plane GP within each family |
| smoothed (library default) | `smoothed` | | independent per image | none |

(sec-grf)=
### 6.1. Gaussian random field of unmodeled mass

The production fits use this likelihood. The residual of image $i$, with family redshift weight $w_i$ and inverse
magnification matrix $A_i$, is

$$
F_i = b_i - \beta_{k(i)} = w_i\, \delta\alpha(\theta_i) + A_i\, \big(d(\theta_i) + e_i\big) .
$$ (eq-grf-residual)

Here, $\delta\alpha$ is the deflection error from mass that the parametric model lacks. We model $\delta\alpha$ as a
curl-free Gaussian random field on the lens potential, with the convergence power spectrum
$P_\kappa(k) = P_0 / [1 + (k/k_t)^\gamma]$ between fixed wavelength cutoffs. The amplitude $\sigma_{\mathrm{GRF}}$ is
the
rms deflection error per component for a source at the reference redshift (`gp_reference_redshift`, 2), and the knee
wavelength $\lambda_{\mathrm{b}} = 2\pi/k_t$ sets the scale of the unmodeled mass. The term $d$ is a correlated
image-plane displacement with an isotropic Gaussian kernel of length $\ell$ and amplitude $\sigma_{\mathrm{GP}}$, which
describes, for example, measurement systematics shared by nearby images. The term $e_i$ is the independent measurement
error, with a scale $\sigma_{\mathrm{meas}}$ that is sampled and shared by all images. The covariance of the stacked
residuals of a block of images is

$$
C = W K_\alpha W + B\,[\sigma_{\mathrm{GP}}^2 K_{\mathrm{img}} + \sigma_{\mathrm{meas}}^2 I]\,B^T
+ \sigma_{\mathrm{fl}}^2 I,
$$ (eq-grf-covariance)

where $W$ holds the redshift weights, $B = \mathrm{blockdiag}(A_i)$, $K_\alpha$ holds the longitudinal and transverse
correlations of the deflection field (tabulated Hankel transforms of $P_\kappa$), and $\sigma_{\mathrm{fl}}$
(`gp_floor_arcsec`, 1.5 mas) is a small smoothness floor.

The option `gp_blocks` sets which images share one field. With `all`, the default for full runs, one mass field covers
every image, and the field-wide translation, convergence and shear are projected out (`gp_knee_project_modes`), because
the sources and the external shear absorb these modes. With `system`, each system (all knots of one galaxy) has an
independent field and no projection is applied. This choice is about five times cheaper per gradient but gives a
different posterior, so we use it only for quick tests. With `family`, each family has an independent field. The
initial search always uses system blocks, while the chain-start step and NUTS use the configured blocks.

The sampled hyperparameters are $\sigma_{\mathrm{GRF}}$, $\lambda_{\mathrm{b}}$, $\ell$, $\sigma_{\mathrm{GP}}$ and
$\sigma_{\mathrm{meas}}$. The slope $\gamma$ is fixed at 4 unless `gp_knee_gamma_prior` frees the slope, with a
`dist.Uniform` or a two-sided `dist.TruncatedNormal` whose bounds also set the range of the tables. A slope of 2
corresponds to truncated isothermal lumps, 4 to NFW-like cusps and 6 to cored lumps. The prior on the knee keeps the
field at the galaxy scale. Left free, the knee moved to 14–25″ and absorbed the halo-scale misfit. The option
`lambda_b_fixed_arcsec` holds the knee at one value instead. The production fits use the following settings.

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
    lambda_b_prior=truncated_lognormal(6.0, 0.6, low=2.5, high=8.0),
    lambda_b_start_arcsec=6.0,
    sigma_grf_prior=truncated_lognormal(0.08, 0.5, low=1e-4, high=0.5),
    sigma_grf_start_arcsec=0.08,
    gp_knee_ell_gp_prior=truncated_lognormal(1.0, 0.4, low=0.4, high=3.0),
    gp_knee_ell_gp_start_arcsec=1.0,
)
config = replace(config, likelihood=grf_likelihood)
```

(sec-image-plane-gp)=
### 6.2. Image-plane Gaussian process

This model keeps the image-plane GP alone, per family, with no source-plane floor. Within a family the image-plane
displacements are correlated, with covariance

$$
S = \sigma_{\mathrm{GP}}^2 (K \otimes I) + \sigma_{\mathrm{meas}}^2 I, \qquad
K_{ij} = \exp\!\left(-\frac{\lvert\theta_i - \theta_j\rvert^2}{2\ell^2}\right) .
$$ (eq-image-plane-gp)

The stacked source-plane residuals have covariance $C = B S B^T + \sigma_{\mathrm{fl}}^2 I$, with
$B = \mathrm{blockdiag}(A_i)$. One Cholesky factor of $C$ per family gives the Gaussian term and the conditional source
position. The sampled scales are $\sigma_{\mathrm{GP}}$ (log-normal with median 0.15″, truncated at 1″),
$\sigma_{\mathrm{meas}}$, and $\ell$ (`gp_correlation_length_prior`, log-normal with median 7″, truncated at 30″).
Setting `gp_correlation_length_prior=None` fixes $\ell$ at `gp_correlation_length_arcsec`.

(sec-smoothed)=
### 6.3. Smoothed local likelihood

The library default compares the source-plane residual of each image with a Gaussian of covariance

$$
C_i = \sigma_{\mathrm{meas}}^2 A_i A_i^T + \sigma_{\mathrm{def}}^2 I + \sigma_{\mathrm{sel}}^2 R_i, \qquad
R_i = (I + A_i A_i^T / \lambda_{\mathrm{sel}}^2)^{-1} .
$$ (eq-smoothed)

The three terms are the propagated measurement error, an isotropic residual-deflection scatter $\sigma_{\mathrm{def}}$,
and a feature-selection term with stretching weight $R_i$. The scales $\sigma_{\mathrm{meas}}$, $\sigma_{\mathrm{def}}$,
$\sigma_{\mathrm{sel}}$ and $\lambda_{\mathrm{sel}}$ are sampled. Setting `sigma_sel_prior=None` drops the selection
term, as the production fits do.

(sec-measurement-errors)=
### 6.4. Measurement errors

There is no fixed astrometric error. The scale $\sigma_{\mathrm{meas}}$ is the only measurement error, shared by every
image and always sampled, with the prior `sigma_meas_prior` (`smoothed` and source-plane modes), `gp_sigma_meas_prior`
(`gp_kernel="image_plane"`) or `gp_knee_sigma_meas_prior` (knee kernel). The catalogs carry no per-image errors, and the
uncertain images take the same error model as the trusted images.

(sec-diagnostic-likelihoods)=
## 7. Diagnostic likelihoods

The modes `source-anisotropic` and `source-isotropic` keep only the Gaussian source-plane term, without $h$. The
isotropic mode also drops the Jacobian from $C$. Both modes are cheap and biased toward demagnifying lenses (§2.1), and
are kept for diagnostics only.

## References

- Bergamini, P., Rosati, P., Mercurio, A., et al. 2019, A&A, 631, A130
- Dempster, A. P., Laird, N. M., & Rubin, D. B. 1977, Journal of the Royal Statistical Society B, 39, 1
- Grillo, C., Suyu, S. H., Rosati, P., et al. 2015, ApJ, 800, 38
- Hoffman, M. D., & Gelman, A. 2014, Journal of Machine Learning Research, 15, 1593
