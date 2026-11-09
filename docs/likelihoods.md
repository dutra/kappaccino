# Likelihoods

The data are the observed positions $\hat\theta_i$ of multiple images, grouped into families: the images of one source
point. This page derives the position likelihood from the lens equation for one image, then for a family. It then
extends the likelihood to members that may not belong to the cluster and to images whose family is unknown, and
assembles the full posterior with every other term (kinematics, priors). The error models that follow (GRF,
image-plane GP, smoothed) differ only in the covariance they assign to the errors in that derivation.

Every step uses the same three rules: a change of variables to go from the source plane to the image plane, the sum
rule to remove a latent quantity (a true position, a membership, a family assignment), and conditional independence to
factor what does not interact.

## The position likelihood

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

### Uncertain members

So far the lens was known up to its parameters $\Theta$. A member catalog, though, is a classification, and some of its
galaxies may not belong to the cluster. Give each member $i$ an indicator $z_i \in \{0, 1\}$: with $z_i = 1$ it is on
the scaling relation, with $z_i = 0$ it has no mass. Its prior is the classifier's membership probability,

$$
p(z) = \prod_i p_i^{z_i} (1 - p_i)^{1 - z_i} ,
$$

and a confirmed member has $p_i = 1$ ([Prior probabilities](#prior-probabilities)).

**The deflection is linear in the indicators.** A member's deflection, its Jacobian and the derivative of the
Jacobian are all proportional to its mass normalization, so

$$
\alpha(\theta;\, \Theta, z) = \alpha_0(\theta;\, \Theta) + \sum_i z_i\, \alpha_i(\theta;\, \Theta),
$$

with $\alpha_0$ the halos, shear, sheet and certain members, and $\alpha_i$ the deflection of uncertain member $i$ at
full mass. The likelihood of the images for one value of the indicators, $p(\hat\theta \mid \Theta, z)$, is the family
likelihood above with this deflection.

**Sum rule.** The indicators are latent, like the true image positions, and are removed the same way:

$$
p(\hat\theta \mid \Theta) = \sum_{z \in \{0,1\}^N} p(z)\; p(\hat\theta \mid \Theta, z) .
$$

The sum has $2^N$ terms and does not factor over members, because every image feels every member. Split the uncertain
members into a small set $\mathcal{E}$ that can decide an image's position (the enumerated members, $K$ of them) and
the rest, $\mathcal{M}$ ([Which members are enumerated](#which-members-are-enumerated)). The sum splits accordingly:

$$
p(\hat\theta \mid \Theta) = \sum_{z_{\mathcal E}} p(z_{\mathcal E})
\underbrace{\sum_{z_{\mathcal M}} p(z_{\mathcal M})\; p(\hat\theta \mid \Theta, z_{\mathcal E}, z_{\mathcal M})}_{\text{inner sum}} .
$$

**The inner sum as a deflection error.** Write each marginalized indicator as its mean plus a fluctuation,
$z_g = p_g + (z_g - p_g)$. The deflection becomes

$$
\alpha = \underbrace{\alpha_0 + \sum_{i \in \mathcal E} z_i\, \alpha_i + \sum_{g \in \mathcal M} p_g\, \alpha_g}_{\bar\alpha(z_{\mathcal E}):\ \text{the model deflection}}
\;+\; \underbrace{\sum_{g \in \mathcal M} (z_g - p_g)\, \alpha_g}_{\eta_{\mathrm{mem}}} .
$$

The fluctuation $\eta_{\mathrm{mem}}$ enters the lens equation exactly where the deflection error $\eta$ of
[One image](#one-image) does. The indicators are independent with variance $p_g(1 - p_g)$, so $\eta_{\mathrm{mem}}$
has zero mean and, between images $j$ and $k$, covariance

$$
C_{\mathrm{mem}}^{(jk)} = \sum_{g \in \mathcal M} p_g (1 - p_g)\, \alpha_g(\theta_j)\, \alpha_g(\theta_k)^T .
$$

It is a sum of many small independent terms, so approximate it as Gaussian. The inner sum is then the integral over
$\eta$ already done in [One image](#one-image), with one more term in its covariance:

$$
\Sigma_\eta \;\to\; \Sigma_\eta + C_{\mathrm{mem}} .
$$

$C_{\mathrm{mem}}$ correlates images $j$ and $k$, because one missing member moves all of its nearby images together.
Like $\eta$, it enters the Gaussian term and, through the conditioning, the smoothed determinant $h$.

**Result.** With $C(z_{\mathcal E})$ the family covariance including $C_{\mathrm{mem}}$,

$$
\boxed{\;p(\hat\theta \mid \Theta) = \sum_{z_{\mathcal E} \in \{0,1\}^K} p(z_{\mathcal E})
\prod_f p\big(\hat\theta_f \mid \beta_f, \Theta;\ \bar\alpha(z_{\mathcal E}),\ \Sigma_\eta + C_{\mathrm{mem}}\big)\;}
$$

a sum of $2^K$ position likelihoods, evaluated as a `logsumexp` over the configurations. The sources and every
continuous parameter are shared by all the configurations. NUTS samples only the continuous parameters; the
indicators never appear as sampled variables.

**Posterior membership.** At a posterior draw, configuration $z_{\mathcal E}$ has weight
$w(z_{\mathcal E}) \propto p(z_{\mathcal E})\, p(\hat\theta \mid \Theta, z_{\mathcal E})$, and the posterior
membership of enumerated member $i$ is the total weight of the configurations with $z_i = 1$, averaged over the draws.
A marginalized member is integrated out without being tracked, so its posterior is its prior: the approximation
propagates the uncertainty these members add to the mass model, but does not update their membership.

**Calibration.** With `membership_calibration_prior`, a shared shift $\delta$ of the log-odds is sampled,
$p_i \to \mathrm{sigmoid}(\mathrm{logit}\,p_i + \delta)$. It changes $p(z_{\mathcal E})$ and $C_{\mathrm{mem}}$ and
gets its own prior factor $p(\delta)$.

With every $p_i = 1$, $\mathcal E$ and $\mathcal M$ are empty, the sum has one term and $C_{\mathrm{mem}} = 0$.

**In the full likelihood**, the uncertain members do not add a factor. They change the lens, so the sum over
$z_{\mathcal E}$ wraps every image term, trusted and uncertain (step 3 of [The full posterior](#the-full-posterior)).

### Uncertain images

So far every image had a known family. Some candidate images do not: they may be a new image of a trusted family, an
image of a new source shared with other candidates, or unrelated to every other image. The trusted families keep their
catalog assignment; only the uncertain images $j = 1, \dots, U$ have an unknown source. Each carries data
$D_j = (\hat\theta_j, \hat z_j, \hat c_j)$: a position, a catalog redshift (with an error $\sigma_{z,j}$, or none) and
colors.

**Sources it can come from.** Call each possible source a slot. The trusted families are slots whose source $\beta_F$
is the one already sampled, at the family redshift $z_F$. The new slots $k = 1, \dots, S$ are extra sources with
parameters $\psi_k = (\beta_k, z_k, c_k)$ (position, redshift and intrinsic colors), all sampled, with the prior of a
field source:

$$
p(\psi_k) = N(\beta_k;\, \mu_\beta, s_\beta^2 I)\; q(z_k, c_k),
$$

where $q$ is a Gaussian mixture fit to a photometric catalog of the field. Image $j$ has a latent label $\ell_j$: an
unrelated source ($\ell_j = 0$), a trusted family $F$, or a new slot $k$, with fixed prior weights $\pi_\ell$.

**One image from one slot.** If image $j$ came from a slot with source $\beta$ at redshift $z$ and colors $c$, its
position term is the [One image](#one-image) result at that redshift. The lensing efficiency $D_{ls}/D_s$ scales the
deflection (and the GRF weight $w$), so one evaluation of the lens at $\hat\theta_j$ serves every slot redshift. Its
redshift and colors are independent data given the source:

$$
p(D_j \mid \beta, z, c) =
\underbrace{N\big(F_j(z);\, 0,\, C_j(z)\big)\, h_j(z)}_{\text{position, at the slot's redshift}}
\;\underbrace{N(\hat z_j;\, z,\, \sigma_{z,j}^2)}_{\text{redshift}}
\;\underbrace{p(\hat c_j \mid c)}_{\text{colors}},
\qquad F_j(z) = b_j(z) - \beta .
$$

The colors scatter about the source's with a same-source scatter plus a broad outlier component. For a trusted slot
the source colors are not sampled: the color term is the predictive given the family's own images.

**An unrelated image.** If image $j$ is the only image of its source, that source is unknown and drawn from the prior,
so the term is the evidence of the image alone:

$$
E_j = \int p(D_j \mid \beta, z, c)\; N(\beta;\, \mu_\beta, s_\beta^2 I)\; q(z, c)\; d\beta\, dz\, dc .
$$

The position term is Gaussian in $\beta$, so the $\beta$ integral is done analytically (with $h_j$ at the conditional
mean of $\beta$), the colors analytically against the mixture, and the redshift on a grid.

**Conditional independence and the sum rule.** Given the lens, the trusted sources and the slot parameters, the
uncertain images are independent: each gets the diagonal block of the configured covariance, so no error correlates
two of them. The labels then sum out one image at a time,

$$
\boxed{\;p(D_U \mid \Theta, \phi, \beta, \psi) = \prod_{j=1}^{U}
\Big[\pi_0\, E_j + \sum_F \pi_F\, p(D_j \mid \beta_F, z_F) + \sum_{k=1}^{S} \pi_k\, p(D_j \mid \psi_k)\Big]\;}
$$

with $U \times (1 + N_F + S)$ terms in all.

**Why this is the sum over family assignments.** Expanding the product of sums gives one term per labeling of the
images. In a labeling, the images that share new slot $k$ all depend on the same $\psi_k$. Integrating $\psi_k$ over
its prior turns their product into the evidence of that group as one family,

$$
\int \prod_{j:\, \ell_j = k} p(D_j \mid \psi_k)\; p(\psi_k)\, d\psi_k = E_{\{j:\, \ell_j = k\}},
$$

and an empty slot integrates to 1. So integrating every new slot gives

$$
\int p(D_U \mid \Theta, \phi, \beta, \psi)\, p(\psi)\, d\psi = \sum_{Z} p(Z) \prod_{G \in Z} E_G ,
$$

the sum over every partition $Z$ of the uncertain images into groups $G$, with the prior $p(Z)$ that the weights
induce. That sum has a Bell number of terms. Keeping $\psi$ as parameters and letting NUTS integrate it gives the
same posterior over the lens with one term per image and slot, and no search over candidate partitions.

**Posterior assignment.** At a posterior draw the labels are independent, with probabilities
$r_{j\ell} \propto \pi_\ell\, p(D_j \mid \ell)$ (with $E_j$ for $\ell = 0$). Their average over the draws is the
assignment probability; sampling a label per image at every draw gives joint draws of the lens, the sources and the
partition. The probability that two images share a source is $\sum_{\ell \ne 0} r_{j\ell}\, r_{k\ell}$, averaged over
the draws.

Like the trusted sources, the new slots are sampled as unit offsets, with a correction term that turns their
unit-normal prior into $p(\psi_k)$.

With no uncertain images, $U = 0$ and the product is 1.

**In the full likelihood**, the uncertain images add one factor, the boxed product, next to the trusted families.
It shares the lens and the trusted sources $\beta_F$ with them (step 2 of [The full posterior](#the-full-posterior)).

## The full posterior

The sampled parameters are the lens $\Theta$ (halos, shear, sheet, scaling relation, member deviations), the error
scales $\phi$ (of $e$ and $\eta$), the trusted sources $\beta = \{\beta_F\}$, the new slots $\psi = \{\psi_k\}$ and
the membership calibration $\delta$. The data are the trusted images $\hat\theta_F$, the uncertain images $D_j$ and
the member dispersions $\hat\sigma_m$. The likelihood is assembled in four steps, each adding one piece to the one
before.

**Step 1: trusted images.** For a fixed lens, the trusted images give

$$
L_1(\Theta, \phi, \beta) = \prod_F p(\hat\theta_F \mid \beta_F, \Theta, \phi) ,
$$

the family likelihood of [A family of images](#a-family-of-images). The product is over families, or over covariance
blocks when the error model correlates families (a shared GRF makes all trusted images one block). $L_1$ is the whole
likelihood of a fit with no uncertain images or members.

**Step 2: add the uncertain images, as a factor.** Given the lens and the sources, the uncertain images are
independent of the trusted ones, so their likelihood multiplies $L_1$:

$$
L_2(\Theta, \phi, \beta, \psi) = L_1(\Theta, \phi, \beta) \times
\underbrace{\prod_j \sum_\ell \pi_\ell\, p(D_j \mid \ell, \Theta, \phi, \beta, \psi)}_{\text{uncertain images}} ,
$$

the boxed result of [Uncertain images](#uncertain-images).

The two factors are not independent of each other in the posterior: they share the lens, and an uncertain image that
joins trusted family $F$ is evaluated at that family's source $\beta_F$. So a new counter-image constrains $\beta_F$
and the lens, exactly as a trusted image would, weighted by its assignment probability.

**Step 3: add the uncertain members, as a sum around everything.** Each member configuration is a different lens, and
every image, trusted or uncertain, sees the same lens. So the members do not add a factor: they change the lens inside
$L_2$, and the configurations are summed around it. With $L_2(\,\cdot\,;\, z_{\mathcal E})$ meaning $L_2$ evaluated
with the deflection $\bar\alpha(z_{\mathcal E})$ and the covariance $C + C_{\mathrm{mem}}$,

$$
L_3(\Theta, \phi, \beta, \psi, \delta) =
\underbrace{\sum_{z_{\mathcal E}} p(z_{\mathcal E} \mid \delta)}_{\text{enumerated members}}\;
L_2(\Theta, \phi, \beta, \psi;\, z_{\mathcal E}) .
$$

The marginalized members are already inside $L_2$: their expected mass is in $\bar\alpha$ and their fluctuation is
$C_{\mathrm{mem}}$ ([Uncertain members](#uncertain-members)). The sum cannot be split into one sum for the trusted
images and another for the uncertain ones, since that would let the two sets of images see different lenses.

**Step 4: add the data and terms that do not depend on the images.** The member dispersions depend only on the lens,
and the soft constraints are prior-like factors, so both multiply $L_3$ outside the membership sum. With the priors,
the sampled density is

$$
\begin{aligned}
p(\Theta, \phi, \beta, \psi, \delta \mid \mathcal{D}) \;\propto\;
& \underbrace{p(\Theta)\, p(\phi)\, p(\delta) \prod_F p(\beta_F) \prod_k p(\psi_k)}_{\text{priors}} \\
& \times \underbrace{\sum_{z_{\mathcal E}} p(z_{\mathcal E} \mid \delta)}_{\text{step 3}}
\Big[\underbrace{\prod_F p(\hat\theta_F \mid \beta_F, \Theta, \phi, z_{\mathcal E})}_{\text{step 1: trusted images}}
\;\underbrace{\prod_j \sum_{\ell} \pi_\ell\, p(D_j \mid \ell, \Theta, \phi, \beta, \psi, z_{\mathcal E})}_{\text{step 2: uncertain images}}\Big] \\
& \times \underbrace{\prod_m p(\hat\sigma_m \mid \Theta)\; B(\Theta)}_{\text{step 4: kinematics, soft constraints}} .
\end{aligned}
$$

Every latent quantity has been removed inside the likelihood, so NUTS samples only continuous parameters:

| latent quantity | removed by | where it is in the likelihood |
|---|---|---|
| true image positions, deflection error $\eta$ | Gaussian integral ([One image](#one-image)) | $N(F;\, 0,\, C)\, h$ of every image |
| marginalized member indicators $z_{\mathcal M}$ | Gaussian approximation ([Uncertain members](#uncertain-members)) | mean in $\bar\alpha$, fluctuation in $C_{\mathrm{mem}}$ |
| enumerated member indicators $z_{\mathcal E}$ | exact sum, $2^K$ terms | outer sum (step 3) |
| uncertain-image labels $\ell_j$ | exact sum, per image ([Uncertain images](#uncertain-images)) | inner sum of each uncertain image (step 2) |

In logs, the target that NUTS adds to the latent priors is

$$
\ln L = \operatorname*{logsumexp}_{z_{\mathcal E}}\Big[\ln p(z_{\mathcal E} \mid \delta)
+ \underbrace{\sum_{\text{blocks}} \ln N(F;\, 0,\, C) + \sum_i \ln h_i}_{\text{trusted images}}
+ \underbrace{\sum_j \operatorname*{logsumexp}_{\ell}\big(\ln \pi_\ell + \ln p(D_j \mid \ell)\big)}_{\text{uncertain images}}\Big]
+ \sum_m \ln p(\hat\sigma_m \mid \Theta) + \ln B(\Theta),
$$

plus the change-of-variables corrections of the sampled source and slot offsets. This is `Evaluator.log_likelihood`:
it evaluates the bracket once per configuration (vectorized over the $2^K$ configurations) and takes the
`logsumexp`. Each step drops out when it is not used: with every $p_i = 1$ the outer sum has one term, and with no
uncertain images the inner product is 1, which leaves step 1.

Each factor:

Priors $p(\Theta)$, $p(\phi)$, $p(\delta)$, $p(\beta_F)$, $p(\psi_k)$
: The halo, shear and scaling-relation priors ({doc}`configuration`), the hierarchical prior of the member deviations
  (each member's offset from the scaling relation, Student-t or normal in units of a sampled population scatter),
  the error-scale priors of the chosen error model, the membership calibration prior, and the source priors of the
  trusted families and new slots.

Trusted images $p(\hat\theta_F \mid \beta_F, \Theta, \phi, z_{\mathcal E})$
: The family likelihood of [A family of images](#a-family-of-images), with the deflection of configuration
  $z_{\mathcal E}$. The error model fixes the covariance $C$, that is, $\Sigma_\eta$ and any correlated displacement
  (the sections below), and $C_{\mathrm{mem}}$ is added to it ([Uncertain members](#uncertain-members)).

Membership $\sum_{z_{\mathcal E}} p(z_{\mathcal E} \mid \delta)$
: The exact sum over the enumerated members ([Uncertain members](#uncertain-members)).

Uncertain images $\prod_j \sum_\ell \pi_\ell\, p(D_j \mid \ell)$
: The slot mixture ([Uncertain images](#uncertain-images)), with the lens of configuration $z_{\mathcal E}$. The
  marginalized members enter it through their expected mass, without $C_{\mathrm{mem}}$.

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
  amplitude $\sigma_{\mathrm{GRF}}$ is the rms deflection error per component for a source at the reference redshift
  (`gp_reference_redshift`, 2). The knee wavelength $\lambda_t = 2\pi/k_t$ sets the scale of the unmodeled mass.

$d$
: A correlated image-plane displacement with an isotropic Gaussian kernel of length $\ell$ and amplitude
  $\sigma_{\mathrm{GP}}$ (for example, measurement systematics shared by nearby images).

$e_i$
: The independent measurement error $\sigma_{\mathrm{meas}}$, sampled and the same for every image.

The covariance of the stacked residuals of a block of images is

$$
C = W K_\alpha W + B\,[\sigma_{\mathrm{GP}}^2 K_{\mathrm{img}} + \sigma_{\mathrm{meas}}^2 I]\,B^T + \sigma_{\mathrm{fl}}^2 I,
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

The sampled hyperparameters are $\sigma_{\mathrm{GRF}}$, the knee $\lambda_t$, the image-plane $\ell$,
$\sigma_{\mathrm{GP}}$ and $\sigma_{\mathrm{meas}}$; the slope $\gamma$ is fixed at 4 unless
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
    sigma_grf_prior=truncated_lognormal(0.08, 0.5, low=1e-4, high=0.5),
    sigma_grf_start_arcsec=0.08,
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
S = \sigma_{\mathrm{GP}}^2 (K \otimes I) + \sigma_{\mathrm{meas}}^2 I, \qquad
K_{ij} = \exp\!\left(-\frac{|\theta_i - \theta_j|^2}{2\ell^2}\right),
$$

and the stacked source-plane residuals have covariance $C = B S B^T + \sigma_{\mathrm{fl}}^2 I$ with
$B = \mathrm{blockdiag}(A_i)$. One Cholesky factor of $C$ per family gives the Gaussian term and the conditional
source position.

The sampled scales are $\sigma_{\mathrm{GP}}$ (log-normal, median 0.15", truncated at 1"), $\sigma_{\mathrm{meas}}$
and $\ell$ (`gp_correlation_length_prior`, log-normal, median 7", truncated at 30"). Set
`gp_correlation_length_prior=None` to fix $\ell$ at `gp_correlation_length_arcsec`.

### Smoothed local likelihood (`smoothed`)

The library default. Each image's source-plane residual is compared with a Gaussian of covariance

$$
C_i = \sigma_{\mathrm{meas}}^2 A_i A_i^T + \sigma_{\mathrm{def}}^2 I + \sigma_{\mathrm{sel}}^2 R_i, \qquad
R_i = (I + A_i A_i^T / \lambda_{\mathrm{sel}}^2)^{-1},
$$

the propagated measurement error $\sigma_{\mathrm{meas}}$, an isotropic residual-deflection scatter
$\sigma_{\mathrm{def}}$, and a feature-selection term with stretching weight $R_i$. $\sigma_{\mathrm{meas}}$,
$\sigma_{\mathrm{def}}$, $\sigma_{\mathrm{sel}}$ and $\lambda_{\mathrm{sel}}$ are sampled. Set `sigma_sel_prior=None`
to drop the selection term (as the production fits do).

### Measurement errors

There is no fixed astrometric error. $\sigma_{\mathrm{meas}}$ is the only measurement error, the same for every image
and always sampled: `sigma_meas_prior` (`smoothed` and source-plane modes), `gp_sigma_meas_prior`
(`gp_kernel="image_plane"`) or `gp_knee_sigma_meas_prior` (knee kernel). Catalogs carry no per-image errors, and
uncertain images take the same error model as the trusted ones.

## Uncertain membership

Member catalogs are built by classification. Spectroscopic members (a redshift within a few thousand km/s of the
cluster) are close to certain. Fainter members come from their colors: a red-sequence cut or a classifier trained on
the spectroscopic members (Grillo et al. 2015 and later CLASH-VLT and MUSE catalogs), with a purity of about
90–95% at the faint end. A published catalog applies a threshold and lists every galaxy above it as a member. With
`MemberConfig(membership="marginalize")` each member instead carries a prior probability $p_i$ of belonging to the
cluster, and the images update it. [Uncertain members](#uncertain-members) derives the likelihood; this section
covers the settings.

### Which members are enumerated

Enumerated members
: Uncertain members that the member classification already made exact (see {doc}`configuration`, Exact and
  far-field members), at most `membership_max_enumerated` (default 4). When there are more, the ones with the
  highest classification score are enumerated, as for `top_exact`. The score is the larger of the truncation term
  and the whitened far-field error over `template_tolerance`, so it measures how much the member matters at the
  images relative to their errors. These members can decide an image's position, so their $2^K$ configurations are
  summed exactly.

Marginalized members
: Every other uncertain member, which enters through its expected mass and the covariance $C_{\mathrm{mem}}$. Under
  the GRF likelihood $C_{\mathrm{mem}}$ adds to $W K_\alpha W$ after the low-order projection, so the shared field's
  blocks gain the cross terms. Under the image-plane GP it adds to each family's covariance, and under the smoothed
  likelihood to each image's $C_i$.

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

## Fitting uncertain images

Candidate images whose family is unknown can be fit with the lens (`PartitionConfig`, `config.partition`). Each
uncertain image may be a new image of a trusted family, an image of a new source it shares with other uncertain images,
or unrelated to every other image. The trusted families of the image catalog are kept as given, and the choice for each
uncertain image is summed out of the likelihood, so NUTS samples the lens with the assignment marginalized
([Uncertain images](#uncertain-images) derives the likelihood; this section covers the settings).

```python
from kappaccino.config import PartitionConfig

config = replace(config, partition=PartitionConfig(catalog_path="uncertain_images.csv",
                                                   population_catalog_path="field_photometry.csv"))
```

The init stage fits the trusted families as usual; the uncertain images enter at the final stage.

### Slot settings

The label weights are fixed: $\pi_0$ = `unrelated_fraction` (0.5), the rest shared equally by the trusted and new
slots. The population mixture $q(z, c)$ is fit to `population_catalog_path` in (redshift, colors), and the source
position prior has width `source_prior_sigma_arcsec` (default: that of the trusted sources). That width sets the cost
of a source, about $\ln(2\pi s_\beta^2)$ in evidence, and so how readily a lone image joins one. The same-source color
scatter is fit on the trusted families unless `color_scatter` is set, and a fraction `outlier_fraction` of the color
terms uses a broad outlier scatter.

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

The final stage writes `final/artifacts/slots.h5` (per-draw option probabilities and sampled assignments, their
averages, the probability that two images share a source, and the new slots' redshifts and positions) and the tables
`uncertain_images.csv` and `uncertain_pairs.csv`. The new slots' sampled parameters (`slot.<k>.*`) are unit offsets.

## Diagnostic likelihoods

`source-anisotropic` and `source-isotropic` keep only the Gaussian source-plane term, without $h$ (the second also
without the Jacobian in $C$). They are cheap and biased toward demagnifying lenses (see [One image](#one-image)), and
are kept for diagnostics.
