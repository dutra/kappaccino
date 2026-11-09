# Model comparison

Every run reports numbers for comparing models fitted to the same data. Use these numbers to answer questions such as "does a
gNFW halo fit better than an NFW halo?" or "does a scaling relation that depends on the distance to the cluster center
fit better than one that does not?".

| number | where | cost | use |
|---|---|---|---|
| BIC | `init/artifacts/model_comparison.json`, always | none beyond the init stage | screening many variants, and comparing with published BIC values |
| WBIC | `final/artifacts/model_comparison.json`, with `SamplingConfig(wbic=True)` | one more NUTS run (about twice the sampling time) | the number to report |

Both estimate $-2 \ln Z$, minus twice the log evidence. Lower is better. The difference between two models,
$\Delta = X_\mathrm{A} - X_\mathrm{B}$, estimates $-2 \ln B_\mathrm{BA}$, so a positive $\Delta$ favors model B.

## Getting the numbers

The init stage always computes BIC. To get WBIC as well, set `wbic=True` in the sampling settings:

```python
from dataclasses import replace
from kappaccino.config import SamplingConfig
from kappaccino.pipeline import run

config = replace(config, sampling=replace(config.sampling, wbic=True))
result = run(config)
```

From the launcher, add `--wbic`:

```bash
xonsh run.xsh M0416_CANUCS2025 --run-label nfw --wbic
```

WBIC can be added to a finished run. Run the same configuration again under the same run name with `wbic=True` (or
`--wbic`). The saved posterior is reloaded and only the WBIC run is added. Changing any other setting under the same
run name is still refused.

`SamplingConfig.wbic_warmup` and `SamplingConfig.wbic_samples` set the length of the WBIC run. They default to the main
run's `warmup` and `samples`.

## Comparing runs

Fit each model under its own run name, then compare the run directories:

```python
from kappaccino.inference.model_comparison import compare

table = compare(["results/m0416/nfw", "results/m0416/gnfw"])
```

The same table from the command line:

```bash
python -m kappaccino.inference.model_comparison results/m0416/nfw results/m0416/gnfw --csv comparison.csv
```

The table has one row per run, with `k`, `n`, `log_likelihood` (at the best fit), `bic`, `delta_bic`, `wbic`,
`wbic_error`, `delta_wbic` and `delta_wbic_error`. Each delta is relative to the lowest value in its column. The error
of a WBIC difference adds the two runs' Monte Carlo errors in quadrature.

`compare` refuses runs fitted to different data. Each record carries a fingerprint of the trusted image positions,
errors, families and redshifts, the uncertain-image labels and the measured member dispersions. Two runs with
different image catalogs, a different trusted/uncertain split or different kinematics have different fingerprints and
cannot be compared by either number.

## A typical workflow

1. **Screen with BIC.** Run the init stage (`run(config, stage="init")`, or `--stage init`) for each variant. The init
   stage takes minutes, so many variants are affordable. Drop the variants that are clearly worse.
2. **Report WBIC.** Run the full fit with `wbic=True` for the few variants that remain.
3. **Calibrate the significance.** Generate mock catalogs from the simpler model's posterior, fit both models to each
   mock, and record the spread of $\Delta\mathrm{WBIC}$. That spread is the threshold for a real preference. It includes
   the Monte Carlo error, the approximation error of WBIC and the extra freedom of the GRF (below).

## Reading the numbers

On the $-2 \ln B$ scale of both numbers, the usual guide (Kass & Raftery 1995) reads a difference of 0 to 2 as not
worth more than a mention, 2 to 6 as positive, 6 to 10 as strong and above 10 as very strong. Treat these thresholds
as a first reading only. The likelihood with the GRF or the image-plane GP is not a sum of independent Gaussian terms,
which the scale assumes, and the approximation error of WBIC can reach several units. A significance quoted in a paper
should come from the mock calibration.

The GRF and the image-plane GP absorb lens-model misfit. A worse mass profile can be partly rescued by a larger
random field, which makes the differences between profiles smaller. Report the fitted GRF amplitude of each model with
the comparison, since the better model should need less of the field. For a headline result, repeat the comparison
with the field fixed.

## How the numbers are defined

### The data log-likelihood

Both numbers use the data log-likelihood $\ln L$
({meth}`~kappaccino.likelihood.evaluator.Evaluator.data_log_likelihood`). It is the sum of the image-position terms,
the uncertain-image mixture and the member kinematics, with every normalization term (the covariance determinants and
the sampled error scales). The source-transport correction, the far-field barrier and the ellipticity caps belong to
the prior side and are left out.

The number of data values is $n$, with two coordinates per trusted image, two per uncertain image and one per
measured member dispersion. The record lists the parts under `n_data`.

### BIC

$$
\mathrm{BIC} = -2 \ln L(\hat\theta) + k \ln n,
$$

with $\hat\theta$ the init stage's best fit and $k$ the number of sampled parameters (Schwarz 1978). Three points
matter in practice.

- **Use BIC for nested models.** The member offsets, the source positions and the GRF make the total $k$ hard to
  interpret, but in a nested comparison (model B is model A with parameters added) the shared parameters cancel and
  $\Delta k$ is the number of added parameters. `parameter_counts` in the record gives $k$ by parameter group, which
  shows $\Delta k$ directly.
- **The best fit maximizes the posterior, not the likelihood.** This is the usual convention of cluster-lens BIC
  values. With informative priors, $\ln L(\hat\theta)$ sits slightly below the maximum likelihood.
- **BIC is not defined for the uncertain-image mixture**, whose assignments make the model singular. The init stage has
  no uncertain images, so its BIC covers the trusted images only. The final stage of a run with uncertain images records
  `bic: null` with the reason.

### WBIC

$$
\mathrm{WBIC} = -2\, \mathrm{E}_\beta\!\left[\ln L\right], \qquad
p_\beta(\theta) \propto L(\theta)^\beta\, \pi(\theta), \qquad \beta = \frac{1}{\ln n},
$$

the mean of $-2 \ln L$ over draws from the tempered posterior $p_\beta$ (Watanabe 2013). The WBIC run samples
$p_\beta$ with NUTS, starting each chain from the last draw of the main chain
({meth}`~kappaccino.likelihood.evaluator.Evaluator.tempered_log_likelihood`). The tempered posterior is broader than
the posterior, so the run explores a wider range of models.

WBIC has three advantages over BIC here.

- **No parameter count.** The tempered posterior sets the effective number of parameters, so the member offsets, the
  GRF and the hyperparameters are counted correctly.
- **Valid for hierarchical and mixture models**, including the uncertain-image slots.
- **Less sensitive to the prior widths** than a Bayes factor from the Savage–Dickey ratio.

WBIC agrees with $-2 \ln Z$ to leading order in $n$. The remainder can be several units for models with many effective
parameters, so a difference of a few units needs the mock calibration. The Monte Carlo error is the standard deviation
of $-2 \ln L$ over the tempered draws divided by the square root of the effective sample size. With about 1,000
effective draws, the error is typically one or two units.

## Limitations

- **Enumerated uncertain members.** With `MemberConfig.membership="marginalize"` and enumerated members, the image terms
  are a sum over member configurations that also holds the source-transport terms, so the data log-likelihood cannot be
  separated. Neither number is computed (the record holds `null` and the reason). Set
  `MemberConfig.membership_max_enumerated=0`, so that the uncertain members enter by the members' expected mass, or use fixed
  membership.
- **Same data only.** Neither number compares a run that ignores the uncertain images with a run that models them,
  because the two runs fit different data. Compare such runs by how well each run predicts held-out trusted images, or
  by how well each run recovers a known truth.

## Output files

`init/artifacts/model_comparison.json`
: `bic` with `bic`, `log_likelihood`, `k`, `n`, `n_data`, `parameter_counts`, `data_fingerprint` and `point`, or
  `bic: null` with `bic_note`.

`final/artifacts/model_comparison.json`
: `run`, the init stage's `bic` (or `null` with `bic_note`) and, with `wbic=True`, `wbic` with `wbic`, `wbic_error`,
  `ess`, `draws`, `dropped_draws`, `beta`, `n`, `n_data`, `data_fingerprint`, `seconds` and `sampler` (the tempered
  run's divergences, largest R-hat, smallest ESS and mean acceptance). The final `summary.json` holds the same record.

`final/artifacts/wbic_draws.h5`
: `log_likelihood` (chain, draw) at the tempered draws, and the tempered `samples` (draw, parameter).

## References

- Schwarz, G. 1978, *Annals of Statistics*, 6, 461 (BIC)
- Kass, R. E. & Raftery, A. E. 1995, *Journal of the American Statistical Association*, 90, 773 (Bayes factors)
- Watanabe, S. 2013, *Journal of Machine Learning Research*, 14, 867 (WBIC)
