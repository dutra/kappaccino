# kappaccino

`kappaccino` builds Bayesian strong-lensing mass models of galaxy clusters with JAX and NumPyro. The lens is a sum of
cluster-scale halos (dPIE or elliptical NFW), an optional external shear and mass sheet, and a population of
cluster-member dPIE subhalos on luminosity scaling relations, each member free to deviate from the relation. The
multiple-image positions constrain the model through an image-plane likelihood that, by default, includes a Gaussian
random field of unmodeled mass. A multi-start MAP search finds the posterior mode, and NUTS samples the posterior from
there.

Documentation: [https://kappaccino.readthedocs.io/](https://kappaccino.readthedocs.io/)

At a high level, `kappaccino` includes:

- dPIE, elliptical NFW, external-shear and mass-sheet lens components in JAX, with exact and far-field member
  evaluation
- member populations on scaling relations with hierarchical per-member deviations
- smoothed local image-plane, image-plane GP and Gaussian-random-field (knee spectrum) likelihoods
- priors as `numpyro.distributions` objects, and a reader for Lenstool `.par` files
- a two-stage pipeline (multi-start MAP, then NUTS) with tables, figures, convergence and lens maps
- a mock-validation harness that generates analytic clusters, fits them and scores the recovery
- catalogs for MACS J0416, Abell S1063 and the simulated FF-SIMS clusters Hera and Ares

## Install

`kappaccino` requires Python 3.11 or newer. Clone this repository, then:

```bash
python -m pip install -e .
```

The dependencies include two forks installed from GitHub branches: `jaxtronomy` from
`dutra/JAXtronomy@feat-pseudojaffe-compact-profiles` (the JAX lens profiles) and `lenstronomy` from
`dutra/lenstronomy@feat-lenscluster-models` (the exact-image solver and plotting).

The image and member catalogs of the supported clusters are under `catalogs/`. Imaging is not tracked; fetch it with
`python scripts/download_imaging.py download` (see the installation page of the documentation). Fits run without
imaging and skip the figures that need it.

## Usage

Set up the CPU chains first (one JAX device per chain), then build a configuration, from a Lenstool parameter file or
in Python, and run it:

```python
from kappaccino.cpu_runtime import configure_cpu_runtime

configure_cpu_runtime(4, "parallel")          # before anything starts JAX

from kappaccino.lenstool_par import config_from_lenstool_par
from kappaccino.pipeline import dry_run, run

config = config_from_lenstool_par(
    "catalogs/hera/source/hera_lenscluster.par",
    image_catalog="catalogs/hera/images.yaml",
    member_catalog="catalogs/hera/members.yaml",
    output_dir="results/demo",
).validate()

print(dry_run(config))                        # model size, member classification, search and sampling sizes
result = run(config)                          # init (multi-start MAP) + final (NUTS, tables, figures)
result.posterior.physical                     # (draw, parameter) in physical units
```

Priors are `numpyro.distributions` objects of the sampled quantities:

```python
from dataclasses import replace
import numpyro.distributions as dist
from kappaccino.config import DPIEHaloConfig
from kappaccino.priors import ellipticity_box, lognormal_from_moments

halo = DPIEHaloConfig("1", 0.0, 0.0, 0.3, 30.0, 10.0, 1500.0, 800.0, z_lens=0.5, priors={
    "x_centre": dist.Uniform(-5.0, 5.0),
    "y_centre": dist.Uniform(-5.0, 5.0),
    **ellipticity_box(0.8),                    # e1, e2
    "core_radius_kpc": dist.LogUniform(0.5, 100.0),
    "v_disp": lognormal_from_moments(800.0, 280.0, low=100.0, high=2200.0),
})
```

Each run writes to `<output_dir>/<run_name>/`: `init/` holds the search and the chain start, and `final/` holds
`posterior.npz`, the summary tables (`posterior_summary.csv`, `image_fit_quality.csv`, `run_summary.txt`, ...), the
figures and the lens maps. The documentation defines every [output file](https://kappaccino.readthedocs.io/en/latest/outputs.html)
and every [posterior parameter](https://kappaccino.readthedocs.io/en/latest/posterior.html) with its units.

## Mock validation

```python
from kappaccino.mock_validation import (MockConfig, MockValidationConfig, MockValidationPathsConfig,
                                        MockValidationRuntimeConfig, run_mock_validation)

campaign = MockValidationConfig(
    mock=MockConfig(mode="production", n_members=10, n_primary_families=4, n_member_families=1),
    paths=MockValidationPathsConfig(output_dir="validation_runs", campaign_name="smoke", run_name="analytic"),
    runtime=MockValidationRuntimeConfig(realizations=1, seed=12345),
)
run_mock_validation(campaign)
```

Each realization draws an analytic cluster, fits it with the same pipeline and writes its recovery of parameters,
magnifications, image positions, subhalo masses and aperture masses; `aggregate_plots/` holds the campaign figures.

## Scripts

`scripts/` holds the data preparation (`convert_lenstool_catalog.py`, `download_imaging.py`), figure regeneration from
saved runs without inference (`regenerate_*.py`, `chain_traces.py`, `make_gp_illustrations.py`), checks and the
paper figures. The scripts page of the documentation describes each.

## Tests

```bash
python -m pytest tests                 # the full suite
python -m pytest tests -m "not slow"   # skip the validation tests that run inference
```

## Data and provenance

`catalogs/README.md` lists where every catalog comes from: the lens models and catalogs of Bergamini et al. (2019, 2023),
Caminha et al. (2016), Rihtarsic et al. (2025, CANUCS), Diego et al. (2024, 2026) and Beauchesne et al. (2025), the
member kinematics of Granata et al. (2026), and the FF-SIMS simulations (Meneghetti et al. 2017). Cite those works when
you use the catalogs.
