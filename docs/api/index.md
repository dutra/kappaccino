# API reference

Every class and function a user calls, with its parameters, units and return values. The configuration classes
describe a fit; the pipeline runs it; the priors, catalog readers and the Lenstool reader build its inputs; and the
model, posterior and analysis objects are what a finished run returns. The remaining modules (the search and sampler,
the far-field series, the likelihood kernels and the plotting modules) are internal.

The package is organized as:

| module or subpackage | contents |
|---|---|
| `kappaccino.config`, `kappaccino.priors` | the configuration of a run and its priors |
| `kappaccino.pipeline`, `kappaccino.cpu_runtime` | running a fit: the two stages, and the CPU devices of the chains |
| `kappaccino.core` | the model: parameter specifications, the model state, inputs and member classification |
| `kappaccino.lens` | lens physics in JAX: profiles, deflections and Jacobians, cosmology, the far-field series |
| `kappaccino.likelihood` | the evaluator and the position likelihoods (smoothed, image-plane GP, GRF) |
| `kappaccino.inference` | the multi-start MAP search and NUTS |
| `kappaccino.io` | catalogs, Lenstool parameter files, sky coordinates, run artifacts and HDF5 arrays |
| `kappaccino.imaging` | HST and JWST images: cutouts, RGB display, photometry |
| `kappaccino.diagnostics` | numerical diagnostics: image recovery, exact images, members, priors, truth recovery |
| `kappaccino.plotting` | figures and the shared matplotlib style |
| `kappaccino.reporting` | tables, text reports and publication macros written after a run |
| `kappaccino.partition` | uncertain images: the source-slot mixture |
| `kappaccino.mock_validation` | analytic mock clusters, their fits and recovery scores |

```{toctree}
:maxdepth: 1

config
priors
pipeline
results
catalogs
lenstool
mock_validation
runtime
```
