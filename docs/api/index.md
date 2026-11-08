# API reference

Every class and function a user calls, with its parameters, units and return values. The configuration classes
describe a fit; the pipeline runs it; the priors, catalog readers and the Lenstool reader build its inputs; and the
model, posterior and analysis objects are what a finished run returns. The remaining modules (`search`, `sampling`,
`farfield`, the likelihood kernels and the plotting modules) are internal.

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
