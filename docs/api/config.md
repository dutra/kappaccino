# Configuration

```{eval-rst}
.. automodule:: kappaccino.config
   :no-members:

Run files
---------

.. autofunction:: kappaccino.run_files.load_run
.. autofunction:: kappaccino.run_files.load_priors
.. autofunction:: kappaccino.run_files.read_run
.. autofunction:: kappaccino.run_files.parse_value

Top level
---------

.. autoclass:: kappaccino.config.LensClusterSolverConfig
.. autoclass:: kappaccino.config.RunPathsConfig
.. autoclass:: kappaccino.config.RuntimeConfig

Lens model
----------

.. autoclass:: kappaccino.config.LensModelConfig
.. autoclass:: kappaccino.config.ReferenceFrameConfig
.. autoclass:: kappaccino.config.CosmologyConfig
.. autoclass:: kappaccino.config.DPIEHaloConfig
.. autoclass:: kappaccino.config.NFWHaloConfig
.. autoclass:: kappaccino.config.ShearHaloConfig
.. autoclass:: kappaccino.config.MassSheetConfig
.. autoclass:: kappaccino.config.MemberPopulationConfig
.. autoclass:: kappaccino.config.IndependentMemberHaloConfig
.. autoclass:: kappaccino.config.ImageConstraintsConfig

Members and scaling relation
----------------------------

.. autoclass:: kappaccino.config.MemberConfig
.. autoclass:: kappaccino.config.ScalingModelConfig

Likelihood, search and sampling
-------------------------------

.. autoclass:: kappaccino.config.LikelihoodConfig
.. autoclass:: kappaccino.config.SearchConfig
.. autoclass:: kappaccino.config.SamplingConfig

Diagnostics and figures
-----------------------

.. autoclass:: kappaccino.config.ImageDiagnosticsConfig
.. autoclass:: kappaccino.config.TruthRecoveryConfig
.. autoclass:: kappaccino.config.ImageCatalogCutoutConfig
.. autoclass:: kappaccino.config.RGBDisplayConfig
```

Priors are `numpyro.distributions` objects; see {doc}`priors`. A configuration can also be read from a Lenstool parameter
file ({doc}`lenstool`).
