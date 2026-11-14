# Model, posterior and analysis

The objects a run returns ({class}`~kappaccino.pipeline.RunResult`) and the helpers that evaluate a fitted model.
{doc}`../posterior` defines every posterior parameter and its units.

```{eval-rst}
Model state and parameters
--------------------------

.. autofunction:: kappaccino.core.state.build_state
.. autofunction:: kappaccino.core.state.with_member_classes
.. autoclass:: kappaccino.core.model.ModelState
   :no-members:
.. autoclass:: kappaccino.core.model.ParameterSpec
   :members: latent_prior, log
.. autoclass:: kappaccino.core.model.FamilyData
   :no-members:
.. autoclass:: kappaccino.core.model.ImageData
   :no-members:
.. autoclass:: kappaccino.core.model.MemberModel
   :no-members:

Latent and physical values
--------------------------

.. autofunction:: kappaccino.core.model.latent_to_physical
.. autofunction:: kappaccino.core.model.physical_to_latent
.. autofunction:: kappaccino.core.model.latent_array_to_physical
.. autofunction:: kappaccino.core.model.convert_theta_to_physical
.. autofunction:: kappaccino.core.model.convert_theta_to_latent
.. autofunction:: kappaccino.core.model.convert_sample_matrix_to_physical
.. autofunction:: kappaccino.core.model.default_theta

Posterior
---------

.. autoclass:: kappaccino.core.model.PosteriorResults
   :no-members:

Evaluating the model
--------------------

.. autoclass:: kappaccino.likelihood.evaluator.Evaluator
   :members: physical, packed_state, packed_state_for_z, component_dispersions, fields, total_fields_at, statistics,
             error_scales, log_likelihood, search_log_likelihood, reported_physical, latent_from_reported_physical,
             source_positions, gp_deflection_correction

Lens maps and truth comparison
------------------------------

.. automodule:: kappaccino.diagnostics.truth_recovery
   :no-members:

.. autofunction:: kappaccino.diagnostics.truth_recovery.truth_maps_on_grid
.. autofunction:: kappaccino.diagnostics.truth_recovery.load_truth_fits
.. autofunction:: kappaccino.diagnostics.truth_recovery.diagnostic_grid
.. autofunction:: kappaccino.diagnostics.truth_recovery.sample_on_grid
.. autofunction:: kappaccino.diagnostics.truth_recovery.grid_arcsec
.. autofunction:: kappaccino.diagnostics.truth_recovery.compute_truth_recovery

Saved artifacts
---------------

.. automodule:: kappaccino.io.artifacts
   :no-members:

.. autofunction:: kappaccino.io.artifacts.load_config
.. autofunction:: kappaccino.io.artifacts.load_posterior
.. autofunction:: kappaccino.io.artifacts.load_start
.. autofunction:: kappaccino.io.artifacts.load_search
.. autofunction:: kappaccino.io.artifacts.load_exact_positions
.. autofunction:: kappaccino.io.artifacts.save_config
.. autofunction:: kappaccino.io.artifacts.save_state
.. autofunction:: kappaccino.io.artifacts.save_search
.. autofunction:: kappaccino.io.artifacts.save_start
.. autofunction:: kappaccino.io.artifacts.save_posterior
.. autofunction:: kappaccino.io.artifacts.spec_to_json
.. autofunction:: kappaccino.io.artifacts.init_signature
.. autofunction:: kappaccino.io.artifacts.write_json
.. autofunction:: kappaccino.io.hdf5.load_arrays
.. autofunction:: kappaccino.io.hdf5.save_arrays
```
