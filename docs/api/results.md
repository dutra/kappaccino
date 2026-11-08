# Model, posterior and analysis

The objects a run returns ({class}`~kappaccino.pipeline.RunResult`) and the helpers that evaluate a fitted model.
{doc}`../posterior` defines every posterior parameter and its units.

```{eval-rst}
Model state and parameters
--------------------------

.. autofunction:: kappaccino.state.build_state
.. autofunction:: kappaccino.state.with_member_classes
.. autoclass:: kappaccino.model.ModelState
   :no-members:
.. autoclass:: kappaccino.model.ParameterSpec
   :members: latent_prior, log
.. autoclass:: kappaccino.model.FamilyData
   :no-members:
.. autoclass:: kappaccino.model.ImageData
   :no-members:
.. autoclass:: kappaccino.model.MemberModel
   :no-members:

Latent and physical values
--------------------------

.. autofunction:: kappaccino.model.latent_to_physical
.. autofunction:: kappaccino.model.physical_to_latent
.. autofunction:: kappaccino.model.latent_array_to_physical
.. autofunction:: kappaccino.model.convert_theta_to_physical
.. autofunction:: kappaccino.model.convert_theta_to_latent
.. autofunction:: kappaccino.model.convert_sample_matrix_to_physical
.. autofunction:: kappaccino.model.default_theta

Posterior
---------

.. autoclass:: kappaccino.model.PosteriorResults
   :no-members:

Evaluating the model
--------------------

.. autoclass:: kappaccino.evaluator.Evaluator
   :members: physical, packed_state, packed_state_for_z, component_dispersions, fields, total_fields_at, statistics,
             error_scales, log_likelihood, search_log_likelihood, reported_physical, latent_from_reported_physical,
             source_positions, gp_deflection_correction

Lens maps and truth comparison
------------------------------

.. automodule:: kappaccino.truth_recovery
   :no-members:

.. autofunction:: kappaccino.truth_recovery.lens_quantities
.. autofunction:: kappaccino.truth_recovery.load_truth_fits
.. autofunction:: kappaccino.truth_recovery.diagnostic_grid
.. autofunction:: kappaccino.truth_recovery.sample_on_grid
.. autofunction:: kappaccino.truth_recovery.grid_arcsec
.. autofunction:: kappaccino.truth_recovery.compute_truth_recovery

Saved artifacts
---------------

.. automodule:: kappaccino.artifacts
   :no-members:

.. autofunction:: kappaccino.artifacts.load_config
.. autofunction:: kappaccino.artifacts.load_posterior
.. autofunction:: kappaccino.artifacts.load_start
.. autofunction:: kappaccino.artifacts.load_search
.. autofunction:: kappaccino.artifacts.load_exact_positions
.. autofunction:: kappaccino.artifacts.save_config
.. autofunction:: kappaccino.artifacts.save_state
.. autofunction:: kappaccino.artifacts.save_search
.. autofunction:: kappaccino.artifacts.save_start
.. autofunction:: kappaccino.artifacts.save_posterior
.. autofunction:: kappaccino.artifacts.spec_to_json
.. autofunction:: kappaccino.artifacts.init_signature
.. autofunction:: kappaccino.artifacts.write_json
```
