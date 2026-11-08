# Model, posterior and analysis

The objects a run returns ({class}`~lenscluster.pipeline.RunResult`) and the helpers that evaluate a fitted model.
{doc}`../posterior` defines every posterior parameter and its units.

```{eval-rst}
Model state and parameters
--------------------------

.. autofunction:: lenscluster.state.build_state
.. autofunction:: lenscluster.state.with_member_classes
.. autoclass:: lenscluster.model.ModelState
   :no-members:
.. autoclass:: lenscluster.model.ParameterSpec
   :members: latent_prior, log
.. autoclass:: lenscluster.model.FamilyData
   :no-members:
.. autoclass:: lenscluster.model.ImageData
   :no-members:
.. autoclass:: lenscluster.model.MemberModel
   :no-members:

Latent and physical values
--------------------------

.. autofunction:: lenscluster.model.latent_to_physical
.. autofunction:: lenscluster.model.physical_to_latent
.. autofunction:: lenscluster.model.latent_array_to_physical
.. autofunction:: lenscluster.model.convert_theta_to_physical
.. autofunction:: lenscluster.model.convert_theta_to_latent
.. autofunction:: lenscluster.model.convert_sample_matrix_to_physical
.. autofunction:: lenscluster.model.default_theta

Posterior
---------

.. autoclass:: lenscluster.model.PosteriorResults
   :no-members:

Evaluating the model
--------------------

.. autoclass:: lenscluster.evaluator.Evaluator
   :members: physical, packed_state, packed_state_for_z, component_dispersions, fields, total_fields_at, statistics,
             error_scales, log_likelihood, search_log_likelihood, reported_physical, latent_from_reported_physical,
             source_positions, gp_deflection_correction

Lens maps and truth comparison
------------------------------

.. automodule:: lenscluster.truth_recovery
   :no-members:

.. autofunction:: lenscluster.truth_recovery.lens_quantities
.. autofunction:: lenscluster.truth_recovery.load_truth_fits
.. autofunction:: lenscluster.truth_recovery.diagnostic_grid
.. autofunction:: lenscluster.truth_recovery.sample_on_grid
.. autofunction:: lenscluster.truth_recovery.grid_arcsec
.. autofunction:: lenscluster.truth_recovery.compute_truth_recovery

Saved artifacts
---------------

.. automodule:: lenscluster.artifacts
   :no-members:

.. autofunction:: lenscluster.artifacts.load_config
.. autofunction:: lenscluster.artifacts.load_posterior
.. autofunction:: lenscluster.artifacts.load_start
.. autofunction:: lenscluster.artifacts.load_search
.. autofunction:: lenscluster.artifacts.load_exact_positions
.. autofunction:: lenscluster.artifacts.save_config
.. autofunction:: lenscluster.artifacts.save_state
.. autofunction:: lenscluster.artifacts.save_search
.. autofunction:: lenscluster.artifacts.save_start
.. autofunction:: lenscluster.artifacts.save_posterior
.. autofunction:: lenscluster.artifacts.spec_to_json
.. autofunction:: lenscluster.artifacts.init_signature
.. autofunction:: lenscluster.artifacts.write_json
```
