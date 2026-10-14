import numpy as np
import pytest

from lenscluster import cluster_solver as cs
from test_scaling_cache_retention import evaluator


def test_invalid_svi_medians_retain_last_valid_center():
    ev, initial = evaluator()
    accepted = initial.copy()
    accepted[0] += 1.
    center, valid, reasons = cs._validated_svi_block_center(ev, accepted, initial)
    assert valid and not reasons
    np.testing.assert_array_equal(center, accepted)
    for bad_value in (np.nan, np.inf):
        invalid = accepted.copy()
        invalid[0] = bad_value
        center, valid, reasons = cs._validated_svi_block_center(ev, invalid, center)
        assert not valid
        np.testing.assert_array_equal(center, accepted)
    recovered = accepted.copy()
    recovered[0] += 1.
    center, valid, _ = cs._validated_svi_block_center(ev, recovered, center)
    assert valid
    np.testing.assert_array_equal(center, recovered)


def test_invalid_first_svi_block_uses_initial_center():
    ev, initial = evaluator()
    center, valid, _ = cs._validated_svi_block_center(ev, initial * np.nan, initial)
    assert not valid
    np.testing.assert_array_equal(center, initial)


def test_invalid_svi_block_without_valid_fallback_stops():
    ev, initial = evaluator()
    for fallback in (None, initial * np.nan):
        with pytest.raises(ValueError, match='no valid center'):
            cs._validated_svi_block_center(ev, initial * np.nan, fallback)


def test_svi_median_rejects_finite_invalid_radius_ordering():
    from dataclasses import replace
    from test_validation import _minimal_two_bin_local_jacobian_state_with_scaling_scatter

    state = _minimal_two_bin_local_jacobian_state_with_scaling_scatter()
    indices = state.packed_lens_spec.cut_ref_param_index.copy()
    indices[1] = 0
    state.packed_lens_spec = replace(state.packed_lens_spec, cut_ref_param_index=indices)
    ev = cs.ClusterJAXEvaluator(state, .1, sampling_engine='full_flat',
                               sample_likelihood_mode='source', source_plane_covariance_mode='unit')
    initial = cs._default_theta(state.parameter_specs)
    initial[0] = 100.
    candidate = initial.copy()
    candidate[0] = 1e-10
    center, accepted, reasons = cs._validated_svi_block_center(ev, candidate, initial)
    assert not accepted
    assert reasons['rs_not_greater_than_ra'] == 1
    np.testing.assert_array_equal(center, initial)
