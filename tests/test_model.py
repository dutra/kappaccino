import numpy as np
import numpyro.distributions as dist

from kappaccino.core.model import ParameterSpec, latent_array_to_physical, latent_to_physical, physical_to_latent


def spec(prior) -> ParameterSpec:
    return ParameterSpec(name="p.x", sample_name="p_x", potential_id="p", profile_type=0, field="x", prior=prior)


def test_log_sampled_prior_round_trip() -> None:
    log_normal = spec(dist.LogNormal(0.0, 1.0))
    assert log_normal.log
    np.testing.assert_allclose(latent_to_physical(physical_to_latent(12.5, log_normal), log_normal), 12.5)
    latent = np.asarray([physical_to_latent(3.0, log_normal), physical_to_latent(6.0, log_normal)])
    np.testing.assert_allclose(latent_array_to_physical(latent, log_normal), [3.0, 6.0])


def test_plain_prior_is_its_own_latent_coordinate() -> None:
    uniform = spec(dist.Uniform(-1.0, 1.0))
    assert not uniform.log
    assert physical_to_latent(0.3, uniform) == 0.3 and latent_to_physical(0.3, uniform) == 0.3
