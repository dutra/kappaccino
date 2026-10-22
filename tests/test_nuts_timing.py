"""NUTS phase boundaries must include completion of asynchronous outputs."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pytest

from lenscluster import cluster_solver as cs, utils


class DeferredArray:
    def __init__(self, name, ready_at, clock, events, error=None):
        self.name, self.ready_at = name, ready_at
        self.clock, self.events, self.error = clock, events, error

    def block_until_ready(self):
        self.events.append('wait:'+self.name)
        self.clock.now = max(self.clock.now, self.ready_at)
        if self.error:
            raise self.error
        return self

    def __array__(self, *args, **kwargs):
        raise AssertionError('timing synchronization must not copy arrays to host')


def test_logged_nuts_phase_waits_for_all_chains_and_collected_outputs(monkeypatch):
    clock, events, logs = SimpleNamespace(now=0.), [], []
    monkeypatch.setattr(utils.time, 'time', lambda: clock.now)
    monkeypatch.setattr(utils, 'log_message', lambda args, message: logs.append(message))
    state = {'z':DeferredArray('state', 2., clock, events)}
    samples = {'x':(DeferredArray('chain0', 6., clock, events),
                    DeferredArray('chain1', 9., clock, events))}
    extra = {'potential_energy':DeferredArray('diagnostics', 11., clock, events)}
    key, init = object(), object()

    class AsyncMCMC:
        last_state = state
        def run(self, rng_key, **kwargs):
            assert rng_key is key
            assert kwargs == {'extra_fields':('potential_energy',), 'init_params':init}
            events.append('dispatch')
            clock.now = 1.
            return None
        def get_samples(self, group_by_chain=False):
            assert group_by_chain is True
            return samples
        def get_extra_fields(self, group_by_chain=False):
            assert group_by_chain is True
            return extra

    result = cs._run_logged_phase(None, 'nuts.run', lambda:cs._run_mcmc_until_ready(
        AsyncMCMC(), key, extra_fields=('potential_energy',), init_params=init))
    assert result is None
    assert clock.now == 11.
    assert events == ['dispatch', 'wait:state', 'wait:chain0', 'wait:chain1', 'wait:diagnostics']
    assert logs[-1] == '[phase] nuts.run end elapsed=11.00s'


def test_asynchronous_failure_is_reported_inside_nuts_run(monkeypatch):
    clock, events, logs = SimpleNamespace(now=0.), [], []
    monkeypatch.setattr(utils.time, 'time', lambda: clock.now)
    monkeypatch.setattr(utils, 'log_message', lambda args, message: logs.append(message))
    class FailingMCMC:
        last_state = DeferredArray('state', 3., clock, events, RuntimeError('device execution failed'))
        def run(self, *args, **kwargs):
            return None
        def get_samples(self, group_by_chain=False):
            return {}
        def get_extra_fields(self, group_by_chain=False):
            return {}
    with pytest.raises(RuntimeError, match='device execution failed'):
        cs._run_logged_phase(None, 'nuts.run', lambda:cs._run_mcmc_until_ready(
            FailingMCMC(), None, extra_fields=(), init_params=None))
    assert logs[-1] == '[phase] nuts.run error elapsed=3.00s RuntimeError: device execution failed'
    assert not any('nuts.run end' in message for message in logs)


@pytest.mark.parametrize('chain_method', ['sequential', 'vectorized', 'parallel'])
def test_real_numpyro_barrier_preserves_samples_and_diagnostics(chain_method):
    if chain_method == 'parallel' and jax.device_count() < 2:
        pytest.skip('run with JAX_NUM_CPU_DEVICES=2 to exercise parallel execution')
    def model():
        numpyro.sample('x', dist.Normal(0., 1.))
    def sampler():
        return MCMC(NUTS(model, max_tree_depth=2, step_size=.1,
                         adapt_step_size=False, adapt_mass_matrix=False),
                    num_warmup=2, num_samples=4, num_chains=2,
                    chain_method=chain_method, progress_bar=False)
    before, after = sampler(), sampler()
    key = jax.random.PRNGKey(218)
    fields = ('potential_energy', 'accept_prob', 'num_steps', 'diverging')
    before.run(key, extra_fields=fields)
    cs._run_mcmc_until_ready(after, key, extra_fields=fields, init_params=None)
    for getter in ('get_samples', 'get_extra_fields'):
        expected = getattr(before, getter)(group_by_chain=True)
        actual = getattr(after, getter)(group_by_chain=True)
        assert all(a.is_ready() for a in jax.tree.leaves(actual))
        for a,b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a,b)
