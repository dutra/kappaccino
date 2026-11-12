"""CPU setup and fresh-process backend/worker setup, without lens inference."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kappaccino import cpu_runtime as cpu


@pytest.mark.parametrize("chains", [1, 2, 4, 8])
@pytest.mark.parametrize("method", ["parallel", "sequential", "vectorized"])
def test_devices_and_blas_follow_chain_method(chains, method):
    plan = cpu.resolve_cpu_plan(chains, method)
    assert plan.chains == chains
    assert plan.jax_cpu_devices == (chains if method == "parallel" else 1)
    assert plan.blas_threads == (1 if method == "parallel" and chains > 1 else None)


def test_parallel_backend_limits_loaded_and_future_blas():
    code = """
import os
import numpy as np
from threadpoolctl import threadpool_info
from kappaccino import cpu_runtime as cpu
plan = cpu.configure_cpu_runtime(4, 'parallel')
assert plan.blas_threads == 1
assert all(os.environ[name] == '1' for name in cpu._BLAS_VARIABLES)
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
a = np.array([[4., 1.], [1., 3.]])
result = jax.pmap(jnp.linalg.cholesky)(jnp.asarray(np.stack([a] * 4)))
np.testing.assert_allclose(result, np.stack([np.linalg.cholesky(a)] * 4), rtol=1e-12)
blas = [p['num_threads'] for p in threadpool_info() if p['user_api'] == 'blas']
assert set(blas) <= {1}          # empty where threadpoolctl cannot see the BLAS (Apple Accelerate); the variables above still hold
assert cpu.configure_cpu_runtime(4, 'parallel') == plan
"""
    env = dict(os.environ, PYTHONPATH=str(Path('src').resolve()), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    subprocess.run([sys.executable, '-c', code], env=env, check=True, timeout=60)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_reject_invalid_chain_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        cpu.resolve_cpu_plan(value)


def test_reject_unknown_chain_method():
    with pytest.raises(ValueError, match="chain method"):
        cpu.resolve_cpu_plan(4, "threaded")


@pytest.mark.parametrize("chains", [1, 2, 4])
def test_fresh_backend_and_spawned_worker_follow_resolved_chains(tmp_path, chains):
    script = tmp_path / "cpu_spawn_check.py"
    script.write_text('''
import json, multiprocessing, os
from kappaccino.cpu_runtime import configure_cpu_runtime
def cpus():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
def inspect_backend():
    import jax
    import jax.numpy as jnp
    from threadpoolctl import threadpool_info
    n = len(jax.devices('cpu'))
    values = jax.pmap(lambda x: x*x + 1)(jnp.arange(n))
    return dict(devices=n, cpus=cpus(), values=values.tolist(),
                blas=[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas'])
def child(q):
    q.put(inspect_backend())
if __name__ == '__main__':
    import sys
    chains = int(sys.argv[1])
    before = cpus()
    plan = configure_cpu_runtime(chains, 'parallel')
    parent = inspect_backend()
    assert configure_cpu_runtime(chains, 'parallel') == plan
    try:
        configure_cpu_runtime(chains + 1, 'parallel')
    except RuntimeError:
        pass
    else:
        raise AssertionError('reconfiguration should require a fresh process')
    ctx = multiprocessing.get_context('spawn')
    q = ctx.Queue(); p = ctx.Process(target=child, args=(q,))
    p.start(); result = q.get(timeout=45); p.join(timeout=45)
    assert p.exitcode == 0
    print(json.dumps(dict(plan=plan.to_dict(), before=before, parent=parent, child=result)))
''')
    env = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
    # Deliberately start with a different device count: typed config wins.
    env["JAX_NUM_CPU_DEVICES"] = "7"
    result = subprocess.run([sys.executable, str(script), str(chains)], env=env,
                            capture_output=True, text=True, check=True, timeout=90)
    report = json.loads(result.stdout)
    assert report["parent"] == report["child"]
    assert report["parent"]["devices"] == chains
    assert report["parent"]["values"] == [i*i+1 for i in range(chains)]
    assert report["parent"]["cpus"] == report["before"]              # the process is not pinned
    if chains > 1:
        assert set(report["parent"]["blas"]) <= {1}


def test_late_backend_setup_fails_actionably():
    code = """
import jax
jax.devices('cpu')
from kappaccino.cpu_runtime import configure_cpu_runtime
try:
    configure_cpu_runtime(1, 'parallel')
except RuntimeError as error:
    assert 'before JAX backend initialization' in str(error)
else:
    raise AssertionError('late setup was silently accepted')
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=45,
                   env=dict(os.environ, PYTHONPATH=str(Path("src").resolve())))


@pytest.mark.parametrize("cluster", ["catalogs/hera", "catalogs/as1063/bergamini2019"])
def test_launcher_uses_resolved_runtime_before_cosmology(cluster):
    code = """
import os, shutil, sys
from kappaccino import cli
from jax._src import xla_bridge
assert not xla_bridge.backends_are_initialized(), 'importing the launcher initialized JAX before CPU setup'
before = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
def check_run(config, stage="both"):
    import jax
    assert config.runtime.chains == 2
    assert len(jax.devices('cpu')) == 2
    assert (sorted(os.sched_getaffinity(0)) if before is not None else None) == before      # not pinned
import kappaccino.pipeline as pipeline
pipeline.run = check_run
try:
    assert cli.main([sys.argv[1], '--results', 'results/pytest_cpu_launcher', '--set', 'runtime.chains=2']) == 0
finally:
    shutil.rmtree('results/pytest_cpu_launcher', ignore_errors=True)
"""
    subprocess.run([sys.executable, "-c", code, cluster], check=True, timeout=45,
                   env=dict(os.environ, PYTHONPATH=str(Path("src").resolve())))


def test_saved_config_with_removed_pinning_field_loads():
    from kappaccino.config import LensClusterSolverConfig, RuntimeConfig, _from_jsonable
    runtime = _from_jsonable({"seed": 3, "chains": 2, "cpu_threads_per_chain": 2}, RuntimeConfig)
    assert runtime == RuntimeConfig(seed=3, chains=2)
    payload = LensClusterSolverConfig().to_json_dict()
    payload["runtime"]["cpu_threads_per_chain"] = 2
    assert LensClusterSolverConfig.from_json_dict(payload).to_json_dict() == LensClusterSolverConfig().to_json_dict()
