"""Set the JAX CPU devices and the BLAS thread limit before JAX initializes its backend.

Parallel chains run one per JAX CPU device.  The process is not pinned to CPUs: XLA sizes its worker pool from the
CPUs the process may use, and the operating system schedules the threads.  BLAS stays single-threaded under parallel
chains to avoid nested parallelism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
import os
import sys

_BLAS_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
_ACTIVE_PLAN = None
_THREADPOOL_LIMITER = None


@dataclass(frozen=True)
class CPUPlan:
    """The CPU setup of one run, as returned by :func:`resolve_cpu_plan`.

    Parameters
    ----------
    chains : int
        Number of NUTS chains in the run.
    chain_method : str
        NumPyro chain method: ``"parallel"``, ``"sequential"``, or ``"vectorized"``.
    jax_cpu_devices : int
        Number of JAX CPU devices to expose (``JAX_NUM_CPU_DEVICES``): ``chains``
        for ``"parallel"``, otherwise 1.
    blas_threads : int or None
        Thread limit applied to BLAS/OpenMP/NumExpr pools: 1 when more than one
        chain runs in parallel, otherwise None (libraries keep their own defaults).
    """

    chains: int
    chain_method: str
    jax_cpu_devices: int
    blas_threads: int | None

    def to_dict(self):
        """Return the plan as a plain dict with one key per field.

        Returns
        -------
        dict
            Field name to value, as produced by :func:`dataclasses.asdict`.
        """
        return asdict(self)


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def resolve_cpu_plan(chains, chain_method="parallel"):
    """The CPU setup of a run, without changing the process.

    Parameters
    ----------
    chains : int
        Number of NUTS chains; must be a positive integer.
    chain_method : str, optional
        NumPyro chain method. ``"parallel"`` (default) runs the chains concurrently,
        one JAX CPU device per chain; ``"vectorized"`` runs them concurrently on one
        device; ``"sequential"`` runs one chain at a time on one device.

    Returns
    -------
    CPUPlan
        The setup. Nothing is applied; see :func:`configure_cpu_runtime`.

    Raises
    ------
    ValueError
        If ``chains`` is not a positive integer, or ``chain_method`` is not one of
        the three allowed values.
    """
    chains = _positive_integer("chains", chains)
    if chain_method not in {"parallel", "sequential", "vectorized"}:
        raise ValueError("Unsupported NUTS chain method for CPU setup.")
    parallel = chain_method == "parallel"
    return CPUPlan(chains, chain_method, chains if parallel else 1, 1 if parallel and chains > 1 else None)


def configure_cpu_runtime(chains, chain_method="parallel"):
    """Resolve and apply the CPU setup of this process before JAX initializes.

    Call once per process, before any JAX computation (including building a
    config, which evaluates cosmology). It sets the BLAS/OpenMP thread variables
    (``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``MKL_NUM_THREADS``,
    ``BLIS_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS``, ``NUMEXPR_NUM_THREADS``) and
    limits already loaded libraries through ``threadpoolctl`` when
    ``plan.blas_threads`` is set, and sets ``JAX_NUM_CPU_DEVICES`` (and
    ``jax_num_cpu_devices`` if JAX is imported). The process is not pinned to
    CPUs. Spawned stages inherit these settings. Existing JAX backends cannot
    change their devices, so late configuration is rejected rather than silently
    ignored.

    Parameters
    ----------
    chains : int
        Number of NUTS chains; must be a positive integer.
    chain_method : str, optional
        ``"parallel"`` (default), ``"sequential"``, or ``"vectorized"``; see
        :func:`resolve_cpu_plan`.

    Returns
    -------
    CPUPlan
        The applied plan. Calling again with the same arguments returns it
        without reapplying.

    Raises
    ------
    RuntimeError
        If a different plan is already applied in this process, or if the JAX
        backend is already initialized.
    ValueError
        If the arguments are invalid (see :func:`resolve_cpu_plan`).
    """
    global _ACTIVE_PLAN, _THREADPOOL_LIMITER
    plan = resolve_cpu_plan(chains, chain_method)
    if _ACTIVE_PLAN == plan:
        return plan
    if _ACTIVE_PLAN is not None:
        raise RuntimeError("CPU setup is already configured differently; start a fresh process to change it.")
    jax = sys.modules.get("jax")
    if jax is not None:
        from jax._src import xla_bridge
        if xla_bridge.backends_are_initialized():
            raise RuntimeError(
                "CPU setup must run before JAX backend initialization. Start a fresh process "
                "and call configure_cpu_runtime with the resolved runtime configuration "
                "before cosmology calculations or inference."
            )
    if plan.blas_threads is not None:
        # Parallel chains already provide concurrency. Limit nested BLAS/OpenMP
        # before XLA creates workers. Changing OpenMP limits in the main thread
        # afterward does not reliably change the limits inherited by existing
        # XLA worker threads.
        for name in _BLAS_VARIABLES:
            os.environ[name] = str(plan.blas_threads)
        # NumPy/SciPy may already be imported while constructing typed configs.
        # Limit those loaded libraries as well as future spawned processes.
        from threadpoolctl import threadpool_limits
        _THREADPOOL_LIMITER = threadpool_limits(limits=plan.blas_threads)
        if "numexpr" in sys.modules:
            sys.modules["numexpr"].set_num_threads(plan.blas_threads)
    os.environ["JAX_NUM_CPU_DEVICES"] = str(plan.jax_cpu_devices)
    if jax is not None:
        jax.config.update("jax_num_cpu_devices", plan.jax_cpu_devices)
    _ACTIVE_PLAN = plan
    return plan
