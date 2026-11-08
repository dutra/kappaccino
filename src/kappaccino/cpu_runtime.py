"""Bound CPU pools before JAX initializes its backend.

The CPU budget is shared by JAX's parallel chains, not a private thread pool
or a pinning assignment for each chain. Linux affinity bounds the worker pools;
BLAS stays single-threaded to avoid nested parallelism.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
import os
from pathlib import Path
import sys

_BLAS_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
_ACTIVE_PLAN = None
_THREADPOOL_LIMITER = None


@dataclass(frozen=True)
class CPUPlan:
    """The CPU layout chosen for one run, as returned by :func:`resolve_cpu_plan`.

    Parameters
    ----------
    chains : int
        Number of NUTS chains in the run.
    chain_method : str
        NumPyro chain method: ``"parallel"``, ``"sequential"``, or ``"vectorized"``.
    threads_per_chain : int or None
        Requested logical CPUs per concurrently running chain. None means the run is
        not pinned and may use every CPU in the process affinity mask.
    jax_cpu_devices : int
        Number of JAX CPU devices to expose (``JAX_NUM_CPU_DEVICES``): ``chains``
        for ``"parallel"``, otherwise 1.
    available_cpu_ids : tuple of int
        Logical CPU ids the process was allowed to use before pinning (the Linux
        affinity mask, or ``range(os.cpu_count())`` where affinity is unsupported).
    selected_cpu_ids : tuple of int
        Logical CPU ids the run is pinned to. Equal to ``available_cpu_ids`` when
        ``threads_per_chain`` is None; otherwise
        ``min(len(available_cpu_ids), concurrent_chains * threads_per_chain)`` CPUs,
        one logical CPU per physical core first, least busy cores first.
    affinity_supported : bool
        Whether the platform can pin the process to CPUs (Linux: True; macOS and
        Windows: False).
    blas_threads : int or None
        Thread limit applied to BLAS/OpenMP/NumExpr pools: 1 when the run is pinned
        or runs more than one parallel chain, otherwise None (libraries keep their
        own defaults).
    """

    chains: int
    chain_method: str
    threads_per_chain: int | None
    jax_cpu_devices: int
    available_cpu_ids: tuple[int, ...]
    selected_cpu_ids: tuple[int, ...]
    affinity_supported: bool
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


def _core_key(cpu):
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    try:
        return ((root / "physical_package_id").read_text().strip(), (root / "core_id").read_text().strip())
    except OSError:
        return ("unknown", str(cpu))


def _physical_core_order(cpu_ids):
    """Use one allowed logical CPU per physical core before SMT siblings."""
    groups = {}
    for cpu in sorted(cpu_ids):
        groups.setdefault(_core_key(cpu), []).append(cpu)
    return tuple(group[i] for i in range(max(map(len, groups.values()), default=0))
                 for group in groups.values() if i < len(group))


def _parse_cpu_list(text):
    cpus = set()
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def cpus_claimed_by_other_runs(markers=("run.xsh", "kappaccino")):
    """Logical CPUs that other solver processes on this machine have pinned themselves to (Linux /proc)."""
    claimed = set()
    proc = Path("/proc")
    if not proc.is_dir():
        return claimed
    me = os.getpid()
    total = os.cpu_count() or 1
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if not any(marker in cmdline for marker in markers):
                continue
            for line in (entry / "status").read_text().splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    cpus = _parse_cpu_list(line.split(":", 1)[1])
                    if len(cpus) < total:                 # a pinned process, not one allowed everywhere
                        claimed |= cpus
        except OSError:
            continue
    return claimed


def cpu_busy_fractions(interval=0.25):
    """Fraction of the last ``interval`` seconds each logical CPU was busy, from two /proc/stat samples (Linux only)."""
    import time

    def sample():
        rows = {}
        try:
            for line in Path("/proc/stat").read_text().splitlines():
                if line.startswith("cpu") and line[3:4].isdigit():
                    fields = [int(v) for v in line.split()[1:]]
                    rows[int(line[3:].split()[0])] = (fields[3] + fields[4], sum(fields))     # idle + iowait, total
        except OSError:
            pass
        return rows

    first = sample()
    if not first:
        return {}
    time.sleep(interval)
    second = sample()
    busy = {}
    for cpu, (idle0, total0) in first.items():
        idle1, total1 = second.get(cpu, (idle0, total0))
        elapsed = total1 - total0
        busy[cpu] = 1.0 - (idle1 - idle0) / elapsed if elapsed > 0 else 0.0
    return busy


def affinity_supported():
    """Whether this platform can pin the process to CPUs (Linux); macOS and Windows cannot."""
    return hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity")


def threads_per_chain_if_supported(threads_per_chain):
    """The ``cpu_threads_per_chain`` setting for this platform: the requested value where CPU affinity exists,
    otherwise None (no pinning), so one run script serves Linux and macOS. Parallel chains still limit BLAS."""
    return threads_per_chain if affinity_supported() else None


def resolve_cpu_plan(chains, chain_method="parallel", threads_per_chain=None, *, avoid=None, load=None):
    """Choose the CPUs of a run without changing the process.

    XLA sizes its thread pool from the affinity mask, so with ``threads_per_chain``
    the run is pinned to ``concurrent * threads_per_chain`` logical CPUs, where
    ``concurrent`` is 1 for ``"sequential"`` and ``chains`` otherwise. One logical
    CPU per physical core is taken before any SMT sibling, least busy cores first
    (a core is as busy as its busiest logical CPU), skipping CPUs already claimed by
    other solver runs. If too few unclaimed cores remain, claimed CPUs are shared.
    Once :func:`configure_cpu_runtime` has applied a plan, a repeated request with
    the same ``(chains, chain_method, threads_per_chain)`` returns that plan
    unchanged.

    Parameters
    ----------
    chains : int
        Number of NUTS chains; must be a positive integer.
    chain_method : str, optional
        NumPyro chain method. ``"parallel"`` (default) runs the chains concurrently,
        one JAX CPU device per chain; ``"vectorized"`` runs them concurrently on one
        device; ``"sequential"`` runs one chain at a time on one device.
    threads_per_chain : int or None, optional
        Logical CPUs per concurrently running chain; must be a positive integer.
        None (default) selects every available CPU and pins nothing.
    avoid : iterable of int or None, optional
        Logical CPU ids to treat as claimed by other runs. None (default) reads
        them from ``/proc`` (processes whose command line contains ``run.xsh`` or
        ``kappaccino`` and that are pinned to a subset of CPUs). Used only when
        ``threads_per_chain`` is set.
    load : mapping of int to float or None, optional
        Busy fraction (dimensionless, 0 to 1) of each logical CPU. None (default)
        samples ``/proc/stat`` over 0.25 seconds (Linux; empty elsewhere, so all
        CPUs count as idle). Used only when ``threads_per_chain`` is set.

    Returns
    -------
    CPUPlan
        The chosen layout. Nothing is applied; see :func:`configure_cpu_runtime`.

    Raises
    ------
    ValueError
        If ``chains`` or ``threads_per_chain`` is not a positive integer, or
        ``chain_method`` is not one of the three allowed values.
    """
    chains = _positive_integer("chains", chains)
    if chain_method not in {"parallel", "sequential", "vectorized"}:
        raise ValueError("Unsupported NUTS chain method for CPU setup.")
    if threads_per_chain is not None:
        threads_per_chain = _positive_integer("cpu_threads_per_chain", threads_per_chain)
    # A repeated request for the configured plan returns it: the core choice depends on the load at launch and on
    # our own narrowed affinity mask, so it must not be recomputed once applied.
    if _ACTIVE_PLAN is not None and (_ACTIVE_PLAN.chains, _ACTIVE_PLAN.chain_method, _ACTIVE_PLAN.threads_per_chain) == (chains, chain_method, threads_per_chain):
        return _ACTIVE_PLAN
    supported = affinity_supported()
    available = tuple(sorted(os.sched_getaffinity(0))) if supported else tuple(range(os.cpu_count() or 1))
    if _ACTIVE_PLAN is not None and available == tuple(sorted(_ACTIVE_PLAN.selected_cpu_ids)):
        available = _ACTIVE_PLAN.available_cpu_ids
    concurrent = 1 if chain_method == "sequential" else chains
    count = len(available) if threads_per_chain is None else min(len(available), concurrent * threads_per_chain)
    if threads_per_chain is None:
        selected = available
    else:
        claimed = set(cpus_claimed_by_other_runs() if avoid is None else avoid)
        free = tuple(cpu for cpu in available if cpu not in claimed)
        pool = free if len(_physical_core_order(free)) >= count else available   # not enough unclaimed cores: share
        busy = cpu_busy_fractions() if load is None else dict(load)
        # A physical core is as busy as its busiest logical CPU, siblings pinned by other runs included; one logical CPU
        # per core first (as _physical_core_order lists them), least busy core first, stable for equal loads.
        siblings = {}
        for cpu in available:
            siblings.setdefault(_core_key(cpu), []).append(cpu)
        core_load = lambda cpu: max(busy.get(c, 0.0) for c in siblings[_core_key(cpu)])
        ordered = _physical_core_order(pool)
        primary = ordered[:len({_core_key(cpu) for cpu in pool})]
        primary = tuple(sorted(primary, key=lambda cpu: (round(core_load(cpu), 2), cpu)))
        selected = (primary + tuple(cpu for cpu in ordered if cpu not in primary))[:count]
    return CPUPlan(chains, chain_method, threads_per_chain,
                   chains if chain_method == "parallel" else 1,
                   available, selected, supported,
                   1 if threads_per_chain is not None or (chain_method == "parallel" and chains > 1) else None)


def configure_cpu_runtime(chains, chain_method="parallel", threads_per_chain=None):
    """Resolve and apply the CPU plan of this process before JAX initializes.

    Call once per process, before any JAX computation (including building a
    config, which evaluates cosmology). It pins the process with
    ``os.sched_setaffinity`` when ``threads_per_chain`` is set, sets the BLAS/OpenMP
    thread variables (``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``,
    ``MKL_NUM_THREADS``, ``BLIS_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS``,
    ``NUMEXPR_NUM_THREADS``) and limits already loaded libraries through
    ``threadpoolctl`` when ``plan.blas_threads`` is set, and sets
    ``JAX_NUM_CPU_DEVICES`` (and ``jax_num_cpu_devices`` if JAX is imported).
    Spawned stages inherit these settings. Existing JAX backends cannot resize
    their pools, so late configuration is rejected rather than silently ignored.

    Parameters
    ----------
    chains : int
        Number of NUTS chains; must be a positive integer.
    chain_method : str, optional
        ``"parallel"`` (default), ``"sequential"``, or ``"vectorized"``; see
        :func:`resolve_cpu_plan`.
    threads_per_chain : int or None, optional
        Logical CPUs per concurrently running chain. None (default) pins nothing.
        Requires CPU affinity support (Linux); use
        :func:`threads_per_chain_if_supported` for scripts shared with macOS.

    Returns
    -------
    CPUPlan
        The applied plan. Calling again with the same arguments returns it
        without reapplying.

    Raises
    ------
    RuntimeError
        If a different plan is already applied in this process, if the JAX
        backend is already initialized, or if ``threads_per_chain`` is set on a
        platform without CPU affinity support.
    ValueError
        If the arguments are invalid (see :func:`resolve_cpu_plan`).
    """
    global _ACTIVE_PLAN, _THREADPOOL_LIMITER
    plan = resolve_cpu_plan(chains, chain_method, threads_per_chain)
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
    if threads_per_chain is not None and not plan.affinity_supported:
        raise RuntimeError("cpu_threads_per_chain requires CPU affinity support; use None on this platform.")
    if threads_per_chain is not None:
        os.sched_setaffinity(0, plan.selected_cpu_ids)
    if plan.blas_threads is not None:
        # Parallel chains already provide concurrency. Limit nested BLAS/OpenMP
        # before XLA creates workers, including on platforms without affinity.
        # Changing OpenMP limits in the main thread afterward does not reliably
        # change the limits inherited by existing XLA worker threads.
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
