"""Shared-memory parallelism over the cores of one node.

Every embarrassingly parallel loop in hopforge (energy / parameter sweeps,
disorder realisations, k-point chunks, Wilson-loop scans, Z2 planes, KPM
random vectors) goes through :func:`pmap`.

* ``n_jobs=None`` uses ``hopforge.parallel.DEFAULT_JOBS`` (environment
  variable ``HOPFORGE_NJOBS``, default 1); ``n_jobs=-1`` uses all cores
  visible to the process (respects SLURM's CPU affinity).
* Workers are forked (Linux/macOS), so closures and lambdas - e.g. shape
  functions - work without pickling.
* Inside every worker BLAS/OpenMP threads are pinned to 1 (threadpoolctl if
  available) to avoid oversubscription: 40 workers x 1 thread on a 40-core node.

>>> import hopforge as hf
>>> hf.parallel.set_default_jobs(40)          # or export HOPFORGE_NJOBS=40
>>> T = system.conductance(energies, n_jobs=40)
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np


def available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        return os.cpu_count() or 1


DEFAULT_JOBS = int(os.environ.get("HOPFORGE_NJOBS", "1"))
_TASK = {}


def set_default_jobs(n: int):
    """Set the default number of worker processes (``-1`` = all cores)."""
    global DEFAULT_JOBS
    DEFAULT_JOBS = int(n)


def resolve_jobs(n_jobs) -> int:
    n = DEFAULT_JOBS if n_jobs is None else int(n_jobs)
    if n < 0:
        n = available_cores()
    return max(1, n)


def _init_worker(threads):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(threads)
    except Exception:
        pass


def _run(i):
    f = _TASK["f"]
    return f(_TASK["items"][i])


def pmap(f, items, n_jobs=None, threads_per_job=1, chunksize=1, ordered=True):
    """Parallel ``[f(x) for x in items]`` over the cores of one node.

    Falls back to a serial loop for ``n_jobs == 1`` or a single item.
    """
    items = list(items)
    n = min(resolve_jobs(n_jobs), len(items))
    if n <= 1 or len(items) <= 1:
        return [f(x) for x in items]
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - Windows
        return [f(x) for x in items]
    _TASK["f"], _TASK["items"] = f, items
    try:
        with ctx.Pool(n, initializer=_init_worker, initargs=(threads_per_job,)) as pool:
            mapper = pool.imap if ordered else pool.imap_unordered
            return list(mapper(_run, range(len(items)), chunksize=chunksize))
    finally:
        _TASK.clear()


def split(array, n_parts):
    """Split an array (e.g. a k-mesh) into ~equal contiguous chunks."""
    n_parts = max(1, min(n_parts, len(array)))
    return np.array_split(array, n_parts)


def chunks_for(n_items, n_jobs, per_job=4):
    """Number of chunks: a few per worker for load balancing."""
    return max(1, min(n_items, resolve_jobs(n_jobs) * per_job))
