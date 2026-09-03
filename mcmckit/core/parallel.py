"""Optional parallel evaluation, shared by the samplers.

The expensive part of Bayesian model updating is almost always the forward model
inside the user's log-likelihood - a finite element run, a time integration, a
modal solve. Everything here exists to spread those calls over cores without
changing what the samplers compute.

Two things are parallelisable in this package:

- **Likelihood evaluations across particles** (TMCMC). Each stage evaluates the
  likelihood for every particle independently.
- **Independent chains** (:func:`~mcmckit.core.multichain.run_chains`).

A single Markov chain is sequential by construction and is not parallelised.

Usage is opt-in and the default is unchanged: ``n_workers=1`` runs serially, in
the calling process, exactly as before.

Backends
--------
``"process"``
    Separate processes. Needed when the likelihood is pure Python, because the
    GIL otherwise serialises it. Requires that the likelihood be *picklable*,
    which rules out lambdas and closures - see :func:`check_picklable`.

``"thread"``
    Threads. No pickling and no process start-up cost, but only helps when the
    likelihood spends its time in code that releases the GIL: NumPy/SciPy linear
    algebra, or an external solver called through a subprocess or C extension.
    That covers most finite element work, and on Windows it avoids the cost of
    re-importing the main module for every worker.

``"auto"``
    ``"process"`` if the callable is picklable, otherwise ``"thread"``.
"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

__all__ = ["WorkerPool", "check_picklable", "limit_blas_threads", "resolve_n_workers"]

_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def limit_blas_threads():
    """Pin each worker process to a single BLAS thread.

    NumPy and SciPy already spread a single large ``dot`` or ``svd`` over every
    core. Running N such workers in parallel therefore asks for N x cores
    threads, and the resulting contention can make the parallel run *slower*
    than the serial one. Since the parallelism here is across likelihood
    evaluations, one BLAS thread per worker is the right split.

    Called as the process-pool initialiser, before the worker imports NumPy.
    """
    for var in _BLAS_THREAD_VARS:
        os.environ[var] = "1"


def resolve_n_workers(n_workers) -> int:
    """Normalise ``n_workers``; ``-1`` means "one per core"."""
    if n_workers is None:
        return 1
    n_workers = int(n_workers)
    if n_workers == -1:
        return os.cpu_count() or 1
    if n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, or -1 for all cores, got {n_workers}")
    return n_workers


def check_picklable(func) -> bool:
    """Whether ``func`` survives a pickle round trip.

    Process workers receive the function by pickling it. Lambdas, closures and
    functions defined inside another function cannot be pickled, and the error
    raised deep inside the executor is unhelpful, so callers check up front.
    """
    try:
        pickle.loads(pickle.dumps(func))
        return True
    except Exception:
        return False


class WorkerPool:
    """A pool held open for the lifetime of a sampler run.

    Creating a process pool is expensive - on Windows each worker re-imports the
    calling module - so a pool rebuilt for every batch of likelihood evaluations
    can easily cost more than it saves. This opens one pool and reuses it for
    every batch, then shuts it down at the end.

    Use it as a context manager::

        with WorkerPool(n_workers=4, backend="auto", func=problem.log_likelihood) as pool:
            values = pool.map(problem.log_likelihood, particles)

    With ``n_workers == 1`` no pool is created at all and ``map`` is a plain
    list comprehension, so the serial path stays free of overhead.
    """

    def __init__(self, n_workers=1, backend="auto", func=None, limit_blas=True):
        self.n_workers = resolve_n_workers(n_workers)
        self._requested_backend = backend
        self.backend = self._choose_backend(backend, func)
        self.limit_blas = limit_blas
        self._executor = None

    def _choose_backend(self, backend, func):
        if self.n_workers == 1:
            return "serial"
        if backend not in ("auto", "process", "thread"):
            raise ValueError(
                f"backend must be 'auto', 'process' or 'thread', got {backend!r}"
            )
        if backend == "auto":
            if func is None or check_picklable(func):
                return "process"
            return "thread"
        if backend == "process" and func is not None and not check_picklable(func):
            raise ValueError(
                "backend='process' needs a picklable log-likelihood, and this one "
                "cannot be pickled. This usually means it is a lambda, a closure, "
                "or defined inside another function. Either move it to module "
                "level, or pass backend='thread' (which needs no pickling and "
                "still helps when the likelihood spends its time in NumPy or an "
                "external solver)."
            )
        return backend

    # -- context manager ------------------------------------------------

    def __enter__(self):
        if self.backend == "process":
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=limit_blas_threads if self.limit_blas else None,
            )
        elif self.backend == "thread":
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)
        return self

    def __exit__(self, *exc):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        return False

    # -- work -----------------------------------------------------------

    def map(self, func, items):
        """Apply ``func`` to every item, in order.

        Falls back to serial evaluation if the pool is not open, so a sampler
        can call this whether or not it is inside the context manager.
        """
        items = list(items)
        if self._executor is None:
            return [func(x) for x in items]
        return list(self._executor.map(func, items))

    def __repr__(self):
        return f"WorkerPool(n_workers={self.n_workers}, backend={self.backend!r})"
