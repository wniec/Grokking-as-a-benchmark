"""Process-pool helpers shared by the screener and the benchmark runner.

Both jobs are embarrassingly parallel over independent runs, and both hit the
same two traps: BLAS oversubscription, and worker processes that must not
inherit a half-initialized NumPy.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterator, Sequence

#: BLAS/OpenMP variables the worker processes inherit.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@contextlib.contextmanager
def single_threaded_blas():
    """Pin BLAS to one thread per process for the duration of the block.

    The networks here are tiny -- a 16x32 matmul -- so threaded BLAS is pure
    overhead even serially, and with a process pool it is actively harmful:
    every worker spins up a full thread pool and the machine spends its time
    context-switching instead of evaluating.  Worker processes are started with
    "spawn" so they import NumPy *after* inheriting these variables; setting
    them post-fork would be too late.
    """
    previous = {name: os.environ.get(name) for name in THREAD_VARS}
    os.environ.update({name: "1" for name in THREAD_VARS})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def map_unordered(
    function: Callable, payloads: Sequence, jobs: int
) -> Iterator:
    """Run ``function`` over ``payloads``, yielding results as they finish.

    Falls back to a plain serial loop when ``jobs <= 1``, so callers do not
    need two code paths.  ``function`` must be importable by name in a fresh
    interpreter (a module-level function), because the pool uses "spawn".
    """
    if jobs <= 1:
        for payload in payloads:
            yield function(payload)
        return

    with single_threaded_blas():
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
            futures = [pool.submit(function, payload) for payload in payloads]
            for future in as_completed(futures):
                yield future.result()
