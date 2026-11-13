#!/usr/bin/env python
"""Run the whole benchmark family on a given optimizer and store CSV results.

This is the single end-to-end entry point of the project::

    python scripts/run_benchmark.py --optimizer cma_es
    python scripts/run_benchmark.py --optimizer cma_es sep_cma_es pso de adam --plots

It loads the curated (screened) suite, runs every problem instance the
requested number of times, and writes ``runs.csv``, ``curves.csv``,
``summary.csv`` and ``ert.csv`` into the output directory -- plus figures with
``--plots`` and IOHprofiler logs with ``--ioh-logs``.
"""

from __future__ import annotations

import os
import sys

# The networks are small enough that threaded BLAS costs more than it saves;
# set this before anything imports NumPy.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from gaab.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["run"] + sys.argv[1:]))
