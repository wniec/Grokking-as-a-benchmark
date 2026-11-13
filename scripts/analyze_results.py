#!/usr/bin/env python
"""Build comparison tables and figures from a benchmark results directory.

python scripts/analyze_results.py --results results/benchmark
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
    sys.exit(main(["analyze"] + sys.argv[1:]))
