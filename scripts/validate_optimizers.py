#!/usr/bin/env python
"""Sanity-check the bundled optimizers against classic BBOB functions.

The suite's own problems cannot tell a correct CMA variant from a subtly broken
one -- nothing there is solved, so everything looks equally bad.  This script
runs the solvers on IOHexperimenter's BBOB functions, where the right answer is
known and the literature says which method should win which function::

    python scripts/validate_optimizers.py --dimensions 128 512

Expect, and the script checks for:

* every step-size-adaptive method solves the sphere to near machine precision;
* ``sep_cma_es`` wins the *separable* ellipsoid -- a diagonal covariance is
  exactly right there;
* ``lm_cma_es`` / ``lm_ma_es`` win the *rotated* ellipsoid and discus, which a
  diagonal covariance cannot follow, at a fraction of full CMA-ES's runtime.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

import ioh  # noqa: E402
import numpy as np  # noqa: E402

from gaab.optimizers import get_optimizer  # noqa: E402

#: BBOB function ids, by the structure each one probes.
FUNCTIONS = {
    "sphere": 1,
    "ellipsoid_separable": 2,
    "rosenbrock": 8,
    "ellipsoid_rotated": 10,
    "discus": 11,
    "different_powers": 14,
}

DEFAULT_SOLVERS = (
    "cma_es",
    "sep_cma_es",
    "lm_cma_es",
    "lm_ma_es",
    "sep_nes",
    "random_search",
)


class _Unbounded(ioh.problem.RealSingleObjective):
    """A BBOB function with its box opened up.

    BBOB is defined on all of R^n and its reference results assume an
    unconstrained search; the benchmark's own in-box clipping would otherwise
    dominate this comparison and hide real differences between the methods.
    """

    def __init__(self, function_id: int, n: int, instance: int = 1) -> None:
        self.inner = ioh.get_problem(function_id, instance, n)
        super().__init__(
            f"bbob_f{function_id}", n, instance, True, ioh.RealBounds(n, -1e12, 1e12)
        )

    def evaluate(self, x):
        return float(self.inner(list(x)))

    @property
    def gap(self) -> float:
        return float(self.state.current_best.y - self.inner.optimum.y)


def run(solvers, dimensions, budget_factor: int, repetitions: int):
    import pandas as pd

    rows = []
    for n in dimensions:
        budget = budget_factor * n
        for label, function_id in FUNCTIONS.items():
            for solver in solvers:
                gaps, seconds = [], []
                for seed in range(repetitions):
                    problem = _Unbounded(function_id, n)
                    rng = np.random.default_rng(seed)
                    started = time.perf_counter()
                    get_optimizer(solver)(
                        problem, budget, rng, x0=rng.uniform(-4, 4, n), sigma0=2.0
                    )
                    seconds.append(time.perf_counter() - started)
                    gaps.append(problem.gap)
                rows.append(
                    {
                        "dimension": n,
                        "function": label,
                        "solver": solver,
                        "budget": budget,
                        "median_gap": float(np.median(gaps)),
                        "best_gap": float(np.min(gaps)),
                        "median_seconds": float(np.median(seconds)),
                        "repetitions": repetitions,
                    }
                )
                print(
                    f"n={n:<5} {label:<20} {solver:<16} "
                    f"gap={rows[-1]['median_gap']:.2e} "
                    f"({rows[-1]['median_seconds']:.1f}s)",
                    flush=True,
                )
    return pd.DataFrame(rows)


def report(frame) -> int:
    """Check the qualitative expectations; returns a process exit code."""
    failures = []
    for n, per_dim in frame.groupby("dimension"):
        pivot = per_dim.pivot_table(
            index="function", columns="solver", values="median_gap"
        )

        for solver in ("cma_es", "sep_cma_es", "lm_cma_es", "lm_ma_es"):
            if solver in pivot and pivot.loc["sphere", solver] > 1e-5:
                failures.append(f"n={n}: {solver} did not solve the sphere")

        if {"lm_cma_es", "sep_cma_es"} <= set(pivot.columns):
            for function in ("ellipsoid_rotated", "discus"):
                if pivot.loc[function, "lm_cma_es"] > pivot.loc[function, "sep_cma_es"]:
                    failures.append(
                        f"n={n}: lm_cma_es lost to sep_cma_es on {function} "
                        "(a limited memory should beat a diagonal on rotated structure)"
                    )

        print(f"\n=== median gap to the optimum, dimension {n} ===")
        print(pivot.map(lambda v: f"{v:.2e}").to_string())

    print()
    for failure in failures:
        print(f"UNEXPECTED: {failure}")
    if not failures:
        print("All expectations held.")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--solvers", nargs="+", default=list(DEFAULT_SOLVERS))
    parser.add_argument("--dimensions", type=int, nargs="+", default=[128])
    parser.add_argument(
        "--budget-factor",
        type=int,
        default=100,
        help="budget = this many evaluations per dimension",
    )
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", default="results/optimizer_validation.csv")
    args = parser.parse_args(argv)

    frame = run(args.solvers, args.dimensions, args.budget_factor, args.reps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    return report(frame)


if __name__ == "__main__":
    sys.exit(main())
