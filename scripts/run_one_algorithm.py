#!/usr/bin/env python
"""Run ONE optimizer on the grokking suite and/or BBOB, and write its results.

This is the unit of work behind the Slurm array job (hpc/): each array task runs
this script for a single algorithm, so the whole optimizer set is spread over the
cluster one algorithm per task. It is also perfectly runnable on its own::

    python scripts/run_one_algorithm.py --algorithm cma_es --benchmark both
    python scripts/run_one_algorithm.py --list          # names you can pass

Output (per algorithm, so array tasks never collide):
    <out>/gaab/<algo>/runs.csv        the grokking-suite campaign (full gaab metrics)
    <out>/bbob/<algo>.csv             one row per (function, dim, instance, rep)

``scripts/aggregate_hpc.py`` then stitches every algorithm's files into one table
and ranks the methods on each benchmark.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

# Pin BLAS before NumPy is imported: each array task owns its cores and
# parallelizes over problems itself, so threaded BLAS is pure oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402

from gaab import analysis as an  # noqa: E402
from gaab.optimizers import (  # noqa: E402
    GRADIENT_BASED,
    OPTIMIZERS,
    get_optimizer,
    optional_dependency,
)
from gaab.parallel import map_unordered  # noqa: E402

#: The canonical roster the HPC sweep runs (every registered black-box method,
#: plus the gradient reference on gaab only). Kept here so the Slurm files and
#: the aggregator share one source of truth -- `--list` prints it.
ALL_ALGORITHMS = tuple(sorted(OPTIMIZERS))


# ---------------------------------------------------------------------------
# BBOB side
# ---------------------------------------------------------------------------
@dataclass
class _Rec:
    """Minimal improvement record so gaab.analysis can score a BBOB curve."""

    evaluations: int
    objective: float
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0


class _Tracker:
    """Wrap an IOH problem, recording the best-so-far gap to the optimum.

    Optimizers only touch ``problem(x)``, ``problem.bounds`` and
    ``problem.meta_data``; everything but ``__call__`` is forwarded to the real
    problem, so this is a transparent proxy that also builds the improvement
    history the AOCC computation needs.
    """

    def __init__(self, problem, f_opt: float) -> None:
        self._problem = problem
        self._f_opt = f_opt
        self.history: list[_Rec] = []
        self._best = np.inf

    def __getattr__(self, name):
        return getattr(self._problem, name)

    def __call__(self, x) -> float:
        y = self._problem(x)
        gap = max(float(y) - self._f_opt, 0.0)
        if gap < self._best:
            self._best = gap
            self.history.append(_Rec(self._problem.state.evaluations, gap))
        return y


def _bbob_worker(payload):
    import ioh

    algo, fid, dim, instance, rep, budget_factor, base_seed = payload
    problem = ioh.get_problem(fid, instance, dim)
    tracker = _Tracker(problem, float(problem.optimum.y))
    budget = int(budget_factor * dim)
    seed = abs(hash((algo, fid, dim, instance, rep, base_seed))) % (2**31)
    rng = np.random.default_rng(seed)

    started = time.perf_counter()
    get_optimizer(algo)(tracker, budget, rng)
    elapsed = time.perf_counter() - started

    used = max(problem.state.evaluations, 1)
    hi = max(tracker.history[0].objective, 1e-7) if tracker.history else 1e-7
    segments = an.segments_from_history(tracker.history, used, hi)
    return {
        "optimizer": algo,
        "benchmark": "bbob",
        "problem": f"f{fid}",
        "fid": fid,
        "dim": dim,
        "instance": instance,
        "rep": rep,
        "budget": budget,
        "evaluations": int(problem.state.evaluations),
        "final_value": float(problem.state.current_best.y),
        "final_gap": float(problem.state.current_best.y - problem.optimum.y),
        "aocc_objective": an.aocc_objective(segments, used, hi=hi, lo=1e-8),
        "seconds": elapsed,
        "seed": seed,
    }


def run_bbob(algo, fids, dims, instances, reps, budget_factor, jobs, seed, out_dir):
    import pandas as pd

    if algo in GRADIENT_BASED:
        print(f"[bbob] skipping {algo}: gradient-based, not a black-box method.")
        return None
    missing = optional_dependency(algo)
    if missing:
        print(f"[bbob] skipping {algo}: optional `{missing}` package not installed.")
        return None

    payloads = [
        (algo, fid, dim, inst, rep, budget_factor, seed)
        for fid in fids for dim in dims for inst in instances for rep in range(reps)
    ]
    print(f"[bbob] {algo}: {len(payloads)} runs "
          f"({len(fids)} functions x {len(dims)} dims x {len(instances)} instances x {reps} reps)")
    rows = []
    started = time.perf_counter()
    for done, row in enumerate(map_unordered(_bbob_worker, payloads, jobs), start=1):
        rows.append(row)
        if done % 50 == 0 or done == len(payloads):
            print(f"[bbob] {algo}: {done}/{len(payloads)} ({time.perf_counter()-started:.0f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{algo}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[bbob] wrote {path}")
    return path


# ---------------------------------------------------------------------------
# gaab side (reuses the full benchmark machinery)
# ---------------------------------------------------------------------------
def run_gaab(algo, suite_path, budget, reps, instances, jobs, seed, out_dir):
    missing = optional_dependency(algo)
    if missing:
        print(f"[gaab] skipping {algo}: optional `{missing}` package not installed.")
        return None

    from gaab.runner import RunConfig, run_suite
    from gaab.suite import load_suite

    suite = load_suite(suite_path, instances=instances)
    config = RunConfig(
        optimizers=[algo], budget=budget, repetitions=reps,
        instances=instances, seed=seed, jobs=jobs,
    )
    algo_dir = os.path.join(out_dir, algo)
    print(f"[gaab] {algo}: {len(suite.specs)} problems, budget {budget}, {reps} reps")
    runs, _ = run_suite(suite, config, output_dir=algo_dir, verbose=True)
    print(f"[gaab] wrote {os.path.join(algo_dir, 'runs.csv')}")
    return os.path.join(algo_dir, "runs.csv")


def _parse_fids(tokens):
    fids: list[int] = []
    for tok in tokens:
        if "-" in tok:
            lo, hi = tok.split("-")
            fids.extend(range(int(lo), int(hi) + 1))
        else:
            fids.append(int(tok))
    return fids


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--algorithm", help="optimizer name (see --list)")
    parser.add_argument("--benchmark", choices=["gaab", "bbob", "both"], default="both")
    parser.add_argument("--list", action="store_true", help="print the algorithm names and exit")
    parser.add_argument("--output-dir", default="results/hpc")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    # gaab
    parser.add_argument("--suite", default=None, help="curated suite JSON (default: shipped)")
    parser.add_argument("--budget", type=int, default=60_000)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--instances", type=int, nargs="+", default=None)
    # bbob
    parser.add_argument("--bbob-fids", nargs="+", default=["1-24"])
    parser.add_argument("--bbob-dims", type=int, nargs="+", default=[5, 20, 40])
    parser.add_argument("--bbob-instances", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--bbob-reps", type=int, default=5)
    parser.add_argument("--bbob-budget-factor", type=int, default=2000,
                        help="BBOB budget = this * dimension")
    args = parser.parse_args(argv)

    if args.list:
        print("\n".join(ALL_ALGORITHMS))
        return 0
    if not args.algorithm:
        parser.error("--algorithm is required (or pass --list)")
    if args.algorithm not in OPTIMIZERS:
        parser.error(f"unknown algorithm {args.algorithm!r}; choose from {list(ALL_ALGORITHMS)}")

    if args.benchmark in ("gaab", "both"):
        run_gaab(args.algorithm, args.suite, args.budget, args.reps,
                 args.instances, args.jobs, args.seed, os.path.join(args.output_dir, "gaab"))
    if args.benchmark in ("bbob", "both"):
        run_bbob(args.algorithm, _parse_fids(args.bbob_fids), args.bbob_dims,
                 args.bbob_instances, args.bbob_reps, args.bbob_budget_factor,
                 args.jobs, args.seed, os.path.join(args.output_dir, "bbob"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
