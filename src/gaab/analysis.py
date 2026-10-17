"""Performance metrics: AOCC, ERT, and the curves the plots are drawn from.

Two families of metric are reported for every run, because these problems have
two axes that come apart -- which is the whole point of using grokking as a
benchmark:

*optimization* (the objective the algorithm is actually minimizing)
    ``aocc_objective``, and ERT to a grid of objective targets.
*generalization* (test accuracy of the incumbent, never part of the objective)
    ``aocc_test_accuracy``, and ERT to test-accuracy targets -- the
    "expected running time to grok".

Everything is computed exactly from a run's improvement history.  The
best-so-far curve is a step function that only changes when the incumbent
improves, so integrating it needs no sampling grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

#: Objective values below this are treated as "solved" when log-scaling.
DEFAULT_LOSS_FLOOR = 1e-2

#: Test-accuracy levels an optimizer is timed against.
DEFAULT_ACCURACY_TARGETS: tuple[float, ...] = (0.25, 0.5, 0.75, 0.95)


@dataclass(frozen=True)
class Segment:
    """A stretch of evaluations over which the incumbent did not change."""

    start: int  # inclusive, 1-based
    stop: int  # inclusive
    objective: float
    train_accuracy: float
    test_accuracy: float

    @property
    def length(self) -> int:
        return self.stop - self.start + 1


def segments_from_history(
    history: Sequence, budget: int, initial_objective: float
) -> list[Segment]:
    """Turn improvement records into the piecewise-constant incumbent curve.

    ``history`` items need ``evaluations``, ``objective``, ``train_accuracy``
    and ``test_accuracy`` attributes (``problem.history`` provides them).
    Evaluations before the first improvement are charged ``initial_objective``
    -- normally the random-guess loss -- and chance-level accuracy.
    """
    records = sorted(history, key=lambda r: r.evaluations)
    out: list[Segment] = []
    cursor = 1
    cur = (initial_objective, 0.0, 0.0)

    for rec in records:
        t = int(rec.evaluations)
        if t > budget:
            break
        if t > cursor:
            out.append(Segment(cursor, t - 1, *cur))
        cur = (float(rec.objective), float(rec.train_accuracy), float(rec.test_accuracy))
        cursor = t
    if cursor <= budget:
        out.append(Segment(cursor, budget, *cur))
    return out


def curve_at(segments: Sequence[Segment], grid: Iterable[int], attr: str) -> np.ndarray:
    """Sample the step function at arbitrary evaluation counts."""
    grid = np.asarray(list(grid), dtype=np.int64)
    starts = np.array([s.start for s in segments])
    values = np.array([getattr(s, attr) for s in segments], dtype=np.float64)
    idx = np.clip(np.searchsorted(starts, grid, side="right") - 1, 0, len(segments) - 1)
    return values[idx]


def budget_grid(budget: int, n_points: int = 120) -> np.ndarray:
    """Log-spaced evaluation counts, for plotting."""
    grid = np.unique(np.round(np.geomspace(1, max(budget, 2), n_points)).astype(np.int64))
    return grid[grid <= budget]


# ---------------------------------------------------------------------------
# AOCC
# ---------------------------------------------------------------------------
def _log_normalize(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map objective values to [0, 1]; 0 = at or below ``lo``, 1 = at ``hi``."""
    v = np.maximum(np.asarray(values, dtype=np.float64), lo)
    span = np.log10(hi) - np.log10(lo)
    return np.clip((np.log10(v) - np.log10(lo)) / span, 0.0, 1.0)


def aocc_objective(
    segments: Sequence[Segment],
    budget: int,
    hi: float,
    lo: float = DEFAULT_LOSS_FLOOR,
) -> float:
    """Area *over* the log-scaled convergence curve, in [0, 1]; higher is better.

    ``AOCC = 1 - (1/B) * sum_t normalized(f_best(t))`` -- the classic
    anytime-performance measure.  Because ``f_best`` is a step function this is
    an exact sum over segments, not a quadrature.
    """
    if budget <= 0 or not segments:
        return 0.0
    if hi <= lo:
        raise ValueError("AOCC needs hi > lo.")
    vals = _log_normalize(np.array([s.objective for s in segments]), lo, hi)
    weights = np.array([s.length for s in segments], dtype=np.float64)
    return float(1.0 - np.sum(vals * weights) / budget)


def aocc_accuracy(segments: Sequence[Segment], budget: int, attr: str = "test_accuracy") -> float:
    """Mean incumbent accuracy over the budget: area *under* the curve, in [0, 1].

    The generalization counterpart of :func:`aocc_objective`.  An optimizer
    that groks early scores high; one that groks at the very last evaluation
    scores near zero, which is the intended pressure.
    """
    if budget <= 0 or not segments:
        return 0.0
    vals = np.array([getattr(s, attr) for s in segments], dtype=np.float64)
    weights = np.array([s.length for s in segments], dtype=np.float64)
    return float(np.sum(vals * weights) / budget)


# ---------------------------------------------------------------------------
# hitting times and ERT
# ---------------------------------------------------------------------------
def hitting_time(
    segments: Sequence[Segment], target: float, attr: str, mode: str = "below"
) -> int | None:
    """First evaluation at which the incumbent reached ``target``."""
    for seg in segments:
        value = getattr(seg, attr)
        hit = value <= target if mode == "below" else value >= target
        if hit:
            return seg.start
    return None


def ert(hitting_times: Sequence[int | None], budget: int) -> tuple[float, float]:
    """Expected running time and success rate over a set of independent runs.

    ``ERT = (sum of evaluations of successful runs + budget * #failures) /
    #successes`` -- the standard restart-based estimator (Hansen et al.); it is
    ``inf`` when no run reached the target.
    """
    times = list(hitting_times)
    if not times:
        return float("inf"), 0.0
    successes = [t for t in times if t is not None]
    n_fail = len(times) - len(successes)
    rate = len(successes) / len(times)
    if not successes:
        return float("inf"), 0.0
    return (sum(successes) + n_fail * budget) / len(successes), rate


def objective_targets(hi: float, lo: float = DEFAULT_LOSS_FLOOR, n: int = 8) -> np.ndarray:
    """A geometric ladder of objective targets between ``hi`` and ``lo``."""
    return np.geomspace(hi, lo, n)


# ---------------------------------------------------------------------------
# aggregation over a results table
# ---------------------------------------------------------------------------
def ert_table(runs, accuracy_targets: Sequence[float] = DEFAULT_ACCURACY_TARGETS):
    """ERT to each test-accuracy target, per (optimizer, problem).

    ``runs`` is the ``runs.csv`` frame written by :mod:`gaab.runner`; it carries
    one ``t_acc_<target>`` column per target (``NaN`` when never reached).
    """
    import pandas as pd

    rows = []
    for (opt, problem), group in runs.groupby(["optimizer", "problem"], sort=False):
        budget = int(group["budget"].max())
        for target in accuracy_targets:
            col = f"t_acc_{target:g}"
            if col not in group:
                continue
            times = [None if pd.isna(v) else int(v) for v in group[col]]
            value, rate = ert(times, budget)
            rows.append(
                {
                    "optimizer": opt,
                    "problem": problem,
                    "n_variables": int(group["n_variables"].iloc[0]),
                    "p": int(group["p"].iloc[0]),
                    "target_test_accuracy": target,
                    "ert": value,
                    "success_rate": rate,
                    "n_runs": len(times),
                    "budget": budget,
                }
            )
    return pd.DataFrame(rows)


def summary_table(runs):
    """Per-optimizer aggregate: the headline table."""

    agg = (
        runs.groupby("optimizer", sort=False)
        .agg(
            problems=("problem", "nunique"),
            runs=("run", "count"),
            mean_aocc_objective=("aocc_objective", "mean"),
            mean_aocc_test_accuracy=("aocc_test_accuracy", "mean"),
            mean_final_objective=("final_objective", "mean"),
            mean_final_test_accuracy=("final_test_accuracy", "mean"),
            mean_best_test_accuracy=("best_test_accuracy", "mean"),
            grokked_fraction=("grokked", "mean"),
        )
        .reset_index()
        .sort_values("mean_aocc_objective", ascending=False)
    )
    return agg
