"""Run a whole benchmark family against one or more optimizers, write CSVs.

This is the module behind ``scripts/run_benchmark.py`` (criterion 6 of the
plan).  Output for a run directory::

    runs.csv     one row per (optimizer, problem, instance, repetition)
    curves.csv   incumbent objective / train / test accuracy on a log-spaced
                 budget grid -- everything the plots need
    summary.csv  per-optimizer aggregate (mean AOCC, grokked fraction, ...)
    ert.csv      ERT and success rate per (optimizer, problem, accuracy target)
    ioh/         optional IOHprofiler-format logs, readable by IOHanalyzer

Every run is seeded from ``(base_seed, problem, instance, repetition)``, so a
rerun reproduces the table exactly and adding an optimizer does not perturb the
others.
"""

from __future__ import annotations

import os
import time
import zlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from gaab import analysis as an
from gaab.parallel import map_unordered
from gaab.optimizers import GRADIENT_BASED, get_optimizer
from gaab.problem import (
    GrokkingProblem,
    GrokkingSuite,
    IncumbentWatcher,
    ProblemSpec,
)


@dataclass
class RunConfig:
    """Everything that defines a benchmark campaign."""

    optimizers: Sequence[str]
    budget: int = 60_000
    repetitions: int = 3
    instances: Sequence[int] | None = None
    seed: int = 42
    x0_mode: str = "init"
    curve_points: int = 120
    accuracy_targets: Sequence[float] = field(
        default_factory=lambda: list(an.DEFAULT_ACCURACY_TARGETS)
    )
    train_threshold: float = 0.99
    test_threshold: float = 0.95
    grokking_delay_factor: float = 2.0
    ioh_logs: bool = False
    jobs: int = 1


def _run_seed(base: int, problem_name: str, instance: int, rep: int) -> int:
    """Deterministic per-run seed that does not depend on iteration order."""
    key = f"{base}|{problem_name}|{instance}|{rep}".encode()
    return int(zlib.crc32(key))


def _row_from_segments(segments, budget, problem, config, meta) -> dict:
    hi = problem.random_guess_loss
    t_fit = an.hitting_time(segments, config.train_threshold, "train_accuracy", "above")
    t_gen = an.hitting_time(segments, config.test_threshold, "test_accuracy", "above")
    delay = float("nan")
    grokked = False
    if t_fit is not None and t_gen is not None:
        delay = t_gen / max(t_fit, 1)
        grokked = delay >= config.grokking_delay_factor

    last = segments[-1]
    row = {
        **meta,
        "budget": budget,
        "final_objective": last.objective,
        "final_train_accuracy": last.train_accuracy,
        "final_test_accuracy": last.test_accuracy,
        "best_test_accuracy": max(s.test_accuracy for s in segments),
        "aocc_objective": an.aocc_objective(segments, budget, hi=hi),
        "aocc_test_accuracy": an.aocc_accuracy(segments, budget, "test_accuracy"),
        "aocc_train_accuracy": an.aocc_accuracy(segments, budget, "train_accuracy"),
        "random_guess_loss": hi,
        "t_fit": t_fit,
        "t_gen": t_gen,
        "grokking_delay": delay,
        "grokked": grokked,
    }
    for target in config.accuracy_targets:
        row[f"t_acc_{target:g}"] = an.hitting_time(segments, target, "test_accuracy", "above")
    for target in an.objective_targets(hi):
        row[f"t_obj_{target:.3g}"] = an.hitting_time(segments, target, "objective", "below")
    return row


def run_single(
    spec: ProblemSpec,
    instance: int,
    optimizer: str,
    config: RunConfig,
    repetition: int,
    problem: GrokkingProblem | None = None,
) -> tuple[dict, list]:
    """One (problem, instance, optimizer, repetition) run.

    Returns the summary row and the sampled curve rows.
    """
    problem = problem or GrokkingProblem(spec, instance=instance)
    solver = get_optimizer(optimizer)
    seed = _run_seed(config.seed, spec.name, instance, repetition)
    rng = np.random.default_rng(seed)

    started = time.perf_counter()
    solver(problem, config.budget, rng, x0_mode=config.x0_mode)
    elapsed = time.perf_counter() - started

    budget = min(config.budget, max(problem.state.evaluations, 1))
    segments = an.segments_from_history(problem.history, budget, problem.random_guess_loss)

    meta = {
        "optimizer": optimizer,
        "gradient_based": optimizer in GRADIENT_BASED,
        "problem": spec.name,
        "problem_id": spec.stable_id,
        "instance": instance,
        "run": repetition,
        "seed": seed,
        "n_variables": spec.n_variables,
        "p": spec.p,
        "rank": spec.rank,
        "reg_strength": spec.reg_strength,
        "train_frac": spec.train_frac,
        "evaluations": problem.state.evaluations,
        "seconds": elapsed,
    }
    row = _row_from_segments(segments, budget, problem, config, meta)

    grid = an.budget_grid(budget, config.curve_points)
    curve_rows = [
        {
            "optimizer": optimizer,
            "problem": spec.name,
            "instance": instance,
            "run": repetition,
            "n_variables": spec.n_variables,
            "evaluations": int(t),
            "objective": float(o),
            "train_accuracy": float(tr),
            "test_accuracy": float(te),
        }
        for t, o, tr, te in zip(
            grid,
            an.curve_at(segments, grid, "objective"),
            an.curve_at(segments, grid, "train_accuracy"),
            an.curve_at(segments, grid, "test_accuracy"),
        )
    ]
    return row, curve_rows


def _worker(payload):
    """Process-pool entry point; rebuilds the problem from its spec."""
    spec_dict, instance, optimizer, config, repetition = payload
    spec = ProblemSpec.from_dict(spec_dict)
    return run_single(spec, instance, optimizer, config, repetition)


def run_suite(
    suite: GrokkingSuite,
    config: RunConfig,
    output_dir: str = "results/benchmark",
    verbose: bool = True,
):
    """Run every (spec, instance, optimizer, repetition) and write the CSVs."""
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    def instances_for(spec: ProblemSpec) -> list[int]:
        # An explicit --instances overrides the suite's curation; otherwise the
        # per-spec instance list chosen by screening wins.
        if config.instances is not None:
            return list(config.instances)
        return list(suite.instances_for(spec))

    jobs = [
        (spec, instance, optimizer, rep)
        for optimizer in config.optimizers
        for spec in suite.specs
        for instance in instances_for(spec)
        for rep in range(config.repetitions)
    ]
    rows: list[dict] = []
    curves: list[dict] = []
    started = time.perf_counter()

    if config.jobs > 1 and config.ioh_logs:
        print(
            "IOH logging needs a single process (loggers are not picklable); "
            "running serially despite jobs > 1.",
            flush=True,
        )

    if config.jobs > 1 and not config.ioh_logs:
        payloads = [(s.to_dict(), i, o, config, r) for s, i, o, r in jobs]
        results = map_unordered(_worker, payloads, config.jobs)
        for done, (row, curve_rows) in enumerate(results, start=1):
            rows.append(row)
            curves.extend(curve_rows)
            if verbose:
                _report(done, len(jobs), row, started)
    else:
        loggers = _make_loggers(config, output_dir) if config.ioh_logs else {}
        done = 0
        for optimizer in config.optimizers:
            for spec in suite.specs:
                for instance in instances_for(spec):
                    problem = GrokkingProblem(spec, instance=instance)
                    logger, incumbent = loggers.get(optimizer, (None, None))
                    if logger is not None:
                        incumbent.problem = problem
                        problem.attach_logger(logger)
                    for rep in range(config.repetitions):
                        row, curve_rows = run_single(
                            spec, instance, optimizer, config, rep, problem=problem
                        )
                        rows.append(row)
                        curves.extend(curve_rows)
                        done += 1
                        if verbose:
                            _report(done, len(jobs), row, started)
                        problem.reset()
                    if logger is not None:
                        problem.detach_logger()
        for logger, _ in loggers.values():
            logger.close()

    runs = pd.DataFrame(rows).sort_values(["optimizer", "problem", "instance", "run"])
    curves_df = pd.DataFrame(curves)
    runs.to_csv(os.path.join(output_dir, "runs.csv"), index=False)
    curves_df.to_csv(os.path.join(output_dir, "curves.csv"), index=False)
    an.summary_table(runs).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    an.ert_table(runs, config.accuracy_targets).to_csv(
        os.path.join(output_dir, "ert.csv"), index=False
    )
    if verbose:
        print(f"\nWrote {len(runs)} runs to {output_dir}/ in {time.perf_counter()-started:.0f}s")
    return runs, curves_df


def _make_loggers(config: RunConfig, output_dir: str) -> dict:
    """One ``Analyzer`` per optimizer, each with its forwarding watch target."""
    import ioh

    root = os.path.join(output_dir, "ioh")
    os.makedirs(root, exist_ok=True)
    loggers = {}
    for name in config.optimizers:
        logger = ioh.logger.Analyzer(
            root=root,
            folder_name=name,
            algorithm_name=name,
            algorithm_info=f"gaab budget={config.budget}",
            store_positions=False,
        )
        incumbent = IncumbentWatcher()
        logger.watch(incumbent, list(GrokkingProblem.WATCHED))
        loggers[name] = (logger, incumbent)
    return loggers


def _report(done: int, total: int, row: dict, started: float) -> None:
    elapsed = time.perf_counter() - started
    eta = elapsed / done * (total - done)
    print(
        f"[{done:>5}/{total}] {row['optimizer']:<18} {row['problem']:<44} "
        f"i{row['instance']} r{row['run']} f={row['final_objective']:.4f} "
        f"test={row['final_test_accuracy']:.2f} aocc={row['aocc_objective']:.3f} "
        f"| eta {eta/60:.1f}m",
        flush=True,
    )
