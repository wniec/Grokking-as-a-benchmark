"""Decide which problems in the family actually grok.

A landscape only belongs in this benchmark if delayed generalization is
*achievable* on it.  That is a property of the problem, not of the optimizer
under test, so it is established once with a gradient-based reference
optimizer (full-batch Adam on the exact same objective, using the analytic
gradients in ``model.py``) and then frozen into a curated suite.

Four things get filtered out; only genuine grokking is kept:

``no_memorization``
    The reference never even fits the training set -- usually the
    regularization strength swamps the cross-entropy.
``no_generalization``
    Train accuracy saturates and test accuracy never follows.
``early_generalization``
    Test accuracy arrives together with train accuracy.  Nothing was delayed,
    so there is no grokking transition for an optimizer to be measured against.
``divergence``
    The reference blew up (the quadratic activation can, at a large init).

What is left is labelled ``grokking``: train fits early, test follows much
later.  ``delay_ratio = t_gen / t_fit`` quantifies "much later".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from gaab.objective import GrokkingObjective
from gaab.parallel import map_unordered
from gaab.problem import GrokkingProblem, ProblemSpec

VERDICTS = (
    "grokking",
    "early_generalization",
    "no_generalization",
    "no_memorization",
    "divergence",
)


@dataclass
class AdamSettings:
    """Reference-optimizer settings.  Shared by every screened problem."""

    steps: int = 60_000
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    eval_every: int = 250
    init_scale: float = 1.0


@dataclass
class GrokkingReport:
    """Verdict for one (spec, instance) pair."""

    problem: str
    problem_id: int
    instance: int
    n_variables: int
    p: int
    rank: int | None
    reg_strength: float
    train_frac: float
    activation: str
    t_fit: int | None
    t_gen: int | None
    delay_ratio: float
    final_train_accuracy: float
    final_test_accuracy: float
    best_test_accuracy: float
    final_objective: float
    verdict: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def adam_trajectory(
    objective: GrokkingObjective,
    theta0: np.ndarray,
    settings: AdamSettings = AdamSettings(),
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Run full-batch Adam, sampling train/test metrics along the way."""
    theta = np.array(theta0, dtype=np.float64, copy=True)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    trace: list[dict[str, float]] = []

    for step in range(1, settings.steps + 1):
        loss, grad = objective.loss_and_grad(theta)
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            break
        m = settings.beta1 * m + (1.0 - settings.beta1) * grad
        v = settings.beta2 * v + (1.0 - settings.beta2) * grad * grad
        m_hat = m / (1.0 - settings.beta1**step)
        v_hat = v / (1.0 - settings.beta2**step)
        theta = theta - settings.lr * m_hat / (np.sqrt(v_hat) + settings.eps)

        if step == 1 or step % settings.eval_every == 0 or step == settings.steps:
            metrics = objective.metrics(theta)
            trace.append({"step": float(step), **metrics.as_dict()})

    return theta, trace


def _first_step(trace: Sequence[dict[str, float]], key: str, threshold: float) -> int | None:
    for row in trace:
        if row[key] >= threshold:
            return int(row["step"])
    return None


def classify(
    trace: Sequence[dict[str, float]],
    settings: AdamSettings,
    train_threshold: float = 0.99,
    test_threshold: float = 0.95,
    early_factor: float = 2.0,
) -> tuple[str, int | None, int | None, float]:
    """Turn a reference trajectory into a verdict and a delay ratio."""
    if not trace:
        return "divergence", None, None, float("nan")
    # A trajectory that stopped early hit a non-finite loss.
    if trace[-1]["step"] < settings.steps:
        return "divergence", None, None, float("nan")

    t_fit = _first_step(trace, "train_accuracy", train_threshold)
    t_gen = _first_step(trace, "test_accuracy", test_threshold)

    if t_fit is None:
        return "no_memorization", None, t_gen, float("nan")
    if t_gen is None:
        return "no_generalization", t_fit, None, float("inf")

    ratio = t_gen / max(t_fit, 1)
    verdict = "grokking" if ratio >= early_factor else "early_generalization"
    return verdict, t_fit, t_gen, float(ratio)


def screen_problem(
    spec: ProblemSpec,
    instance: int = 1,
    settings: AdamSettings = AdamSettings(),
    train_threshold: float = 0.99,
    test_threshold: float = 0.95,
    early_factor: float = 2.0,
    return_trace: bool = False,
):
    """Screen a single (spec, instance) pair with the reference optimizer."""
    problem = GrokkingProblem(spec, instance=instance, track_history=False)
    theta0 = problem.initial_theta(scale=settings.init_scale)
    _, trace = adam_trajectory(problem.objective, theta0, settings)
    verdict, t_fit, t_gen, ratio = classify(
        trace, settings, train_threshold, test_threshold, early_factor
    )
    last = trace[-1] if trace else {}

    report = GrokkingReport(
        problem=spec.name,
        problem_id=spec.stable_id,
        instance=instance,
        n_variables=spec.n_variables,
        p=spec.p,
        rank=spec.rank,
        reg_strength=spec.reg_strength,
        train_frac=spec.train_frac,
        activation=spec.activation,
        t_fit=t_fit,
        t_gen=t_gen,
        delay_ratio=ratio,
        final_train_accuracy=float(last.get("train_accuracy", float("nan"))),
        final_test_accuracy=float(last.get("test_accuracy", float("nan"))),
        best_test_accuracy=float(max((r["test_accuracy"] for r in trace), default=float("nan"))),
        final_objective=float(last.get("objective", float("nan"))),
        verdict=verdict,
    )
    return (report, trace) if return_trace else report


def _screen_worker(payload):
    """Process-pool entry point; rebuilds the spec from its dict."""
    spec_dict, instance, settings, thresholds = payload
    spec = ProblemSpec.from_dict(spec_dict)
    return screen_problem(spec, instance, settings, *thresholds).as_row()


def screen_specs(
    specs: Sequence[ProblemSpec],
    instances: Sequence[int] = (1, 2, 3),
    settings: AdamSettings = AdamSettings(),
    train_threshold: float = 0.99,
    test_threshold: float = 0.95,
    early_factor: float = 2.0,
    jobs: int = 1,
    verbose: bool = True,
):
    """Screen a whole family; returns a ``pandas.DataFrame`` of reports.

    Screening runs are independent, so ``jobs > 1`` spreads them over processes
    -- worth it: a full grid is thousands of gradient-descent runs.
    """
    import pandas as pd

    thresholds = (train_threshold, test_threshold, early_factor)
    payloads = [
        (spec.to_dict(), instance, settings, thresholds)
        for spec in specs
        for instance in instances
    ]
    rows = []
    for done, row in enumerate(map_unordered(_screen_worker, payloads, jobs), start=1):
        rows.append(row)
        if verbose:
            print(
                f"[{done:>5}/{len(payloads)}] {row['problem']} i{row['instance']} "
                f"n={row['n_variables']:<5} -> {row['verdict']} "
                f"(t_fit={row['t_fit']}, t_gen={row['t_gen']})",
                flush=True,
            )
    return pd.DataFrame(rows)


def filter_specs(
    specs: Sequence[ProblemSpec],
    reports,
    min_fraction: float = 1 / 3,
    instance_level: bool = True,
):
    """Keep what groks; report why the rest was dropped.

    A landscape frequently groks under some train/test splits and not others,
    so filtering happens at the (spec, instance) level by default: a spec is
    kept if at least ``min_fraction`` of its instances grok, and only its
    grokking instances are retained.  Set ``instance_level=False`` to keep
    every instance of a surviving spec instead.

    Returns ``(kept_specs, spec_instances, summary_frame)``.  The summary is
    worth keeping around: "why was this problem dropped" is the first question
    anyone asks of a curated suite.
    """

    if len(reports) == 0:
        raise ValueError("No screening reports to filter on.")

    reports = reports.copy()
    reports["is_grokking"] = reports["verdict"] == "grokking"
    grouped = reports.groupby("problem", sort=False)
    summary = grouped.agg(
        n_variables=("n_variables", "first"),
        p=("p", "first"),
        rank=("rank", "first"),
        reg_strength=("reg_strength", "first"),
        train_frac=("train_frac", "first"),
        instances=("instance", "count"),
        grokking_instances=("is_grokking", "sum"),
        grokking_fraction=("is_grokking", "mean"),
        median_delay_ratio=("delay_ratio", "median"),
        median_t_fit=("t_fit", "median"),
        median_t_gen=("t_gen", "median"),
        mean_final_test_accuracy=("final_test_accuracy", "mean"),
    ).reset_index()
    # The modal verdict explains the rejects at a glance.
    modal = grouped["verdict"].agg(lambda s: s.value_counts().idxmax())
    summary = summary.merge(modal.rename("modal_verdict"), on="problem")
    summary["kept"] = summary["grokking_fraction"] >= min_fraction

    keep_names = set(summary.loc[summary["kept"], "problem"])
    kept = [s for s in specs if s.name in keep_names]

    spec_instances: dict[str, list[int]] = {}
    for name in keep_names:
        rows = reports[reports["problem"] == name]
        chosen = rows.loc[rows["is_grokking"], "instance"] if instance_level else rows["instance"]
        spec_instances[name] = sorted(int(i) for i in chosen)

    summary["kept_instances"] = summary["problem"].map(
        lambda n: ",".join(str(i) for i in spec_instances.get(n, []))
    )
    return kept, spec_instances, summary
