"""Keep only problems whose *global* minimum is the generalizing solution.

Screening (``screening.py``) establishes that delayed generalization is
*reachable* on a landscape: the reference optimizer, from one init, groks.  But
the benchmark makes a stronger implicit promise.  An optimizer under test is
scored by how low it drives

    f(theta) = CE_train(theta) + reg * mean(theta ** 2),

and grokking is only a meaningful target if the *lowest* point of ``f`` is the
generalizing one.  If some memorizing solution sits lower in ``f``, then a
perfect optimizer is *rewarded* for not generalizing -- the grokking we observe
is an artifact of the Adam path, not a property of the objective, and the
problem does not belong in the suite.

The global minimum of a non-convex landscape cannot be certified, so it is
approximated by the lowest ``f`` found over a pool of candidate minima:

* a geometric ladder of random restarts.  ``scale`` is the Omnigrok ``alpha =
  w / w0`` knob (``model.initial_theta``): small scales settle in generalizing
  basins, large scales in memorizing ones, so the ladder samples both axes of
  the generalize/memorize dichotomy directly;
* explicitly constructed *min-norm memorizers*: fit the train set with no
  regularization (a high-norm memorizer), then descend the real ``f`` from
  there -- Adam lowers the weight norm while holding the train fit, finding the
  lowest-``f`` point reachable inside the memorizing basin.  This is the
  hardest competitor the generalizing solution can face.

A ``(spec, instance)`` pair is **kept** iff the lowest-``f`` candidate found is
a generalizing one -- equivalently, no memorizing minimum attains ``f`` at or
below the best generalizing minimum (up to ``margin``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from gaab.data import make_dataset
from gaab.objective import GrokkingObjective, Metrics
from gaab.parallel import map_unordered
from gaab.problem import ProblemSpec
from gaab.screening import AdamSettings, adam_trajectory

#: Candidate verdicts.  Only the first keeps a problem in the suite.
GLOBAL_VERDICTS = (
    "generalizing_global",  # the lowest-f minimum found generalizes -> keep
    "memorizing_global",  # a memorizing minimum attains f <= the best generalizer
    "no_generalizer",  # no candidate generalized at all
)


@dataclass
class GateSettings:
    """How the candidate pool is built for one ``(spec, instance)`` pair."""

    n_restarts: int = 64
    n_memorizers: int = 4
    scale_lo: float = 0.25
    scale_hi: float = 4.0
    #: init scale for the no-reg interpolators that seed the memorizer descent.
    memorizer_scale: float = 2.0
    train_threshold: float = 0.99
    test_threshold: float = 0.95
    #: how much lower in ``f`` the best generalizer must sit than the best
    #: memorizer to count as the global minimum.  ``0.0`` means "strictly
    #: lower"; a small positive value guards against descent noise.
    margin: float = 0.0
    seed: int = 0


@dataclass
class Candidate:
    """One converged minimum of ``f`` and what it generalized to."""

    source: str  # "restart" or "memorizer"
    scale: float
    objective: float
    train_accuracy: float
    test_accuracy: float
    weight_norm: float
    reg_penalty: float
    finite: bool

    def classify(self, train_threshold: float, test_threshold: float) -> str:
        if not self.finite or self.train_accuracy < train_threshold:
            return "unfit"
        if self.test_accuracy >= test_threshold:
            return "generalizing"
        return "memorizing"


@dataclass
class GlobalMinReport:
    """Global-minimum verdict for one ``(spec, instance)`` pair."""

    problem: str
    problem_id: int
    instance: int
    n_variables: int
    p: int
    rank: int | None
    reg_strength: float
    train_frac: float
    activation: str
    n_candidates: int
    n_generalizing: int
    n_memorizing: int
    best_gen_objective: float
    best_gen_test_accuracy: float
    best_gen_weight_norm: float
    best_mem_objective: float
    best_mem_test_accuracy: float
    best_mem_weight_norm: float
    global_objective: float
    global_generalizes: bool
    margin: float
    verdict: str
    keep: bool

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def _objectives(
    spec: ProblemSpec, instance: int
) -> tuple[GrokkingObjective, GrokkingObjective]:
    """Build the target ``f`` and its no-reg sibling on the same data.

    The gate reasons about the *native* objective ``f``; any benchmark-time
    random-subspace projection (``spec.projection_dim``) is a reparameterization
    that cannot change which weight vector globally minimizes ``f``, so it is
    deliberately ignored here.
    """
    data = make_dataset(p=spec.p, op=spec.op, train_frac=spec.train_frac, seed=instance)
    target = GrokkingObjective(spec.model_config, data, reg_strength=spec.reg_strength)
    pure = GrokkingObjective(spec.model_config, data, reg_strength=0.0)
    return target, pure


def _as_candidate(
    source: str, scale: float, metrics: Metrics, finite: bool
) -> Candidate:
    return Candidate(
        source=source,
        scale=float(scale),
        objective=float(metrics.objective),
        train_accuracy=float(metrics.train_accuracy),
        test_accuracy=float(metrics.test_accuracy),
        weight_norm=float(metrics.weight_norm),
        reg_penalty=float(metrics.reg_penalty),
        finite=bool(finite and np.isfinite(metrics.objective)),
    )


def _descend(objective: GrokkingObjective, theta0: np.ndarray, settings: AdamSettings):
    """Descend ``f`` to convergence; return ``(final_metrics, finite, theta)``."""
    theta, trace = adam_trajectory(objective, theta0, settings)
    # A trajectory that stopped before the last step hit a non-finite loss.
    finite = bool(trace) and trace[-1]["step"] >= settings.steps
    metrics = objective.metrics(theta)
    return metrics, finite, theta


def collect_candidates(
    spec: ProblemSpec,
    instance: int,
    settings: AdamSettings = AdamSettings(),
    gate: GateSettings = GateSettings(),
) -> list[Candidate]:
    """Build the pool of candidate minima of ``f`` for one problem instance."""
    target, pure = _objectives(spec, instance)
    model = target.model
    rng_scales = np.geomspace(gate.scale_lo, gate.scale_hi, num=max(gate.n_restarts, 1))

    candidates: list[Candidate] = []

    # 0. The reference init screening itself used (scale=1.0, seed=instance).
    #    That exact init is *known* to grok -- it is why this problem passed
    #    screening -- and full-batch Adam is deterministic, so descending it here
    #    reproduces the generalizing solution screening already found.  Without
    #    it the pool can miss a generalizer the suite already guarantees exists,
    #    and the problem is dropped as ``no_generalizer`` for no real reason.
    theta_ref = model.initial_theta(seed=instance, scale=1.0)
    metrics, finite, _ = _descend(target, theta_ref, settings)
    candidates.append(_as_candidate("reference", 1.0, metrics, finite))

    # 1. Random restarts spread across the init-scale (generalize<->memorize) axis.
    for i, scale in enumerate(rng_scales):
        theta0 = model.initial_theta(
            seed=gate.seed + 1000 * instance + i, scale=float(scale)
        )
        metrics, finite, _ = _descend(target, theta0, settings)
        candidates.append(_as_candidate("restart", scale, metrics, finite))

    # 2. Explicit min-norm memorizers: interpolate train with no reg (high-norm
    #    memorizer), then descend the real f from there to squeeze the norm down
    #    without leaving the memorizing basin.
    for j in range(gate.n_memorizers):
        seed = gate.seed + 50_000 + 1000 * instance + j
        theta_mem0 = model.initial_theta(seed=seed, scale=gate.memorizer_scale)
        interp, finite0, theta_interp = _descend(pure, theta_mem0, settings)
        if not finite0:
            # The no-reg fit diverged (quad activation at large init can);
            # it contributes no usable memorizer.
            candidates.append(
                _as_candidate("memorizer", gate.memorizer_scale, interp, False)
            )
            continue
        metrics, finite, _ = _descend(target, theta_interp, settings)
        candidates.append(
            _as_candidate("memorizer", gate.memorizer_scale, metrics, finite)
        )

    return candidates


def assess(
    candidates: Sequence[Candidate],
    gate: GateSettings = GateSettings(),
) -> tuple[str, bool, dict[str, float]]:
    """Reduce a candidate pool to a verdict plus the numbers behind it."""
    usable = [c for c in candidates if c.finite]
    gens = [
        c
        for c in usable
        if c.classify(gate.train_threshold, gate.test_threshold) == "generalizing"
    ]
    mems = [
        c
        for c in usable
        if c.classify(gate.train_threshold, gate.test_threshold) == "memorizing"
    ]

    best_gen = min(gens, key=lambda c: c.objective, default=None)
    best_mem = min(mems, key=lambda c: c.objective, default=None)
    global_cand = min(usable, key=lambda c: c.objective, default=None)

    best_gen_f = best_gen.objective if best_gen else float("inf")
    best_mem_f = best_mem.objective if best_mem else float("inf")

    if best_gen is None:
        verdict = "no_generalizer"
    elif best_mem_f - gate.margin <= best_gen_f:
        # A memorizing minimum is as deep as (or deeper than) the best
        # generalizer: the objective does not single out generalization.
        verdict = "memorizing_global"
    else:
        verdict = "generalizing_global"

    numbers = {
        "n_candidates": float(len(usable)),
        "n_generalizing": float(len(gens)),
        "n_memorizing": float(len(mems)),
        "best_gen_objective": best_gen_f,
        "best_gen_test_accuracy": best_gen.test_accuracy if best_gen else float("nan"),
        "best_gen_weight_norm": best_gen.weight_norm if best_gen else float("nan"),
        "best_mem_objective": best_mem_f,
        "best_mem_test_accuracy": best_mem.test_accuracy if best_mem else float("nan"),
        "best_mem_weight_norm": best_mem.weight_norm if best_mem else float("nan"),
        "global_objective": global_cand.objective if global_cand else float("inf"),
    }
    global_generalizes = bool(
        global_cand is not None
        and global_cand.classify(gate.train_threshold, gate.test_threshold)
        == "generalizing"
    )
    return verdict, global_generalizes, numbers


def gate_problem(
    spec: ProblemSpec,
    instance: int = 1,
    settings: AdamSettings = AdamSettings(),
    gate: GateSettings = GateSettings(),
    return_candidates: bool = False,
):
    """Run the full global-minimum gate on one ``(spec, instance)`` pair."""
    candidates = collect_candidates(spec, instance, settings, gate)
    verdict, global_generalizes, numbers = assess(candidates, gate)
    report = GlobalMinReport(
        problem=spec.name,
        problem_id=spec.stable_id,
        instance=instance,
        n_variables=spec.n_variables,
        p=spec.p,
        rank=spec.rank,
        reg_strength=spec.reg_strength,
        train_frac=spec.train_frac,
        activation=spec.activation,
        n_candidates=int(numbers["n_candidates"]),
        n_generalizing=int(numbers["n_generalizing"]),
        n_memorizing=int(numbers["n_memorizing"]),
        best_gen_objective=numbers["best_gen_objective"],
        best_gen_test_accuracy=numbers["best_gen_test_accuracy"],
        best_gen_weight_norm=numbers["best_gen_weight_norm"],
        best_mem_objective=numbers["best_mem_objective"],
        best_mem_test_accuracy=numbers["best_mem_test_accuracy"],
        best_mem_weight_norm=numbers["best_mem_weight_norm"],
        global_objective=numbers["global_objective"],
        global_generalizes=global_generalizes,
        margin=gate.margin,
        verdict=verdict,
        keep=verdict == "generalizing_global",
    )
    return (report, candidates) if return_candidates else report


def _gate_worker(payload):
    """Process-pool entry point; rebuilds the spec from its dict."""
    spec_dict, instance, settings, gate = payload
    spec = ProblemSpec.from_dict(spec_dict)
    return gate_problem(spec, instance, settings, gate).as_row()


def gate_specs(
    specs: Sequence[ProblemSpec],
    spec_instances: dict[str, Sequence[int]] | None = None,
    instances: Sequence[int] = (1,),
    settings: AdamSettings = AdamSettings(),
    gate: GateSettings = GateSettings(),
    jobs: int = 1,
    verbose: bool = True,
):
    """Run the gate over a family; return a ``pandas.DataFrame`` of reports.

    ``spec_instances`` (as carried by a curated ``GrokkingSuite``) selects which
    instances to gate per spec; anything unlisted falls back to ``instances``.
    """
    import pandas as pd

    spec_instances = spec_instances or {}
    payloads = [
        (spec.to_dict(), int(instance), settings, gate)
        for spec in specs
        for instance in spec_instances.get(spec.name, instances)
    ]
    rows = []
    for done, row in enumerate(map_unordered(_gate_worker, payloads, jobs), start=1):
        rows.append(row)
        if verbose:
            tag = "KEEP" if row["keep"] else "drop"
            print(
                f"[{done:>4}/{len(payloads)}] {row['problem']} i{row['instance']} "
                f"-> {row['verdict']:>19} ({tag})  "
                f"gen_f={row['best_gen_objective']:.4g} mem_f={row['best_mem_objective']:.4g}",
                flush=True,
            )
    return pd.DataFrame(rows)


def filter_by_globalmin(suite, reports):
    """Keep only ``(spec, instance)`` pairs whose global minimum generalizes.

    Returns ``(kept_specs, spec_instances, summary)``.  A spec survives if at
    least one of its gated instances is kept; only the kept instances are
    retained.  ``summary`` is one row per spec, with the modal verdict so the
    reason a problem was dropped is legible at a glance.
    """
    import pandas as pd

    if len(reports) == 0:
        raise ValueError("No gate reports to filter on.")

    kept_rows = reports[reports["keep"]]
    spec_instances: dict[str, list[int]] = {}
    for name, grp in kept_rows.groupby("problem", sort=False):
        spec_instances[name] = sorted(int(i) for i in grp["instance"])

    keep_names = set(spec_instances)
    kept = [s for s in suite.specs if s.name in keep_names]

    grouped = reports.groupby("problem", sort=False)
    summary = grouped.agg(
        n_variables=("n_variables", "first"),
        p=("p", "first"),
        rank=("rank", "first"),
        reg_strength=("reg_strength", "first"),
        train_frac=("train_frac", "first"),
        instances=("instance", "count"),
        kept_instances=("keep", "sum"),
        keep_fraction=("keep", "mean"),
        median_best_gen_objective=("best_gen_objective", "median"),
        median_best_mem_objective=("best_mem_objective", "median"),
    ).reset_index()
    modal = grouped["verdict"].agg(lambda s: s.value_counts().idxmax())
    summary = summary.merge(modal.rename("modal_verdict"), on="problem")
    summary["kept"] = summary["problem"].isin(keep_names)
    summary["kept_instance_ids"] = summary["problem"].map(
        lambda n: ",".join(str(i) for i in spec_instances.get(n, []))
    )
    return kept, spec_instances, summary
