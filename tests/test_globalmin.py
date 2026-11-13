import numpy as np
import pandas as pd
import pytest

from gaab.globalmin import (
    GLOBAL_VERDICTS,
    Candidate,
    GateSettings,
    assess,
    collect_candidates,
    filter_by_globalmin,
    gate_problem,
)
from gaab.problem import GrokkingSuite, ProblemSpec
from gaab.screening import AdamSettings

FAST = AdamSettings(steps=300, eval_every=100)
TINY = GateSettings(n_restarts=3, n_memorizers=1)


def cand(source, objective, train, test, norm=1.0, finite=True):
    return Candidate(
        source=source,
        scale=1.0,
        objective=objective,
        train_accuracy=train,
        test_accuracy=test,
        weight_norm=norm,
        reg_penalty=0.0,
        finite=finite,
    )


def test_candidate_classify():
    assert cand("r", 1.0, 1.0, 1.0).classify(0.99, 0.95) == "generalizing"
    assert cand("r", 1.0, 1.0, 0.3).classify(0.99, 0.95) == "memorizing"
    assert cand("r", 1.0, 0.5, 0.1).classify(0.99, 0.95) == "unfit"
    assert cand("r", 1.0, 1.0, 1.0, finite=False).classify(0.99, 0.95) == "unfit"


def test_assess_keeps_when_generalizer_is_the_deepest():
    candidates = [
        cand("restart", objective=0.10, train=1.0, test=1.0),  # generalizer, lowest f
        cand("memorizer", objective=0.25, train=1.0, test=0.2),  # memorizer, higher f
    ]
    verdict, global_gen, nums = assess(candidates, GateSettings())
    assert verdict == "generalizing_global"
    assert global_gen is True
    assert nums["best_gen_objective"] == pytest.approx(0.10)
    assert nums["best_mem_objective"] == pytest.approx(0.25)


def test_assess_rejects_when_a_memorizer_sits_lower():
    candidates = [
        cand("restart", objective=0.30, train=1.0, test=1.0),  # generalizer
        cand("memorizer", objective=0.20, train=1.0, test=0.1),  # deeper memorizer
    ]
    verdict, global_gen, _ = assess(candidates, GateSettings())
    assert verdict == "memorizing_global"
    assert global_gen is False


def test_assess_rejects_when_nothing_generalizes():
    candidates = [cand("restart", 0.2, 1.0, 0.1), cand("memorizer", 0.3, 1.0, 0.2)]
    verdict, global_gen, _ = assess(candidates, GateSettings())
    assert verdict == "no_generalizer"
    assert global_gen is False


def test_assess_margin_tightens_the_bar():
    candidates = [
        cand("restart", objective=0.19, train=1.0, test=1.0),  # generalizer
        cand("memorizer", objective=0.20, train=1.0, test=0.1),  # barely higher
    ]
    # Zero margin keeps it (generalizer is strictly lower)...
    assert assess(candidates, GateSettings(margin=0.0))[0] == "generalizing_global"
    # ...but a margin wider than the gap rejects it.
    assert assess(candidates, GateSettings(margin=0.05))[0] == "memorizing_global"


def test_unfit_candidates_do_not_count_as_minima():
    candidates = [
        cand("restart", objective=0.01, train=0.5, test=0.0),  # unfit, lowest f
        cand("restart", objective=0.10, train=1.0, test=1.0),  # generalizer
    ]
    verdict, _, nums = assess(candidates, GateSettings())
    assert verdict == "generalizing_global"
    assert nums["n_generalizing"] == 1 and nums["n_memorizing"] == 0


def test_gate_problem_smoke():
    spec = ProblemSpec(
        p=7, embed_dim=4, hidden_dim=16, activation="quad", reg_strength=1e-3
    )
    report = gate_problem(spec, instance=1, settings=FAST, gate=TINY)
    assert report.verdict in GLOBAL_VERDICTS
    assert report.problem == spec.name
    assert report.keep == (report.verdict == "generalizing_global")
    assert report.n_candidates <= 1 + TINY.n_restarts + TINY.n_memorizers


def test_collect_candidates_tags_and_counts():
    spec = ProblemSpec(
        p=7, embed_dim=4, hidden_dim=16, activation="quad", reg_strength=1e-3
    )
    candidates = collect_candidates(spec, instance=1, settings=FAST, gate=TINY)
    assert len(candidates) == 1 + TINY.n_restarts + TINY.n_memorizers
    assert {c.source for c in candidates} <= {"reference", "restart", "memorizer"}
    assert sum(c.source == "reference" for c in candidates) == 1
    assert sum(c.source == "memorizer" for c in candidates) == TINY.n_memorizers


def test_filter_by_globalmin_keeps_only_passing_instances():
    specs = [
        ProblemSpec(p=7, embed_dim=4, hidden_dim=16),
        ProblemSpec(p=11, embed_dim=4, hidden_dim=16),
    ]
    suite = GrokkingSuite(specs, instances=[1, 2])
    rows = []
    # spec 0: instance 1 keeps, instance 2 drops; spec 1: both drop.
    outcomes = {
        (specs[0].name, 1): ("generalizing_global", True),
        (specs[0].name, 2): ("memorizing_global", False),
        (specs[1].name, 1): ("memorizing_global", False),
        (specs[1].name, 2): ("no_generalizer", False),
    }
    for spec in specs:
        for inst in (1, 2):
            verdict, keep = outcomes[(spec.name, inst)]
            rows.append(
                {
                    "problem": spec.name,
                    "problem_id": spec.stable_id,
                    "instance": inst,
                    "n_variables": spec.n_variables,
                    "p": spec.p,
                    "rank": spec.rank,
                    "reg_strength": spec.reg_strength,
                    "train_frac": spec.train_frac,
                    "best_gen_objective": 0.1,
                    "best_mem_objective": 0.2,
                    "verdict": verdict,
                    "keep": keep,
                }
            )
    reports = pd.DataFrame(rows)

    kept, spec_instances, summary = filter_by_globalmin(suite, reports)
    assert [s.name for s in kept] == [specs[0].name]
    assert spec_instances[specs[0].name] == [1]
    assert summary.loc[summary["problem"] == specs[0].name, "kept"].item()
    assert not summary.loc[summary["problem"] == specs[1].name, "kept"].item()


def test_filter_by_globalmin_needs_reports():
    suite = GrokkingSuite([ProblemSpec(p=7)], instances=[1])
    with pytest.raises(ValueError):
        filter_by_globalmin(suite, pd.DataFrame())
