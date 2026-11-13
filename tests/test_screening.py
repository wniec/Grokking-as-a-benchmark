import numpy as np
import pytest

from gaab.problem import ProblemSpec
from gaab.screening import (
    AdamSettings,
    VERDICTS,
    adam_trajectory,
    classify,
    filter_specs,
    screen_problem,
    screen_specs,
)

FAST = AdamSettings(steps=400, eval_every=100)


def trace(pairs, steps):
    """Build a trace with the given (train_acc, test_acc) at each sample."""
    return [
        {
            "step": float(s),
            "objective": 1.0,
            "train_loss": 1.0,
            "train_accuracy": tr,
            "test_loss": 1.0,
            "test_accuracy": te,
            "reg_penalty": 0.0,
            "weight_norm": 1.0,
        }
        for s, (tr, te) in zip(range(100, steps + 1, 100), pairs)
    ]


def test_classify_grokking_needs_a_delay():
    settings = AdamSettings(steps=400)
    pairs = [(0.5, 0.1), (1.0, 0.1), (1.0, 0.3), (1.0, 1.0)]
    verdict, t_fit, t_gen, ratio = classify(trace(pairs, 400), settings)
    assert (verdict, t_fit, t_gen) == ("grokking", 200, 400)
    assert ratio == pytest.approx(2.0)


def test_classify_early_generalization_when_both_arrive_together():
    settings = AdamSettings(steps=400)
    pairs = [(0.5, 0.1), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
    verdict, _, _, _ = classify(trace(pairs, 400), settings)
    assert verdict == "early_generalization"


def test_classify_no_generalization():
    settings = AdamSettings(steps=400)
    pairs = [(1.0, 0.1)] * 4
    verdict, t_fit, t_gen, ratio = classify(trace(pairs, 400), settings)
    assert verdict == "no_generalization" and t_gen is None and np.isinf(ratio)


def test_classify_no_memorization():
    settings = AdamSettings(steps=400)
    verdict, _, _, _ = classify(trace([(0.2, 0.1)] * 4, 400), settings)
    assert verdict == "no_memorization"


def test_classify_divergence_on_a_truncated_trace():
    settings = AdamSettings(steps=400)
    assert classify(trace([(1.0, 1.0)], 100), settings)[0] == "divergence"
    assert classify([], settings)[0] == "divergence"


def test_early_factor_moves_the_boundary():
    settings = AdamSettings(steps=400)
    pairs = [(0.5, 0.1), (1.0, 0.1), (1.0, 1.0), (1.0, 1.0)]
    assert classify(trace(pairs, 400), settings, early_factor=1.5)[0] == "grokking"
    assert (
        classify(trace(pairs, 400), settings, early_factor=2.0)[0]
        == "early_generalization"
    )


def test_adam_trajectory_reduces_the_objective():
    from gaab.objective import build_objective

    obj = build_objective(p=7, embed_dim=4, hidden_dim=16, activation="quad", seed=0)
    theta0 = obj.initial_theta(0)
    theta, tr = adam_trajectory(obj, theta0, FAST)
    assert len(tr) >= 2
    assert obj(theta) < obj(theta0)
    assert not np.array_equal(theta, theta0)


def test_screen_problem_returns_a_known_verdict():
    spec = ProblemSpec(p=7, embed_dim=4, hidden_dim=16, activation="quad")
    report = screen_problem(spec, instance=1, settings=FAST)
    assert report.verdict in VERDICTS
    assert report.problem == spec.name and report.n_variables == spec.n_variables


def test_filter_keeps_only_grokking_and_records_the_instances():
    specs = [
        ProblemSpec(p=7, embed_dim=4, hidden_dim=16),
        ProblemSpec(p=11, embed_dim=4, hidden_dim=16),
    ]
    reports = screen_specs(specs, instances=(1, 2), settings=FAST, verbose=False)
    # Force a known outcome so the filter itself is what is under test.
    reports.loc[:, "verdict"] = [
        "grokking",
        "no_generalization",
        "no_generalization",
        "no_generalization",
    ]
    kept, spec_instances, summary = filter_specs(specs, reports, min_fraction=0.5)
    assert [s.name for s in kept] == [specs[0].name]
    assert spec_instances[specs[0].name] == [1]
    assert set(summary.columns) >= {"kept", "modal_verdict", "grokking_fraction"}
    assert not summary.loc[summary["problem"] == specs[1].name, "kept"].item()


def test_filter_can_keep_every_instance_of_a_survivor():
    specs = [ProblemSpec(p=7, embed_dim=4, hidden_dim=16)]
    reports = screen_specs(specs, instances=(1, 2), settings=FAST, verbose=False)
    reports.loc[:, "verdict"] = ["grokking", "no_generalization"]
    _, spec_instances, _ = filter_specs(
        specs, reports, min_fraction=0.5, instance_level=False
    )
    assert spec_instances[specs[0].name] == [1, 2]


def test_filter_needs_reports():
    import pandas as pd

    with pytest.raises(ValueError):
        filter_specs([ProblemSpec(p=7)], pd.DataFrame())
