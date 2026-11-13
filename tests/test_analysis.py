import math

import numpy as np
import pytest

from gaab import analysis as an
from gaab.problem import ImprovementRecord


def rec(evaluations, objective, train=0.0, test=0.0):
    return ImprovementRecord(
        evaluations=evaluations,
        objective=objective,
        train_loss=objective,
        train_accuracy=train,
        test_loss=objective,
        test_accuracy=test,
        reg_penalty=0.0,
        weight_norm=0.0,
    )


def test_segments_tile_the_whole_budget():
    segments = an.segments_from_history(
        [rec(3, 1.0), rec(7, 0.5)], budget=10, initial_objective=2.0
    )
    assert [(s.start, s.stop, s.objective) for s in segments] == [
        (1, 2, 2.0),
        (3, 6, 1.0),
        (7, 10, 0.5),
    ]
    assert sum(s.length for s in segments) == 10


def test_segments_ignore_records_past_the_budget():
    segments = an.segments_from_history(
        [rec(2, 1.0), rec(50, 0.1)], budget=10, initial_objective=2.0
    )
    assert segments[-1].objective == 1.0
    assert sum(s.length for s in segments) == 10


def test_segments_with_no_improvements_are_flat():
    segments = an.segments_from_history([], budget=5, initial_objective=2.0)
    assert len(segments) == 1 and segments[0].objective == 2.0


def test_curve_at_samples_the_step_function():
    segments = an.segments_from_history(
        [rec(3, 1.0), rec(7, 0.5)], budget=10, initial_objective=2.0
    )
    got = an.curve_at(segments, [1, 2, 3, 6, 7, 10], "objective")
    assert list(got) == [2.0, 2.0, 1.0, 1.0, 0.5, 0.5]


def test_aocc_is_one_when_the_target_is_hit_immediately():
    segments = an.segments_from_history(
        [rec(1, 1e-6)], budget=100, initial_objective=2.0
    )
    assert an.aocc_objective(segments, 100, hi=2.0, lo=1e-2) == pytest.approx(1.0)


def test_aocc_is_zero_when_nothing_improves():
    segments = an.segments_from_history([], budget=100, initial_objective=2.0)
    assert an.aocc_objective(segments, 100, hi=2.0, lo=1e-2) == pytest.approx(0.0)


def test_aocc_matches_a_hand_computed_value():
    # Half the budget at the ceiling, half at the floor.
    segments = an.segments_from_history(
        [rec(51, 1e-3)], budget=100, initial_objective=2.0
    )
    assert an.aocc_objective(segments, 100, hi=2.0, lo=1e-2) == pytest.approx(0.5)


def test_aocc_is_monotone_in_how_early_the_improvement_lands():
    early = an.segments_from_history([rec(10, 1e-3)], budget=100, initial_objective=2.0)
    late = an.segments_from_history([rec(90, 1e-3)], budget=100, initial_objective=2.0)
    assert an.aocc_objective(early, 100, hi=2.0) > an.aocc_objective(late, 100, hi=2.0)


def test_accuracy_aocc_is_the_mean_incumbent_accuracy():
    segments = an.segments_from_history(
        [rec(51, 0.5, test=1.0)], budget=100, initial_objective=2.0
    )
    assert an.aocc_accuracy(segments, 100) == pytest.approx(0.5)


def test_hitting_time_below_and_above():
    segments = an.segments_from_history(
        [rec(3, 1.0, test=0.4), rec(7, 0.5, test=0.9)], budget=10, initial_objective=2.0
    )
    assert an.hitting_time(segments, 0.6, "objective", "below") == 7
    assert an.hitting_time(segments, 0.3, "test_accuracy", "above") == 3
    assert an.hitting_time(segments, 0.99, "test_accuracy", "above") is None


def test_ert_uses_the_restart_estimator():
    # Two successes at 100 and 300, one failure charged the full 1000 budget.
    value, rate = an.ert([100, 300, None], budget=1000)
    assert value == pytest.approx((100 + 300 + 1000) / 2)
    assert rate == pytest.approx(2 / 3)


def test_ert_is_infinite_without_a_success():
    value, rate = an.ert([None, None], budget=1000)
    assert math.isinf(value) and rate == 0.0


def test_budget_grid_is_increasing_and_bounded():
    grid = an.budget_grid(1000, 40)
    assert grid[0] == 1 and grid[-1] <= 1000
    assert np.all(np.diff(grid) > 0)


def test_objective_targets_descend_from_hi_to_lo():
    targets = an.objective_targets(2.0, lo=1e-2, n=5)
    assert targets[0] == pytest.approx(2.0) and targets[-1] == pytest.approx(1e-2)
    assert np.all(np.diff(targets) < 0)


def test_aocc_rejects_an_inverted_range():
    segments = an.segments_from_history([], budget=10, initial_objective=1.0)
    with pytest.raises(ValueError):
        an.aocc_objective(segments, 10, hi=1e-3, lo=1.0)
