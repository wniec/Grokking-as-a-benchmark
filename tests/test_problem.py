import ioh
import numpy as np
import pytest

from gaab.problem import GrokkingProblem, GrokkingSuite, ProblemSpec

SPEC = ProblemSpec(p=7, embed_dim=4, hidden_dim=8, reg_strength=0.1, activation="quad")


def test_problem_is_an_ioh_real_problem():
    problem = GrokkingProblem(SPEC, instance=2)
    assert isinstance(problem, ioh.problem.RealSingleObjective)
    assert problem.meta_data.n_variables == SPEC.n_variables
    assert problem.meta_data.instance == 2
    assert problem.meta_data.name == SPEC.name
    assert problem.meta_data.optimization_type == ioh.OptimizationType.MIN


def test_bounds_match_the_spec():
    problem = GrokkingProblem(ProblemSpec(p=7, lb=-1.5, ub=1.5))
    assert np.allclose(problem.bounds.lb, -1.5)
    assert np.allclose(problem.bounds.ub, 1.5)


def test_evaluation_counter_and_best_tracking():
    problem = GrokkingProblem(SPEC)
    rng = np.random.default_rng(0)
    values = [problem(list(rng.normal(0, 0.3, problem.n_variables))) for _ in range(25)]
    assert problem.state.evaluations == 25
    assert problem.state.current_best.y == pytest.approx(min(values))


def test_objective_matches_the_underlying_callable():
    problem = GrokkingProblem(SPEC)
    x = problem.initial_theta()
    assert problem(list(x)) == pytest.approx(problem.objective(x))


def test_history_records_only_improvements():
    problem = GrokkingProblem(SPEC)
    rng = np.random.default_rng(1)
    for _ in range(60):
        problem(list(rng.normal(0, 0.3, problem.n_variables)))
    values = [r.objective for r in problem.history]
    assert values == sorted(values, reverse=True)
    assert len(values) == len(set(values))
    assert problem.history[-1].objective == pytest.approx(problem.state.current_best.y)


def test_history_evaluation_indices_are_one_based_and_in_range():
    problem = GrokkingProblem(SPEC)
    rng = np.random.default_rng(2)
    for _ in range(30):
        problem(list(rng.normal(0, 0.3, problem.n_variables)))
    assert problem.history[0].evaluations == 1
    assert all(1 <= r.evaluations <= 30 for r in problem.history)


def test_watched_properties_track_the_incumbent():
    problem = GrokkingProblem(SPEC)
    assert np.isnan(problem.test_accuracy)
    problem(list(problem.initial_theta()))
    for name in GrokkingProblem.WATCHED:
        assert np.isfinite(getattr(problem, name))
    assert 0.0 <= problem.test_accuracy <= 1.0


def test_reset_clears_state_and_history():
    problem = GrokkingProblem(SPEC)
    problem(list(problem.initial_theta()))
    problem.reset()
    assert problem.state.evaluations == 0
    assert problem.history == []
    assert np.isnan(problem.test_accuracy)


def test_logger_round_trip(tmp_path):
    problem = GrokkingProblem(SPEC)
    logger = ioh.logger.Analyzer(
        root=str(tmp_path), folder_name="run", algorithm_name="test"
    )
    problem.attach_logger(logger)
    logger.watch(problem, list(GrokkingProblem.WATCHED))
    rng = np.random.default_rng(0)
    for _ in range(20):
        problem(list(rng.normal(0, 0.3, problem.n_variables)))
    problem.detach_logger()
    logger.close()
    written = list((tmp_path / "run").rglob("*"))
    assert any(path.suffix == ".json" for path in written)


def test_spec_rejects_composite_p():
    with pytest.raises(ValueError):
        ProblemSpec(p=15)


def test_spec_name_and_id_are_stable():
    a, b = ProblemSpec(p=11, rank=2), ProblemSpec(p=11, rank=2)
    assert a.name == b.name and a.stable_id == b.stable_id
    assert ProblemSpec(p=11, rank=4).stable_id != a.stable_id


def test_spec_dict_round_trip():
    spec = ProblemSpec(p=13, embed_dim=4, rank=3, reg_strength=1.5, train_frac=0.7)
    assert ProblemSpec.from_dict(spec.to_dict()) == spec


def test_suite_iteration_and_per_spec_instances():
    specs = [ProblemSpec(p=7), ProblemSpec(p=11)]
    suite = GrokkingSuite(specs, instances=[1, 2])
    assert len(suite) == 4
    suite.spec_instances = {specs[1].name: [3]}
    assert [(p.spec.p, p.instance_id) for p in suite] == [(7, 1), (7, 2), (11, 3)]
    assert len(suite) == 3


def test_suite_save_load_round_trip(tmp_path):
    specs = [ProblemSpec(p=7), ProblemSpec(p=11)]
    suite = GrokkingSuite(specs, instances=[1, 2], spec_instances={specs[0].name: [2]})
    path = tmp_path / "suite.json"
    suite.save(str(path))
    loaded = GrokkingSuite.load(str(path))
    assert [s.name for s in loaded.specs] == [s.name for s in suite.specs]
    assert loaded.instances_for(loaded.specs[0]) == [2]
    assert loaded.instances_for(loaded.specs[1]) == [1, 2]


def test_instances_change_the_split_but_not_the_dimension():
    a, b = GrokkingProblem(SPEC, instance=1), GrokkingProblem(SPEC, instance=2)
    assert a.n_variables == b.n_variables
    assert not np.array_equal(a.objective.data.train_a, b.objective.data.train_a)


def test_incumbent_watcher_forwards_to_the_live_problem():
    from gaab.problem import IncumbentWatcher

    watcher = IncumbentWatcher()
    assert np.isnan(watcher.test_accuracy)

    first = GrokkingProblem(SPEC)
    first(list(first.initial_theta()))
    watcher.problem = first
    assert watcher.train_loss == pytest.approx(first.train_loss)

    # Repointing is what a shared logger needs; a second `watch` would not.
    second = GrokkingProblem(ProblemSpec(p=11, embed_dim=4, hidden_dim=8))
    second(list(second.initial_theta()))
    watcher.problem = second
    assert watcher.train_loss == pytest.approx(second.train_loss)
    assert watcher.train_loss != pytest.approx(first.train_loss)


def test_incumbent_watcher_exposes_every_watched_property():
    from gaab.problem import IncumbentWatcher

    watcher = IncumbentWatcher()
    for name in GrokkingProblem.WATCHED:
        assert isinstance(getattr(IncumbentWatcher, name), property)
        assert np.isnan(getattr(watcher, name))
