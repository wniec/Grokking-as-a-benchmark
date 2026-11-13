import numpy as np
import pytest

from gaab.optimizers import (
    GRADIENT_BASED,
    OPTIMIZERS,
    OPTIONAL_OPTIMIZERS,
    get_optimizer,
    initial_point,
)
from gaab.problem import GrokkingProblem, ProblemSpec

SPEC = ProblemSpec(p=7, embed_dim=4, hidden_dim=8, reg_strength=0.5, activation="quad")
# Every non-gradient method that has no optional dependency, so it can run here.
BLACK_BOX = sorted(set(OPTIMIZERS) - OPTIONAL_OPTIMIZERS - GRADIENT_BASED)


@pytest.mark.parametrize("name", BLACK_BOX)
def test_optimizer_respects_the_budget(name):
    problem = GrokkingProblem(SPEC)
    budget = 250
    get_optimizer(name)(problem, budget, np.random.default_rng(0))
    assert problem.state.evaluations == budget


@pytest.mark.parametrize("name", BLACK_BOX)
def test_optimizer_returns_a_finite_in_bounds_point(name):
    problem = GrokkingProblem(SPEC)
    x = get_optimizer(name)(problem, 250, np.random.default_rng(1))
    assert x.shape == (problem.n_variables,)
    assert np.all(np.isfinite(x))
    assert np.all(x >= problem.bounds.lb - 1e-9)
    assert np.all(x <= problem.bounds.ub + 1e-9)


@pytest.mark.parametrize("name", BLACK_BOX)
def test_optimizer_is_reproducible_from_its_seed(name):
    results = []
    for _ in range(2):
        problem = GrokkingProblem(SPEC)
        get_optimizer(name)(problem, 200, np.random.default_rng(7))
        results.append(problem.state.current_best.y)
    assert results[0] == pytest.approx(results[1])


@pytest.mark.parametrize("name", sorted(set(BLACK_BOX) - {"random_search"}))
def test_optimizer_improves_on_its_starting_point(name):
    problem = GrokkingProblem(SPEC)
    start = float(problem.objective(initial_point(problem, np.random.default_rng(0))))
    get_optimizer(name)(problem, 1500, np.random.default_rng(0))
    assert problem.state.current_best.y < start


def test_gradient_reference_beats_the_black_box_methods_on_the_objective():
    def best(name, budget=1500):
        problem = GrokkingProblem(SPEC)
        get_optimizer(name)(problem, budget, np.random.default_rng(0))
        return problem.state.current_best.y

    assert best("adam") < min(best(n) for n in ("random_search", "pso", "de"))


def test_adam_needs_a_gradient_capable_problem():
    class Opaque:
        pass

    with pytest.raises(TypeError):
        get_optimizer("adam")(Opaque(), 10, np.random.default_rng(0))


def test_gradient_based_registry_is_accurate():
    assert GRADIENT_BASED <= set(OPTIMIZERS)
    assert "adam" in GRADIENT_BASED
    assert "cma_es" not in GRADIENT_BASED


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError):
        get_optimizer("gradient_descent_but_better")


def test_initial_point_modes_stay_in_bounds():
    problem = GrokkingProblem(SPEC)
    for mode in ("init", "uniform", "zeros"):
        x = initial_point(problem, np.random.default_rng(0), mode)
        assert np.all(x >= problem.bounds.lb) and np.all(x <= problem.bounds.ub)


def test_default_benchmark_roster_drops_de_and_uniform_random_search():
    from gaab.optimizers import DEFAULT_BENCHMARK_OPTIMIZERS as roster

    assert "de" not in roster
    assert "random_search" not in roster  # the uniform floor is dropped
    assert "local_random_search" in roster  # the (1+1) climber stays
    assert {"adam", "lm_cma_es", "lm_ma_es", "sep_cma_es", "cma_es"} <= set(roster)
    assert set(roster) <= set(OPTIMIZERS)
