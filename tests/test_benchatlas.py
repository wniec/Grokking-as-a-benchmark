import pickle

import numpy as np
import pytest

from gaab.benchatlas import (
    _build_objective,
    _GaabObjective,
    generalization_probe,
    suite,
    to_problem,
)
from gaab.problem import GrokkingSuite, ProblemSpec

SPEC = ProblemSpec(p=7, embed_dim=4, hidden_dim=16, activation="quad", reg_strength=1e-3)


def _rows(spec, n=5, seed=0):
    rng = np.random.default_rng(seed)
    d = spec.n_variables
    return rng.uniform(spec.lb, spec.ub, size=(n, d))


# --- core adapter: no bench_atlas needed -----------------------------------

def test_vectorized_objective_matches_scalar():
    fn = _GaabObjective(SPEC, instance=1)
    obj = _build_objective(SPEC, instance=1)
    X = _rows(SPEC)
    got = fn(X)
    assert got.shape == (len(X),)
    expected = np.array([obj(row) for row in X])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0)


def test_objective_is_picklable_and_stable():
    fn = _GaabObjective(SPEC, instance=2)
    X = _rows(SPEC, seed=1)
    before = fn(X)
    restored = pickle.loads(pickle.dumps(fn))
    # The rebuilt objective is dropped on pickling and rebuilt lazily.
    assert restored.__dict__.get("_obj") is None
    np.testing.assert_allclose(restored(X), before, rtol=1e-12, atol=0)


def test_generalization_probe_maps_unit_box_and_is_an_accuracy():
    probe = generalization_probe(SPEC, instance=1)
    obj = _build_objective(SPEC, instance=1)
    # u = 0 -> native lb, u = 1 -> native ub.
    d = SPEC.n_variables
    u = np.vstack([np.zeros(d), np.ones(d), np.full(d, 0.5)])
    acc = probe(u)
    assert acc.shape == (3,)
    assert np.all((acc >= 0.0) & (acc <= 1.0))
    expected_lb = obj.metrics(np.full(d, SPEC.lb)).test_accuracy
    assert acc[0] == pytest.approx(expected_lb)


def test_projected_spec_builds_at_projected_dimension():
    spec = ProblemSpec(p=7, embed_dim=4, hidden_dim=16, activation="quad",
                       reg_strength=1e-3, projection_dim=12)
    assert spec.n_variables == 12
    fn = _GaabObjective(spec, instance=1)
    X = _rows(spec, n=3)
    assert X.shape[1] == 12
    assert fn(X).shape == (3,)


# --- Problem-returning parts: need bench_atlas -----------------------------

def test_to_problem_round_trips_through_the_unit_box():
    ba = pytest.importorskip("bench_atlas.problems")
    problem = to_problem(SPEC, instance=1)
    assert problem.dimension == SPEC.n_variables
    assert problem.id.suite == "gaab"
    assert problem.id.function == SPEC.stable_id
    assert problem.f_opt is None

    # A unit point must evaluate to the native objective at its rescaled point.
    obj = _build_objective(SPEC, instance=1)
    u = np.random.default_rng(0).uniform(0, 1, size=(4, SPEC.n_variables))
    native = ba.from_unit(u, SPEC.lb, SPEC.ub)
    expected = np.array([obj(row) for row in native])
    np.testing.assert_allclose(problem(u), expected, rtol=1e-10, atol=0)
    assert problem.evaluations == 4


def test_suite_adapts_every_curated_instance():
    pytest.importorskip("bench_atlas.problems")
    gsuite = GrokkingSuite([SPEC, ProblemSpec(p=5, embed_dim=4, hidden_dim=16)],
                           instances=[1, 2])
    problems = suite(gsuite)
    assert len(problems) == 4
    assert {p.id.suite for p in problems} == {"gaab"}
    assert len({p.id for p in problems}) == 4
