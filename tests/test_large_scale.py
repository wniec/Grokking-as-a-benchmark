"""Behavioural tests for the limited-memory CMA variants.

These do not check formulas -- they check the property that justifies the
methods existing: a limited memory of search directions learns non-separable
structure that a diagonal covariance cannot, at a cost that stays well under
full CMA-ES's O(n^2) per sample.
"""

import numpy as np
import ioh
import pytest

from gaab.optimizers import LARGE_SCALE, OPTIMIZERS, OPTIONAL_OPTIMIZERS, get_optimizer

# The large-scale methods that are always available (no optional dependency).
LARGE_SCALE_SOLVERS = sorted(LARGE_SCALE - OPTIONAL_OPTIMIZERS)
# The CMA/MA-ES family among them -- these converge on the sphere within 100 n
# evaluations. Separable NES is a natural-gradient method that needs more, so it
# gets its own convergence test below.
CMA_FAMILY = sorted(set(LARGE_SCALE_SOLVERS) - {"sep_nes"})


def quadratic(n, condition=1e6, seed=0, rotate=True):
    """``sum_i c_i z_i^2`` with ``z = R x`` -- ill-conditioned, optionally rotated.

    The box is opened up because these are classic unbounded test functions;
    the suite's own problems keep their box.
    """
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.standard_normal((n, n)))[0] if rotate else np.eye(n)
    scale = condition ** (np.arange(n) / max(n - 1, 1))

    class Quadratic(ioh.problem.RealSingleObjective):
        def __init__(self):
            super().__init__("quadratic", n, 1, True, ioh.RealBounds(n, -1e12, 1e12))

        def evaluate(self, x):
            z = R @ np.asarray(x, dtype=float)
            return float(scale @ (z * z))

    return Quadratic()


def solve(name, problem, budget, seed, sigma0=2.0):
    rng = np.random.default_rng(seed)
    n = problem.meta_data.n_variables
    get_optimizer(name)(problem, budget, rng, x0=rng.uniform(-4, 4, n), sigma0=sigma0)
    return problem.state.current_best.y


def median_over_seeds(name, n, budget, seeds=(0, 1, 2), **kw):
    return float(
        np.median([solve(name, quadratic(n, seed=s, **kw), budget, s) for s in seeds])
    )


@pytest.mark.parametrize("name", CMA_FAMILY)
def test_cma_family_solves_the_sphere_to_high_precision(name):
    value = median_over_seeds(name, 64, 6400, condition=1.0, rotate=False)
    assert value < 1e-6


def test_separable_nes_converges_on_the_sphere():
    # NES follows the natural gradient and needs more evaluations than CMA to
    # drive the sphere down, but it does converge.
    value = median_over_seeds("sep_nes", 64, 20_000, condition=1.0, rotate=False)
    assert value < 1e-2


def test_separable_nes_beats_full_cma_on_a_separable_problem():
    # A per-coordinate method is exactly right on a separable landscape -- the
    # flip side of failing on the rotated one.
    nes = median_over_seeds("sep_nes", 64, 12_000, rotate=False)
    full = median_over_seeds("cma_es", 64, 12_000, rotate=False)
    assert nes < full


@pytest.mark.parametrize("name", ["lm_cma_es", "lm_ma_es"])
def test_limited_memory_beats_diagonal_on_a_rotated_problem(name):
    """The reason these exist: a diagonal covariance cannot follow a rotation."""
    limited = median_over_seeds(name, 64, 12_000)
    diagonal = median_over_seeds("sep_cma_es", 64, 12_000)
    assert limited < diagonal


@pytest.mark.parametrize("name", LARGE_SCALE_SOLVERS)
def test_registered_and_budget_respecting(name):
    assert name in OPTIMIZERS
    problem = quadratic(32)
    solve(name, problem, 500, seed=0)
    assert problem.state.evaluations == 500


@pytest.mark.parametrize("name", ["lm_cma_es", "lm_ma_es"])
def test_memory_size_does_not_change_the_contract(name):
    problem = quadratic(32)
    rng = np.random.default_rng(0)
    x = get_optimizer(name)(problem, 600, rng, x0=np.ones(32), sigma0=1.0, memory=3)
    assert x.shape == (32,) and np.all(np.isfinite(x))
    assert problem.state.evaluations == 600


def test_large_scale_registry_is_accurate():
    assert LARGE_SCALE <= set(OPTIMIZERS)
    assert {"lm_cma_es", "lm_ma_es", "sep_cma_es", "sep_nes"} <= LARGE_SCALE
    assert "cma_es" not in LARGE_SCALE  # full O(n^2) covariance is not large-scale


def test_optional_large_scale_methods_are_flagged():
    # The Nevergrad-backed methods are large-scale but optional; a caller can
    # tell they need a dependency before trying to run them.
    from gaab.optimizers import NEVERGRAD_METHODS, optional_dependency

    assert set(NEVERGRAD_METHODS) <= (LARGE_SCALE & OPTIONAL_OPTIMIZERS)
    for name in NEVERGRAD_METHODS:
        assert optional_dependency(name) == "nevergrad"
    assert optional_dependency("sep_nes") is None  # always available


def test_cholesky_memory_inverts_its_own_transform():
    """``Ainv(A z) == z`` is what makes the stored directions meaningful."""
    from gaab.optimizers import _CholeskyMemory

    rng = np.random.default_rng(0)
    n = 24
    memory = _CholeskyMemory(n, memory=4, c_1=0.05)
    for generation in range(6):
        memory.update(rng.standard_normal(n), generation, n_steps=1)
    assert len(memory.vectors) == 4  # capped at the memory size

    z = rng.standard_normal(n)
    assert np.allclose(memory.inverse_transform(memory.transform(z)), z, atol=1e-8)


def test_cholesky_memory_evicts_the_most_redundant_direction():
    from gaab.optimizers import _CholeskyMemory

    rng = np.random.default_rng(1)
    memory = _CholeskyMemory(8, memory=3, c_1=0.05)
    for generation in (0, 10, 11):  # 10 and 11 are near-duplicates in time
        memory.update(rng.standard_normal(8), generation, n_steps=5)
    memory.update(rng.standard_normal(8), 40, n_steps=5)
    # The tight pair loses its older member; the well-separated 0 survives.
    assert memory.saved_at == [0, 11, 40]


def test_cholesky_memory_ignores_a_degenerate_direction():
    from gaab.optimizers import _CholeskyMemory

    memory = _CholeskyMemory(8, memory=3, c_1=0.05)
    memory.update(np.zeros(8), 0, n_steps=1)
    assert memory.vectors == []
