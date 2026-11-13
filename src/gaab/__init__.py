"""GaaB -- Grokking as a Benchmark.

A suite of IOH-compatible continuous optimization problems whose search space
is the weights of a small neural network learning ``a + b = c (mod p)``.  What
makes them unusual as benchmark functions is that every problem carries a
second, hidden axis -- generalization to held-out pairs -- that no optimizer is
allowed to optimize, and that on these landscapes arrives long after the
training loss has been driven down.  Only problems where that delayed
transition is achievable are kept in the suite.

Quick start::

    from gaab import load_suite, get_optimizer
    import numpy as np

    suite = load_suite()
    problem = next(iter(suite))            # a real ioh.problem.RealSingleObjective
    get_optimizer("cma_es")(problem, 20_000, np.random.default_rng(0))
    print(problem.state.current_best.y, problem.test_accuracy)
"""

from gaab.data import ModularArithmeticData, PRIMES, is_prime, make_dataset
from gaab.model import GrokkingMLP, ModelConfig, n_parameters
from gaab.objective import GrokkingObjective, build_objective
from gaab.optimizers import OPTIMIZERS, get_optimizer
from gaab.projection import ProjectedObjective
from gaab.problem import (
    GrokkingProblem,
    GrokkingSuite,
    IncumbentWatcher,
    ProblemSpec,
)
from gaab.suite import candidate_specs, candidate_suite, load_suite

__version__ = "0.1.0"

__all__ = [
    "GrokkingMLP",
    "GrokkingObjective",
    "GrokkingProblem",
    "GrokkingSuite",
    "IncumbentWatcher",
    "ModelConfig",
    "ModularArithmeticData",
    "OPTIMIZERS",
    "PRIMES",
    "ProblemSpec",
    "ProjectedObjective",
    "build_objective",
    "candidate_specs",
    "candidate_suite",
    "get_optimizer",
    "is_prime",
    "load_suite",
    "make_dataset",
    "n_parameters",
    "__version__",
]
