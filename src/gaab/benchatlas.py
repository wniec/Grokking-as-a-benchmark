"""Expose GaaB problems as ``bench-atlas`` problems.

`bench-atlas <https://github.com/helix-agh/bench-atlas>`_ measures continuous
black-box benchmarks -- ELA, Local Optima Networks, an optimizer portfolio with
normalized AOCC, evaluation cost, diversity -- over suites it wraps from IOH
(BBOB, MA-BBOB, GNBG).  Rather than reimplement those analyses here, this module
turns a :class:`~gaab.problem.ProblemSpec` into a ``bench_atlas.problems.Problem``
so the grokking suite plugs in next to ``bbob`` and ``gnbg`` and is measured in
the *same* space as the reference benchmarks.

Two domain details are bridged here:

* **Box.**  A ``bench_atlas`` problem lives on ``[0, 1]^d``; a GaaB problem lives
  on ``[spec.lb, spec.ub]^n``.  ``from_function`` takes the native box and does
  the affine rescale itself, so we hand it the objective on native coordinates.
* **Pickling.**  ``bench_atlas`` sends problems to parallel workers, and a
  :class:`~gaab.problem.GrokkingProblem` subclasses an IOH problem, which cannot
  be pickled.  So the callables here carry only the (frozen, picklable)
  ``ProblemSpec`` and the instance, and rebuild the plain-NumPy objective lazily
  in each worker -- the same pattern ``bench_atlas`` uses for IOH problems.

There is no known analytic optimum, so ``x_opt`` / ``f_opt`` are left unset;
``bench_atlas``'s AOCC falls back to the best value any portfolio run finds.

**Next step -- generalization-AOCC.**  ``bench_atlas`` scores a run by area over
the *objective* convergence curve.  For grokking the question is whether the
optimizer actually generalizes, not just lowers ``f``.  :func:`generalization_probe`
returns a callable that maps stored run points (which ``bench_atlas`` keeps on
``[0, 1]^d``) to held-out **test accuracy**, the primitive a test-accuracy AOCC
is built from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from gaab.data import make_dataset
from gaab.objective import GrokkingObjective
from gaab.problem import GrokkingSuite, ProblemSpec

if TYPE_CHECKING:  # bench_atlas is an optional dependency, imported lazily below.
    from bench_atlas.problems import Problem

__all__ = ["to_problem", "suite", "generalization_probe"]


def _build_objective(spec: ProblemSpec, instance: int):
    """Rebuild a GaaB objective from its spec, without the IOH wrapper.

    Mirrors :meth:`GrokkingProblem.__init__` (dataset seeded by the instance,
    optional random-subspace projection) but returns the plain-NumPy objective,
    which is picklable and all these analyses need.
    """
    data = make_dataset(
        p=spec.p, op=spec.op, train_frac=spec.train_frac, seed=int(instance)
    )
    objective: Any = GrokkingObjective(
        spec.model_config, data, reg_strength=spec.reg_strength
    )
    if spec.projection_dim is not None:
        from gaab.projection import ProjectedObjective

        objective = ProjectedObjective(
            objective, dim=spec.projection_dim, seed=int(instance)
        )
    return objective


class _SpecCallable:
    """Base for picklable callables that lazily rebuild the objective per worker."""

    def __init__(self, spec: ProblemSpec, instance: int) -> None:
        self.spec = spec
        self.instance = int(instance)
        self._obj: Any = None

    def _ensure(self) -> Any:
        if self._obj is None:
            self._obj = _build_objective(self.spec, self.instance)
        return self._obj

    # Drop the rebuilt objective when pickling; workers rebuild it on first use.
    def __getstate__(self) -> dict[str, Any]:
        return {"spec": self.spec, "instance": self.instance}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.spec = state["spec"]
        self.instance = state["instance"]
        self._obj = None


class _GaabObjective(_SpecCallable):
    """Vectorized GaaB objective on native coordinates: ``(n, d) -> (n,)``."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        obj = self._ensure()
        x = np.asarray(x, dtype=float)
        return np.array([float(obj(row)) for row in x], dtype=float)


class _GaabTestAccuracy(_SpecCallable):
    """Held-out test accuracy of unit-box points: ``(n, d) on [0,1]^d -> (n,)``.

    Takes points on ``[0, 1]^d`` (the domain ``bench_atlas`` stores runs in) and
    maps them back to native coordinates before reading test accuracy, so it can
    be applied directly to a stored best-so-far curve.
    """

    def __init__(self, spec: ProblemSpec, instance: int) -> None:
        super().__init__(spec, instance)
        self.lb = float(spec.lb)
        self.ub = float(spec.ub)

    def __call__(self, u: np.ndarray) -> np.ndarray:
        obj = self._ensure()
        x = self.lb + np.asarray(u, dtype=float) * (self.ub - self.lb)
        return np.array([float(obj.metrics(row).test_accuracy) for row in x], dtype=float)


def to_problem(
    spec: ProblemSpec, instance: int = 1, suite_name: str = "gaab"
) -> "Problem":
    """Adapt one ``(spec, instance)`` pair to a ``bench_atlas`` problem.

    The identifier is ``(suite_name, spec.stable_id, instance, n_variables)``.
    The search dimension is ``spec.n_variables`` (already the projected dimension
    when ``spec.projection_dim`` is set), and the native box ``[spec.lb, spec.ub]``
    is rescaled to ``[0, 1]^d`` by ``from_function``.
    """
    from bench_atlas.problems import ProblemId, from_function

    dimension = int(spec.n_variables)
    problem_id = ProblemId(
        suite=suite_name,
        function=int(spec.stable_id),
        instance=int(instance),
        dimension=dimension,
    )
    return from_function(
        _GaabObjective(spec, instance),
        id=problem_id,
        lb=spec.lb,
        ub=spec.ub,
        name=f"{spec.name}_i{int(instance)}",
    )


def suite(gsuite: GrokkingSuite, suite_name: str = "gaab") -> list["Problem"]:
    """Adapt a whole :class:`~gaab.problem.GrokkingSuite` to a problem list.

    Honors per-spec curated instances (:meth:`GrokkingSuite.instances_for`), so
    the result matches exactly what the GaaB suite benchmarks -- the shape
    ``bench_atlas.suites`` expects (a flat ``list[Problem]``).
    """
    return [
        to_problem(spec, instance, suite_name)
        for spec in gsuite.specs
        for instance in gsuite.instances_for(spec)
    ]


def generalization_probe(spec: ProblemSpec, instance: int = 1) -> _GaabTestAccuracy:
    """A picklable callable ``[0,1]^d -> test accuracy`` for generalization-AOCC.

    ``bench_atlas`` scores objective convergence; grokking also needs the
    generalization curve.  Apply this to a stored run's best-so-far points (kept
    on ``[0, 1]^d``) to get the test-accuracy trajectory an area-over-the-curve
    generalization score is computed from.
    """
    return _GaabTestAccuracy(spec, instance)
