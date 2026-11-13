"""Exploratory Landscape Analysis (ELA) features for GaaB problems.

ELA (Mersmann et al. 2011) describes a continuous optimization problem by a
vector of cheap, sample-based numerical features -- measures of modality,
curvature, the distribution of objective values, nearest-better clustering,
dispersion and information content -- computed from a space-filling design of
``(x, f(x))`` pairs.  It is the descriptor space the optimization-benchmarking
community uses to compare problem *families* (most prominently BBOB) and to
train algorithm selectors.

Putting GaaB problems in this same space lets the paper make a quantitative
claim it otherwise can only assert: *where* the grokking problems sit relative
to BBOB, and whether they occupy a region the classical benchmark does not
cover.  The rigorous, exact-gradient diagnostics in :mod:`gaab.landscape` answer
"what is the landscape really like"; ELA answers "how does it look to the
sample-based tools everyone else already uses", so the two are complementary.

``pflacco`` is an optional dependency (it pins older numpy/pandas); import this
module only when you actually want ELA features::

    from gaab.ela import ela_features
    from gaab.problem import GrokkingProblem
    from gaab.suite import load_suite

    problem = GrokkingProblem(load_suite().specs[0], instance=1)
    feats = ela_features(problem)          # dict of ~40 named features
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

# Default space-filling design size, as a multiple of the problem dimension.
# 50*d is the common rule of thumb for ELA; the information-content and
# nearest-better-clustering cells are O(n^2 d), so this is the main cost lever.
DEFAULT_N_PER_DIM = 50

# Feature set name -> (pflacco calculator, needs a seed argument).  These are
# the standard low-cost cells that do not require extra function evaluations
# beyond the initial design (so no cell-mapping / local-search features here).
_FEATURE_SETS: dict[str, tuple[str, bool]] = {
    "meta": ("calculate_ela_meta", False),
    "distribution": ("calculate_ela_distribution", False),
    "pca": ("calculate_pca", False),
    "nbc": ("calculate_nbc", False),
    "dispersion": ("calculate_dispersion", False),
    "information_content": ("calculate_information_content", True),
}

# ``meta`` (the linear/quadratic model-fit features) is deliberately NOT in the
# defaults: it fits a full quadratic surrogate with O(d^2) interaction terms,
# which needs n >= d^2 samples to be well posed and is otherwise rank-deficient
# and very slow.  That is fine for BBOB's low dimensions but not for GaaB's
# d = 72..199 (it would need ~40k samples at the top end).  It is kept available
# for low-dimensional use -- e.g. ELA on a ``ProjectedObjective`` subspace.
DEFAULT_SETS: tuple[str, ...] = (
    "distribution",
    "pca",
    "nbc",
    "dispersion",
    "information_content",
)

ALL_SETS: tuple[str, ...] = tuple(_FEATURE_SETS)


def _require_pflacco() -> None:
    try:
        import pflacco  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "ELA features require the optional dependency 'pflacco' "
            "(`uv pip install pflacco`)."
        ) from exc


def _calculator(set_name: str) -> tuple[Callable[..., dict], bool]:
    func_name, needs_seed = _FEATURE_SETS[set_name]
    from pflacco import classical_ela_features as cef

    return getattr(cef, func_name), needs_seed


def problem_bounds(problem: Any) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper bounds for the sampling box, read from the ioh problem."""
    bounds = problem.bounds
    lb = np.asarray(bounds.lb, dtype=float)
    ub = np.asarray(bounds.ub, dtype=float)
    return lb, ub


def initial_design(
    problem: Any,
    n: int | None = None,
    *,
    n_per_dim: int = DEFAULT_N_PER_DIM,
    seed: int = 0,
    sample_type: str = "lhs",
) -> tuple["pd.DataFrame", np.ndarray]:
    """Sample a space-filling design in the problem's box and evaluate it.

    Returns ``(X, y)`` where ``X`` is a pandas DataFrame of shape ``(n, dim)``
    (the format pflacco expects) and ``y`` the objective values.
    """
    _require_pflacco()
    from pflacco.sampling import create_initial_sample

    dim = int(problem.meta_data.n_variables)
    if n is None:
        n = n_per_dim * dim
    lb, ub = problem_bounds(problem)
    # pflacco wants scalar bounds when the box is a hypercube, which ours is.
    lower = float(lb.min())
    upper = float(ub.max())
    X = create_initial_sample(
        dim,
        n=n,
        lower_bound=lower,
        upper_bound=upper,
        sample_type=sample_type,
        seed=seed,
    )
    y = np.fromiter((problem(row) for row in X.to_numpy()), dtype=float, count=len(X))
    return X, y


def ela_features(
    problem: Any,
    *,
    n: int | None = None,
    n_per_dim: int = DEFAULT_N_PER_DIM,
    seed: int = 0,
    sets: tuple[str, ...] = DEFAULT_SETS,
    sample_type: str = "lhs",
    include_runtime: bool = False,
) -> dict[str, float]:
    """Compute ELA features for one GrokkingProblem (or any ioh problem).

    ``sets`` selects which feature groups to compute (see ``DEFAULT_SETS``).
    A failing group is recorded as ``<group>.error`` rather than aborting the
    whole vector, so one unstable cell never costs the rest.
    """
    _require_pflacco()
    unknown = set(sets) - set(_FEATURE_SETS)
    if unknown:
        raise ValueError(f"unknown ELA feature set(s): {sorted(unknown)}")

    X, y = initial_design(
        problem,
        n=n,
        n_per_dim=n_per_dim,
        seed=seed,
        sample_type=sample_type,
    )
    feats: dict[str, float] = {}
    for name in sets:
        calc, needs_seed = _calculator(name)
        try:
            out = calc(X, y, seed=seed) if needs_seed else calc(X, y)
        except Exception as exc:  # noqa: BLE001 - record, don't abort the vector
            feats[f"{name}.error"] = 1.0
            feats[f"{name}.error_msg"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]
            continue
        for key, val in out.items():
            if key.endswith("costs_runtime") and not include_runtime:
                continue
            feats[key] = val
    return feats


def bbob_problem(fid: int, dim: int, iid: int = 1) -> Any:
    """A BBOB function as an ioh problem, for computing ELA in the same space."""
    import ioh

    return ioh.get_problem(
        fid, instance=iid, dimension=dim, problem_class=ioh.ProblemClass.BBOB
    )


def bbob_features(
    fid: int,
    dim: int,
    iid: int = 1,
    **kwargs: Any,
) -> dict[str, float]:
    """ELA features for a BBOB function, computed exactly as for GaaB problems."""
    return ela_features(bbob_problem(fid, dim, iid), **kwargs)
