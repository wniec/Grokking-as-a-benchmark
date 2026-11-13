"""Reference optimizers, all speaking the same IOH-problem interface.

Every optimizer has the signature::

    solver(problem, budget, rng, x0=None, **kwargs) -> np.ndarray

``problem`` is any callable IOH real problem (``GrokkingProblem`` in practice),
``budget`` is a hard cap on function evaluations, and the return value is the
best point found.  Nothing here reads the objective's internals, so a
third-party algorithm can be dropped in by registering it in ``OPTIMIZERS``.

The one exception is ``adam``, the gradient-based reference: it *does* use the
analytic gradients, and is included precisely so the plots show how much of the
gap to grokking is inherent to the landscape rather than to black-box search.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

Solver = Callable[..., np.ndarray]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def problem_bounds(problem) -> tuple[np.ndarray, np.ndarray]:
    lb = np.asarray(problem.bounds.lb, dtype=np.float64)
    ub = np.asarray(problem.bounds.ub, dtype=np.float64)
    return lb, ub


def initial_point(problem, rng: np.random.Generator, mode: str = "init") -> np.ndarray:
    """Where to start.

    ``init`` uses the network's own fan-in-scaled initialization, which is the
    honest starting point for a neural-network landscape: a uniform draw from
    the box starts every run at an absurd weight scale and measures rescaling
    ability rather than optimization.
    """
    lb, ub = problem_bounds(problem)
    n = problem.meta_data.n_variables
    if mode == "init" and hasattr(problem, "initial_theta"):
        return np.clip(problem.initial_theta(seed=int(rng.integers(1 << 30))), lb, ub)
    if mode == "zeros":
        return np.zeros(n)
    return rng.uniform(lb, ub, size=n)


def default_sigma0(x0: np.ndarray) -> float:
    """Initial step size for a step-size-adaptive method.

    The usual "a quarter of the domain width" rule is badly wrong here: the box
    is [-2, 2] but a network initialization has an RMS weight around 0.2, so a
    quarter-width step scatters every offspring into a region where the loss
    carries no signal and the whole budget goes on shrinking sigma back down.
    Scaling to the RMS of the starting point puts the first population inside
    the basin the initialization already sits in.
    """
    rms = float(np.sqrt(np.mean(np.asarray(x0, dtype=np.float64) ** 2)))
    return max(0.3 * rms, 1e-3)


class _BudgetedProblem:
    """Counts evaluations and stops the solver once the budget is spent."""

    class Exhausted(Exception):
        pass

    def __init__(self, problem, budget: int) -> None:
        self.problem = problem
        self.budget = int(budget)
        self.used = 0
        self.best_x: np.ndarray | None = None
        self.best_y = np.inf
        self.lb, self.ub = problem_bounds(problem)

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def __call__(self, x: np.ndarray) -> float:
        if self.used >= self.budget:
            raise self.Exhausted
        x = np.clip(np.asarray(x, dtype=np.float64), self.lb, self.ub)
        y = float(self.problem(x.tolist()))
        self.used += 1
        if y < self.best_y:
            self.best_y, self.best_x = y, x.copy()
        return y

    def batch(self, xs: np.ndarray) -> np.ndarray:
        return np.array([self(x) for x in xs])


def _finish(budgeted: _BudgetedProblem, fallback: np.ndarray) -> np.ndarray:
    return budgeted.best_x if budgeted.best_x is not None else fallback


# ---------------------------------------------------------------------------
# solvers
# ---------------------------------------------------------------------------
def random_search(
    problem, budget: int, rng: np.random.Generator, x0=None, **kw
) -> np.ndarray:
    """Uniform sampling in the box. The floor every other method must beat."""
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    x0 = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0)
    )
    try:
        f(x0)
        while f.remaining > 0:
            f(rng.uniform(f.lb, f.ub, size=n))
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, x0)


#: 1/5th-success rule: multiply sigma by ``c`` after a success and by
#: ``c ** -0.25`` after a failure, which settles at a 20% success rate.
_STEP_UP = 1.5**0.25
_STEP_DOWN = _STEP_UP**-0.25


def local_random_search(
    problem, budget: int, rng: np.random.Generator, x0=None, sigma0: float = 0.1, **kw
) -> np.ndarray:
    """(1+1) Gaussian hill climber with a 1/5th-success step-size rule."""
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    x = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float)
    )
    sigma = float(sigma0)
    try:
        y = f(x)
        while f.remaining > 0:
            cand = x + sigma * rng.normal(size=n)
            y_cand = f(cand)
            if y_cand <= y:
                x, y, sigma = cand, y_cand, sigma * _STEP_UP
            else:
                sigma *= _STEP_DOWN
            sigma = float(np.clip(sigma, 1e-12, 1e2))
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, x)


def _cma_weights(n: int, lam: int) -> tuple[int, np.ndarray, float]:
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w**2)
    return mu, w, float(mueff)


def cma_es(
    problem,
    budget: int,
    rng: np.random.Generator,
    x0=None,
    sigma0: float | None = None,
    popsize: int | None = None,
    **kw,
) -> np.ndarray:
    """(mu/mu_w, lambda)-CMA-ES with full covariance adaptation.

    A textbook implementation (Hansen, 2016) rather than a wrapper, so the
    benchmark has no optional dependency; ``pycma_es`` below defers to the
    reference ``cma`` package when it is installed.
    """
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    xmean = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float).copy()
    )
    sigma = float(sigma0 if sigma0 is not None else default_sigma0(xmean))

    lam = int(popsize or 4 + math.floor(3 * math.log(n)))
    mu, w, mueff = _cma_weights(n, lam)

    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    invsqrtC = np.eye(n)
    eigeneval = 0
    counteval = 0

    try:
        while f.remaining > 0:
            z = rng.standard_normal((lam, n))
            y = (z * D) @ B.T
            xs = xmean + sigma * y
            fitness = f.batch(xs)
            counteval += lam

            order = np.argsort(fitness)
            y_sel = y[order[:mu]]
            y_w = w @ y_sel
            xmean = xmean + sigma * y_w

            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y_w)
            hsig = float(
                np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * counteval / lam))
                < (1.4 + 2.0 / (n + 1)) * chiN
            )
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w

            rank_one = np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C
            rank_mu = (y_sel * w[:, None]).T @ y_sel
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = float(np.clip(sigma, 1e-16, 1e4))

            # Eigen-decomposition is O(n^3); amortize it as Hansen prescribes.
            if counteval - eigeneval > lam / (c1 + cmu) / n / 10.0:
                eigeneval = counteval
                C = np.triu(C) + np.triu(C, 1).T
                eigvals, B = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                D = np.sqrt(eigvals)
                invsqrtC = (B / D) @ B.T
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, xmean)


def sep_cma_es(
    problem,
    budget: int,
    rng: np.random.Generator,
    x0=None,
    sigma0: float | None = None,
    popsize: int | None = None,
    **kw,
) -> np.ndarray:
    """Separable CMA-ES (diagonal covariance): O(n) per sample.

    The right baseline at the top of this suite's dimension range, where full
    covariance adaptation cannot pay for itself within a realistic budget.
    """
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    xmean = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float).copy()
    )
    sigma = float(sigma0 if sigma0 is not None else default_sigma0(xmean))

    lam = int(popsize or 4 + math.floor(3 * math.log(n)))
    mu, w, mueff = _cma_weights(n, lam)

    cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
    cs = (mueff + 2.0) / (n + mueff + 5.0)
    # Sep-CMA-ES speeds the learning rates up by (n + 2) / 3 (Ros & Hansen, 2008).
    boost = (n + 2.0) / 3.0
    c1 = min(1.0, boost * 2.0 / ((n + 1.3) ** 2 + mueff))
    cmu = min(
        1.0 - c1,
        boost * 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff),
    )
    damps = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
    chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    pc = np.zeros(n)
    ps = np.zeros(n)
    d = np.ones(n)  # diagonal standard deviations
    counteval = 0

    try:
        while f.remaining > 0:
            z = rng.standard_normal((lam, n))
            y = z * d
            fitness = f.batch(xmean + sigma * y)
            counteval += lam

            order = np.argsort(fitness)
            y_sel = y[order[:mu]]
            y_w = w @ y_sel
            xmean = xmean + sigma * y_w

            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (y_w / d)
            hsig = float(
                np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * counteval / lam))
                < (1.4 + 2.0 / (n + 1)) * chiN
            )
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w

            c_diag = d**2
            c_diag = (
                (1 - c1 - cmu) * c_diag
                + c1 * (pc**2 + (1 - hsig) * cc * (2 - cc) * c_diag)
                + cmu * (w @ (y_sel**2))
            )
            d = np.sqrt(np.maximum(c_diag, 1e-20))

            sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = float(np.clip(sigma, 1e-16, 1e4))
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, xmean)


def pso(
    problem,
    budget: int,
    rng: np.random.Generator,
    x0=None,
    swarm_size: int | None = None,
    inertia: float = 0.729,
    cognitive: float = 1.49445,
    social: float = 1.49445,
    **kw,
) -> np.ndarray:
    """Standard constriction-coefficient particle swarm optimization."""
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    swarm = int(swarm_size or min(40, 10 + 2 * int(math.sqrt(n))))
    span = f.ub - f.lb

    centre = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float)
    )
    # Seed the swarm around a valid network initialization rather than
    # uniformly in the box, for the reason given in `initial_point`.
    scale = 0.5 * float(np.mean(np.abs(centre))) + 1e-3
    x = centre + rng.normal(0.0, scale, size=(swarm, n))
    x[0] = centre
    x = np.clip(x, f.lb, f.ub)
    v = rng.uniform(-0.1, 0.1, size=(swarm, n)) * span

    try:
        p_best = x.copy()
        p_val = f.batch(x)
        g_idx = int(np.argmin(p_val))
        g_best, g_val = p_best[g_idx].copy(), float(p_val[g_idx])

        while f.remaining > 0:
            r1 = rng.random((swarm, n))
            r2 = rng.random((swarm, n))
            v = inertia * v + cognitive * r1 * (p_best - x) + social * r2 * (g_best - x)
            v = np.clip(v, -0.5 * span, 0.5 * span)
            x = np.clip(x + v, f.lb, f.ub)

            vals = f.batch(x)
            improved = vals < p_val
            p_best[improved] = x[improved]
            p_val[improved] = vals[improved]
            g_idx = int(np.argmin(p_val))
            if p_val[g_idx] < g_val:
                g_best, g_val = p_best[g_idx].copy(), float(p_val[g_idx])
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, centre)


def differential_evolution(
    problem,
    budget: int,
    rng: np.random.Generator,
    x0=None,
    pop_size: int | None = None,
    F: float = 0.5,
    CR: float = 0.9,
    **kw,
) -> np.ndarray:
    """DE/rand/1/bin."""
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    npop = int(pop_size or min(50, 10 + 2 * int(math.sqrt(n))))

    centre = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float)
    )
    scale = 0.5 * float(np.mean(np.abs(centre))) + 1e-3
    pop = np.clip(centre + rng.normal(0.0, scale, size=(npop, n)), f.lb, f.ub)
    pop[0] = centre

    try:
        vals = f.batch(pop)
        while f.remaining > 0:
            for i in range(npop):
                idxs = rng.choice(npop, 3, replace=False)
                while i in idxs:
                    idxs = rng.choice(npop, 3, replace=False)
                a, b, c = pop[idxs]
                mutant = np.clip(a + F * (b - c), f.lb, f.ub)
                cross = rng.random(n) < CR
                cross[rng.integers(n)] = True
                trial = np.where(cross, mutant, pop[i])
                y = f(trial)
                if y <= vals[i]:
                    pop[i], vals[i] = trial, y
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, centre)


def adam(
    problem,
    budget: int,
    rng: np.random.Generator,
    x0=None,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    **kw,
) -> np.ndarray:
    """Gradient-based reference. NOT a black-box method -- reported separately.

    It spends one "evaluation" per step so its curve is directly comparable on
    the same x-axis, and it is the ceiling the black-box methods are measured
    against: on these landscapes it is the only method that reliably groks.
    """
    if not hasattr(problem, "objective"):
        raise TypeError("adam needs a GrokkingProblem (it uses analytic gradients).")
    f = _BudgetedProblem(problem, budget)
    x = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float).copy()
    )
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    try:
        for step in range(1, budget + 1):
            f(x)  # keeps IOH's counter and the improvement history honest
            _, g = problem.objective.loss_and_grad(x)
            if not np.all(np.isfinite(g)):
                break
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g * g
            x = x - lr * (m / (1 - beta1**step)) / (
                np.sqrt(v / (1 - beta2**step)) + eps
            )
            x = np.clip(x, f.lb, f.ub)
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, x)


# ---------------------------------------------------------------------------
# large-scale variants
#
# Full CMA-ES stores and eigendecomposes an n x n covariance matrix, which is
# what makes it the wrong tool at the top of this suite's dimension range.  The
# two methods below keep a *limited memory* of m ~ 4 + 3 ln n direction vectors
# instead, so an iteration costs O(m n lambda) rather than O(n^2 lambda), and
# unlike `sep_cma_es` they can still learn non-separable structure -- which a
# neural-network landscape is full of.
# ---------------------------------------------------------------------------
def _memory_size(n: int) -> int:
    return int(4 + math.floor(3 * math.log(n)))


def lm_ma_es(
    problem,
    budget: int,
    rng: np.random.Generator,
    x0=None,
    sigma0: float | None = None,
    popsize: int | None = None,
    memory: int | None = None,
    **kw,
) -> np.ndarray:
    """LM-MA-ES (Loshchilov, Glasmachers & Beyer, 2019).

    Limited-memory matrix adaptation: the sampling transform is a product of
    ``m`` rank-one corrections built from the same weighted step ``wz`` at
    ``m`` different learning rates, so the memory spans several time scales at
    once.  No covariance matrix is ever formed.
    """
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    xmean = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float).copy()
    )
    sigma = float(sigma0 if sigma0 is not None else default_sigma0(xmean))

    lam = int(popsize or 4 + math.floor(3 * math.log(n)))
    mu, w, mueff = _cma_weights(n, lam)
    m = int(memory or _memory_size(n))

    c_sigma = min(1.0, 2.0 * lam / n)
    # Geometrically spaced learning rates: M[0] tracks the newest direction,
    # M[m-1] averages over the longest horizon.
    c_d = np.array([1.0 / (1.5**i * n) for i in range(m)])
    c_c = np.clip(np.array([lam / (4.0**i * n) for i in range(m)]), 0.0, 1.0)

    M = np.zeros((m, n))
    p_sigma = np.zeros(n)
    generation = 0

    try:
        while f.remaining > 0:
            Z = rng.standard_normal((lam, n))
            D = Z.copy()
            for i in range(min(generation, m)):
                # D <- (1 - c_d) D + c_d (D M_i^T) M_i, one rank-one correction
                # applied to the whole population at once.
                D = (1.0 - c_d[i]) * D + c_d[i] * np.outer(D @ M[i], M[i])

            fitness = f.batch(xmean + sigma * D)
            order = np.argsort(fitness)[:mu]

            wz = w @ Z[order]
            xmean = xmean + sigma * (w @ D[order])

            p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(
                mueff * c_sigma * (2 - c_sigma)
            ) * wz
            for i in range(m):
                M[i] = (1 - c_c[i]) * M[i] + math.sqrt(
                    mueff * c_c[i] * (2 - c_c[i])
                ) * wz

            sigma *= math.exp(0.5 * c_sigma * (float(p_sigma @ p_sigma) / n - 1.0))
            sigma = float(np.clip(sigma, 1e-16, 1e4))
            generation += 1
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, xmean)


class _CholeskyMemory:
    """The limited-memory Cholesky factor ``A`` that LM-CMA samples through.

    ``A`` is held as a product of rank-one factors ``F_t = a I + beta_t v v^T``
    (``a = sqrt(1 - c_1)``), one per remembered search direction.  Choosing
    ``beta_t`` so that ``F_t F_t^T`` contributes exactly ``c_1 p p^T`` makes the
    product reproduce the rank-one covariance update ``C <- (1 - c_1) C +
    c_1 p p^T`` -- without ever storing ``C``.  Both ``A z`` and ``A^-1 x`` are
    then O(m n).
    """

    def __init__(self, n: int, memory: int, c_1: float) -> None:
        self.n = n
        self.memory = memory
        self.c_1 = c_1
        self.a = math.sqrt(1.0 - c_1)
        self.vectors: list[np.ndarray] = []
        self.beta: list[float] = []
        self.saved_at: list[int] = []

    def transform(self, z: np.ndarray) -> np.ndarray:
        """``A z``: apply the newest factor first, the oldest last."""
        x = z
        for v, b in zip(reversed(self.vectors), reversed(self.beta)):
            x = self.a * x + b * (x @ v) * v
        return x

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """``A^-1 x`` via Sherman-Morrison on each factor, oldest first."""
        z = x
        inv_a = 1.0 / self.a
        for v, b in zip(self.vectors, self.beta):
            delta = b / (self.a * (self.a + b * float(v @ v)))
            z = inv_a * z - delta * (z @ v) * v
        return z

    def _evict(self, n_steps: int) -> int:
        """Which slot to overwrite, keeping the remembered times well spread.

        Directions saved a similar number of iterations apart are largely
        redundant, so the pair with the smallest temporal gap loses its older
        member; if every gap is already wide, the oldest goes.
        """
        gaps = np.diff(self.saved_at)
        if len(gaps) and gaps.min() < n_steps:
            return int(np.argmin(gaps))
        return 0

    def update(self, p_c: np.ndarray, generation: int, n_steps: int) -> None:
        v = self.inverse_transform(p_c)
        norm2 = float(v @ v)
        if not np.isfinite(norm2) or norm2 <= 1e-300:
            return
        beta = (self.a / norm2) * (
            math.sqrt(1.0 + self.c_1 * norm2 / (1.0 - self.c_1)) - 1.0
        )
        if not np.isfinite(beta):
            return

        if len(self.vectors) < self.memory:
            slot = len(self.vectors)
            self.vectors.append(v)
            self.beta.append(beta)
            self.saved_at.append(generation)
            return

        slot = self._evict(n_steps)
        del self.vectors[slot], self.beta[slot], self.saved_at[slot]
        self.vectors.append(v)
        self.beta.append(beta)
        self.saved_at.append(generation)


def lm_cma_es(
    problem,
    budget: int,
    rng: np.random.Generator,
    x0=None,
    sigma0: float | None = None,
    popsize: int | None = None,
    memory: int | None = None,
    n_steps: int | None = None,
    **kw,
) -> np.ndarray:
    """LM-CMA-ES (Loshchilov, 2015) with the population success rule.

    Keeps ``m`` past evolution-path snapshots as the factors of a limited-memory
    Cholesky factor (:class:`_CholeskyMemory`), and adapts the step size by
    comparing the current population's fitness ranks against the previous one --
    no conjugate evolution path, which is what would need the full ``C``.
    """
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    xmean = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float).copy()
    )
    sigma = float(sigma0 if sigma0 is not None else default_sigma0(xmean))

    lam = int(popsize or 4 + math.floor(3 * math.log(n)))
    mu, w, mueff = _cma_weights(n, lam)
    m = int(memory or _memory_size(n))
    steps = int(n_steps if n_steps is not None else max(1, n // m))

    c_c = 0.5 / math.sqrt(n)
    c_1 = 1.0 / (10.0 * math.log(n + 1.0))
    # Population success rule: c_sigma smooths the rank comparison, z_target is
    # the success level the step size is driven towards.
    c_sigma, d_sigma, z_target = 0.3, 1.0, 0.25

    memory_A = _CholeskyMemory(n, m, c_1)
    p_c = np.zeros(n)
    s = 0.0
    previous: np.ndarray | None = None
    generation = 0

    try:
        while f.remaining > 0:
            Z = rng.standard_normal((lam, n))
            D = np.array([memory_A.transform(z) for z in Z])
            fitness = f.batch(xmean + sigma * D)

            order = np.argsort(fitness)
            step = w @ D[order[:mu]]
            xmean = xmean + sigma * step

            p_c = (1 - c_c) * p_c + math.sqrt(c_c * (2 - c_c) * mueff) * step
            if generation % steps == 0:
                memory_A.update(p_c, generation, steps)

            if previous is not None:
                # Rank the two populations jointly; a positive mean rank shift
                # means the new population is winning, so the step grows.
                joint = np.concatenate([fitness, previous])
                ranks = np.empty(len(joint))
                ranks[np.argsort(joint)] = np.arange(len(joint))
                shift = float(np.sum(ranks[lam:] - ranks[:lam])) / (lam * lam)
                s = (1 - c_sigma) * s + c_sigma * (shift - z_target)
                sigma *= math.exp(s / d_sigma)
                sigma = float(np.clip(sigma, 1e-16, 1e4))
            previous = fitness
            generation += 1
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, xmean)


def separable_nes(
    problem,
    budget: int,
    rng: np.random.Generator,
    x0=None,
    sigma0: float | None = None,
    popsize: int | None = None,
    **kw,
) -> np.ndarray:
    """Separable Natural Evolution Strategy (Wierstra et al., 2014).

    A different family from CMA/MA-ES: it follows the *natural gradient* of the
    expected fitness under a Gaussian with a per-coordinate standard deviation.
    The separable (diagonal) parameterization makes every step O(n), so like
    ``sep_cma_es`` it is a large-scale method -- but it cannot represent
    correlations, which is exactly the axis this benchmark's rugged landscapes
    probe.
    """
    f = _BudgetedProblem(problem, budget)
    n = problem.meta_data.n_variables
    mean = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float).copy()
    )
    sigma = float(sigma0 if sigma0 is not None else default_sigma0(mean)) * np.ones(n)

    lam = int(popsize or 4 + math.floor(3 * math.log(n)))
    eta_mean = 1.0
    eta_sigma = (3.0 + math.log(n)) / (5.0 * math.sqrt(n))

    # Fitness shaping: rank-based utilities that sum to zero (Wierstra 2014).
    ranks = np.arange(1, lam + 1)
    raw = np.maximum(0.0, math.log(lam / 2.0 + 1.0) - np.log(ranks))
    utilities = raw / raw.sum() - 1.0 / lam

    try:
        while f.remaining > 0:
            noise = rng.standard_normal((lam, n))
            fitness = f.batch(mean + sigma * noise)

            util = np.empty(lam)
            util[np.argsort(fitness)] = utilities  # best-ranked gets highest utility

            grad_mean = util @ noise
            grad_sigma = util @ (noise * noise - 1.0)
            mean = mean + eta_mean * sigma * grad_mean
            sigma = sigma * np.exp(0.5 * eta_sigma * grad_sigma)
            sigma = np.clip(sigma, 1e-16, 1e4)
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, mean)


#: Nevergrad optimizers to expose, keyed by our registry name.  Chosen for the
#: large-scale focus: NGOpt is Nevergrad's self-configuring meta-optimizer
#: (picks large-scale methods in high dimension); the rest are its explicit
#: large-scale / CMA family.  All are optional -- skipped if Nevergrad is absent.
NEVERGRAD_METHODS: dict[str, str] = {
    "ng_ngopt": "NGOpt",
    "ng_diagonal_cma": "DiagonalCMA",
    "ng_rescaled_cma": "RescaledCMA",
    "ng_cma": "CMA",
}


def make_nevergrad(method: str) -> "Solver":
    """Build a solver backed by a named Nevergrad optimizer (lazy import)."""

    def solver(problem, budget, rng, x0=None, **kw):
        try:
            import nevergrad as ng
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                f"{method} requires the optional `nevergrad` package "
                "(`pip install nevergrad`)."
            ) from exc

        f = _BudgetedProblem(problem, budget)
        start = (
            initial_point(problem, rng, kw.get("x0_mode", "init"))
            if x0 is None
            else np.asarray(x0, float)
        )
        param = ng.p.Array(init=np.clip(start, f.lb, f.ub))
        param.set_bounds(f.lb, f.ub, method="clipping")
        opt = ng.optimizers.registry[method](parametrization=param, budget=budget)
        try:
            while f.remaining > 0:
                cand = opt.ask()
                opt.tell(cand, f(np.asarray(cand.value, dtype=float)))
        except _BudgetedProblem.Exhausted:
            pass
        return _finish(f, start)

    solver.__name__ = method
    solver.__doc__ = f"Nevergrad's {method} (optional; needs `pip install nevergrad`)."
    return solver


def pycma_es(
    problem, budget: int, rng: np.random.Generator, x0=None, sigma0=None, **kw
):
    """Adapter for the reference ``cma`` package, when it is installed."""
    try:
        import cma
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pycma_es requires the optional `cma` package.") from exc

    f = _BudgetedProblem(problem, budget)
    start = (
        initial_point(problem, rng, kw.get("x0_mode", "init"))
        if x0 is None
        else np.asarray(x0, float)
    )
    sigma = float(sigma0 if sigma0 is not None else default_sigma0(start))
    es = cma.CMAEvolutionStrategy(
        list(start),
        sigma,
        {
            "bounds": [list(f.lb), list(f.ub)],
            "maxfevals": budget,
            "seed": int(rng.integers(1, 1 << 30)),
            "verbose": -9,
        },
    )
    try:
        while not es.stop() and f.remaining > 0:
            xs = es.ask()
            es.tell(xs, [f(x) for x in xs])
    except _BudgetedProblem.Exhausted:
        pass
    return _finish(f, start)


#: Name -> solver. Extend this to benchmark your own optimizer.
OPTIMIZERS: dict[str, Solver] = {
    "random_search": random_search,
    "local_random_search": local_random_search,
    "cma_es": cma_es,
    "sep_cma_es": sep_cma_es,
    "lm_cma_es": lm_cma_es,
    "lm_ma_es": lm_ma_es,
    "sep_nes": separable_nes,
    "pso": pso,
    "de": differential_evolution,
    "pycma_es": pycma_es,
    "adam": adam,
}

# Optional Nevergrad-backed large-scale methods; registered by name so they show
# up in `--list` and the sweep everywhere, but they raise (and callers skip) if
# Nevergrad is not installed.
OPTIMIZERS.update(
    {name: make_nevergrad(ng_name) for name, ng_name in NEVERGRAD_METHODS.items()}
)

#: The optimizers the shipped experiment runs.  ``de`` and the uniform
#: ``random_search`` floor are deliberately excluded: on this suite both sit at
#: the bottom on every metric and never approach a grokking transition, so they
#: only pad the campaign.  ``local_random_search`` (the (1+1) hill climber) is
#: kept as the cheap baseline that actually moves.
DEFAULT_BENCHMARK_OPTIMIZERS: tuple[str, ...] = (
    "adam",
    "lm_cma_es",
    "lm_ma_es",
    "sep_cma_es",
    "cma_es",
    "pso",
    "local_random_search",
)

#: Methods that use gradients; excluded from black-box-only comparisons.
GRADIENT_BASED = frozenset({"adam"})

#: Methods designed for high dimension: limited-memory, diagonal, or diagonal +
#: rank-one covariance, costing O(n) or O(m n) per sample instead of full
#: CMA-ES's O(n^2).  Includes the optional Nevergrad large-scale methods.
LARGE_SCALE = frozenset(
    {"sep_cma_es", "lm_cma_es", "lm_ma_es", "sep_nes"} | set(NEVERGRAD_METHODS)
)

#: Optimizers that need an optional third-party package.  ``optional_dependency``
#: returns the missing one (or None), so runners/tests can skip cleanly instead
#: of failing when it is absent.
OPTIONAL_OPTIMIZERS = frozenset({"pycma_es"} | set(NEVERGRAD_METHODS))


def optional_dependency(name: str) -> str | None:
    """The optional package ``name`` needs but that is not importable, or None."""
    module = (
        "cma"
        if name == "pycma_es"
        else "nevergrad"
        if name in NEVERGRAD_METHODS
        else None
    )
    if module is None:
        return None
    try:
        __import__(module)
        return None
    except ImportError:
        return module


def get_optimizer(name: str) -> Solver:
    if name not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer {name!r}; choose from {sorted(OPTIMIZERS)}."
        )
    return OPTIMIZERS[name]
