"""Fitness-landscape diagnostics: plateaus, saddle points, and local optima.

The grokking objectives ship exact gradients (``model.loss_and_grad``), so their
landscapes can be characterized *rigorously* rather than through surrogate
features: a Hessian is a finite difference of the analytic gradient (``n``
gradient calls for the full ``n x n`` matrix, cheap at ``n <= 200``), and its
eigen-spectrum tells local minimum from saddle directly.

Three questions, three tools:

* **Plateaus** -- how much of the landscape is (near) flat?  ``gradient_norms``
  samples ``||grad f||``; a large fraction near zero, together with the flat
  stretches in ``line_profile`` / ``plane_slice``, is the evidence.
* **Local optima** -- does gradient descent from different starts reach
  *different* objective levels?  ``restart_minima`` runs many descents;
  distinct converged values (and barriers between them, ``line_profile``) mean
  multiple basins.
* **Saddle points** -- do the critical points have negative curvature?
  ``find_critical_point`` lands on a point with ``||grad|| ~ 0`` of arbitrary
  index (it minimizes ``||grad||^2``), and ``spectrum`` counts the negative
  Hessian eigenvalues -- the saddle index.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaab.objective import GrokkingObjective


# ---------------------------------------------------------------------------
# curvature: finite-difference Hessian from the analytic gradient
# ---------------------------------------------------------------------------
def _grad(obj: GrokkingObjective, theta: np.ndarray) -> np.ndarray:
    return obj.loss_and_grad(theta)[1]


def hessian_vector_product(
    obj: GrokkingObjective, theta: np.ndarray, v: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """``H v`` by a central difference of the exact gradient -- two grad calls."""
    theta = np.asarray(theta, dtype=np.float64)
    scale = eps * (1.0 + np.linalg.norm(theta)) / max(np.linalg.norm(v), 1e-12)
    return (_grad(obj, theta + scale * v) - _grad(obj, theta - scale * v)) / (2.0 * scale)


def hessian(obj: GrokkingObjective, theta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """The full symmetric Hessian by differencing the gradient along each axis."""
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.size
    step = eps * (1.0 + np.linalg.norm(theta)) / np.sqrt(n)
    H = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = step
        H[:, j] = (_grad(obj, theta + e) - _grad(obj, theta - e)) / (2.0 * step)
    return 0.5 * (H + H.T)  # symmetrize away FD asymmetry


@dataclass
class Spectrum:
    """The Hessian eigen-spectrum at a point, and what it says about the point."""

    eigenvalues: np.ndarray
    min: float
    max: float
    n_negative: int
    index: float           # fraction of eigenvalues that are negative
    n_flat: int            # eigenvalues indistinguishable from zero (a flat plateau direction)

    @property
    def kind(self) -> str:
        if self.n_negative == 0:
            return "minimum"
        if self.n_negative == len(self.eigenvalues):
            return "maximum"
        return "saddle"


def spectrum(H: np.ndarray, flat_tol: float = 1e-6) -> Spectrum:
    eig = np.linalg.eigvalsh(H)
    scale = max(abs(eig[0]), abs(eig[-1]), 1e-12)
    tol = flat_tol * scale
    return Spectrum(
        eigenvalues=eig,
        min=float(eig[0]),
        max=float(eig[-1]),
        n_negative=int(np.sum(eig < -tol)),
        index=float(np.mean(eig < -tol)),
        n_flat=int(np.sum(np.abs(eig) <= tol)),
    )


# ---------------------------------------------------------------------------
# descent to critical points
# ---------------------------------------------------------------------------
@dataclass
class Descent:
    theta: np.ndarray
    objective: float
    grad_norm: float
    steps: int
    trajectory: np.ndarray  # (k, 2): step, objective at sampled points


def _adam(obj, theta, steps, lr, sample_every, tol):
    theta = np.array(theta, dtype=np.float64, copy=True)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    traj = []
    grad_norm = np.inf
    for t in range(1, steps + 1):
        loss, g = obj.loss_and_grad(theta)
        grad_norm = float(np.linalg.norm(g))
        if t == 1 or t % sample_every == 0:
            traj.append((t, loss))
        if not np.isfinite(loss) or grad_norm < tol:
            break
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta = theta - lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return theta, grad_norm, t, np.array(traj)


def descend_to_minimum(
    obj: GrokkingObjective,
    theta0: np.ndarray,
    steps: int = 20_000,
    lr: float = 1e-2,
    sample_every: int = 100,
    tol: float = 1e-6,
) -> Descent:
    """Adam descent until the gradient is tiny -- lands in a local basin."""
    theta, gn, t, traj = _adam(obj, theta0, steps, lr, sample_every, tol)
    return Descent(theta, float(obj(theta)), gn, t, traj)


def find_critical_point(
    obj: GrokkingObjective,
    theta0: np.ndarray,
    steps: int = 500,
    eps: float = 1e-5,
    tol: float = 1e-7,
) -> Descent:
    """Minimize ``phi = 1/2 ||grad f||^2`` to reach a critical point of *any* index.

    Ordinary descent only finds minima; ``phi`` has gradient ``H g`` (one extra
    Hessian-vector product per step), so following it can settle on a saddle,
    which is what exposes the negative-curvature structure.  A backtracking line
    search on ``phi`` keeps it stable regardless of the local conditioning that
    made a fixed step size diverge.
    """
    theta = np.array(theta0, dtype=np.float64, copy=True)
    g = _grad(obj, theta)
    phi = 0.5 * float(g @ g)
    step = 1.0
    t = 0
    for t in range(1, steps + 1):
        grad_norm = float(np.linalg.norm(g))
        if grad_norm < tol:
            break
        direction = hessian_vector_product(obj, theta, g, eps)  # grad(phi) = H g
        dnorm = float(np.linalg.norm(direction))
        if dnorm < 1e-30:
            break
        # Backtracking: shrink the step until phi actually decreases.
        accepted = False
        for _ in range(30):
            cand = theta - step * direction
            g_cand = _grad(obj, cand)
            phi_cand = 0.5 * float(g_cand @ g_cand)
            if phi_cand < phi:
                theta, g, phi = cand, g_cand, phi_cand
                step *= 1.5  # grow again while it keeps working
                accepted = True
                break
            step *= 0.5
        if not accepted:  # cannot make progress -> we are at a critical point
            break
    return Descent(theta, float(obj(theta)), float(np.linalg.norm(g)), t, np.empty((0, 2)))


def restart_minima(
    obj: GrokkingObjective,
    n_restarts: int = 32,
    scale: float = 1.0,
    seed: int = 0,
    steps: int = 20_000,
    lr: float = 1e-2,
) -> list[Descent]:
    """Descend from many random initializations -- one basin per restart."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_restarts):
        theta0 = obj.initial_theta(seed=int(rng.integers(1 << 30)), scale=scale)
        out.append(descend_to_minimum(obj, theta0, steps=steps, lr=lr))
    return out


def count_distinct(values, rel_tol: float = 1e-3) -> int:
    """Number of distinct objective levels among converged descents."""
    vals = np.sort(np.asarray(values, dtype=np.float64))
    if len(vals) == 0:
        return 0
    scale = max(abs(vals[0]), 1e-9)
    distinct = 1
    ref = vals[0]
    for x in vals[1:]:
        if abs(x - ref) > rel_tol * scale:
            distinct += 1
            ref = x
    return distinct


# ---------------------------------------------------------------------------
# plateaus: gradient-norm sampling and low-dimensional slices
# ---------------------------------------------------------------------------
def gradient_norms(
    obj: GrokkingObjective, thetas: np.ndarray
) -> np.ndarray:
    return np.array([np.linalg.norm(obj.loss_and_grad(t)[1]) for t in np.atleast_2d(thetas)])


def sample_points(obj, n_points, scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([obj.initial_theta(seed=int(rng.integers(1 << 30)), scale=scale)
                     for _ in range(n_points)])


def line_profile(
    obj: GrokkingObjective, a: np.ndarray, b: np.ndarray, n: int = 101, extend: float = 0.0
):
    """Objective along the segment ``a -> b`` (optionally extended past the ends).

    A flat stretch is a plateau; a bump between two minima is a barrier that
    proves they sit in different basins.
    """
    lo, hi = -extend, 1.0 + extend
    alphas = np.linspace(lo, hi, n)
    a, b = np.asarray(a, float), np.asarray(b, float)
    values = np.array([obj(a + alpha * (b - a)) for alpha in alphas])
    return alphas, values


def characterize(
    obj: GrokkingObjective,
    restarts: int = 16,
    n_critical: int = 10,
    descent_steps: int = 10_000,
    critical_steps: int = 150,
    seed: int = 0,
) -> dict:
    """Scalar plateau / saddle / local-optima metrics for one objective.

    Shared by the single-problem figures and the suite-wide sweep, so the two
    report the same numbers.  Everything here is a population-summarisable
    scalar; the figure script computes the extra curves it needs separately.
    """
    descents = restart_minima(obj, n_restarts=restarts, seed=seed, steps=descent_steps)
    finals = np.array([d.objective for d in descents])
    global_f = float(finals.min())
    reached = finals <= global_f + 1e-2 * max(abs(global_f), 1.0)
    best = min(descents, key=lambda d: d.objective)
    min_spec = spectrum(hessian(obj, best.theta))

    rng = np.random.default_rng(seed + 1)
    saddle_indices, kinds = [], []
    for _ in range(n_critical):
        cp = find_critical_point(
            obj, obj.initial_theta(seed=int(rng.integers(1 << 30)), scale=1.0),
            steps=critical_steps,
        )
        if cp.grad_norm < 0.2:  # keep only genuine (near-)critical points
            sp = spectrum(hessian(obj, cp.theta))
            saddle_indices.append(sp.index)
            kinds.append(sp.kind)

    return {
        "n_variables": int(obj.n_variables),
        "n_descents": len(finals),
        "n_distinct_minima": count_distinct(finals, rel_tol=1e-2),
        "frac_reached_global": float(reached.mean()),
        "global_objective": round(global_f, 5),
        "worst_basin_gap": round(float(finals.max() - global_f), 5),
        "n_critical_points": len(saddle_indices),
        "frac_saddles": float(np.mean([k == "saddle" for k in kinds])) if kinds else float("nan"),
        "median_saddle_index": float(np.median(saddle_indices)) if saddle_indices else float("nan"),
        "min_index": round(min_spec.index, 4),
        "min_flat_directions": int(min_spec.n_flat),
    }


def plane_slice(
    obj: GrokkingObjective,
    center: np.ndarray,
    span: float = 1.0,
    n: int = 41,
    seed: int = 0,
):
    """Objective on a 2D grid over two random orthonormal directions.

    ``span`` is measured in units of the center's norm, so the window matches
    the scale the optimizer actually lives at.
    """
    rng = np.random.default_rng(seed)
    center = np.asarray(center, float)
    d = center.size
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    w = rng.standard_normal(d)
    w -= (w @ u) * u
    w /= np.linalg.norm(w)

    radius = span * (np.linalg.norm(center) + 1.0)
    ts = np.linspace(-radius, radius, n)
    Z = np.empty((n, n))
    for i, a in enumerate(ts):
        for j, b in enumerate(ts):
            Z[i, j] = obj(center + a * u + b * w)
    return ts, Z
