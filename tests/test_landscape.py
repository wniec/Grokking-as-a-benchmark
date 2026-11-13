"""Tests for the landscape diagnostics -- the numerics must be trustworthy, since
the paper's plateau/saddle/local-optima claims rest on them."""

import numpy as np
import pytest

from gaab import landscape as L
from gaab.objective import build_objective


class Quadratic:
    """f(x) = 1/2 x^T A x + b^T x -- a mock objective with a known Hessian A."""

    def __init__(self, A, b):
        self.A = np.asarray(A, float)
        self.b = np.asarray(b, float)
        self.n_variables = self.A.shape[0]

    def loss_and_grad(self, x):
        x = np.asarray(x, float)
        return float(0.5 * x @ self.A @ x + self.b @ x), self.A @ x + self.b

    def __call__(self, x):
        x = np.asarray(x, float)
        return float(0.5 * x @ self.A @ x + self.b @ x)

    def initial_theta(self, seed=0, scale=1.0):
        return np.random.default_rng(seed).standard_normal(self.n_variables) * scale


def indefinite_quadratic(n=10, n_negative=3, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
    eig = np.concatenate(
        [-np.linspace(0.5, 3, n_negative), np.linspace(0.5, 5, n - n_negative)]
    )
    A = Q @ np.diag(eig) @ Q.T
    return Quadratic(0.5 * (A + A.T), rng.standard_normal(n)), eig


def test_hessian_matches_a_known_quadratic():
    q, _ = indefinite_quadratic(12, seed=1)
    H = L.hessian(q, np.random.default_rng(0).standard_normal(12))
    assert np.allclose(H, q.A, atol=1e-6)
    assert np.allclose(H, H.T)  # symmetric


def test_hessian_vector_product_matches():
    q, _ = indefinite_quadratic(12, seed=2)
    x = np.random.default_rng(0).standard_normal(12)
    v = np.random.default_rng(1).standard_normal(12)
    assert np.allclose(L.hessian_vector_product(q, x, v), q.A @ v, atol=1e-5)


def test_spectrum_classifies_minimum_saddle_and_counts_negatives():
    q, eig = indefinite_quadratic(10, n_negative=3, seed=3)
    sp = L.spectrum(q.A)
    assert sp.kind == "saddle"
    assert sp.n_negative == 3
    assert sp.index == pytest.approx(0.3)

    spd = np.diag([0.5, 1.0, 2.0, 3.0])  # positive definite
    assert L.spectrum(spd).kind == "minimum"
    assert L.spectrum(spd).n_negative == 0


def test_spectrum_counts_flat_directions():
    H = np.diag([-2.0, 0.0, 0.0, 1.0, 3.0])
    sp = L.spectrum(H)
    assert sp.n_flat == 2 and sp.n_negative == 1


def test_find_critical_point_reaches_a_known_saddle():
    q, _ = indefinite_quadratic(
        12, n_negative=3, seed=4
    )  # nonsingular -> unique critical point
    true_star = np.linalg.solve(q.A, -q.b)
    cp = L.find_critical_point(
        q, np.random.default_rng(0).standard_normal(12) * 0.5, steps=400
    )
    assert cp.grad_norm < 1e-2
    assert np.linalg.norm(cp.theta - true_star) < 1e-2
    assert L.spectrum(L.hessian(q, cp.theta)).n_negative == 3


def test_descend_reduces_a_convex_objective():
    q = Quadratic(np.diag([1.0, 2.0, 4.0]), np.zeros(3))
    d = L.descend_to_minimum(q, np.array([3.0, -2.0, 1.0]), steps=3000, lr=5e-2)
    assert d.objective < 1e-3 and d.grad_norm < 1e-2


def test_count_distinct_levels():
    assert L.count_distinct([0.0, 0.0, 0.5, 0.5, 0.9], rel_tol=1e-2) == 3
    assert L.count_distinct([1.0, 1.0, 1.0]) == 1
    assert L.count_distinct([]) == 0


def test_line_profile_hits_the_endpoints():
    q = Quadratic(np.diag([1.0, 1.0]), np.zeros(2))
    a, b = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    alphas, vals = L.line_profile(q, a, b, n=11)
    assert alphas[0] == 0.0 and alphas[-1] == 1.0
    assert vals[0] == pytest.approx(q(a)) and vals[-1] == pytest.approx(q(b))


def test_hessian_eigen_plane_uses_true_curvature():
    # anisotropic quadratic: known eigenvalues 9 (steep) and 1 (flat), min at 0
    q = Quadratic(np.diag([9.0, 1.0]), np.zeros(2))
    ts, Z, (lam_max, lam_min) = L.hessian_eigen_plane(q, np.zeros(2), n=21, span=1.0)
    assert lam_max == pytest.approx(9.0, abs=1e-3)
    assert lam_min == pytest.approx(1.0, abs=1e-3)
    centre = Z[len(ts) // 2, len(ts) // 2]
    assert centre == pytest.approx(0.0, abs=1e-9)  # minimum sits at the centre
    assert Z.min() == pytest.approx(centre)  # and is the lowest on the grid


def test_trajectory_variance_measures_path_dimension():
    rng = np.random.default_rng(0)
    # a path that lives in 2 of 5 dimensions -> ~all variance in the first 2
    iters = np.zeros((50, 5))
    iters[:, 0] = rng.standard_normal(50) * 5.0  # dominant direction
    iters[:, 1] = rng.standard_normal(50) * 1.0  # secondary
    ratios = L.trajectory_variance(iters, center=np.zeros(5))
    assert ratios.shape == (5,)
    assert ratios[0] >= ratios[1] >= ratios[2]  # descending
    assert ratios[:2].sum() == pytest.approx(1.0, abs=1e-6)
    assert np.sum(ratios) == pytest.approx(1.0, abs=1e-9)


def test_trajectory_pca_plane_recovers_the_path_plane():
    # iterates living in the x-y plane of a 4D space -> 2 PCs capture all variance
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((40, 2))
    iterates = np.zeros((40, 4))
    iterates[:, :2] = coords
    q = Quadratic(np.diag([1.0, 1.0, 1.0, 1.0]), np.zeros(4))
    xs, ys, Z, path_xy, var_ratio = L.trajectory_pca_plane(
        q, iterates, center=np.zeros(4), n=15
    )
    assert var_ratio == pytest.approx(1.0, abs=1e-6)  # path is exactly 2D
    assert path_xy.shape == (40, 2)
    assert Z.shape == (15, 15) and np.all(np.isfinite(Z))


# --- one end-to-end smoke test on a real (tiny) grokking objective ---------
def test_landscape_primitives_run_on_a_grokking_objective():
    obj = build_objective(
        p=5,
        embed_dim=2,
        hidden_dim=6,
        reg_strength=0.01,
        train_frac=0.8,
        activation="quad",
        seed=0,
    )
    theta = obj.initial_theta(0)
    H = L.hessian(obj, theta)
    assert H.shape == (obj.n_variables, obj.n_variables)
    assert np.allclose(H, H.T, atol=1e-6)

    norms = L.gradient_norms(obj, L.sample_points(obj, 5, seed=0))
    assert norms.shape == (5,) and np.all(norms >= 0)

    mins = L.restart_minima(obj, n_restarts=3, steps=500, seed=0)
    assert len(mins) == 3 and all(np.isfinite(m.objective) for m in mins)
