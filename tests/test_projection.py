"""Tests for the random-subspace reparameterization (arbitrary search dim)."""

import numpy as np
import pytest

from gaab.objective import build_objective
from gaab.problem import GrokkingProblem, ProblemSpec
from gaab.projection import ProjectedObjective


def base_objective(**kw):
    return build_objective(
        p=7,
        embed_dim=2,
        hidden_dim=6,
        activation="quad",
        reg_strength=1e-2,
        train_frac=0.6,
        seed=0,
        **kw,
    )


@pytest.mark.parametrize("dim", [8, 40, 400])
def test_search_dimension_is_the_projection_dim(dim):
    obj = ProjectedObjective(base_objective(), dim=dim, seed=1)
    assert obj.n_variables == dim
    z = obj.initial_theta(0)
    assert z.shape == (dim,)
    assert obj.expand(z).shape == (obj.n_full,)


def test_orthonormal_projection_when_dim_below_model():
    base = base_objective()
    obj = ProjectedObjective(base, dim=20, seed=2)
    assert obj.dim < obj.n_full
    assert np.allclose(obj.P.T @ obj.P, np.eye(20), atol=1e-10)  # isometry


def test_over_complete_projection_is_allowed():
    base = base_objective()
    obj = ProjectedObjective(base, dim=base.n_variables + 50, seed=3)
    assert obj.n_variables == base.n_variables + 50  # d > n_full is fine


def test_zero_latent_sits_at_the_anchor():
    base = base_objective()
    obj = ProjectedObjective(base, dim=30, seed=4)
    assert np.allclose(obj.expand(np.zeros(30)), base.initial_theta(seed=4))


def test_gradient_pulls_back_through_the_projection():
    obj = ProjectedObjective(base_objective(), dim=50, seed=5)
    z = obj.initial_theta(1, scale=1.0)
    loss, grad = obj.loss_and_grad(z)
    assert loss == pytest.approx(obj(z))
    # exact chain rule: grad_z == P^T grad_theta
    _, grad_theta = obj.base.loss_and_grad(obj.expand(z))
    assert np.allclose(grad, obj.P.T @ grad_theta)
    # and it matches finite differences
    rng = np.random.default_rng(0)
    eps = 1e-6
    for i in rng.choice(obj.n_variables, 8, replace=False):
        zp, zm = z.copy(), z.copy()
        zp[i] += eps
        zm[i] -= eps
        assert (obj(zp) - obj(zm)) / (2 * eps) == pytest.approx(grad[i], abs=1e-6)


def test_metrics_pass_through_to_the_base():
    base = base_objective()
    obj = ProjectedObjective(base, dim=40, seed=6)
    z = obj.initial_theta(2)
    assert obj.metrics(z).as_dict() == base.metrics(obj.expand(z)).as_dict()
    assert obj.random_guess_loss == base.random_guess_loss


def test_rejects_degenerate_dimension():
    with pytest.raises(ValueError):
        ProjectedObjective(base_objective(), dim=0)


# --- integration with ProblemSpec / GrokkingProblem ------------------------
def test_problem_spec_exposes_the_projected_dimension():
    spec = ProblemSpec(p=11, embed_dim=2, hidden_dim=6, projection_dim=64)
    assert spec.n_variables == 64
    assert spec.model_n_variables == 129  # the network itself is unchanged
    assert "proj64" in spec.name
    assert ProblemSpec.from_dict(spec.to_dict()) == spec  # round-trips


def test_projected_problem_has_the_right_ioh_dimension():
    spec = ProblemSpec(
        p=11,
        embed_dim=2,
        hidden_dim=6,
        projection_dim=64,
        reg_strength=1e-3,
        train_frac=0.7,
        activation="quad",
    )
    problem = GrokkingProblem(spec, instance=1)
    assert problem.n_variables == 64
    assert problem.meta_data.n_variables == 64
    assert len(problem.bounds.lb) == 64
    x = problem.initial_theta()
    assert problem(list(x)) > 0  # callable at the projected dimension


def test_instances_differ_through_the_projection():
    spec = ProblemSpec(p=11, embed_dim=2, hidden_dim=6, projection_dim=40)
    a = GrokkingProblem(spec, instance=1).objective
    b = GrokkingProblem(spec, instance=2).objective
    assert not np.allclose(a.theta0, b.theta0) or not np.allclose(a.P, b.P)
