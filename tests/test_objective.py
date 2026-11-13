import numpy as np
import pytest

from gaab.objective import build_objective


def test_objective_is_train_loss_plus_direct_regularization():
    obj = build_objective(p=7, embed_dim=4, hidden_dim=8, reg_strength=0.25, seed=0)
    theta = obj.initial_theta(1)
    metrics = obj.metrics(theta)
    assert obj(theta) == pytest.approx(
        metrics.train_loss + 0.25 * float(np.mean(theta**2))
    )
    assert metrics.objective == pytest.approx(obj(theta))


def test_regularization_only_changes_the_objective_not_the_train_loss():
    kwargs = dict(p=7, embed_dim=4, hidden_dim=8, seed=0)
    plain = build_objective(reg_strength=0.0, **kwargs)
    regularized = build_objective(reg_strength=1.0, **kwargs)
    theta = plain.initial_theta(2)
    assert plain.metrics(theta).train_loss == pytest.approx(
        regularized.metrics(theta).train_loss
    )
    assert regularized(theta) > plain(theta)


def test_test_metrics_never_enter_the_objective():
    obj = build_objective(p=7, embed_dim=4, hidden_dim=8, reg_strength=0.1, seed=0)
    theta = obj.initial_theta(0)
    m = obj.metrics(theta)
    assert m.objective == pytest.approx(obj(theta))
    assert 0.0 <= m.test_accuracy <= 1.0


def test_gradient_includes_the_regularizer():
    obj = build_objective(p=7, embed_dim=4, hidden_dim=8, reg_strength=0.5, seed=0)
    theta = obj.initial_theta(1)
    loss, grad = obj.loss_and_grad(theta)
    assert loss == pytest.approx(obj(theta))
    eps = 1e-6
    for i in (0, 5, 17):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric = (obj(plus) - obj(minus)) / (2 * eps)
        assert numeric == pytest.approx(grad[i], abs=1e-6)


def test_random_guess_loss_is_log_p():
    obj = build_objective(p=11, seed=0)
    assert obj.random_guess_loss == pytest.approx(np.log(11))


def test_negative_regularization_is_rejected():
    with pytest.raises(ValueError):
        build_objective(p=7, reg_strength=-1.0)
