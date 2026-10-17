"""The scalar function a continuous optimizer actually minimizes.

    f(theta) = CE_train(theta) + lambda * mean(theta ** 2)

The regularization term is a *direct factor of the loss*, not a weight-decay
term bolted onto an update rule.  That distinction is the point: it must be
part of the landscape so that a black-box optimizer -- which has no update rule
to decay -- sees the same problem a gradient method does.  ``lambda`` is
therefore a landscape knob, and sweeping it moves a problem between "memorizes
and never generalizes" and "groks".

Test-set quantities are computed but never enter ``f``.  Grokking is precisely
the gap between the objective the optimizer minimizes and the generalization it
does or does not eventually acquire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gaab.data import ModularArithmeticData, make_dataset
from gaab.model import GrokkingMLP, ModelConfig


@dataclass(frozen=True)
class Metrics:
    """Everything observable about one parameter vector."""

    objective: float
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
    reg_penalty: float
    weight_norm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "objective": self.objective,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "test_loss": self.test_loss,
            "test_accuracy": self.test_accuracy,
            "reg_penalty": self.reg_penalty,
            "weight_norm": self.weight_norm,
        }


def _cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean cross-entropy and accuracy, computed stably."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    n = y.shape[0]
    loss = float(-log_probs[np.arange(n), y].mean())
    acc = float((logits.argmax(axis=1) == y).mean())
    return loss, acc


class GrokkingObjective:
    """Dataset + model + regularizer, as a plain NumPy callable."""

    def __init__(
        self,
        model_config: ModelConfig,
        data: ModularArithmeticData,
        reg_strength: float = 0.0,
    ) -> None:
        if data.p != model_config.p:
            raise ValueError("Model and dataset must agree on p.")
        if reg_strength < 0.0:
            raise ValueError("reg_strength must be non-negative.")
        self.model = GrokkingMLP(model_config)
        self.data = data
        self.reg_strength = float(reg_strength)

    @property
    def n_variables(self) -> int:
        return self.model.n_params

    @property
    def random_guess_loss(self) -> float:
        """Cross-entropy of the uniform predictor -- the natural ``f`` ceiling."""
        return float(np.log(self.data.p))

    def penalty(self, theta: np.ndarray) -> float:
        return float(np.mean(np.asarray(theta) ** 2))

    def __call__(self, theta: np.ndarray) -> float:
        d = self.data
        logits = self.model.logits(theta, d.train_a, d.train_b)
        loss, _ = _cross_entropy(logits, d.train_y)
        if self.reg_strength:
            loss += self.reg_strength * self.penalty(theta)
        return loss

    def metrics(self, theta: np.ndarray) -> Metrics:
        d = self.data
        train_loss, train_acc = _cross_entropy(
            self.model.logits(theta, d.train_a, d.train_b), d.train_y
        )
        test_loss, test_acc = _cross_entropy(
            self.model.logits(theta, d.test_a, d.test_b), d.test_y
        )
        pen = self.penalty(theta)
        return Metrics(
            objective=train_loss + self.reg_strength * pen,
            train_loss=train_loss,
            train_accuracy=train_acc,
            test_loss=test_loss,
            test_accuracy=test_acc,
            reg_penalty=pen,
            weight_norm=float(np.linalg.norm(theta)),
        )

    def loss_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Objective and its exact gradient (used only by the screener)."""
        d = self.data
        loss, grad = self.model.loss_and_grad(theta, d.train_a, d.train_b, d.train_y)
        if self.reg_strength:
            theta = np.asarray(theta, dtype=np.float64)
            loss += self.reg_strength * self.penalty(theta)
            grad = grad + (2.0 * self.reg_strength / theta.size) * theta
        return loss, grad

    def initial_theta(self, seed: int = 0, scale: float = 1.0) -> np.ndarray:
        return self.model.initial_theta(seed=seed, scale=scale)


def build_objective(
    p: int,
    embed_dim: int = 8,
    hidden_dim: int = 32,
    rank: int | None = None,
    reg_strength: float = 0.0,
    train_frac: float = 0.5,
    op: str = "add",
    activation: str = "relu",
    rank_targets: tuple[str, ...] = ("W1",),
    seed: int = 0,
) -> GrokkingObjective:
    config = ModelConfig(
        p=p,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        rank=rank,
        activation=activation,
        rank_targets=rank_targets,
    )
    data = make_dataset(p=p, op=op, train_frac=train_frac, seed=seed)
    return GrokkingObjective(config, data, reg_strength=reg_strength)
