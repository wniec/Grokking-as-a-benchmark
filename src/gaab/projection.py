"""Random-subspace reparameterization -- optimize in *any* dimension.

A low-rank affine map decouples the search dimension from the model's parameter
count.  The optimizer works on ``z`` in ``R^d``; the model's weights are

    theta = theta0 + P @ z ,      P in R^{n_full x d}

for a fixed random projection ``P`` and an anchor ``theta0`` (the network's own
initialization).  This is the "intrinsic dimension" construction of Li et al.
(2018): any ``d`` is allowed -- far below the parameter count (a hard low-rank
restriction of the search) or above it (an over-complete search) -- so the
benchmark can present a problem of an arbitrary dimensionality while the
underlying grokking task is unchanged.

The reparameterization is exact and cheap: the gradient pulls back as
``grad_z = P^T grad_theta`` (one extra mat-vec), so screening, the analytic
landscape tools, and the IOH wrapper all apply through it without change.
"""

from __future__ import annotations

import numpy as np

from gaab.objective import GrokkingObjective, Metrics


class ProjectedObjective:
    """Wrap an objective so the search space is a ``d``-dim random subspace.

    Exposes the same interface as :class:`gaab.objective.GrokkingObjective`
    (``loss_and_grad``, ``metrics``, ``__call__``, ``initial_theta``,
    ``n_variables``), but every method takes the latent ``z`` in ``R^d`` and
    evaluates the base objective at ``theta0 + P z``.
    """

    def __init__(
        self,
        base: GrokkingObjective,
        dim: int,
        seed: int = 0,
        orthonormal: bool = True,
    ) -> None:
        if dim < 1:
            raise ValueError("projection dim must be >= 1.")
        self.base = base
        self.dim = int(dim)
        self.n_full = base.n_variables
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        # Anchor at the network's own initialization, so z = 0 is a valid start.
        self.theta0 = base.initial_theta(seed=self.seed)

        if orthonormal and self.dim <= self.n_full:
            # Orthonormal columns (P^T P = I): ||P z|| = ||z||, so the latent
            # space is isometric to the subspace and the landscape in z is not
            # distorted by the projection.
            Q, _ = np.linalg.qr(rng.standard_normal((self.n_full, self.dim)))
            self.P = Q[:, : self.dim]
        else:
            # d > n_full (over-complete) cannot be orthonormal; use unit-norm
            # random columns scaled so a typical step keeps the weight scale.
            P = rng.standard_normal((self.n_full, self.dim))
            P /= np.linalg.norm(P, axis=0, keepdims=True) + 1e-12
            self.P = P / np.sqrt(max(self.dim / self.n_full, 1.0))

    # -- the projected affine map -------------------------------------------
    def expand(self, z: np.ndarray) -> np.ndarray:
        """Latent ``z`` -> full parameter vector ``theta0 + P z``."""
        return self.theta0 + self.P @ np.asarray(z, dtype=np.float64).reshape(-1)

    # -- interface mirroring GrokkingObjective ------------------------------
    @property
    def n_variables(self) -> int:
        return self.dim

    @property
    def random_guess_loss(self) -> float:
        return self.base.random_guess_loss

    @property
    def reg_strength(self) -> float:
        return self.base.reg_strength

    def __call__(self, z: np.ndarray) -> float:
        return self.base(self.expand(z))

    def loss_and_grad(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        loss, grad_theta = self.base.loss_and_grad(self.expand(z))
        return loss, self.P.T @ grad_theta  # chain rule through the linear map

    def metrics(self, z: np.ndarray) -> Metrics:
        return self.base.metrics(self.expand(z))

    def initial_theta(self, seed: int = 0, scale: float = 1.0) -> np.ndarray:
        """A latent start.  ``z = 0`` sits at the anchor; ``scale`` adds jitter."""
        if scale == 0.0:
            return np.zeros(self.dim)
        rng = np.random.default_rng(seed)
        # Small isotropic jitter so restarts differ without leaving the basin.
        return 0.1 * scale * rng.standard_normal(self.dim)
