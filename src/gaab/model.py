"""The neural network whose weights form the search space.

A deliberately small embedding-MLP decoder for ``a OP b = c (mod p)``::

    x      = concat(E[a], E[b])          (N, 2 * embed_dim)
    h      = act(x @ W1 + b1)            (N, hidden_dim)
    logits = h @ W2 + b2                 (N, p)

Any weight matrix named in ``rank_targets`` is replaced by a rank-``r`` product
``U @ V``.  That is the "low rank approximation" knob: it reshapes the loss
landscape and *may* shrink the parameter count -- but a rank that exceeds the
natural bottleneck of a matrix makes it grow instead (``(m + n) * r`` vs
``m * n``), so it is "possibly, but not necessarily" fewer parameters.

*Which* matrix is restricted matters more than the rank does.  ``W1`` tolerates
a hard bottleneck: the Fourier solution only needs a handful of frequencies, so
a few directions out of ``2 * embed_dim`` carry it.  ``W2`` does not -- it is
the readout onto ``p`` classes, and squeezing it through ``r < p`` directions
caps how many classes can be told apart, which is why screening rejects almost
every ``W2``-factorized configuration as ``no_generalization``.  Hence the
default ``rank_targets = ("W1",)``.

Everything is NumPy and works on flat parameter vectors, because that is what a
continuous black-box optimizer hands us.  Analytic gradients are provided too:
they are not used by the benchmark objective, only by the gradient-based
reference optimizer that decides which problems grok (see ``screening.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIVATIONS = ("relu", "tanh", "gelu", "quad")

#: Weight matrices a rank restriction can be applied to.
RANK_TARGETS = ("E", "W1", "W2")


def _act(name: str, z: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh(z)
    if name == "gelu":  # tanh approximation, as in the original GELU paper
        c = 0.7978845608028654  # sqrt(2 / pi)
        return 0.5 * z * (1.0 + np.tanh(c * (z + 0.044715 * z**3)))
    if name == "quad":
        # Gromov (2023) shows a two-layer MLP with a quadratic activation has
        # closed-form Fourier solutions for modular addition; it is by far the
        # most reliable way to make a *small* network grok.
        return z * z
    raise ValueError(f"Unknown activation {name!r}; choose from {ACTIVATIONS}.")


def _act_grad(name: str, z: np.ndarray, a: np.ndarray) -> np.ndarray:
    if name == "relu":
        return (z > 0.0).astype(z.dtype)
    if name == "tanh":
        return 1.0 - a * a
    if name == "gelu":
        c = 0.7978845608028654
        inner = c * (z + 0.044715 * z**3)
        t = np.tanh(inner)
        return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * c * (
            1.0 + 3 * 0.044715 * z**2
        )
    if name == "quad":
        return 2.0 * z
    raise ValueError(f"Unknown activation {name!r}.")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the decoder; fixes the dimension of the search space."""

    p: int
    embed_dim: int = 8
    hidden_dim: int = 32
    rank: int | None = None
    activation: str = "relu"
    rank_targets: tuple[str, ...] = ("W1",)

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}.")
        if self.rank is not None and self.rank < 1:
            raise ValueError("rank must be >= 1 or None (full rank).")
        for name in ("embed_dim", "hidden_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1.")
        unknown = set(self.rank_targets) - set(RANK_TARGETS)
        if unknown:
            raise ValueError(
                f"Unknown rank target(s) {sorted(unknown)}; choose from {RANK_TARGETS}."
            )
        # A frozen dataclass has to go through object.__setattr__ to normalize.
        object.__setattr__(
            self,
            "rank_targets",
            tuple(t for t in RANK_TARGETS if t in self.rank_targets),
        )


@dataclass
class _Block:
    """One parameter tensor's slice of the flat vector."""

    name: str
    shape: tuple[int, ...]
    start: int
    fan_in: int

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def stop(self) -> int:
        return self.start + self.size


class GrokkingMLP:
    """Flat-vector view of the decoder above.

    The only state is the *layout*: which slice of ``theta`` is which tensor.
    All methods take ``theta`` explicitly so the object is safe to share across
    parallel evaluations.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.blocks: list[_Block] = []
        self._cursor = 0

        p, d, h = config.p, config.embed_dim, config.hidden_dim
        targets = set(config.rank_targets)
        self._add_matrix("E", (p, d), fan_in=d, factorize="E" in targets)
        self._add_matrix("W1", (2 * d, h), fan_in=2 * d, factorize="W1" in targets)
        self._add("b1", (h,), fan_in=1)
        self._add_matrix("W2", (h, p), fan_in=h, factorize="W2" in targets)
        self._add("b2", (p,), fan_in=1)

        self.n_params = self._cursor
        self.index: dict[str, _Block] = {b.name: b for b in self.blocks}

    # -- layout construction ------------------------------------------------
    def _add(self, name: str, shape: tuple[int, ...], fan_in: int) -> None:
        self.blocks.append(_Block(name, shape, self._cursor, fan_in))
        self._cursor += int(np.prod(shape))

    def _add_matrix(
        self, name: str, shape: tuple[int, int], fan_in: int, factorize: bool
    ) -> None:
        """Store ``shape`` either densely or as a rank-r product ``U @ V``."""
        r = self.config.rank
        m, n = shape
        if not factorize or r is None or r >= min(m, n):
            # A rank at or above min(m, n) is not a restriction, so we keep the
            # dense parameterization instead of wasting parameters on it.
            self._add(name, shape, fan_in)
            return
        self._add(f"{name}_u", (m, r), fan_in=fan_in)
        self._add(f"{name}_v", (r, n), fan_in=r)

    def is_factorized(self, name: str) -> bool:
        return f"{name}_u" in self.index

    # -- parameter access ---------------------------------------------------
    def unpack(self, theta: np.ndarray) -> dict[str, np.ndarray]:
        theta = np.asarray(theta, dtype=np.float64).reshape(-1)
        if theta.size != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {theta.size}.")
        return {b.name: theta[b.start : b.stop].reshape(b.shape) for b in self.blocks}

    def matrix(self, params: dict[str, np.ndarray], name: str) -> np.ndarray:
        """Materialize a weight matrix, collapsing the low-rank factors."""
        if self.is_factorized(name):
            return params[f"{name}_u"] @ params[f"{name}_v"]
        return params[name]

    def initial_theta(self, seed: int = 0, scale: float = 1.0) -> np.ndarray:
        """Standard fan-in-scaled Gaussian init, flattened.

        ``scale`` is the Omnigrok-style ``alpha = w / w0`` knob: it multiplies
        every tensor after the usual init.
        """
        rng = np.random.default_rng(seed)
        theta = np.empty(self.n_params, dtype=np.float64)
        for b in self.blocks:
            std = scale / np.sqrt(b.fan_in)
            theta[b.start : b.stop] = rng.normal(0.0, std, size=b.size)
        return theta

    # -- forward / backward -------------------------------------------------
    def logits(self, theta: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        params = self.unpack(theta)
        E = self.matrix(params, "E")
        x = np.concatenate([E[a], E[b]], axis=1)
        z1 = x @ self.matrix(params, "W1") + params["b1"]
        h = _act(self.config.activation, z1)
        return h @ self.matrix(params, "W2") + params["b2"]

    def loss_and_grad(
        self, theta: np.ndarray, a: np.ndarray, b: np.ndarray, y: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Mean cross-entropy and its gradient w.r.t. the flat vector."""
        cfg = self.config
        params = self.unpack(theta)
        E = self.matrix(params, "E")
        W1 = self.matrix(params, "W1")
        W2 = self.matrix(params, "W2")

        x = np.concatenate([E[a], E[b]], axis=1)
        z1 = x @ W1 + params["b1"]
        h = _act(cfg.activation, z1)
        logits = h @ W2 + params["b2"]

        n = y.shape[0]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        denom = exp.sum(axis=1, keepdims=True)
        log_probs = shifted - np.log(denom)
        loss = float(-log_probs[np.arange(n), y].mean())

        dlogits = exp / denom
        dlogits[np.arange(n), y] -= 1.0
        dlogits /= n

        grad = np.zeros(self.n_params, dtype=np.float64)

        def write(name: str, value: np.ndarray) -> None:
            blk = self.index[name]
            grad[blk.start : blk.stop] = value.reshape(-1)

        def write_matrix(name: str, d_full: np.ndarray) -> None:
            """Push a gradient w.r.t. the dense matrix through the factors."""
            if self.is_factorized(name):
                u, v = params[f"{name}_u"], params[f"{name}_v"]
                write(f"{name}_u", d_full @ v.T)
                write(f"{name}_v", u.T @ d_full)
            else:
                write(name, d_full)

        write_matrix("W2", h.T @ dlogits)
        write("b2", dlogits.sum(axis=0))

        dh = dlogits @ W2.T
        dz1 = dh * _act_grad(cfg.activation, z1, h)
        write_matrix("W1", x.T @ dz1)
        write("b1", dz1.sum(axis=0))

        dx = dz1 @ W1.T
        dE = np.zeros_like(E)
        d = cfg.embed_dim
        np.add.at(dE, a, dx[:, :d])
        np.add.at(dE, b, dx[:, d:])
        write_matrix("E", dE)

        return loss, grad

    def summary(self) -> dict[str, object]:
        return {
            "n_params": self.n_params,
            "blocks": {b.name: tuple(b.shape) for b in self.blocks},
            "dense_n_params": _dense_size(self.config),
        }


def _dense_size(config: ModelConfig) -> int:
    """Parameter count the same architecture would have at full rank."""
    p, d, h = config.p, config.embed_dim, config.hidden_dim
    return p * d + 2 * d * h + h + h * p + p


def n_parameters(config: ModelConfig) -> int:
    return GrokkingMLP(config).n_params
