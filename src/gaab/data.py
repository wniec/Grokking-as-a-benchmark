"""P-modular arithmetic datasets, the substrate every benchmark problem sits on.

Mirrors the classic grokking setup (Power et al., 2022): for a prime modulus
``p`` every pair ``(a, b) in {0, ..., p-1}^2`` is an example with label
``c = (a OP b) mod p``.  A random ``train_frac`` of the ``p * p`` pairs is used
for training, the rest is held out.  The held-out fraction is the *only* source
of missing data -- generalization to it is what "grokking" means here.

Unlike the reference torch implementation this module is pure NumPy: a single
objective evaluation must be cheap enough to spend a five-figure budget on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OPS = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
}

#: Primes small enough that the induced search space stays tractable for
#: black-box optimizers.  ``p = 3`` is the floor: its table is only 9 pairs, so
#: whether it groks is marginal (screening decides per configuration), but it is
#: allowed.  ``p = 2`` is excluded -- Z_2 addition is a single XOR bit with no
#: room for a held-out split to mean anything.
PRIMES: tuple[int, ...] = (3, 5, 7, 11, 13, 17, 19, 23)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


@dataclass(frozen=True)
class ModularArithmeticData:
    """A train/test split of the full ``p x p`` modular-arithmetic table."""

    train_a: np.ndarray
    train_b: np.ndarray
    train_y: np.ndarray
    test_a: np.ndarray
    test_b: np.ndarray
    test_y: np.ndarray
    p: int
    op: str

    @property
    def n_train(self) -> int:
        return int(self.train_a.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.test_a.shape[0])


def make_dataset(
    p: int = 11,
    op: str = "add",
    train_frac: float = 0.5,
    seed: int = 0,
) -> ModularArithmeticData:
    """Build the ``a OP b = c (mod p)`` table and split it.

    Parameters
    ----------
    p:
        Prime modulus.  Non-primes are rejected: modular *addition* is what the
        benchmark is about, and the group structure that makes grokking happen
        (and makes the Fourier-feature solution exist) needs ``Z_p`` to be a
        field.
    train_frac:
        Fraction of the ``p * p`` pairs used for training.
    seed:
        Controls which pairs land in the training set.  This is the knob the
        IOH *instance* id turns.
    """
    if not is_prime(p):
        raise ValueError(f"p must be prime, got {p}.")
    if op not in OPS:
        raise ValueError(f"Unknown op {op!r}; choose from {sorted(OPS)}.")
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must lie in (0, 1), got {train_frac}.")

    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    y = OPS[op](a, b, p)

    n = p * p
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    # Both splits must be non-empty for the train/test distinction to mean
    # anything; clamp rather than silently producing a degenerate problem.
    n_train = int(np.clip(n_train, 1, n - 1))
    tr, te = perm[:n_train], perm[n_train:]

    return ModularArithmeticData(
        train_a=a[tr].astype(np.int64),
        train_b=b[tr].astype(np.int64),
        train_y=y[tr].astype(np.int64),
        test_a=a[te].astype(np.int64),
        test_b=b[te].astype(np.int64),
        test_y=y[te].astype(np.int64),
        p=p,
        op=op,
    )
