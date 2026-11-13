"""The benchmark family: candidate grids and the curated, screened suite.

``candidate_specs`` enumerates a grid over the four landscape knobs -- the prime
``p``, the network width, the low-rank ``rank``, and the regularization strength
-- and ``screening.filter_specs`` then throws away everything that does not grok.
The survivors are shipped as a JSON file so a benchmark run is reproducible
without re-screening.
"""

from __future__ import annotations

import itertools
import os
from typing import Sequence

from gaab.problem import GrokkingSuite, ProblemSpec

#: Directory holding curated (already screened) suites.
SUITES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "suites",
)

DEFAULT_SUITE_FILE = os.path.join(SUITES_DIR, "grokking_v1.json")

#: The parameter ceiling for the shipped suite.  Every problem an optimizer
#: sees is at most this many variables; the low-rank knob is what keeps a
#: p-modular-addition network under it while still grokking.
DEFAULT_MAX_VARIABLES = 200

#: Grid presets.  ``default`` is the grid the shipped suite is screened from --
#: capped at ``DEFAULT_MAX_VARIABLES`` and reaching down to ``p = 3``; ``quick``
#: is a smoke-test subset; ``full`` widens the grid but keeps the same ceiling.
#: ``large`` is the deliberate exception: it lifts the ceiling for
#: stress-testing large-scale optimizers on the grokking task itself, and is
#: never part of the default suite.  ``max_variables`` travels with the preset.
PRESETS: dict[str, dict[str, Sequence]] = {
    # p = 11 with a low-rank trunk is the reliable small grokker (n <= 187), so
    # the smoke test actually finds something to keep.
    "quick": {
        "p": (11,),
        "shape": ((2, 8), (2, 12)),
        "rank": (4, 6),
        "rank_targets": (("W1", "W2"),),
        "reg_strength": (
            1e-05,
            3e-05,
            0.0001,
            0.0003,
            0.001,
            0.003,
            0.01,
            0.02,
            0.03,
            0.05,
        ),
        "train_frac": (0.6,),
        "max_variables": DEFAULT_MAX_VARIABLES,
    },
    "default": {
        "p": (3, 5, 7, 11, 13, 17),
        "shape": (
            (2, 6),
            (2, 8),
            (2, 12),
            (3, 6),
            (3, 8),
            (4, 6),
            (4, 8),
            (4, 12),
        ),
        "rank": (None, 3, 4, 6),
        "rank_targets": (("W1",), ("W1", "W2")),
        "reg_strength": (
            1e-05,
            3e-05,
            0.0001,
            0.0003,
            0.001,
            0.003,
            0.01,
            0.02,
            0.03,
            0.05,
        ),
        # Per-p train/test split: lower p needs more training data to grok
        # (scripts/trainfrac_sweep.py). p=13 keeps both 0.6 (mid nets) and 0.8
        # (small nets), which grok at different splits.
        "train_frac": {
            3: (0.8,),
            5: (0.8,),
            7: (0.8,),
            11: (0.7,),
            13: (0.6, 0.8),
            17: (0.8,),
            "default": (0.6,),
        },
        "max_variables": DEFAULT_MAX_VARIABLES,
    },
    "full": {
        "p": (3, 5, 7, 11, 13, 17, 19, 23),
        "shape": (
            (2, 6),
            (2, 8),
            (2, 12),
            (3, 6),
            (3, 8),
            (3, 12),
            (4, 6),
            (4, 8),
            (4, 12),
            (4, 16),
        ),
        "rank": (None, 2, 3, 4, 6),
        "rank_targets": (("W1",), ("W1", "W2"), ("E", "W1")),
        "reg_strength": (
            1e-05,
            3e-05,
            0.0001,
            0.0003,
            0.001,
            0.003,
            0.01,
            0.02,
            0.03,
            0.05,
        ),
        "train_frac": {
            3: (0.7, 0.8),
            5: (0.7, 0.8),
            7: (0.7, 0.8),
            11: (0.6, 0.7, 0.8),
            13: (0.6, 0.7, 0.8),
            17: (0.6, 0.7, 0.8),
            "default": (0.6, 0.7, 0.8),
        },
        "max_variables": DEFAULT_MAX_VARIABLES,
    },
    # Above the 200-parameter ceiling ON PURPOSE: for exercising LM-CMA-ES /
    # LM-MA-ES on high-dimensional grokking problems.  Opt in explicitly; it is
    # never screened into the shipped suite.
    "large": {
        "p": (17, 23),
        "shape": ((8, 32), (8, 64), (12, 64)),
        "rank": (None, 4, 8, 16),
        "rank_targets": (("W1",), ("W1", "W2")),
        "reg_strength": (
            1e-05,
            3e-05,
            0.0001,
            0.0003,
            0.001,
            0.003,
            0.01,
            0.02,
            0.03,
            0.05,
        ),
        "train_frac": (0.6,),
        "max_variables": 4000,
    },
}

#: Keys of a preset that are architecture axes rather than grid axes.
_NON_AXIS_KEYS = frozenset({"max_variables"})


def candidate_specs(
    preset: str = "default",
    op: str = "add",
    activation: str = "quad",
    max_variables: int | None = None,
    **overrides: Sequence,
) -> list[ProblemSpec]:
    """Enumerate the unscreened grid for ``preset``.

    ``activation`` defaults to ``quad``: a quadratic activation is the one that
    makes a *small* network grok reliably (Gromov, 2023), and small is what a
    black-box optimizer needs.  ``max_variables`` defaults to the preset's own
    ceiling and drops configurations whose search space is larger.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose from {sorted(PRESETS)}.")
    grid = dict(PRESETS[preset])
    grid.update(overrides)
    if max_variables is None:
        max_variables = grid.get("max_variables")
    axes = {k: v for k, v in grid.items() if k not in _NON_AXIS_KEYS}

    # ``train_frac`` may be a flat sequence (one schedule for every prime) or a
    # per-p mapping ``{p: (fracs,), "default": (fracs,)}``.  The split that groks
    # depends on p -- a smaller prime has fewer pairs, so it needs a larger
    # fraction to keep enough training data (see scripts/trainfrac_sweep.py) --
    # so the shipped default gives lower p a higher split.
    tf_axis = axes["train_frac"]

    def fracs_for(prime: int) -> Sequence[float]:
        if isinstance(tf_axis, dict):
            return tf_axis.get(prime, tf_axis.get("default", (0.6,)))
        return tf_axis

    specs: list[ProblemSpec] = []
    seen: set[str] = set()
    for p in axes["p"]:
        for (
            embed_dim,
            hidden_dim,
        ), rank, targets, reg, train_frac in itertools.product(
            axes["shape"],
            axes["rank"],
            axes["rank_targets"],
            axes["reg_strength"],
            fracs_for(p),
        ):
            spec = ProblemSpec(
                p=p,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                rank=rank,
                rank_targets=tuple(targets),
                reg_strength=float(reg),
                train_frac=float(train_frac),
                op=op,
                activation=activation,
            )
            # Full-rank specs are identical whatever the targets say, and a rank
            # at or above a matrix's bottleneck degenerates to dense -- both
            # collapse onto a name already emitted.
            if spec.name in seen:
                continue
            if max_variables is not None and spec.n_variables > max_variables:
                continue
            seen.add(spec.name)
            specs.append(spec)

    return [s.with_id(i + 1) for i, s in enumerate(specs)]


def load_suite(
    path: str | None = None, instances: Sequence[int] | None = None
) -> GrokkingSuite:
    """Load the curated suite (default: the shipped ``grokking_v1.json``)."""
    path = path or DEFAULT_SUITE_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No curated suite at {path}. Run `python -m gaab.cli screen` first, "
            "or pass an explicit --suite path."
        )
    suite = GrokkingSuite.load(path)
    if instances is not None:
        suite.instances = list(instances)
    return suite


def candidate_suite(
    preset: str = "default", instances: Sequence[int] = (1, 2, 3), **kwargs
) -> GrokkingSuite:
    """The unscreened grid as a suite -- useful for screening it."""
    return GrokkingSuite(candidate_specs(preset, **kwargs), instances=instances)
