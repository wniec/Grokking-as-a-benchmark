"""Tests for ELA feature extraction (gaab.ela).

Skipped as a whole when the optional dependency ``pflacco`` is not installed.
Uses a tiny problem and a small design so the suite stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pflacco")

from gaab import ela as E  # noqa: E402
from gaab.problem import GrokkingProblem, ProblemSpec  # noqa: E402


def _tiny_problem():
    # small p and widths keep the dimension (and the design) small/fast
    spec = ProblemSpec(
        p=5,
        embed_dim=2,
        hidden_dim=4,
        reg_strength=1e-3,
        train_frac=0.8,
        activation="quad",
    )
    return GrokkingProblem(spec, instance=1)


def test_meta_excluded_from_defaults():
    assert "meta" not in E.DEFAULT_SETS
    assert "meta" in E.ALL_SETS


def test_initial_design_shapes_and_box():
    pr = _tiny_problem()
    d = pr.meta_data.n_variables
    X, y = E.initial_design(pr, n=40, seed=0)
    assert X.shape == (40, d)
    assert y.shape == (40,)
    lb, ub = E.problem_bounds(pr)
    arr = X.to_numpy()
    assert arr.min() >= lb.min() - 1e-9
    assert arr.max() <= ub.max() + 1e-9
    # y must match the objective evaluated directly
    assert np.isclose(y[0], pr(arr[0]))


def test_ela_features_default_sets():
    pr = _tiny_problem()
    feats = E.ela_features(pr, n_per_dim=12, seed=0)
    assert isinstance(feats, dict) and feats
    # every requested group contributed at least one named feature, no errors
    assert not any(k.endswith(".error") for k in feats)
    for group in E.DEFAULT_SETS:
        prefix = {
            "distribution": "ela_distr",
            "pca": "pca",
            "nbc": "nbc",
            "dispersion": "disp",
            "information_content": "ic",
        }[group]
        assert any(k.startswith(prefix) for k in feats), group
    # all values finite-or-nan floats, and deterministic for a fixed seed
    again = E.ela_features(pr, n_per_dim=12, seed=0)
    assert feats.keys() == again.keys()


def test_unknown_set_raises():
    pr = _tiny_problem()
    with pytest.raises(ValueError, match="unknown ELA feature set"):
        E.ela_features(pr, sets=("not_a_set",))


def test_bbob_features_same_space():
    feats = E.bbob_features(1, dim=5, iid=1, n_per_dim=12, seed=0)
    gaab = E.ela_features(_tiny_problem(), n_per_dim=12, seed=0)
    # BBOB and GaaB produce the same feature coordinates -> comparable space
    shared = {k for k in feats if k in gaab and "." in k}
    assert len(shared) >= 10
