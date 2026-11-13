import numpy as np
import pytest

from gaab.data import OPS, is_prime, make_dataset


def test_rejects_composite_modulus():
    with pytest.raises(ValueError):
        make_dataset(p=9)


def test_is_prime():
    assert [n for n in range(2, 20) if is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]
    assert not is_prime(1) and not is_prime(0) and not is_prime(-7)


def test_split_covers_every_pair_exactly_once():
    d = make_dataset(p=11, train_frac=0.6, seed=3)
    assert d.n_train + d.n_test == 11 * 11
    pairs = set(zip(d.train_a.tolist(), d.train_b.tolist()))
    pairs |= set(zip(d.test_a.tolist(), d.test_b.tolist()))
    assert len(pairs) == 11 * 11


def test_labels_are_modular_addition():
    d = make_dataset(p=13, seed=0)
    assert np.array_equal(d.train_y, (d.train_a + d.train_b) % 13)
    assert np.array_equal(d.test_y, (d.test_a + d.test_b) % 13)


@pytest.mark.parametrize("op", sorted(OPS))
def test_other_ops(op):
    d = make_dataset(p=7, op=op, seed=1)
    assert np.array_equal(d.test_y, OPS[op](d.test_a, d.test_b, 7))


def test_seed_controls_the_split():
    a = make_dataset(p=11, seed=1)
    b = make_dataset(p=11, seed=2)
    c = make_dataset(p=11, seed=1)
    assert np.array_equal(a.train_a, c.train_a)
    assert not np.array_equal(a.train_a, b.train_a)


def test_both_splits_stay_non_empty():
    d = make_dataset(p=5, train_frac=0.001)
    assert d.n_train >= 1 and d.n_test >= 1
