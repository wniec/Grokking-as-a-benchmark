import numpy as np
import pytest

from gaab.data import make_dataset
from gaab.model import ACTIVATIONS, GrokkingMLP, ModelConfig, _dense_size, n_parameters


def test_full_rank_parameter_count():
    cfg = ModelConfig(p=11, embed_dim=8, hidden_dim=32)
    assert n_parameters(cfg) == _dense_size(cfg) == 11 * 8 + 16 * 32 + 32 + 32 * 11 + 11


def test_low_rank_shrinks_when_rank_is_below_the_bottleneck():
    dense = n_parameters(ModelConfig(p=11, embed_dim=8, hidden_dim=32))
    small = n_parameters(
        ModelConfig(p=11, embed_dim=8, hidden_dim=32, rank=2, rank_targets=("W1", "W2"))
    )
    assert small < dense


def test_rank_at_or_above_the_bottleneck_degenerates_to_dense():
    cfg = ModelConfig(
        p=11, embed_dim=8, hidden_dim=32, rank=64, rank_targets=("W1", "W2")
    )
    model = GrokkingMLP(cfg)
    assert model.n_params == _dense_size(cfg)
    assert not model.is_factorized("W1")


def test_low_rank_can_also_grow_the_parameter_count():
    # W1 is (2 * embed_dim, hidden) = (4, 4): a rank-3 factorization costs
    # (4 + 4) * 3 = 24 parameters where the dense matrix costs 16.  "possibly,
    # but not necessarily" fewer parameters.
    cfg = ModelConfig(p=3, embed_dim=2, hidden_dim=4, rank=3, rank_targets=("W1",))
    model = GrokkingMLP(cfg)
    assert model.is_factorized("W1")
    assert model.index["W1_u"].size + model.index["W1_v"].size > 4 * 4
    assert model.n_params > _dense_size(cfg)


def test_rank_targets_select_which_matrices_are_factorized():
    for targets, factorized in [
        ((), set()),
        (("W1",), {"W1"}),
        (("W1", "W2"), {"W1", "W2"}),
        (("E", "W1"), {"E", "W1"}),
    ]:
        model = GrokkingMLP(
            ModelConfig(p=11, embed_dim=8, hidden_dim=32, rank=2, rank_targets=targets)
        )
        assert {m for m in ("E", "W1", "W2") if model.is_factorized(m)} == factorized


def test_rank_targets_are_normalized_and_validated():
    # Order is canonical, so ("W1", "E") and ("E", "W1") are the same landscape.
    assert ModelConfig(p=7, rank_targets=("W1", "E")).rank_targets == ("E", "W1")
    with pytest.raises(ValueError):
        ModelConfig(p=7, rank_targets=("W3",))


def test_restricting_the_readout_costs_more_than_restricting_the_trunk():
    """W2 is the p-way readout; W1 is the one that tolerates a bottleneck."""
    base = dict(p=17, embed_dim=4, hidden_dim=16, rank=4)
    trunk = GrokkingMLP(ModelConfig(**base, rank_targets=("W1",)))
    both = GrokkingMLP(ModelConfig(**base, rank_targets=("W1", "W2")))
    assert both.n_params < trunk.n_params
    # The readout's rank caps how many classes can be separated at all.
    params = both.unpack(both.initial_theta(0))
    assert np.linalg.matrix_rank(both.matrix(params, "W2")) <= 4


def test_factorized_matrix_materializes_at_the_right_shape():
    model = GrokkingMLP(
        ModelConfig(p=11, embed_dim=8, hidden_dim=32, rank=3, rank_targets=("W1", "W2"))
    )
    params = model.unpack(model.initial_theta(0))
    assert model.matrix(params, "W1").shape == (16, 32)
    assert np.linalg.matrix_rank(model.matrix(params, "W1")) <= 3


def test_unpack_rejects_wrong_length():
    model = GrokkingMLP(ModelConfig(p=7))
    with pytest.raises(ValueError):
        model.unpack(np.zeros(model.n_params + 1))


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize(
    "rank,targets",
    [(None, ("W1",)), (2, ("W1",)), (2, ("W1", "W2")), (2, ("E", "W1", "W2"))],
)
def test_gradient_matches_finite_differences(activation, rank, targets):
    cfg = ModelConfig(
        p=7,
        embed_dim=4,
        hidden_dim=8,
        rank=rank,
        activation=activation,
        rank_targets=targets,
    )
    model = GrokkingMLP(cfg)
    data = make_dataset(p=7, train_frac=0.5, seed=1)
    theta = model.initial_theta(seed=3)
    args = (data.train_a, data.train_b, data.train_y)
    _, grad = model.loss_and_grad(theta, *args)

    rng = np.random.default_rng(0)
    eps = 1e-6
    for i in rng.choice(model.n_params, 8, replace=False):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += eps
        minus[i] -= eps
        numeric = (
            model.loss_and_grad(plus, *args)[0] - model.loss_and_grad(minus, *args)[0]
        ) / (2 * eps)
        assert numeric == pytest.approx(grad[i], abs=1e-6)


def test_init_scale_rescales_the_weights():
    model = GrokkingMLP(ModelConfig(p=7))
    small = np.linalg.norm(model.initial_theta(0, scale=1.0))
    large = np.linalg.norm(model.initial_theta(0, scale=4.0))
    assert large == pytest.approx(4.0 * small, rel=1e-9)


def test_unknown_activation_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(p=7, activation="swish")
