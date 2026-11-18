import numpy as np
import pytest

from tensor_dmt.tucker.randomized import tucker_randomize_sum
from tests._helpers import dense_tucker, make_tucker, relative_error


@pytest.mark.parametrize("sketch", ("kron", "krp"))
@pytest.mark.parametrize("strategy", ("heur_coarse", "heur_fine"))
@pytest.mark.parametrize("sparse_inputs", (False, True))
def test_randomized_sum_matches_small_dense_sum(
    sketch: str,
    strategy: str,
    sparse_inputs: bool,
) -> None:
    tensors = (
        make_tucker(
            shape=(6, 5, 4),
            rank=(2, 2, 2),
            seed=20,
            sparse_core=sparse_inputs,
            orthogonal_factors=True,
        ),
        make_tucker(
            shape=(6, 5, 4),
            rank=(3, 1, 2),
            seed=21,
            sparse_core=sparse_inputs,
            orthogonal_factors=True,
        ),
    )
    weights = np.array([1.25, -0.5])
    expected = sum(
        weight * dense_tucker(tensor)
        for weight, tensor in zip(weights, tensors, strict=True)
    )

    result = tucker_randomize_sum(
        tensors,
        weights,
        p=4,
        tol=1e-10,
        sketch=sketch,
        rng=np.random.default_rng(22),
        strategy=strategy,
    )

    assert result.shape == tensors[0].shape
    assert isinstance(result.core, np.ndarray)
    assert np.all(np.isfinite(dense_tucker(result)))
    assert relative_error(dense_tucker(result), expected) < 1e-8


@pytest.mark.parametrize("sketch", ("kron", "krp"))
def test_fine_strategy_can_orthogonalize_nonorthogonal_inputs(sketch: str) -> None:
    tensors = (
        make_tucker(shape=(6, 5, 4), rank=(2, 2, 2), seed=23),
        make_tucker(shape=(6, 5, 4), rank=(2, 3, 1), seed=24),
    )
    weights = np.array([0.75, -1.5])
    expected = sum(
        weight * dense_tucker(tensor)
        for weight, tensor in zip(weights, tensors, strict=True)
    )

    result = tucker_randomize_sum(
        tensors,
        weights,
        p=4,
        tol=1e-10,
        sketch=sketch,
        rng=np.random.default_rng(25),
        strategy="heur_fine",
        orthogonalize=True,
    )

    assert relative_error(dense_tucker(result), expected) < 1e-8


@pytest.mark.parametrize("sketch", ("kron", "krp"))
def test_randomized_sum_is_reproducible_for_a_fixed_seed(sketch: str) -> None:
    tensors = (
        make_tucker(shape=(5, 4, 3), rank=(2, 2, 2), seed=26),
        make_tucker(shape=(5, 4, 3), rank=(1, 2, 2), seed=27),
    )
    weights = (1.0, -0.25)

    first = tucker_randomize_sum(
        tensors,
        weights,
        p=3,
        tol=1e-8,
        sketch=sketch,
        rng=np.random.default_rng(28),
        strategy="heur_coarse",
    )
    second = tucker_randomize_sum(
        tensors,
        weights,
        p=3,
        tol=1e-8,
        sketch=sketch,
        rng=np.random.default_rng(28),
        strategy="heur_coarse",
    )

    np.testing.assert_allclose(dense_tucker(first), dense_tucker(second))


@pytest.mark.parametrize("sketch", ("kron", "krp"))
def test_zero_weight_does_not_change_randomized_sum(sketch: str) -> None:
    retained = make_tucker(
        shape=(5, 4, 3),
        rank=(2, 2, 2),
        seed=29,
        sparse_core=True,
        orthogonal_factors=True,
    )
    ignored = make_tucker(
        shape=(5, 4, 3),
        rank=(3, 1, 2),
        seed=30,
        sparse_core=True,
        orthogonal_factors=True,
    )

    result = tucker_randomize_sum(
        (retained, ignored),
        (1.0, 0.0),
        p=4,
        tol=1e-10,
        sketch=sketch,
        rng=np.random.default_rng(31),
        strategy="heur_coarse",
    )

    assert relative_error(dense_tucker(result), dense_tucker(retained)) < 1e-8


def test_randomized_sum_validates_inputs() -> None:
    tensor = make_tucker(seed=32)

    with pytest.raises(ValueError, match="Size mismatch"):
        tucker_randomize_sum(
            (tensor,),
            (1.0, 2.0),
            p=2,
            tol=1e-8,
            sketch="kron",
            rng=np.random.default_rng(33),
        )

    with pytest.raises(ValueError, match="sketch"):
        tucker_randomize_sum(
            (tensor,),
            (1.0,),
            p=2,
            tol=1e-8,
            sketch="invalid",
            rng=np.random.default_rng(33),
        )
