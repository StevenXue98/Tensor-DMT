from itertools import product

import numpy as np
import pytest
import scipy.sparse as sps
import sparse
from tensorly.tenalg import multi_mode_dot

from tensor_dmt.tucker.core import (
    tucker_axby,
    tucker_inner_prod,
    tucker_kron_op,
    tucker_norm,
    tucker_rounding,
    tucker_scale,
    tucker_sum,
    tucker_summands_kron_op,
)
from tests._helpers import dense_tucker, make_tucker, relative_error


CORE_STORAGE_PAIRS = list(product((False, True), repeat=2))


@pytest.mark.parametrize(("x_sparse", "y_sparse"), CORE_STORAGE_PAIRS)
def test_inner_product_and_norm_match_dense_ground_truth(
    x_sparse: bool,
    y_sparse: bool,
) -> None:
    x = make_tucker(seed=1, sparse_core=x_sparse)
    y = make_tucker(seed=2, sparse_core=y_sparse)
    x_dense = dense_tucker(x)
    y_dense = dense_tucker(y)

    assert tucker_inner_prod(x, y) == pytest.approx(np.vdot(x_dense, y_dense))
    assert tucker_norm(x) == pytest.approx(np.linalg.norm(x_dense))
    assert tucker_norm(y) == pytest.approx(np.linalg.norm(y_dense))


@pytest.mark.parametrize(("x_sparse", "y_sparse"), CORE_STORAGE_PAIRS)
def test_inner_product_rank_swap_matches_dense_ground_truth(
    x_sparse: bool,
    y_sparse: bool,
) -> None:
    x = make_tucker(rank=(3, 3, 2), seed=15, sparse_core=x_sparse)
    y = make_tucker(rank=(1, 2, 1), seed=16, sparse_core=y_sparse)

    assert tucker_inner_prod(x, y) == pytest.approx(
        np.vdot(dense_tucker(x), dense_tucker(y))
    )


@pytest.mark.parametrize("sparse_core", (False, True))
def test_scale_matches_dense_and_preserves_core_storage(sparse_core: bool) -> None:
    tensor = make_tucker(seed=3, sparse_core=sparse_core)
    scaled = tucker_scale(tensor, -2.5)

    np.testing.assert_allclose(dense_tucker(scaled), -2.5 * dense_tucker(tensor))
    assert isinstance(scaled.core, sparse.COO) is sparse_core


@pytest.mark.parametrize("sparse_core", (False, True))
@pytest.mark.parametrize("sparse_operators", (False, True))
def test_kron_operator_matches_dense_ground_truth_and_preserves_core(
    sparse_core: bool,
    sparse_operators: bool,
) -> None:
    tensor = make_tucker(seed=4, sparse_core=sparse_core)
    rng = np.random.default_rng(5)
    dense_operators = [rng.standard_normal((n, n)) for n in tensor.shape]
    operators = (
        [sps.csr_matrix(operator) for operator in dense_operators]
        if sparse_operators
        else dense_operators
    )

    result = tucker_kron_op(operators, tensor)
    expected = multi_mode_dot(dense_tucker(tensor), dense_operators)

    np.testing.assert_allclose(dense_tucker(result), expected)
    assert isinstance(result.core, sparse.COO) is sparse_core


def test_summands_kron_operator_returns_each_term() -> None:
    tensor = make_tucker(seed=6, sparse_core=True)
    identities = [np.eye(n) for n in tensor.shape]
    scalings = [2.0 * np.eye(n) for n in tensor.shape]

    results = tucker_summands_kron_op((identities, scalings), tensor)

    assert len(results) == 2
    np.testing.assert_allclose(dense_tucker(results[0]), dense_tucker(tensor))
    np.testing.assert_allclose(
        dense_tucker(results[1]),
        (2.0 ** len(tensor.shape)) * dense_tucker(tensor),
    )


@pytest.mark.parametrize(("x_sparse", "y_sparse"), CORE_STORAGE_PAIRS)
def test_axby_is_exact_and_returns_sparse_core(
    x_sparse: bool,
    y_sparse: bool,
) -> None:
    x = make_tucker(rank=(2, 2, 2), seed=7, sparse_core=x_sparse)
    y = make_tucker(rank=(1, 3, 2), seed=8, sparse_core=y_sparse)

    result = tucker_axby(1.25, x, -0.75, y)
    expected = 1.25 * dense_tucker(x) - 0.75 * dense_tucker(y)

    assert isinstance(result.core, sparse.COO)
    np.testing.assert_allclose(dense_tucker(result), expected)


@pytest.mark.parametrize("sparse_core", (False, True))
def test_rounding_matches_input_and_returns_dense_core(sparse_core: bool) -> None:
    tensor = make_tucker(seed=9, sparse_core=sparse_core)

    rounded = tucker_rounding(tensor, 1e-12)

    assert isinstance(rounded.core, np.ndarray)
    assert relative_error(dense_tucker(rounded), dense_tucker(tensor)) < 1e-11


def test_rounding_truncates_a_known_small_component() -> None:
    core = np.zeros((2, 2, 2))
    core[0, 0, 0] = 1.0
    core[1, 1, 1] = 1e-8
    tensor = make_tucker(
        shape=(3, 3, 3),
        rank=(2, 2, 2),
        seed=10,
        orthogonal_factors=True,
    )
    tensor = type(tensor)((core, tensor.factors))

    rounded = tucker_rounding(tensor, 1e-6)

    assert rounded.rank == (1, 1, 1)
    assert relative_error(dense_tucker(rounded), dense_tucker(tensor)) <= 1e-6


@pytest.mark.parametrize(("x_sparse", "y_sparse"), CORE_STORAGE_PAIRS)
def test_sum_matches_weighted_dense_sum(
    x_sparse: bool,
    y_sparse: bool,
) -> None:
    x = make_tucker(rank=(2, 2, 2), seed=11, sparse_core=x_sparse)
    y = make_tucker(rank=(1, 3, 2), seed=12, sparse_core=y_sparse)

    result = tucker_sum((x, y), np.array([0.5, -1.5]), 1e-12)
    expected = 0.5 * dense_tucker(x) - 1.5 * dense_tucker(y)

    assert isinstance(result.core, np.ndarray)
    assert relative_error(dense_tucker(result), expected) < 1e-11


def test_shape_and_weight_validation() -> None:
    x = make_tucker(shape=(4, 5, 3), seed=13)
    y = make_tucker(shape=(4, 5, 4), seed=14)

    with pytest.raises(ValueError, match="Size mismatch"):
        tucker_inner_prod(x, y)

    with pytest.raises(ValueError, match="Size mismatch"):
        tucker_kron_op((np.eye(4),), x)

    with pytest.raises(ValueError, match="Size mismatch"):
        tucker_sum((x, y), (1.0,), 1e-8)
