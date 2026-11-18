from collections.abc import Sequence

import numpy as np
import pytest
import scipy.sparse as sps
import sparse
from scipy.sparse.linalg import splu
from tensorly.tenalg import multi_mode_dot
from tensorly.tucker_tensor import TuckerTensor

from tensor_dmt.tucker.core import tucker_kron_op, tucker_sum
from tensor_dmt.tucker.gmres import RoundingSum, preconditioner, tucker_GMRES
from tensor_dmt.tucker.randomized import tucker_randomize_sum
from tests._helpers import dense_tucker, make_tucker


def deterministic_sum(
    tensors: Sequence[TuckerTensor],
    weights: Sequence[float] | np.ndarray,
    tolerance: float,
) -> TuckerTensor:
    return tucker_sum(tensors, weights, tolerance)


def make_randomized_sum(sketch: str) -> RoundingSum:
    rng = np.random.default_rng(41)

    def randomized_sum(
        tensors: Sequence[TuckerTensor],
        weights: Sequence[float] | np.ndarray,
        tolerance: float,
    ) -> TuckerTensor:
        return tucker_randomize_sum(
            tensors,
            weights,
            p=4,
            tol=tolerance,
            sketch=sketch,
            rng=rng,
            strategy="heur_coarse",
        )

    return randomized_sum


@pytest.mark.parametrize("sparse_core", (False, True))
def test_preconditioner_solves_the_first_factor(sparse_core: bool) -> None:
    tensor = make_tucker(
        shape=(4, 3),
        rank=(2, 2),
        seed=39,
        sparse_core=sparse_core,
    )
    matrix = sps.diags([1.0, 2.0, 3.0, 4.0], format="csc")

    result = preconditioner(splu(matrix), tensor)
    expected_factors = list(tensor.factors)
    expected_factors[0] = np.linalg.solve(matrix.toarray(), tensor.factors[0])
    expected = TuckerTensor((sparse.asnumpy(tensor.core), expected_factors))

    np.testing.assert_allclose(dense_tucker(result), dense_tucker(expected))
    assert isinstance(result.core, sparse.COO) is sparse_core


@pytest.mark.parametrize(
    "rounding_sum",
    (
        deterministic_sum,
        make_randomized_sum("kron"),
        make_randomized_sum("krp"),
    ),
    ids=("deterministic", "kron", "krp"),
)
def test_gmres_solves_a_small_separable_system(rounding_sum: RoundingSum) -> None:
    diagonal_1 = np.diag([1.0, 1.5, 2.0, 3.0])
    diagonal_2 = np.diag([0.75, 1.25, 2.5])
    rhs = make_tucker(
        shape=(4, 3),
        rank=(2, 2),
        seed=40,
        orthogonal_factors=True,
    )

    def operator(tensor: TuckerTensor) -> Sequence[TuckerTensor]:
        return (tucker_kron_op((diagonal_1, diagonal_2), tensor),)

    result = tucker_GMRES(
        operator,
        rhs,
        tol=1e-10,
        max_iter=12,
        prec=lambda tensor: tensor,
        rounding_tol=1e-12,
        rounding_sum=rounding_sum,
    )

    solution = dense_tucker(result.solution)
    residual = multi_mode_dot(solution, (diagonal_1, diagonal_2)) - dense_tucker(rhs)
    relative_residual = np.linalg.norm(residual) / np.linalg.norm(dense_tucker(rhs))

    assert relative_residual < 1e-8
    assert result.ranks[-1] == result.solution.rank
    assert result.timer.total >= 0.0
