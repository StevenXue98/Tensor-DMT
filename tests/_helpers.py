from collections.abc import Sequence

import numpy as np
import sparse
import tensorly as tl
from tensorly.tucker_tensor import TuckerTensor


def make_tucker(
    shape: Sequence[int] = (4, 5, 3),
    rank: Sequence[int] = (2, 3, 2),
    *,
    seed: int = 0,
    sparse_core: bool = False,
    orthogonal_factors: bool = False,
) -> TuckerTensor:
    rng = np.random.default_rng(seed)
    core = rng.standard_normal(tuple(rank))
    core.ravel()[::3] = 0.0

    factors = []
    for mode_size, mode_rank in zip(shape, rank, strict=True):
        factor = rng.standard_normal((mode_size, mode_rank))
        if orthogonal_factors:
            factor, _ = np.linalg.qr(factor, mode="reduced")
        factors.append(factor)

    if sparse_core:
        core = sparse.COO.from_numpy(core)

    return TuckerTensor((core, factors))


def dense_tucker(tensor: TuckerTensor) -> np.ndarray:
    core = sparse.asnumpy(tensor.core)
    return tl.tucker_to_tensor((core, tensor.factors))


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.linalg.norm(expected)
    if denominator == 0:
        return float(np.linalg.norm(actual - expected))
    return float(np.linalg.norm(actual - expected) / denominator)
