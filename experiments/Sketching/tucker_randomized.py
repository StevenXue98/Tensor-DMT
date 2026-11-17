import sparse
import numpy as np
import tensorly as tl
from numpy.typing import NDArray
from tucker_core import tucker_scale
from collections.abc import Sequence
from tensorly.tucker_tensor import TuckerTensor
from tensorly.tenalg.core_tenalg import mode_dot, multi_mode_dot, unfolding_dot_khatri_rao

__all__ = [
    "tucker_subranks_kron",
    "tucker_randomize_sum",
]



def tucker_subranks_kron(
        effective_ranks: Sequence[int] | NDArray[np.int_],
        oversampling: int
    ) -> NDArray[np.int_]:

    if len(effective_ranks) < 2:
        raise ValueError("Tensor has to have at least two modes.")

    #==========================================================================
    #   Calculate Subranks via heuristic
    #==========================================================================
    d                 = len(effective_ranks)
    oversampled_ranks = np.asarray(effective_ranks) + oversampling
    product_root      = np.prod(oversampled_ranks) ** (1 / (d - 1))
    subranks          = np.ceil(product_root / oversampled_ranks).astype(np.int_)
    return subranks


def tucker_randomize_sum(
        W: Sequence[TuckerTensor],
        a: Sequence[float] | NDArray[np.float64],
        p: int,
        tol: float,
        sketch: str,
        rng: np.random.Generator,
        strategy: str = "heur_coarse",
        *,
        orthogonalize: bool = False,
    ) -> TuckerTensor:
    #==========================================================================
    #   Input Validation
    #==========================================================================
    if len(W) != len(a):
        raise ValueError(
            f"Size mismatch: 'W' has length {len(W)}, "
            f"but 'a' has length {len(a)}."
        )

    if len(W) == 0:
        raise ValueError("'W' and 'a' cannot be empty.")

    if not all(isinstance(W[i], TuckerTensor) for i in range(len(W))):
        raise TypeError("W must contain only TuckerTensor objects.")

    shape = W[0].shape
    if any(tensor.shape != shape for tensor in W[1:]):
        raise ValueError("All Tucker tensors must have the same shape.")

    if p < 0:
        raise ValueError("'p' has to be non-negative.")

    if tol < 0:
        raise ValueError("'tol' has to be non-negative.")

    if sketch not in {"kron", "krp"}:
        raise ValueError("sketch must be 'kron' or 'krp'.")

    if strategy not in {"heur_coarse", "heur_fine"}:
        raise ValueError("strategy must be 'heur_coarse' or 'heur_fine'.")

    #==========================================================================
    #   Setup
    #==========================================================================
    num_tensors = len(W)
    dim         = len(W[0].factors)

    Tensors     = []
    CoreTensors = []
    weights     = np.zeros(num_tensors)
    n           = np.asarray(W[0].shape, dtype=np.int_)

    for i in range(num_tensors):
        scaled_tensor = tucker_scale(W[i], a[i])
        core = sparse.asnumpy(scaled_tensor.core)

        if orthogonalize and strategy == "heur_fine":
            factors = list(scaled_tensor.factors)
            for mode in range(dim):
                Q, R = np.linalg.qr(factors[mode], mode="reduced")
                factors[mode] = Q
                core = mode_dot(core, R, mode=mode)
            scaled_tensor = TuckerTensor((core, factors))

        Tensors.append(scaled_tensor)
        CoreTensors.append(core)

        if strategy == "heur_coarse":
            weights[i] = np.linalg.norm(core.ravel())

    #==========================================================================
    #   Phase 1: Rank Computation
    #==========================================================================
    effective_ranks = np.zeros(dim, dtype=np.int_)

    for d in range(dim):
        blocks = []

        if strategy == "heur_coarse":
            # Strategy: Explicit Concatenation
            blocks = [weights[i] * Tensors[i].factors[d] for i in range(num_tensors)]

        elif strategy == "heur_fine":
            # Strategy: Directional Core SVD (Assuming Orthogonal Factors)
            blocks = []
            for i in range(num_tensors):
                G_unfolded = tl.unfold(CoreTensors[i], d)
                L, S, _    = np.linalg.svd(G_unfolded, full_matrices=False)
                blocks.append((Tensors[i].factors[d] @ L) * S)

        V_d = np.concatenate(blocks, axis=1)
        B_d = V_d.T @ V_d

        evs = np.linalg.eigvalsh(B_d)
        evs = np.maximum(evs, 0.0)

        eigsum_p1 = np.cumsum(evs)
        thresh_p1 = (tol ** 2 * eigsum_p1[-1]) / dim

        effective_ranks[d] = max(1, len(eigsum_p1) - np.searchsorted(eigsum_p1, thresh_p1, side='right'))

    #==========================================================================
    #   Phase 2: Subrank Selection
    #==========================================================================
    if sketch == "kron":
        sketch_ranks = tucker_subranks_kron(effective_ranks, p)
    else:
        sketch_ranks = np.full(dim, np.max(effective_ranks) + p, dtype=np.int_)

    #==========================================================================
    #   Phase 3: Kronecker/KRP Sketching
    #==========================================================================
    Omega         = [rng.standard_normal((n[d], sketch_ranks[d])) for d in range(dim)]
    mode_sketches = []

    for t in range(num_tensors):
        sketched_factors = []

        for d in range(dim):
            sketched_factor = Omega[d].T @ Tensors[t].factors[d]
            sketched_factors.append(sketched_factor if sketch=="kron" else sketched_factor.T)

        for d in range(dim):
            if sketch == "kron":
                Y = multi_mode_dot(CoreTensors[t], sketched_factors, skip=d)
                mode_sketch = Tensors[t].factors[d] @ tl.unfold(Y, d)
            else:
                Y = unfolding_dot_khatri_rao(CoreTensors[t], (None, sketched_factors), d)
                mode_sketch = Tensors[t].factors[d] @ Y

            if t == 0:
                mode_sketches.append(mode_sketch)
            else:
                mode_sketches[d] += mode_sketch

    #==========================================================================
    #   Phase 4: Randomized Range Finder
    #==========================================================================
    U_hat = []
    for d in range(dim):
        Q_hat, _ = np.linalg.qr(mode_sketches[d], mode='reduced')
        U_hat.append(Q_hat)

    #==========================================================================
    #   Phase 5: Form Intermediate Core
    #==========================================================================
    core_sz = tuple(U.shape[1] for U in U_hat)
    G_hat   = np.zeros(core_sz)

    for t in range(num_tensors):
        W_proj = [U_hat[d].T @ Tensors[t].factors[d] for d in range(dim)]
        G_hat += multi_mode_dot(CoreTensors[t], W_proj)

    FinalCore = G_hat

    #==========================================================================
    #   Phase 6: ST-HOSVD (Core Truncation)
    #==========================================================================
    norm_core_sq = np.linalg.norm(FinalCore.ravel()) ** 2
    eigsumthresh = (tol ** 2 * norm_core_sq) / dim

    V = []
    for j in range(dim):
        V_j, s, _ = np.linalg.svd(tl.unfold(FinalCore, j), full_matrices=False)
        eigsum    = np.cumsum((s ** 2)[::-1])
        r_j       = max(1, len(eigsum) - np.searchsorted(eigsum, eigsumthresh, side='right'))
        V_j       = V_j[:, :r_j]
        FinalCore = mode_dot(FinalCore, V_j.T, mode=j)
        V.append(V_j)

    #==========================================================================
    #   Phase 7: Factor Update
    #==========================================================================
    Ui_final = [U_hat[d] @ V[d] for d in range(dim)]
    Xapprox = TuckerTensor((FinalCore, Ui_final))

    return Xapprox

