import os
os.environ["SPARSE_AUTO_DENSIFY"] = "0"

import sparse
import numpy as np
import tensorly as tl
import scipy.sparse as sps
from numpy.typing import NDArray
from collections.abc import Sequence
from tensorly.tucker_tensor import TuckerTensor
from tensorly.tenalg.core_tenalg import mode_dot

DenseMatrix = NDArray[np.float64]
OperatorMatrix = DenseMatrix | sps.spmatrix | sps.sparray

__all__ = [
    "DenseMatrix",
    "OperatorMatrix",
    "tucker_scale",
    "tucker_inner_prod",
    "tucker_norm",
    "tucker_kron_op",
    "tucker_summands_kron_op",
    "tucker_rounding",
    "tucker_axby",
    "tucker_sum",
]



def tucker_scale(Y: TuckerTensor, alpha: float) -> TuckerTensor:
    #==========================================================================
    #   Scale Tensor
    #==========================================================================
    new_core = alpha * Y.core
    new_factors = list(Y.factors)
    X = TuckerTensor((new_core, new_factors))

    return X


def tucker_inner_prod(X: TuckerTensor, Y:TuckerTensor) -> float:
    #==========================================================================
    #   Input Validation
    #==========================================================================
    if X.shape != Y.shape:
        raise ValueError(
            f"Size mismatch: Tucker tensor 'X' has shape {X.shape}, "
            f"but Tucker tensor 'Y' has shape {Y.shape}."
        )
    #==========================================================================
    #   Compute Inner Product
    #==========================================================================
    if np.prod(X.rank) > np.prod(Y.rank):
        X, Y = Y, X

    W = [X.factors[i].T @ Y.factors[i] for i in range(len(X.rank))]

    J = tl.tucker_to_tensor((Y.core, W))

    if isinstance(X.core, sparse.COO):
        return np.vdot(
            X.core.data,
            J[tuple(X.core.coords)],
        )

    return np.vdot(X.core, J)


def tucker_norm(X: TuckerTensor) -> float:
    return np.sqrt(tucker_inner_prod(X, X))


def tucker_kron_op(A: Sequence[OperatorMatrix], Y: TuckerTensor) -> TuckerTensor:
    #==========================================================================
    #   Input Validation
    #==========================================================================
    N           = len(A)
    num_modes_Y = len(Y.factors)
    if N != num_modes_Y:
        raise ValueError(
            f"Size mismatch: 'A' has {len(A)} operator matrices, "
            f"but Tucker tensor 'Y' has {num_modes_Y} modes."
        )
    #==========================================================================
    #   Apply Operator
    #==========================================================================
    X_factors = [A[i] @ Y.factors[i] for i in range(N)]

    X = TuckerTensor((Y.core, X_factors))

    return X


def tucker_summands_kron_op(A: Sequence[Sequence[OperatorMatrix]], Y: TuckerTensor) -> Sequence[TuckerTensor]:
    #==========================================================================
    #   Apply Mode-Specific Operators
    #==========================================================================
    return [tucker_kron_op(A[i], Y) for i in range(len(A))]


def tucker_rounding(Y: TuckerTensor, tol: float):
    #==========================================================================
    #   Setup
    #==========================================================================
    U = list(Y.factors)
    N = len(U)
    
    # EXPLICITLY DENSIFY THE CORE:
    # If Y came from TuckerAxby, its core is an sptensor. The subsequent 
    # ttm operations with dense R matrices will destroy sparsity anyway. 
    # Converting to a dense tensor immediately avoids massive overhead.
    core = sparse.asnumpy(Y.core)
    
    #==========================================================================
    #   Step 1: Orthogonalize factors via QR
    #==========================================================================
    for i in range(N):
        Q, R = np.linalg.qr(U[i])
        U[i] = Q
        core = mode_dot(core, R, i)
    
    #==========================================================================
    #   Step 2: Truncate core via SVD
    #==========================================================================
    
    # PRE-CALCULATE ERROR BUDGET: 
    # The total allowable Frobenius norm error is split equally among the N modes.
    norm_core_sq = np.linalg.norm(core)**2
    eigsumthresh = (tol**2 * norm_core_sq) / N
    
    for i in range(N):
        core_i      = tl.unfold(core, i)
        V, s, _     = np.linalg.svd(core_i, full_matrices=False)
        s_sq        = s**2
        eigsum      = np.cumsum(s_sq[::-1])
        rank_i      = max(1, len(eigsum) - np.searchsorted(eigsum, eigsumthresh, side='right'))
        V_trunc     = V[:, :rank_i]
        U[i]        = U[i] @ V_trunc
        core        = mode_dot(core, V_trunc.T, i)
    
    X = TuckerTensor((core, U))

    return X


def tucker_axby(alpha: float, X: TuckerTensor, beta: float, Y: TuckerTensor, tol=None):
    #==========================================================================
    #   Scale inputs
    #==========================================================================
    X  = tucker_scale(X, alpha)
    Y  = tucker_scale(Y, beta)

    G1 = X.core
    G2 = Y.core
    U1 = X.factors
    U2 = Y.factors
    
    #==========================================================================
    #   Concatenate factor matrices
    #==========================================================================
    U_new = [np.concatenate((U1[i], U2[i]), axis=1) for i in range(len(U1))]
    
    #==========================================================================
    #   Construct the sparse core tensor
    #==========================================================================
    spG1 = sparse.COO(G1)
    spG2 = sparse.COO(G2)
    
    offset = np.asarray(G1.shape)[:, None]

    coords = np.concatenate([spG1.coords, spG2.coords + offset], axis=1)

    data = np.concatenate([spG1.data, spG2.data])
    new_shape = tuple(np.add(G1.shape, G2.shape))

    G_new = sparse.COO(coords, data, shape=new_shape)
    Z = TuckerTensor((G_new, U_new))
    
    #==========================================================================
    #   Truncate if tolerance is provided (Eager Rounding)
    #==========================================================================
    if tol is not None:
        Z = tucker_rounding(Z, tol)

    return Z


def tucker_sum(
        S: Sequence[TuckerTensor],
        a: Sequence[float] | NDArray[np.float64],
        tol: float
    ) -> TuckerTensor:
    #==========================================================================
    #   Input Validation
    #==========================================================================
    if len(S) != len(a):
        raise ValueError(
            f"Size mismatch: 'S' has {len(S)} tensors, "
            f"but 'a' has {len(a)} floats."
        )

    if len(S) == 0:
        raise ValueError(f"Summand arrays cannot be empty.")

    if tol < 0:
        raise ValueError(f"Tolerance 'tol' must be non-negative.")
    
    #==========================================================================
    #   Compute Sum and Round
    #==========================================================================
    X = tucker_scale(S[0], a[0])

    for i in range(1, len(S)):
       X = tucker_axby(1.0, X, a[i], S[i])

    X = tucker_rounding(X, tol)

    return X
