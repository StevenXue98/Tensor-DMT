import numpy as np
from time import perf_counter
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy.sparse.linalg import SuperLU
from collections.abc import Callable, Sequence
from tensorly.tucker_tensor import TuckerTensor

from .core import (
    tucker_scale,
    tucker_inner_prod,
    tucker_norm,
    tucker_rounding,
)

__all__ = [
    "TuckerOperator",
    "TuckerPreconditioner",
    "RoundingSum",
    "TuckerTimer",
    "TuckerGMRESResult",
    "preconditioner",
    "tucker_GMRES",
]


TuckerOperator = Callable[
    [TuckerTensor],
    Sequence[TuckerTensor],
]

TuckerPreconditioner = Callable[
    [TuckerTensor],
    TuckerTensor,
]

RoundingSum = Callable[
    [Sequence[TuckerTensor], Sequence[float] | NDArray[np.float64], float],
    TuckerTensor,
]


@dataclass
class TuckerTimer:
    round_sum: float = 0.0
    operator: float = 0.0
    preconditioner: float = 0.0
    other: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.round_sum
            + self.operator
            + self.preconditioner
            + self.other
        )


@dataclass
class TuckerGMRESResult:
    solution: TuckerTensor
    ranks: list[tuple[int, ...]]
    timer: TuckerTimer


def preconditioner(lu: SuperLU, Y: TuckerTensor) -> TuckerTensor:
    new_factors = list(Y.factors)
    new_factors[0] = lu.solve(new_factors[0])
    return TuckerTensor((Y.core, new_factors))


def tucker_GMRES(
        A: TuckerOperator,
        b: TuckerTensor,
        *,
        tol: float,
        max_iter: int,
        prec: TuckerPreconditioner,
        rounding_tol: float,
        rounding_sum: RoundingSum
    ) -> TuckerGMRESResult:

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    #==========================================================================
    #   Initialize Timers & Variables
    #==========================================================================
    k             = 0
    normb         = tucker_norm(b)
    timer         = TuckerTimer()

    start         = perf_counter()
    r             = b
    normres       = tucker_norm(r)

    if normres == 0:
        raise ValueError("Residual has norm 0.")

    V             = [tucker_scale(r, 1 / normres)]
    timer.other  += perf_counter() - start

    start         = perf_counter()
    H             = np.zeros((max_iter + 1, max_iter))
    rK            = np.zeros(max_iter + 1)
    rK[0]         = normres
    ranks         = [V[0].rank]
    timer.other  += perf_counter() - start

    R:     NDArray[np.float64] | None = None
    t_vec: NDArray[np.float64] | None = None

    #==========================================================================
    #   Main Krylov Iteration Loop
    #==========================================================================
    while k < max_iter and normres > tol * normb:
        start                 = perf_counter()
        w                     = prec(V[k])
        timer.preconditioner += perf_counter() - start

        start                 = perf_counter()
        summands              = A(w)
        timer.operator       += perf_counter() - start

        start                 = perf_counter()
        delta                 = 0.01 * tol / (normres * normb)
        w                     = rounding_sum(summands, np.ones(len(summands)), delta)
        timer.round_sum      += perf_counter() - start

        start                 = perf_counter()
        h                     = np.zeros(k + 2)
        for i in range(k + 1):
            h[i] = tucker_inner_prod(V[i], w)
        timer.other          += perf_counter() - start

        start                 = perf_counter()
        terms                 = V + [w]
        weights               = np.concatenate((-h[:k + 1], np.array([1.0])))
        w                     = rounding_sum(terms, weights, delta)
        timer.round_sum      += perf_counter() - start

        start                 = perf_counter()
        h[k + 1]              = tucker_norm(w)

        if h[k + 1] < 2e-16:
            raise RuntimeError("GMRES breakdown")

        V.append(tucker_scale(w, 1 / h[k + 1]))
        ranks.append(V[k + 1].rank)

        H[:k + 2, k]          = h
        Q, R                  = np.linalg.qr(H[:k + 2, :k + 1], mode="complete")
        t_vec                 = Q.T @ rK[:k + 2]
        normres               = np.abs(t_vec[k + 1])

        print(f"Iteration {k + 1}: Residual = {normres:e}")
        timer.other          += perf_counter() - start

        k += 1

    #==========================================================================
    #   Reconstruct Final Solution
    #==========================================================================
    if R is None or t_vec is None:
        raise RuntimeError("GMRES terminated before completing an iteration")

    start                 = perf_counter()
    y                     = np.linalg.solve(R[:k, :k], t_vec[:k])
    timer.other          += perf_counter() - start

    start                 = perf_counter()
    x                     = rounding_sum(V[:k], y[:k], rounding_tol)
    timer.round_sum      += perf_counter() - start

    start                 = perf_counter()
    x                     = prec(x)
    timer.preconditioner += perf_counter() - start

    start                 = perf_counter()
    x                     = tucker_rounding(x, rounding_tol)
    ranks.append(x.rank)
    timer.other          += perf_counter() - start

    return TuckerGMRESResult(x, ranks, timer)
