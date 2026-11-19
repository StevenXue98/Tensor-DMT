import numpy as np
import scipy.sparse as sps

from typing import Literal
from time import perf_counter
from dataclasses import dataclass
from scipy.sparse.linalg import splu
from cookie_data import load_cookie_data
from tensorly.tucker_tensor import TuckerTensor
from tensor_dmt.tucker.gmres import TuckerTimer, RoundingSum, preconditioner, tucker_GMRES
from tensor_dmt.tucker.core import OperatorMatrix, tucker_scale, tucker_norm, tucker_axby, tucker_summands_kron_op


__all__ = [
    "Method",
    "CookieExperimentConfig",
    "CookieTrialRecord",
    "run_cookie_problem",
]


Method = Literal["naive", "kron", "krp"]


@dataclass(frozen=True)
class CookieExperimentConfig:
    sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    repeats: int = 3
    tol: float = 1e-5
    rounding_tol: float = 1e-7
    max_iter: int = 20
    oversampling: int = 5
    seed: int = 12


@dataclass
class CookieTrialRecord:
    n: int
    repeat: int
    method: Method
    elapsed: float
    residual: float
    ranks: list[tuple[int, ...]]
    timer: TuckerTimer


def run_cookie_problem(
        config: CookieExperimentConfig,
        solvers: tuple[tuple[Method, RoundingSum], ...],
    ) -> list[CookieTrialRecord]: 
    data = load_cookie_data()
    records: list[CookieTrialRecord] = []
    
    lu = splu((data.A0 + data.A1 + data.A2 + data.A3).tocsc())
    prec = lambda x: preconditioner(lu, x)

    for n in config.sizes:
        parameter_values = np.linspace(1.0, 10.0, n)

        D = sps.diags(parameter_values, format="csr")
        B = sps.eye(n, format="csr")
        c = np.ones((n, 1))
        A: tuple[tuple[OperatorMatrix, ...], ...] = (
            (data.A0, B, B, B),
            (data.A1, D, B, B),
            (data.A2, B, D, B),
            (data.A3, B, B, D),
        )

        core = np.ones((1, 1, 1, 1))
        b = TuckerTensor((core, [data.a0[:, None], c, c, c]))
        b = tucker_scale(b, 1.0 / tucker_norm(b))
        operator = lambda x: tucker_summands_kron_op(A, x)

        for repeat in range(config.repeats):
            for method, solver in solvers:
                start = perf_counter()

                result = tucker_GMRES(
                    operator,
                    b,
                    tol=config.tol,
                    max_iter=config.max_iter,
                    prec=prec,
                    rounding_tol=config.rounding_tol,
                    rounding_sum=solver,
                )

                elapsed = perf_counter() - start

                summands = tucker_summands_kron_op(A, result.solution)
                residual = summands[0]

                for summand in summands[1:]:
                    residual = tucker_axby(1.0, residual, 1.0, summand)

                residual = tucker_axby(1.0, residual, -1.0, b)
                residual_norm = max(float(np.real(tucker_norm(residual))), np.finfo(float).eps)

                records.append(
                    CookieTrialRecord(
                        n=n,
                        repeat=repeat,
                        method=method,
                        elapsed=elapsed,
                        residual=residual_norm,
                        ranks=result.ranks,
                        timer=result.timer,
                    )
                )

    return records
