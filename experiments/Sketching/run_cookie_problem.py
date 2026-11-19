import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import argparse
import numpy as np

from tensor_dmt.tucker.core import tucker_sum
from tensor_dmt.tucker.gmres import RoundingSum
from tensor_dmt.tucker.randomized import tucker_randomize_sum
from cookie_problem import Method, CookieExperimentConfig, run_cookie_problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=("kron", "krp", "naive"),
        default=("kron", "krp", "naive"),
    )
    args = parser.parse_args()

    config = CookieExperimentConfig(
        sizes=(8,),
        repeats=1,
        tol=1e-5,
        rounding_tol=1e-7,
        max_iter=20,
        oversampling=5,
        seed=12,
    )
    rng = np.random.default_rng(config.seed)
    
    solver_naive = lambda W, a, tol: tucker_sum(W, a, tol)
    solver_kron = lambda W, a, tol: tucker_randomize_sum(W, a, config.oversampling, tol, "kron", rng, strategy="heur_fine")
    solver_krp = lambda W, a, tol: tucker_randomize_sum(W, a, config.oversampling, tol, "krp", rng, strategy="heur_fine")
    available_solvers: tuple[tuple[Method, RoundingSum], ...] = (
        ("kron", solver_kron), 
        ("krp", solver_krp), 
        ("naive", solver_naive),
    )
    solvers = tuple(solver for solver in available_solvers if solver[0] in args.solvers)

    records = run_cookie_problem(config, solvers)

    for record in records:
        print(
            f"n={record.n}, "
            f"method={record.method}, "
            f"time={record.elapsed:.3f}s, "
            f"residual={record.residual:.10e}, "
            f"final rank={record.ranks[-1]}"
        )


if __name__ == "__main__":
    main()