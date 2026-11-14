import numpy as np
from collections.abc import Sequence
from scipy.sparse.linalg import SuperLU, splu
from tensorly.tucker_tensor import TuckerTensor

from tucker_core import (
    OperatorMatrix,
    tucker_norm,
    tucker_scale,
    tucker_rounding,
    tucker_sum,
)



def preconditioner(lu: SuperLU, Y: TuckerTensor) -> TuckerTensor:
    new_factors = list(Y.factors)
    new_factors[0] = lu.solve(new_factors[0])
    return TuckerTensor((Y.core, new_factors))


def tucker_GMRES(
          A: TuckerTensor, 
          b: TuckerTensor, 
          tol: float, 
          max_iter: int, 
          prec, 
          tols, 
          rounding_sum
          ):
    pass