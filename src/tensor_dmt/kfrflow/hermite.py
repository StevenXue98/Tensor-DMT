import numpy as np
from functools import lru_cache
from time import perf_counter

from .flow import FlowResult
from scipy.linalg import solve
from scipy.special import factorial
from numpy.polynomial.hermite_e import hermevander
from itertools import combinations_with_replacement


@lru_cache(maxsize=16)
def _full_product_powers(d, max_degree):
    n = max_degree + 1
    return np.indices((n,) * d).reshape(d, -1).T


@lru_cache(maxsize=16)
def _total_degree_powers(d, max_degree):
    indices = []
    for degree in range(1, max_degree + 1):
        for axes in combinations_with_replacement(range(d), degree):
            indices.append(np.bincount(axes, minlength=d))
    return np.asarray(indices, dtype=int).reshape(-1, d)


def hermite_velocity(
    X, 
    log_ratio, 
    max_degree=4, 
    nugget=1e-8, 
    reg_order=None, 
    total_degree=True
):
    J, d = X.shape # (number of particles, dimension)
    degrees = np.arange(max_degree + 1)
    normalization = np.sqrt(factorial(degrees, exact=False))
    
    # construct Hermite function basis
    polynomials = hermevander(X, max_degree) / normalization # divide by sqrt(k!)
    damping = np.exp(-0.25 * X[..., None] ** 2)
    H = polynomials * damping
    dH = -0.5 * X[..., None] * H
    dH[..., 1:] += H[..., :-1] * np.sqrt(degrees[1:])

    if total_degree:
        indices = np.vstack([
            np.zeros((1, d), dtype=int),
            _total_degree_powers(d, max_degree),
        ])
    else:
        indices = _full_product_powers(d, max_degree)

    num_features = len(indices)
    # Each table evaluates one coordinate across all particles and retained features
    values = [H[:, k, indices[:, k]] for k in range(d)]
    Phi = values[0].copy(order="C")
    for factor in values[1:]:
        Phi *= factor

    derivatives = [dH[:, k, indices[:, k]] for k in range(d)]
    A = np.empty((J, num_features, d), dtype=H.dtype)

    for k in range(d):
        A[:, :, k] = derivatives[k]
        for m, factor in enumerate(values):
            if m != k:
                A[:, :, k] *= factor

    # construct gram matrix M
    G = A.transpose(0, 2, 1).reshape(J * d, num_features)
    M = G.T @ G

    # construct RHS b
    q = log_ratio(X)
    q = q - q.mean()
    b = Phi.T @ q

    # add regularizer and solve for coefficients alpha
    if reg_order is not None:
        penalty = (1 + indices.sum(axis=1).astype(H.dtype)) ** reg_order
    else:
        penalty = 1.0
    diagonal = np.diag_indices_from(M)
    M[diagonal] += nugget * penalty
    alpha = solve(M, b, assume_a="pos", overwrite_a=True, check_finite=False)

    # construct velocity field
    V = A.swapaxes(1, 2) @ alpha

    return V


def hfrflow_euler(
    X,
    log_ratio,
    *,
    max_degree=4,
    nugget=1e-8,
    reg_order=None,
    total_degree=True,
    delta_t=0.01,
    steps=100,
    save_history=False,
    save_every=25,
) -> FlowResult:
    start = perf_counter()
    history = [X.copy()] if save_history else []
    times = [0.0] if save_history else []

    for step in range(1, steps + 1):
        V = hermite_velocity(X, log_ratio, max_degree, nugget, reg_order, total_degree)
        X = X + delta_t * V
        if save_history and (step % save_every == 0 or step == steps):
            history.append(X.copy())
            times.append(step * delta_t)

    return FlowResult(
        final_particles=X,
        history=history,
        times=times,
        method_name=f"Hermite {'total-degree' if total_degree else 'full-product'} (p={max_degree})",
        elapsed_seconds=perf_counter() - start,
    )
