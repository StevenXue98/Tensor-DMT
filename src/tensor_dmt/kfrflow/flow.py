import numpy as np
from time import perf_counter

from dataclasses import dataclass


@dataclass
class FlowResult:
    final_particles: np.ndarray
    history: list[np.ndarray]
    times: list[float]
    method_name: str
    elapsed_seconds: float


def kfrflow_velocity(X, log_ratio, kernel, nugget=1e-8):
    # construct kernel, kernel gradient, and stiffness matrices
    K, grad_K = kernel(X)
    M = np.einsum('ijk,ilk->jl', grad_K, grad_K)

    # construct log density ratio
    q = log_ratio(X)
    q = q - q.mean()

    # construct RHS b and solve for coefficients alpha
    b = K.T @ q
    alpha = np.linalg.solve(M + nugget * np.eye(M.shape[0]), b)

    # construct velocity field
    V = np.einsum('ijk,j->ik', grad_K, alpha)

    return V


def kfrflow_euler(
    X,
    log_ratio,
    kernel,
    nugget=1e-8,
    delta_t=0.01,
    steps=100,
    save_history=False,
    save_every=25,
) -> FlowResult:
    start = perf_counter()
    history = [X.copy()] if save_history else []
    times = [0.0] if save_history else []

    for step in range(1, steps + 1):
        V = kfrflow_velocity(X, log_ratio, kernel, nugget)
        X = X + delta_t * V
        if save_history and (step % save_every == 0 or step == steps):
            history.append(X.copy())
            times.append(step * delta_t)

    return FlowResult(
        final_particles=X,
        history=history,
        times=times,
        method_name="KFRFlow",
        elapsed_seconds=perf_counter() - start,
    )


def _importance_step(X, log_ratio, kernel, nugget=1e-8, delta_t=0.01):
    K, grad_K = kernel(X)
    
    log_weights = delta_t * log_ratio(X)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()

    a = K.mean(axis=0)
    b = K.T @ weights

    M = np.einsum("ijk,ilk->jl", grad_K, grad_K) / X.shape[0]
    coefficients = np.linalg.solve(M + nugget * np.eye(M.shape[0]), b - a)

    increment = np.einsum("ijk,j->ik", grad_K, coefficients)
    X = X + increment

    return X


def kfrflow_importance(
    X,
    log_ratio,
    kernel,
    nugget=1e-8,
    delta_t=0.01,
    steps=100,
    save_history=False,
    save_every=25,
) -> FlowResult:
    start = perf_counter()
    history = [X.copy()] if save_history else []
    times = [0.0] if save_history else []

    for step in range(1, steps + 1):
        X = _importance_step(X, log_ratio, kernel, nugget=nugget, delta_t=delta_t)
        if save_history and (step % save_every == 0 or step == steps):
            history.append(X.copy())
            times.append(step * delta_t)

    return FlowResult(
        final_particles=X,
        history=history,
        times=times,
        method_name="KFRFlow-I",
        elapsed_seconds=perf_counter() - start,
    )
