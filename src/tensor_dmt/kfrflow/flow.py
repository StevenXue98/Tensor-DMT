import numpy as np

from dataclasses import dataclass


@dataclass
class FlowResult:
    final_particles: np.ndarray
    history: list[np.ndarray]
    times: list[float]


def velocity(X, log_ratio, kernel, nugget=1e-8):
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


def forward_euler(
    X,
    log_ratio,
    kernel,
    nugget=1e-8,
    delta_t=0.01,
    steps=100,
    save_history=False,
    save_every=25,
) -> FlowResult:
    history = [X.copy()] if save_history else []
    times = [0.0] if save_history else []

    for step in range(1, steps + 1):
        V = velocity(X, log_ratio, kernel, nugget)
        X = X + delta_t * V
        if save_history and (step % save_every == 0 or step == steps):
            history.append(X.copy())
            times.append(step * delta_t)

    return FlowResult(
        final_particles=X,
        history=history,
        times=times,
    )