import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tensor_dmt.kfrflow.metrics import imq_ksd
from tensor_dmt.kfrflow.hermite import hfrflow_euler
from tensor_dmt.kfrflow.flow import kfrflow_importance
from tensor_dmt.kfrflow.kernels import rational_quadratic_kernel


def funnel_log_ratio(X: np.ndarray) -> np.ndarray:
    dimension = X.shape[-1]
    x1 = X[..., 0]
    remaining_sq = np.sum(X[..., 1:] ** 2, axis=-1)

    log_target = (
        -x1**2 / 18
        # - 0.5 * (dimension - 1) * x1
        - 0.5 * np.exp(-x1) * remaining_sq
    )
    log_reference = -0.5 * np.sum(X**2, axis=-1)

    return log_target - log_reference


def funnel_score(X: np.ndarray) -> np.ndarray:
    x1 = X[..., 0]
    remaining = X[..., 1:]
    exp_neg_x1 = np.exp(-x1)

    score = np.empty_like(X)
    score[..., 0] = (
        -x1 / 9
        + 0.5 * exp_neg_x1 * np.sum(remaining**2, axis=-1)
    )
    score[..., 1:] = -exp_neg_x1[..., None] * remaining

    return score


def main():
    dimensions = (5, 10, 15, 20)
    num_particles = 100
    delta_t = 0.01
    steps = 100
    repeats = 30

    nuggets = {
        5: 1e-2,
        10: 1e-3,
        15: 1e-3,
        20: 1e-3,
    }

    rng = np.random.default_rng(0)
    dimension = dimensions[0]
    X_initial = rng.standard_normal((num_particles, dimension))

    kernel_result = kfrflow_importance(
        X_initial.copy(),
        funnel_log_ratio,
        rational_quadratic_kernel,
        nugget=nuggets[dimension],
        delta_t=delta_t,
        steps=steps,
        save_history=True,
        save_every=25,
    )

    X_initial = rng.standard_normal((num_particles * 100, dimension))

    hermite_result = hfrflow_euler(
        X_initial.copy(),
        funnel_log_ratio,
        max_degree=5,
        nugget=1e-3,
        reg_order=None,
        total_degree=False,
        delta_t=delta_t,
        steps=steps,
        save_history=True,
        save_every=25,
    )

    kernel_ksd_history = [
        imq_ksd(samples, funnel_score(samples))
        for samples in kernel_result.history
    ]

    hermite_ksd_history = [
        imq_ksd(samples, funnel_score(samples))
        for samples in hermite_result.history
    ]

    print(f"kernel ksd: {kernel_ksd_history}\nhermite ksd:{hermite_ksd_history}")


if __name__ == "__main__":
    main()
