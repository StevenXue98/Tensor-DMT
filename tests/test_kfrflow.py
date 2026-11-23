from pathlib import Path

import numpy as np

from tensor_dmt.kfrflow.flow import kfrflow_euler, kfrflow_importance
from tensor_dmt.kfrflow.kernels import matern32_kernel, rational_quadratic_kernel


REFERENCE_PATH = Path(__file__).with_name("data") / "kfrflow_donut_julia.csv"
FUNNEL_REFERENCE_PATH = (
    Path(__file__).with_name("data") / "kfrflow_importance_funnel_julia.csv"
)
DIMENSION = 2
NUM_PARTICLES = 20
NUM_STEPS = 100
DELTA_T = 0.01
NUGGET = 1e-8


def donut_log_ratio(X: np.ndarray) -> np.ndarray:
    return -8.0 * (np.linalg.norm(X, axis=-1) - 2.0) ** 2


def funnel_log_ratio(X: np.ndarray) -> np.ndarray:
    x1 = X[..., 0]
    remaining_sq = np.sum(X[..., 1:] ** 2, axis=-1)
    log_target = -x1**2 / 18 - 0.5 * np.exp(-x1) * remaining_sq
    log_reference = -0.5 * np.sum(X**2, axis=-1)
    return log_target - log_reference


def test_kfrflow_euler_matches_julia_donut_history() -> None:
    assert REFERENCE_PATH.exists(), (
        "Generate the Julia reference with "
        "`julia tests/reference/generate_kfrflow_donut_reference.jl`."
    )

    reference = np.loadtxt(REFERENCE_PATH, delimiter=",")
    reference_times = reference[:, 0]
    reference_history = reference[:, 1:].reshape(
        NUM_STEPS + 1,
        NUM_PARTICLES,
        DIMENSION,
    )

    result = kfrflow_euler(
        reference_history[0].copy(),
        donut_log_ratio,
        matern32_kernel,
        nugget=NUGGET,
        delta_t=DELTA_T,
        steps=NUM_STEPS,
        save_history=True,
        save_every=1,
    )

    python_history = np.stack(result.history)

    np.testing.assert_allclose(
        result.times,
        reference_times,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        python_history,
        reference_history,
        rtol=1e-9,
        atol=1e-10,
    )


def test_kfrflow_importance_matches_julia_funnel_history() -> None:
    dimension = 5
    num_particles = 20

    assert FUNNEL_REFERENCE_PATH.exists(), (
        "Generate the Julia reference with "
        "`julia tests/reference/"
        "generate_kfrflow_importance_funnel_reference.jl`."
    )

    reference = np.loadtxt(FUNNEL_REFERENCE_PATH, delimiter=",")
    reference_times = reference[:, 0]
    reference_history = reference[:, 1:].reshape(
        NUM_STEPS + 1,
        num_particles,
        dimension,
    )

    result = kfrflow_importance(
        reference_history[0].copy(),
        funnel_log_ratio,
        rational_quadratic_kernel,
        nugget=1e-2,
        delta_t=DELTA_T,
        steps=NUM_STEPS,
        save_history=True,
        save_every=1,
    )

    python_history = np.stack(result.history)

    np.testing.assert_allclose(
        result.times,
        reference_times,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        python_history,
        reference_history,
        rtol=1e-9,
        atol=1e-10,
    )
