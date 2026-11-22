from pathlib import Path

import numpy as np

from tensor_dmt.kfrflow.flow import forward_euler
from tensor_dmt.kfrflow.kernels import matern32_kernel


REFERENCE_PATH = Path(__file__).with_name("data") / "kfrflow_donut_julia.csv"
DIMENSION = 2
NUM_PARTICLES = 20
NUM_STEPS = 100
DELTA_T = 0.01
NUGGET = 1e-8


def donut_log_ratio(X: np.ndarray) -> np.ndarray:
    return -8.0 * (np.linalg.norm(X, axis=-1) - 2.0) ** 2


def test_forward_euler_matches_julia_donut_history() -> None:
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

    result = forward_euler(
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
