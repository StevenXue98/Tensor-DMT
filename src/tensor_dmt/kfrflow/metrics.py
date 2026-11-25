import numpy as np


def imq_ksd(
    samples: np.ndarray,
    score: np.ndarray,
    *,
    c: float = 1.0,
    beta: float = -0.5,
) -> float:
    J, d = samples.shape

    diff_tensor = samples[:, None, :] - samples[None, :, :]
    diff_norm_sq = np.sum(diff_tensor**2, axis=-1)
    base = c**2 + diff_norm_sq
    kernel = base**beta

    score_inner = score @ score.T
    score_term = score_inner * kernel

    gradient_scale = 2 * beta * base ** (beta - 1)

    score_difference = score[None, :, :] - score[:, None, :]
    gradient_term = gradient_scale * np.sum(
        score_difference * diff_tensor,
        axis=-1,
    )

    trace_term = (
        -2 * beta * d * base ** (beta - 1)
        -4
        * beta
        * (beta - 1)
        * diff_norm_sq
        * base ** (beta - 2)
    )

    stein_kernel = score_term + gradient_term + trace_term
    ksd_squared = np.mean(stein_kernel)

    return float(np.sqrt(max(ksd_squared, 0.0)))
