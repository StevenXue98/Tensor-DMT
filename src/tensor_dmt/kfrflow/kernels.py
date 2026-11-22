import numpy as np


def matern32_kernel(
    X: np.ndarray,
    centers: np.ndarray | None = None,
    bandwidth: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:
    if centers is None:
        centers = X

    diff_tensor = X[:, None, :] - centers[None, :, :]
    diff_norms = np.linalg.norm(diff_tensor, axis=2)

    # construct kernel_matrix
    kernel = (1 + np.sqrt(3) / bandwidth * diff_norms) * np.exp(- np.sqrt(3) / bandwidth * diff_norms)

    # construct kernel_gradient
    kernel_gradient = (-3 / bandwidth**2 * np.exp(-np.sqrt(3) / bandwidth * diff_norms))[:, :, None] * (diff_tensor)

    return kernel, kernel_gradient
