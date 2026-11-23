import numpy as np

from scipy.spatial.distance import pdist


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


def median_bandwidth(X: np.ndarray) -> float:
    distances = pdist(X)
    median_distance = np.median(distances)
    bandwidth = median_distance / (2 * np.sqrt(np.log(X.shape[0])))

    return float(bandwidth)


def rational_quadratic_kernel(
    X: np.ndarray,
    centers: np.ndarray | None = None,
    alpha: float = 0.5,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if centers is None:
        centers = X

    if bandwidth is None:
        bandwidth = median_bandwidth(X)

    diff_tensor = X[:, None, :] - centers[None, :, :]
    diff_norm_sq = np.sum(diff_tensor**2, axis=2)

    # construct kernel_matrix and kernel_gradient
    base = 1 + diff_norm_sq / (2 * alpha * bandwidth**2)
    kernel = base ** (-alpha)
    kernel_gradient = (-base ** (-alpha - 1) / bandwidth**2)[:, :, None] * diff_tensor

    return kernel, kernel_gradient