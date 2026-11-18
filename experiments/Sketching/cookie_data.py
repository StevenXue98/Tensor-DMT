import numpy as np
import scipy.sparse as sps

from pathlib import Path
from numpy.typing import NDArray
from dataclasses import dataclass


__all__ = [
    "CookieData",
    "load_cookie_data",
    "read_coordinate_matrix",
    "read_vector",
]


COOKIE_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "references"
    / "SupSmiPGetal26"
    / "Data"
    / "Cookie"
)


@dataclass(frozen=True)
class CookieData:
    A0: sps.csr_matrix
    A1: sps.csr_matrix
    A2: sps.csr_matrix
    A3: sps.csr_matrix
    a0: NDArray[np.float64]


def read_coordinate_matrix(
    path: str | Path,
    *,
    shape: tuple[int, int] | None = None,
) -> sps.csr_matrix:
    """Read a zero-based ``row column value`` file as a CSR matrix."""
    entries = np.loadtxt(path, dtype=np.float64, ndmin=2)

    if entries.shape[1] != 3:
        raise ValueError(f"Expected three columns in {path}, found {entries.shape[1]}")

    row_indices = entries[:, 0].astype(np.intp)
    col_indices = entries[:, 1].astype(np.intp)
    values = entries[:, 2]

    if np.any(row_indices < 0) or np.any(col_indices < 0):
        raise ValueError(f"Matrix indices must be non-negative in {path}")

    if shape is None:
        shape = (int(row_indices.max()) + 1, int(col_indices.max()) + 1)

    matrix = sps.csr_matrix(
        (values, (row_indices, col_indices)),
        shape=shape,
    )
    matrix.eliminate_zeros()
    return matrix


def read_vector(path: str | Path) -> NDArray[np.float64]:
    """Read one floating-point value per line into a one-dimensional array."""
    return np.loadtxt(path, dtype=np.float64, ndmin=1).reshape(-1)


def load_cookie_data(data_dir: str | Path = COOKIE_DATA_DIR) -> CookieData:
    """Load the fixed spatial matrices and right-hand-side factor."""
    data_dir = Path(data_dir)
    a0 = read_vector(data_dir / "b.txt")
    matrix_shape = (a0.size, a0.size)

    return CookieData(
        A0=read_coordinate_matrix(data_dir / "A0.txt", shape=matrix_shape),
        A1=read_coordinate_matrix(data_dir / "A1.txt", shape=matrix_shape),
        A2=read_coordinate_matrix(data_dir / "A2.txt", shape=matrix_shape),
        A3=read_coordinate_matrix(data_dir / "A3.txt", shape=matrix_shape),
        a0=a0,
    )
