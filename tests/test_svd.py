from __future__ import annotations

import numpy as np

from semantic_stego.svd.svd_from_scratch import svd_decompose, svd_reconstruct
from semantic_stego.svd.svd_utils import compute_reconstruction_error


def test_svd_reconstructs_small_matrix() -> None:
    matrix = np.array([[3.0, 1.0], [0.0, 2.0], [1.0, 1.0]])
    U, S, Vt = svd_decompose(matrix)
    reconstructed = svd_reconstruct(U, S, Vt)
    assert np.allclose(matrix, reconstructed, atol=1e-5)


def test_singular_values_non_negative() -> None:
    _, S, _ = svd_decompose(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.all(S >= 0)


def test_reconstruction_error_below_threshold() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    U, S, Vt = svd_decompose(matrix)
    error = compute_reconstruction_error(matrix, U, S, Vt)
    assert error < 1e-5


def test_close_to_numpy_reference() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]])
    _, s_custom, _ = svd_decompose(matrix)
    _, s_numpy, _ = np.linalg.svd(matrix, full_matrices=False)
    assert np.allclose(s_custom, s_numpy, atol=1e-5)
