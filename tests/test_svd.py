from __future__ import annotations

import numpy as np

from semantic_stego.svd.svd_from_scratch import svd_decompose, svd_reconstruct
from semantic_stego.svd.svd_utils import compute_reconstruction_error, effective_singular_capacity, select_singular_indices


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


def test_effective_capacity_filters_weak_singular_values() -> None:
    singular_values = np.array([180.0, 140.0, 35.0, 20.0, 10.0, 4.0])
    assert effective_singular_capacity(singular_values, "high_energy", 10.0) == 2
    assert effective_singular_capacity(singular_values, "mid_energy", 10.0) == 0
    assert effective_singular_capacity(singular_values, "low_energy", 10.0) == 0


def test_select_singular_indices_respects_effective_capacity() -> None:
    singular_values = np.array([180.0, 130.0, 85.0, 30.0, 20.0, 5.0])
    indices = select_singular_indices(singular_values, payload_len=4, band="high_energy", delta=10.0)
    assert np.array_equal(indices, np.array([0, 1], dtype=int))


def test_effective_capacity_rejects_close_singular_neighbors() -> None:
    singular_values = np.array([120.0, 115.0, 90.0, 30.0, 20.0, 5.0])
    assert effective_singular_capacity(singular_values, "high_energy", 10.0) == 0
