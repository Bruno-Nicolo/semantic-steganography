from __future__ import annotations

import numpy as np


def svd_decompose(A: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(A, dtype=np.float64)
    m, n = matrix.shape

    if m >= n:
        gram = matrix.T @ matrix
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        V = eigenvectors[:, order]
        S = np.sqrt(eigenvalues)
        U = np.zeros((m, n), dtype=np.float64)
        for index, sigma in enumerate(S):
            if sigma > eps:
                U[:, index] = (matrix @ V[:, index]) / sigma
        U = _orthonormalize_columns(U)
        Vt = V.T
        return U, S, Vt

    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    U = eigenvectors[:, order]
    S = np.sqrt(eigenvalues)
    V = np.zeros((n, m), dtype=np.float64)
    for index, sigma in enumerate(S):
        if sigma > eps:
            V[:, index] = (matrix.T @ U[:, index]) / sigma
    V = _orthonormalize_columns(V)
    Vt = V.T
    return U, S, Vt


def svd_reconstruct(U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    return U @ np.diag(S) @ Vt


def _orthonormalize_columns(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    output = matrix.copy()
    for index in range(output.shape[1]):
        column = output[:, index]
        for prev in range(index):
            column = column - np.dot(output[:, prev], column) * output[:, prev]
        norm = np.linalg.norm(column)
        if norm > eps:
            output[:, index] = column / norm
        else:
            output[:, index] = _build_fallback_basis_vector(output, index, eps)
    return output


def _build_fallback_basis_vector(matrix: np.ndarray, index: int, eps: float) -> np.ndarray:
    size = matrix.shape[0]
    for basis_index in range(size):
        candidate = np.zeros(size, dtype=np.float64)
        candidate[basis_index] = 1.0
        for prev in range(index):
            candidate = candidate - np.dot(matrix[:, prev], candidate) * matrix[:, prev]
        norm = np.linalg.norm(candidate)
        if norm > eps:
            return candidate / norm
    return np.zeros(size, dtype=np.float64)
