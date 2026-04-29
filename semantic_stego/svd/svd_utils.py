from __future__ import annotations

import numpy as np

from semantic_stego.svd.svd_from_scratch import svd_reconstruct


def select_singular_indices(S: np.ndarray, payload_len: int, band: str) -> np.ndarray:
    n = len(S)
    if payload_len <= 0:
        return np.array([], dtype=int)
    payload_len = min(payload_len, n)

    if band == "high_energy":
        start = 0
    elif band == "mid_energy":
        start = max(0, n // 2 - payload_len // 2)
    elif band == "low_energy":
        start = max(0, n - payload_len)
    else:
        raise ValueError(f"Unsupported SVD band: {band}")

    stop = min(n, start + payload_len)
    return np.arange(start, stop, dtype=int)


def compute_reconstruction_error(A: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> float:
    reconstructed = svd_reconstruct(U, S, Vt)
    denom = np.linalg.norm(A)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(A - reconstructed) / denom)
