from __future__ import annotations

import numpy as np

from semantic_stego.svd.svd_from_scratch import svd_reconstruct

MIN_SIGMA_DELTA_FACTOR = 4.0
MIN_NEIGHBOR_GAP_DELTA_FACTOR = 0.5


def roi_svd_capacity(height: int, width: int) -> int:
    return max(0, min(height, width))


def select_singular_indices(S: np.ndarray, payload_len: int, band: str, delta: float, strength_mode: str = "absolute") -> np.ndarray:
    candidates = select_eligible_singular_indices(S, band, delta, strength_mode)
    if payload_len <= 0 or len(candidates) == 0:
        return np.array([], dtype=int)
    return candidates[: min(payload_len, len(candidates))]


def effective_singular_capacity(S: np.ndarray, band: str, delta: float, strength_mode: str = "absolute") -> int:
    return int(len(select_eligible_singular_indices(S, band, delta, strength_mode)))


def select_eligible_singular_indices(S: np.ndarray, band: str, delta: float, strength_mode: str = "absolute") -> np.ndarray:
    n = len(S)
    if n == 0:
        return np.array([], dtype=int)

    if band == "high_energy":
        start = 0
        stop = max(1, n // 3)
    elif band == "mid_energy":
        start = n // 3
        stop = max(start + 1, (2 * n) // 3)
    elif band == "low_energy":
        start = (2 * n) // 3
        stop = n
    else:
        raise ValueError(f"Unsupported SVD band: {band}")

    band_indices = np.arange(start, min(stop, n), dtype=int)
    if len(band_indices) == 0:
        return band_indices

    eligible = []
    for index in band_indices:
        item_delta = _singular_delta(S, index, delta, strength_mode)
        min_sigma = max(item_delta * MIN_SIGMA_DELTA_FACTOR, 1.0)
        min_neighbor_gap = item_delta * MIN_NEIGHBOR_GAP_DELTA_FACTOR
        if S[index] >= min_sigma and _has_stable_neighbor_gaps(S, index, min_neighbor_gap):
            eligible.append(index)
    return np.asarray(eligible, dtype=int)


def _singular_delta(S: np.ndarray, index: int, delta: float, strength_mode: str) -> float:
    if strength_mode == "absolute":
        return float(delta)
    if strength_mode == "proportional_singular":
        return max(float(S[index]) * float(delta), 1e-9)
    raise ValueError(f"Unsupported embedding strength mode: {strength_mode}")


def _has_stable_neighbor_gaps(S: np.ndarray, index: int, min_neighbor_gap: float) -> bool:
    if index > 0 and (S[index - 1] - S[index]) <= min_neighbor_gap:
        return False
    if index < len(S) - 1 and (S[index] - S[index + 1]) <= min_neighbor_gap:
        return False
    return True


def compute_reconstruction_error(A: np.ndarray, U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> float:
    reconstructed = svd_reconstruct(U, S, Vt)
    denom = np.linalg.norm(A)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(A - reconstructed) / denom)
