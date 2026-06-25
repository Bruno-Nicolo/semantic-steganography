from __future__ import annotations

from time import perf_counter

import numpy as np

from semantic_stego.config.schemas import EmbeddingMetadata, EmbeddingResult, ROI
from semantic_stego.data.image_io import apply_gray_delta_to_rgb, crop_roi, paste_roi, rgb_to_gray
from semantic_stego.stego.payload import fit_payload_to_capacity
from semantic_stego.svd.svd_from_scratch import svd_decompose, svd_reconstruct
from semantic_stego.svd.svd_utils import compute_reconstruction_error, effective_singular_capacity, select_singular_indices


class SvdEmbedder:
    def __init__(self, payload_policy: str = "truncate_message", repetition_factor: int = 3):
        self.payload_policy = payload_policy
        if repetition_factor <= 0:
            raise ValueError("repetition_factor must be positive")
        self.repetition_factor = repetition_factor

    def embed(
        self,
        image: np.ndarray,
        roi: ROI,
        payload_bits: np.ndarray,
        band: str,
        strength: float,
        strength_mode: str = "absolute",
        mode: str = "qim",
    ) -> EmbeddingResult:
        roi_patch = crop_roi(image, roi)
        gray_channel = rgb_to_gray(roi_patch).astype(np.float64)

        svd_start = perf_counter()
        U, S, Vt = svd_decompose(gray_channel)
        svd_time_ms = (perf_counter() - svd_start) * 1000.0

        symbol_capacity = effective_singular_capacity(S, band, strength, strength_mode)
        payload_capacity = symbol_capacity // self.repetition_factor
        fitted_bits, truncated, dropped = fit_payload_to_capacity(payload_bits, payload_capacity, self.payload_policy)
        coded_bits = np.repeat(fitted_bits, self.repetition_factor)
        indices = select_singular_indices(S, len(coded_bits), band, strength, strength_mode)
        stego_s = S.copy()
        delta = _resolve_delta(S, indices, strength, strength_mode)

        decomposition_error = compute_reconstruction_error(gray_channel, U, S, Vt)
        embed_start = perf_counter()
        stego_s[indices] = _embed_qim_bits(stego_s[indices], coded_bits, delta)
        stego_gray = np.clip(np.rint(svd_reconstruct(U, stego_s, Vt)), 0, 255)
        gray_delta = stego_gray - gray_channel
        stego_roi = apply_gray_delta_to_rgb(roi_patch, gray_delta)
        stego_image = paste_roi(image, roi, stego_roi)
        embedding_time_ms = (perf_counter() - embed_start) * 1000.0

        metadata = EmbeddingMetadata(
            roi=roi,
            band=band,
            indices=indices,
            payload_len=len(fitted_bits),
            repetition_factor=self.repetition_factor,
            strength=float(np.mean(delta)) if np.size(delta) else float(strength),
            mode=mode,
            qim_delta=delta,
            channel="gray",
        )
        return EmbeddingResult(
            stego_image=stego_image,
            metadata=metadata,
            embedded_bits=fitted_bits,
            requested_bits=len(payload_bits),
            svd_time_ms=svd_time_ms,
            embedding_time_ms=embedding_time_ms,
            svd_reconstruction_error=decomposition_error,
            payload_bits_capacity=payload_capacity,
            payload_bits_dropped=dropped,
            payload_truncated=truncated,
        )


def _resolve_delta(S: np.ndarray, indices: np.ndarray, strength: float, strength_mode: str) -> float | np.ndarray:
    if strength_mode == "absolute":
        return float(strength)
    if strength_mode == "proportional_singular":
        return np.maximum(S[indices].astype(float) * float(strength), 1e-9)
    raise ValueError(f"Unsupported embedding strength mode: {strength_mode}")


def _embed_qim_bits(values: np.ndarray, bits: np.ndarray, delta: float | np.ndarray) -> np.ndarray:
    quantized = values.copy()
    for index, bit in enumerate(bits.astype(int)):
        quantized[index] = _nearest_qim_codeword(float(quantized[index]), _delta_at(delta, index), bit)
    return quantized


def _delta_at(delta: float | np.ndarray, index: int) -> float:
    if np.isscalar(delta):
        return float(delta)
    return float(delta[index])


def _nearest_qim_codeword(value: float, delta: float, bit: int) -> float:
    if delta <= 0:
        raise ValueError("QIM delta must be positive")
    scaled = value / delta
    rounded = int(np.rint(scaled))
    if rounded % 2 != bit:
        lower = rounded - 1
        upper = rounded + 1
        rounded = lower if abs(scaled - lower) <= abs(scaled - upper) else upper
    if rounded < 0:
        rounded = bit
    return float(rounded * delta)
