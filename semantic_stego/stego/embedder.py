from __future__ import annotations

from time import perf_counter

import numpy as np

from semantic_stego.config.schemas import EmbeddingMetadata, EmbeddingResult, ROI
from semantic_stego.data.image_io import crop_roi, paste_roi, rgb_to_ycrcb, ycrcb_to_rgb
from semantic_stego.stego.payload import fit_payload_to_capacity
from semantic_stego.svd.svd_from_scratch import svd_decompose, svd_reconstruct
from semantic_stego.svd.svd_utils import compute_reconstruction_error, select_singular_indices


class SvdEmbedder:
    def __init__(self, payload_policy: str = "truncate_message"):
        self.payload_policy = payload_policy

    def embed(
        self,
        image: np.ndarray,
        roi: ROI,
        payload_bits: np.ndarray,
        band: str,
        strength: float,
        mode: str = "qim",
    ) -> EmbeddingResult:
        roi_patch = crop_roi(image, roi)
        roi_ycc = rgb_to_ycrcb(roi_patch)
        y_channel = roi_ycc[:, :, 0].astype(np.float64)

        svd_start = perf_counter()
        U, S, Vt = svd_decompose(y_channel)
        svd_time_ms = (perf_counter() - svd_start) * 1000.0

        capacity = len(S)
        fitted_bits, truncated, dropped = fit_payload_to_capacity(payload_bits, capacity, self.payload_policy)
        indices = select_singular_indices(S, len(fitted_bits), band)
        stego_s = S.copy()
        delta = float(strength)

        decomposition_error = compute_reconstruction_error(y_channel, U, S, Vt)
        embed_start = perf_counter()
        stego_s[indices] = _embed_qim_bits(stego_s[indices], fitted_bits, delta)
        stego_y = svd_reconstruct(U, stego_s, Vt)
        roi_ycc[:, :, 0] = np.clip(np.rint(stego_y), 0, 255).astype(np.uint8)
        stego_roi = ycrcb_to_rgb(roi_ycc)
        stego_image = paste_roi(image, roi, stego_roi)
        embedding_time_ms = (perf_counter() - embed_start) * 1000.0

        metadata = EmbeddingMetadata(
            roi=roi,
            band=band,
            indices=indices,
            payload_len=len(fitted_bits),
            strength=delta,
            mode=mode,
            qim_delta=delta,
        )
        return EmbeddingResult(
            stego_image=stego_image,
            metadata=metadata,
            embedded_bits=fitted_bits,
            requested_bits=len(payload_bits),
            svd_time_ms=svd_time_ms,
            embedding_time_ms=embedding_time_ms,
            svd_reconstruction_error=decomposition_error,
            payload_bits_capacity=capacity,
            payload_bits_dropped=dropped,
            payload_truncated=truncated,
        )


def _embed_qim_bits(values: np.ndarray, bits: np.ndarray, delta: float) -> np.ndarray:
    quantized = values.copy()
    for index, bit in enumerate(bits.astype(int)):
        normalized = quantized[index] / delta
        if bit == 0:
            candidate_q = int(np.floor(normalized / 2.0) * 2)
        else:
            candidate_q = int(np.ceil((normalized - 1.0) / 2.0) * 2 + 1)
        quantized[index] = candidate_q * delta
        if quantized[index] < 0:
            quantized[index] = 0.0
    return quantized
