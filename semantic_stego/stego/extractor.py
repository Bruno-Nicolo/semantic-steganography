from __future__ import annotations

from time import perf_counter

import numpy as np

from semantic_stego.config.schemas import EmbeddingMetadata, ExtractionResult
from semantic_stego.data.image_io import crop_roi, rgb_to_gray, rgb_to_ycrcb
from semantic_stego.stego.embedder import _nearest_qim_codeword
from semantic_stego.svd.svd_from_scratch import svd_decompose


class SvdExtractor:
    def extract(
        self,
        stego_or_attacked_image: np.ndarray,
        metadata: EmbeddingMetadata,
        original_image: np.ndarray | None,
        decoder_type: str,
    ) -> ExtractionResult:
        start = perf_counter()
        stego_s = self._extract_singular_values(stego_or_attacked_image, metadata)
        if decoder_type == "non_blind" and original_image is None:
            raise ValueError("non_blind decoder requires original_image")
        if decoder_type not in {"non_blind", "blind"}:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")

        if decoder_type == "non_blind":
            original_s = self._extract_singular_values(original_image, metadata)
            symbol_bits = _decode_non_blind_bits(stego_s[metadata.indices], original_s[metadata.indices], metadata.qim_delta)
        else:
            symbol_bits = _decode_qim_bits(stego_s[metadata.indices], metadata.qim_delta)

        bits = _decode_repetition(symbol_bits, metadata.repetition_factor)

        elapsed_ms = (perf_counter() - start) * 1000.0
        return ExtractionResult(bits=bits[: metadata.payload_len], extraction_time_ms=elapsed_ms)

    def _extract_singular_values(self, image: np.ndarray, metadata: EmbeddingMetadata) -> np.ndarray:
        roi_patch = crop_roi(image, metadata.roi)
        if metadata.channel == "gray":
            channel = rgb_to_gray(roi_patch).astype(np.float64)
        else:
            roi_ycc = rgb_to_ycrcb(roi_patch)
            channel = roi_ycc[:, :, 0].astype(np.float64)
        _, S, _ = svd_decompose(channel)
        return S


def _decode_qim_bits(values: np.ndarray, delta: float) -> np.ndarray:
    bits = np.zeros(len(values), dtype=np.uint8)
    for index, value in enumerate(values.astype(float)):
        even = _nearest_qim_codeword(float(value), delta, 0)
        odd = _nearest_qim_codeword(float(value), delta, 1)
        bits[index] = np.uint8(0 if abs(value - even) <= abs(value - odd) else 1)
    return bits


def _decode_non_blind_bits(values: np.ndarray, original_values: np.ndarray, delta: float) -> np.ndarray:
    bits = np.zeros(len(values), dtype=np.uint8)
    for index, original_value in enumerate(original_values.astype(float)):
        observed = float(values[index])
        even = _nearest_qim_codeword(original_value, delta, 0)
        odd = _nearest_qim_codeword(original_value, delta, 1)
        bits[index] = np.uint8(0 if abs(observed - even) <= abs(observed - odd) else 1)
    return bits


def _decode_repetition(bits: np.ndarray, repetition_factor: int) -> np.ndarray:
    if repetition_factor <= 1 or len(bits) == 0:
        return bits.astype(np.uint8, copy=True)

    usable = (len(bits) // repetition_factor) * repetition_factor
    grouped = bits[:usable].reshape(-1, repetition_factor)
    threshold = (repetition_factor // 2) + 1
    return (grouped.sum(axis=1) >= threshold).astype(np.uint8)
