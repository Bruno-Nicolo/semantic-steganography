from __future__ import annotations

from time import perf_counter

import numpy as np

from semantic_stego.config.schemas import EmbeddingMetadata, ExtractionResult
from semantic_stego.data.image_io import crop_roi, rgb_to_ycrcb
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
            bits = _decode_non_blind_bits(stego_s[metadata.indices], original_s[metadata.indices], metadata.qim_delta)
        else:
            bits = _decode_qim_bits(stego_s[metadata.indices], metadata.qim_delta)

        elapsed_ms = (perf_counter() - start) * 1000.0
        return ExtractionResult(bits=bits[: metadata.payload_len], extraction_time_ms=elapsed_ms)

    def _extract_singular_values(self, image: np.ndarray, metadata: EmbeddingMetadata) -> np.ndarray:
        roi_patch = crop_roi(image, metadata.roi)
        roi_ycc = rgb_to_ycrcb(roi_patch)
        y_channel = roi_ycc[:, :, 0].astype(np.float64)
        _, S, _ = svd_decompose(y_channel)
        return S


def _decode_qim_bits(values: np.ndarray, delta: float) -> np.ndarray:
    quantized = np.rint(values / delta).astype(int)
    return (quantized % 2).astype(np.uint8)


def _decode_non_blind_bits(values: np.ndarray, original_values: np.ndarray, delta: float) -> np.ndarray:
    return (values >= original_values).astype(np.uint8)
