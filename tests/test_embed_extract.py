from __future__ import annotations

import numpy as np

from semantic_stego.config.schemas import ROI
from semantic_stego.stego.embedder import SvdEmbedder
from semantic_stego.stego.extractor import SvdExtractor


def _image() -> np.ndarray:
    return np.random.default_rng(42).integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_embedding_modifies_image_and_preserves_shape() -> None:
    image = _image()
    roi = ROI(0, 0, 64, 64, "full_image", None, "full_image", None, 0)
    bits = np.array([0, 1, 1, 0, 1, 0, 1, 1], dtype=np.uint8)
    result = SvdEmbedder().embed(image, roi, bits, "mid_energy", 10.0)
    assert result.stego_image.shape == image.shape
    assert not np.array_equal(result.stego_image, image)


def test_non_blind_recovers_payload_without_attack() -> None:
    image = _image()
    roi = ROI(0, 0, 64, 64, "full_image", None, "full_image", None, 0)
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    embed_result = SvdEmbedder().embed(image, roi, bits, "high_energy", 20.0)
    extracted = SvdExtractor().extract(embed_result.stego_image, embed_result.metadata, image, "non_blind")
    assert np.array_equal(extracted.bits, embed_result.embedded_bits)


def test_blind_recovers_small_payload_clean() -> None:
    image = _image()
    roi = ROI(0, 0, 64, 64, "full_image", None, "full_image", None, 0)
    bits = np.array([1, 0], dtype=np.uint8)
    embed_result = SvdEmbedder().embed(image, roi, bits, "high_energy", 20.0)
    extracted = SvdExtractor().extract(embed_result.stego_image, embed_result.metadata, None, "blind")
    assert np.array_equal(extracted.bits, embed_result.embedded_bits)
