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


def test_non_blind_recovers_payload_with_proportional_strength() -> None:
    image = _image()
    roi = ROI(0, 0, 64, 64, "full_image", None, "full_image", None, 0)
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)

    embed_result = SvdEmbedder().embed(image, roi, bits, "high_energy", 0.05, strength_mode="proportional_singular")
    extracted = SvdExtractor().extract(embed_result.stego_image, embed_result.metadata, image, "non_blind")

    assert np.array_equal(extracted.bits, embed_result.embedded_bits)
    assert not np.isscalar(embed_result.metadata.qim_delta)


def test_blind_recovers_small_payload_clean() -> None:
    image = _image()
    roi = ROI(0, 0, 64, 64, "full_image", None, "full_image", None, 0)
    bits = np.array([1, 0], dtype=np.uint8)
    embed_result = SvdEmbedder().embed(image, roi, bits, "high_energy", 20.0)
    extracted = SvdExtractor().extract(embed_result.stego_image, embed_result.metadata, None, "blind")
    assert np.array_equal(extracted.bits, embed_result.embedded_bits)


def test_non_blind_recovers_repeated_payload_across_multiple_clean_images() -> None:
    rng = np.random.default_rng(123)
    roi = ROI(0, 0, 64, 64, "full_image", None, "full_image", None, 0)
    embedder = SvdEmbedder()
    extractor = SvdExtractor()

    for _ in range(10):
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        bits = rng.integers(0, 2, size=4, dtype=np.uint8)
        embed_result = embedder.embed(image, roi, bits, "high_energy", 10.0)
        assert len(embed_result.embedded_bits) == len(bits)

        extracted = extractor.extract(embed_result.stego_image, embed_result.metadata, image, "non_blind")
        assert np.array_equal(extracted.bits, embed_result.embedded_bits)


def test_payload_capacity_accounts_for_repetition_factor() -> None:
    image = _image()
    roi = ROI(0, 0, 64, 64, "full_image", None, "full_image", None, 0)
    bits = np.zeros(16, dtype=np.uint8)

    repeated = SvdEmbedder(repetition_factor=3).embed(image, roi, bits, "high_energy", 10.0)
    unrepeated = SvdEmbedder(repetition_factor=1).embed(image, roi, bits, "high_energy", 10.0)

    assert repeated.payload_bits_capacity <= unrepeated.payload_bits_capacity // 3 + 1
    assert repeated.metadata.repetition_factor == 3
    assert unrepeated.metadata.repetition_factor == 1
