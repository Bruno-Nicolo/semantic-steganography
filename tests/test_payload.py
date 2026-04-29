from __future__ import annotations

import numpy as np
import pytest

from semantic_stego.stego.payload import PayloadCapacityError, bits_to_text, fit_payload_to_capacity, random_bits, text_to_bits


def test_text_bits_roundtrip() -> None:
    bits = text_to_bits("ciao")
    assert bits_to_text(bits).startswith("ciao")


def test_random_bits_reproducible() -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    assert np.array_equal(random_bits(16, rng_a), random_bits(16, rng_b))


def test_truncate_payload() -> None:
    bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
    fitted, truncated, dropped = fit_payload_to_capacity(bits, 3, "truncate_message")
    assert truncated is True
    assert dropped == 2
    assert np.array_equal(fitted, np.array([0, 1, 1], dtype=np.uint8))


def test_payload_capacity_error() -> None:
    with pytest.raises(PayloadCapacityError):
        fit_payload_to_capacity(np.ones(5, dtype=np.uint8), 2, "skip_image")
