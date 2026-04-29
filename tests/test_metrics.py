from __future__ import annotations

import math

import numpy as np

from semantic_stego.metrics.image_metrics import compute_psnr
from semantic_stego.metrics.message_metrics import bit_error_rate, exact_match


def test_psnr_identical_is_infinite_or_high() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    psnr = compute_psnr(image, image)
    assert math.isinf(psnr) or psnr > 90


def test_ber_zero_for_identical_bits() -> None:
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    assert bit_error_rate(bits, bits) == 0.0


def test_ber_correct_for_known_errors() -> None:
    a = np.array([0, 1, 1, 0], dtype=np.uint8)
    b = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert bit_error_rate(a, b) == 0.5


def test_exact_match() -> None:
    a = np.array([0, 1, 1, 0], dtype=np.uint8)
    b = np.array([0, 1, 1, 0], dtype=np.uint8)
    c = np.array([0, 1, 0, 0], dtype=np.uint8)
    assert exact_match(a, b) is True
    assert exact_match(a, c) is False
