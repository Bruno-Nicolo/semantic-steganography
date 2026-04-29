from __future__ import annotations

import numpy as np
import pytest

from semantic_stego.attacks.attacks import apply_gaussian_blur, apply_gaussian_noise, apply_jpeg_compression


def _image() -> np.ndarray:
    return np.full((32, 32, 3), 127, dtype=np.uint8)


def test_attacks_keep_shape_and_uint8() -> None:
    image = _image()
    for output in [apply_gaussian_noise(image, 5), apply_gaussian_blur(image, 3), apply_jpeg_compression(image, 30)]:
        assert output.shape == image.shape
        assert output.dtype == np.uint8


def test_jpeg_changes_image_low_quality() -> None:
    image = np.random.default_rng(42).integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    compressed = apply_jpeg_compression(image, 30)
    assert not np.array_equal(compressed, image)


def test_blur_rejects_even_kernel() -> None:
    with pytest.raises(ValueError):
        apply_gaussian_blur(_image(), 4)


def test_noise_is_clipped() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    noisy = apply_gaussian_noise(image, 100)
    assert noisy.min() >= 0
    assert noisy.max() <= 255
