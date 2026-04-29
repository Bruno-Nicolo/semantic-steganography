from __future__ import annotations

import cv2
import numpy as np

from semantic_stego.data.image_io import to_uint8


def apply_attack(image: np.ndarray, attack_type: str, params: dict) -> np.ndarray:
    if attack_type == "none":
        return image.copy()
    if attack_type == "gaussian_noise":
        return apply_gaussian_noise(
            image,
            sigma=float(params["sigma"]),
            mean=float(params.get("mean", 0.0)),
            rng=params.get("rng"),
        )
    if attack_type == "gaussian_blur":
        return apply_gaussian_blur(image, kernel_size=int(params["kernel_size"]), sigma=params.get("sigma"))
    if attack_type == "jpeg_compression":
        return apply_jpeg_compression(image, quality=int(params["quality"]))
    raise ValueError(f"Unsupported attack type: {attack_type}")


def apply_gaussian_noise(image: np.ndarray, sigma: float, mean: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    generator = rng or np.random.default_rng()
    noisy = image.astype(np.float32) + generator.normal(mean, sigma, size=image.shape)
    return to_uint8(noisy)


def apply_gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float | None = None) -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("Gaussian blur kernel_size must be odd")
    sigma_value = 0 if sigma is None else sigma
    return cv2.GaussianBlur(to_uint8(image), (kernel_size, kernel_size), sigmaX=sigma_value)


def apply_jpeg_compression(image: np.ndarray, quality: int) -> np.ndarray:
    success, encoded = cv2.imencode(".jpg", cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
