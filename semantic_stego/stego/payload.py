from __future__ import annotations

import numpy as np


def text_to_bits(text: str, encoding: str = "utf-8") -> np.ndarray:
    payload = text.encode(encoding)
    return np.unpackbits(np.frombuffer(payload, dtype=np.uint8))


def bits_to_text(bits: np.ndarray, encoding: str = "utf-8") -> str:
    array = np.asarray(bits, dtype=np.uint8)
    if len(array) == 0:
        return ""
    remainder = len(array) % 8
    if remainder:
        array = np.pad(array, (0, 8 - remainder))
    payload = np.packbits(array).tobytes()
    return payload.decode(encoding, errors="ignore")


def random_bits(n_bits: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n_bits, endpoint=False, dtype=np.uint8)


def fit_payload_to_capacity(bits: np.ndarray, capacity: int, policy: str) -> tuple[np.ndarray, bool, int]:
    if len(bits) <= capacity:
        return bits.copy(), False, 0
    if policy == "truncate_message":
        dropped = len(bits) - capacity
        return bits[:capacity].copy(), True, dropped
    if policy == "skip_image":
        raise PayloadCapacityError("Payload exceeds ROI capacity")
    if policy == "raise_error":
        raise PayloadCapacityError("Payload exceeds ROI capacity")
    raise ValueError(f"Unsupported payload policy: {policy}")


class PayloadCapacityError(RuntimeError):
    pass
