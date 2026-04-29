from __future__ import annotations

import numpy as np


def bit_errors(original_bits: np.ndarray, recovered_bits: np.ndarray) -> int:
    a = np.asarray(original_bits, dtype=np.uint8)
    b = np.asarray(recovered_bits, dtype=np.uint8)
    overlap = min(len(a), len(b))
    errors = int(np.count_nonzero(a[:overlap] != b[:overlap]))
    errors += abs(len(a) - len(b))
    return errors


def bit_error_rate(original_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    total = max(len(original_bits), len(recovered_bits), 1)
    return bit_errors(original_bits, recovered_bits) / total


def exact_match(original_bits: np.ndarray, recovered_bits: np.ndarray) -> bool:
    return len(original_bits) == len(recovered_bits) and bit_errors(original_bits, recovered_bits) == 0


def character_accuracy(original_text: str, recovered_text: str) -> float:
    if not original_text:
        return 1.0 if not recovered_text else 0.0
    overlap = min(len(original_text), len(recovered_text))
    correct = sum(1 for index in range(overlap) if original_text[index] == recovered_text[index])
    return correct / len(original_text)
