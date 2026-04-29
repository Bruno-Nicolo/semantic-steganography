from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from semantic_stego.config.schemas import ROI
from semantic_stego.data.image_io import crop_roi


def compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(original, modified, data_range=255))


def compute_ssim(original: np.ndarray, modified: np.ndarray) -> float:
    min_dim = min(original.shape[0], original.shape[1])
    if min_dim < 3:
        return 1.0 if np.array_equal(original, modified) else 0.0
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    return float(structural_similarity(original, modified, channel_axis=2, data_range=255, win_size=win_size))


def compute_roi_metrics(original: np.ndarray, modified: np.ndarray, roi: ROI) -> dict[str, float]:
    original_roi = crop_roi(original, roi)
    modified_roi = crop_roi(modified, roi)
    return {
        "PSNR_roi": compute_psnr(original_roi, modified_roi),
        "SSIM_roi": compute_ssim(original_roi, modified_roi),
    }
