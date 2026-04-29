from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from semantic_stego.config.schemas import ROI


def read_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def crop_roi(image: np.ndarray, roi: ROI) -> np.ndarray:
    return image[roi.y1:roi.y2, roi.x1:roi.x2].copy()


def paste_roi(image: np.ndarray, roi: ROI, roi_patch: np.ndarray) -> np.ndarray:
    output = image.copy()
    output[roi.y1:roi.y2, roi.x1:roi.x2] = roi_patch
    return output


def to_float32(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def rgb_to_ycrcb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(image), cv2.COLOR_RGB2YCrCb)


def ycrcb_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(image), cv2.COLOR_YCrCb2RGB)


def draw_roi(image: np.ndarray, roi: ROI, color: tuple[int, int, int] = (255, 0, 0)) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (roi.x1, roi.y1), (roi.x2, roi.y2), color, 2)
    return output
