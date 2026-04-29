from __future__ import annotations

import numpy as np

from semantic_stego.config.schemas import Detection
from semantic_stego.detection.roi_selector import select_roi


def _detections() -> list[Detection]:
    return [
        Detection(0, 0, 10, 10, 0.9, 0, "a"),
        Detection(0, 0, 20, 15, 0.8, 1, "b"),
        Detection(5, 5, 8, 8, 0.7, 2, "c"),
    ]


def test_largest_selects_max_area() -> None:
    roi = select_roi((30, 30, 3), _detections(), "largest", np.random.default_rng(42))
    assert roi is not None
    assert (roi.x2 - roi.x1) * (roi.y2 - roi.y1) == 300


def test_smallest_selects_min_area() -> None:
    roi = select_roi((30, 30, 3), _detections(), "smallest", np.random.default_rng(42))
    assert roi is not None
    assert (roi.x2 - roi.x1) * (roi.y2 - roi.y1) == 9


def test_random_is_reproducible() -> None:
    roi_a = select_roi((30, 30, 3), _detections(), "random", np.random.default_rng(42))
    roi_b = select_roi((30, 30, 3), _detections(), "random", np.random.default_rng(42))
    assert roi_a == roi_b


def test_full_image_returns_whole_image() -> None:
    roi = select_roi((40, 50, 3), [], "full_image", np.random.default_rng(42))
    assert roi is not None
    assert (roi.x1, roi.y1, roi.x2, roi.y2) == (0, 0, 50, 40)


def test_none_when_no_detection_for_yolo_strategy() -> None:
    roi = select_roi((40, 50, 3), [], "largest", np.random.default_rng(42))
    assert roi is None
