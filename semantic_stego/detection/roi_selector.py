from __future__ import annotations

import numpy as np

from semantic_stego.config.schemas import Detection, ROI


def select_roi(
    image_shape: tuple[int, int, int],
    detections: list[Detection],
    strategy: str,
    rng: np.random.Generator,
    min_roi_area: int | None = None,
) -> ROI | None:
    height, width = image_shape[:2]
    if strategy == "full_image":
        return ROI(0, 0, width, height, "full_image", None, "full_image", None, len(detections))

    valid = [d for d in detections if d.area > 0 and (min_roi_area is None or d.area >= min_roi_area)]
    if not valid:
        return None

    if strategy == "largest":
        detection = max(valid, key=lambda item: item.area)
    elif strategy == "smallest":
        detection = min(valid, key=lambda item: item.area)
    elif strategy == "random":
        detection = valid[int(rng.integers(0, len(valid)))]
    else:
        raise ValueError(f"Unsupported ROI strategy: {strategy}")

    return ROI(
        x1=detection.x1,
        y1=detection.y1,
        x2=detection.x2,
        y2=detection.y2,
        strategy=strategy,
        class_id=detection.class_id,
        class_name=detection.class_name,
        confidence=detection.confidence,
        num_detections=len(detections),
    )
