from __future__ import annotations

from time import perf_counter

import numpy as np
from ultralytics import YOLO

from semantic_stego.config.schemas import Detection


class YoloDetector:
    def __init__(self, model_name: str, confidence_threshold: float, image_size: int = 640):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        self._model = YOLO(self.model_name)
        return self._model

    def detect(self, image: np.ndarray) -> tuple[list[Detection], float]:
        model = self._load_model()
        start = perf_counter()
        results = model.predict(image, imgsz=self.image_size, verbose=False)
        elapsed_ms = (perf_counter() - start) * 1000.0

        detections: list[Detection] = []
        height, width = image.shape[:2]
        for result in results:
            names = result.names
            for box in result.boxes:
                confidence = float(box.conf.item())
                if confidence < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                class_id = int(box.cls.item())
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=confidence,
                        class_id=class_id,
                        class_name=str(names[class_id]),
                    )
                )
        return detections, elapsed_ms
