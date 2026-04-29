from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ExperimentConfig:
    coco_root: Path
    split: str
    output_dir: Path
    max_images: int | None
    image_size: int
    yolo_model: str
    confidence_threshold: float
    roi_strategies: list[str]
    svd_bands: list[str]
    decoders: list[str]
    attacks: list[str]
    jpeg_qualities: list[int]
    noise_sigmas: list[float]
    blur_kernels: list[int]
    payload_text: str | None
    payload_bits: int
    payload_seed: int
    embedding_strength: float
    seed: int
    save_images: bool = False
    save_roi_debug: bool = False
    min_roi_area: int | None = None
    skip_no_detection: bool = True
    payload_policy: str = "truncate_message"


@dataclass(slots=True)
class ImageRecord:
    image_id: str
    image_path: Path


@dataclass(slots=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(slots=True)
class ROI:
    x1: int
    y1: int
    x2: int
    y2: int
    strategy: str
    class_id: int | None
    class_name: str | None
    confidence: float | None
    num_detections: int = 0

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(slots=True)
class AttackConfig:
    attack_type: str
    strength: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EmbeddingMetadata:
    roi: ROI
    band: str
    indices: np.ndarray
    payload_len: int
    strength: float
    mode: str
    qim_delta: float
    channel: str = "Y"


@dataclass(slots=True)
class EmbeddingResult:
    stego_image: np.ndarray
    metadata: EmbeddingMetadata
    embedded_bits: np.ndarray
    requested_bits: int
    svd_time_ms: float
    embedding_time_ms: float
    svd_reconstruction_error: float
    payload_bits_capacity: int
    payload_bits_dropped: int
    payload_truncated: bool


@dataclass(slots=True)
class ExtractionResult:
    bits: np.ndarray
    extraction_time_ms: float
