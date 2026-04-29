from __future__ import annotations

from pathlib import Path

from semantic_stego.config.schemas import ExperimentConfig

DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_ROI_STRATEGIES = ["largest", "smallest", "random", "full_image"]
DEFAULT_SVD_BANDS = ["high_energy", "mid_energy", "low_energy"]
DEFAULT_DECODERS = ["non_blind", "blind"]
DEFAULT_ATTACKS = ["none", "gaussian_noise", "gaussian_blur", "jpeg_compression"]
DEFAULT_JPEG_QUALITIES = [90, 70, 50, 30]
DEFAULT_NOISE_SIGMAS = [5, 10, 20]
DEFAULT_BLUR_KERNELS = [3, 5, 7]
DEFAULT_PAYLOAD_POLICY = "truncate_message"
DEFAULT_SKIP_NO_DETECTION = True
DEFAULT_IMAGE_SIZE = 640
DEFAULT_YOLO_MODEL = "yolov8n.pt"


def build_default_debug_config() -> ExperimentConfig:
    return ExperimentConfig(
        coco_root=Path("data/coco"),
        split="val2017",
        output_dir=Path("outputs/debug"),
        max_images=10,
        image_size=DEFAULT_IMAGE_SIZE,
        yolo_model=DEFAULT_YOLO_MODEL,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        roi_strategies=["largest", "full_image"],
        svd_bands=["mid_energy"],
        decoders=["non_blind"],
        attacks=["none"],
        jpeg_qualities=[90],
        noise_sigmas=[5],
        blur_kernels=[3],
        payload_text=None,
        payload_bits=64,
        payload_seed=42,
        embedding_strength=10.0,
        seed=42,
        save_images=False,
        save_roi_debug=True,
        min_roi_area=None,
        skip_no_detection=DEFAULT_SKIP_NO_DETECTION,
        payload_policy=DEFAULT_PAYLOAD_POLICY,
    )
