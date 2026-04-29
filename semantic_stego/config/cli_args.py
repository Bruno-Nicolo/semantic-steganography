from __future__ import annotations

import argparse
from pathlib import Path

from semantic_stego.config.defaults import (
    DEFAULT_ATTACKS,
    DEFAULT_BLUR_KERNELS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DECODERS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_JPEG_QUALITIES,
    DEFAULT_NOISE_SIGMAS,
    DEFAULT_PAYLOAD_POLICY,
    DEFAULT_ROI_STRATEGIES,
    DEFAULT_SVD_BANDS,
    DEFAULT_YOLO_MODEL,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic steganography experiments on COCO.")
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/debug"))
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--roi-strategies", nargs="+", default=DEFAULT_ROI_STRATEGIES)
    parser.add_argument("--svd-bands", nargs="+", default=DEFAULT_SVD_BANDS)
    parser.add_argument("--decoders", nargs="+", default=DEFAULT_DECODERS)
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--jpeg-qualities", nargs="+", type=int, default=DEFAULT_JPEG_QUALITIES)
    parser.add_argument("--noise-sigmas", nargs="+", type=float, default=DEFAULT_NOISE_SIGMAS)
    parser.add_argument("--blur-kernels", nargs="+", type=int, default=DEFAULT_BLUR_KERNELS)
    parser.add_argument("--payload-text", default=None)
    parser.add_argument("--payload-bits", type=int, default=128)
    parser.add_argument("--payload-seed", type=int, default=42)
    parser.add_argument("--embedding-strength", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-roi-area", type=int, default=None)
    parser.add_argument("--payload-policy", choices=["truncate_message", "skip_image", "raise_error"], default=DEFAULT_PAYLOAD_POLICY)
    parser.add_argument("--skip-no-detection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-roi-debug", action="store_true")
    return parser
