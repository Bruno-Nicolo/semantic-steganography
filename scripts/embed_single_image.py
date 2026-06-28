from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from semantic_stego.config.defaults import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IMAGE_SIZE, DEFAULT_PAYLOAD_POLICY, DEFAULT_REPETITION_FACTOR, DEFAULT_YOLO_MODEL
from semantic_stego.data.image_io import draw_roi, read_image_rgb, save_image_rgb
from semantic_stego.detection.roi_selector import select_roi
from semantic_stego.detection.yolo_detector import YoloDetector
from semantic_stego.metrics.image_metrics import compute_roi_metrics
from semantic_stego.stego.embedder import SvdEmbedder
from semantic_stego.stego.payload import random_bits, text_to_bits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SVD/QIM steganography to a single image.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "single_image")
    parser.add_argument("--payload-text", default=None, help="Text payload. If omitted, random bits are used.")
    parser.add_argument("--payload-bits", type=int, default=128, help="Random payload length when --payload-text is omitted.")
    parser.add_argument("--payload-seed", type=int, default=42)
    parser.add_argument("--roi-strategy", choices=["full_image", "largest", "smallest", "random"], default="full_image")
    parser.add_argument("--svd-band", choices=["high_energy", "mid_energy", "low_energy"], default="mid_energy")
    parser.add_argument("--embedding-strength", type=float, default=20.0)
    parser.add_argument("--embedding-strength-mode", choices=["absolute", "proportional_singular"], default="absolute")
    parser.add_argument("--repetition-factor", type=int, default=DEFAULT_REPETITION_FACTOR)
    parser.add_argument("--payload-policy", choices=["truncate_message", "skip_image", "raise_error"], default=DEFAULT_PAYLOAD_POLICY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--min-roi-area", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_image_rgb(args.image)
    rng = np.random.default_rng(args.seed)
    payload_rng = np.random.default_rng(args.payload_seed)
    payload_bits = text_to_bits(args.payload_text) if args.payload_text is not None else random_bits(args.payload_bits, payload_rng)

    detections = []
    yolo_time_ms = 0.0
    if args.roi_strategy != "full_image":
        detector = YoloDetector(args.yolo_model, args.confidence_threshold, args.image_size)
        detections, yolo_time_ms = detector.detect(image)

    roi = select_roi(image.shape, detections, args.roi_strategy, rng, args.min_roi_area)
    if roi is None:
        raise SystemExit(f"No valid ROI found for strategy '{args.roi_strategy}'.")

    embedder = SvdEmbedder(payload_policy=args.payload_policy, repetition_factor=args.repetition_factor)
    result = embedder.embed(
        image=image,
        roi=roi,
        payload_bits=payload_bits,
        band=args.svd_band,
        strength=args.embedding_strength,
        strength_mode=args.embedding_strength_mode,
        mode="qim",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stego_path = args.output_dir / f"{args.image.stem}_stego.png"
    roi_path = args.output_dir / f"{args.image.stem}_roi.png"
    metadata_path = args.output_dir / f"{args.image.stem}_metadata.json"

    save_image_rgb(stego_path, result.stego_image)
    save_image_rgb(roi_path, draw_roi(image, roi))

    metadata = {
        "input_image": str(args.image),
        "stego_image": str(stego_path),
        "roi_debug_image": str(roi_path),
        "roi": asdict(roi),
        "payload_text": args.payload_text,
        "payload_bits_requested": int(len(payload_bits)),
        "payload_bits_embedded": int(len(result.embedded_bits)),
        "payload_bits_capacity": int(result.payload_bits_capacity),
        "payload_bits_dropped": int(result.payload_bits_dropped),
        "payload_truncated": bool(result.payload_truncated),
        "svd_band": args.svd_band,
        "embedding_strength": args.embedding_strength,
        "embedding_strength_mode": args.embedding_strength_mode,
        "embedding_delta_mean": result.metadata.strength,
        "repetition_factor": args.repetition_factor,
        "yolo_time_ms": yolo_time_ms,
        "embedding_time_ms": result.embedding_time_ms,
        "svd_time_ms": result.svd_time_ms,
        "numpy_svd_time_ms": result.numpy_svd_time_ms,
        "svd_reconstruction_error": result.svd_reconstruction_error,
        "metrics": compute_roi_metrics(image, result.stego_image, roi),
    }
    metadata_path.write_text(json.dumps(jsonify(metadata), indent=2), encoding="utf-8")

    print(f"Stego image: {stego_path}")
    print(f"ROI debug: {roi_path}")
    print(f"Metadata: {metadata_path}")


def jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
