from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


RESULT_COLUMNS = [
    "run_id", "dataset", "image_id", "image_path", "image_width", "image_height",
    "roi_strategy", "roi_class_id", "roi_class_name", "roi_confidence", "roi_x1", "roi_y1", "roi_x2", "roi_y2",
    "roi_width", "roi_height", "roi_area", "roi_area_ratio", "num_detections",
    "svd_band", "decoder_type", "embedding_strength", "payload_bits", "payload_text", "payload_seed",
    "payload_bits_requested", "payload_bits_capacity", "payload_bits_embedded", "payload_bits_dropped",
    "payload_retention_ratio", "payload_truncated", "payload_success_ratio", "bpp_roi", "bpp_image",
    "attack_type", "attack_strength", "attack_param_sigma", "attack_param_kernel", "attack_param_quality",
    "PSNR_full", "PSNR_roi", "SSIM_full", "SSIM_roi", "bit_errors", "total_bits", "BER", "exact_match",
    "character_accuracy", "yolo_time_ms", "embedding_time_ms", "extraction_time_ms", "svd_time_ms", "attack_time_ms",
    "total_time_ms", "svd_reconstruction_error", "status", "error_message",
]


class ResultWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_csv_path = self.output_dir / "results.csv"
        self.results_jsonl_path = self.output_dir / "results.jsonl"
        self.failures_jsonl_path = self.output_dir / "failures.jsonl"
        self._csv_file = self.results_csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=RESULT_COLUMNS)
        self._csv_writer.writeheader()

    def save_config(self, config: Any) -> None:
        with (self.output_dir / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(_jsonify(config), handle, indent=2)

    def write_result(self, row: dict[str, Any]) -> None:
        normalized = {column: row.get(column) for column in RESULT_COLUMNS}
        self._csv_writer.writerow(normalized)
        self._csv_file.flush()
        with self.results_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonify(normalized)) + "\n")
        if normalized.get("status", "success") != "success":
            with self.failures_jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(_jsonify(normalized)) + "\n")

    def close(self) -> None:
        self._csv_file.close()


def _jsonify(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
