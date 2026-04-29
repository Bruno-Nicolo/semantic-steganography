from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from semantic_stego.attacks.attacks import apply_attack
from semantic_stego.config.schemas import ExperimentConfig, ROI
from semantic_stego.data.coco_loader import CocoImageLoader
from semantic_stego.data.image_io import draw_roi, read_image_rgb, save_image_rgb
from semantic_stego.detection.roi_selector import select_roi
from semantic_stego.detection.yolo_detector import YoloDetector
from semantic_stego.experiments.grid import build_attack_grid
from semantic_stego.experiments.result_writer import ResultWriter
from semantic_stego.metrics.image_metrics import compute_psnr, compute_roi_metrics, compute_ssim
from semantic_stego.metrics.message_metrics import bit_error_rate, bit_errors, character_accuracy, exact_match
from semantic_stego.metrics.timing import Timer
from semantic_stego.stego.embedder import SvdEmbedder
from semantic_stego.stego.extractor import SvdExtractor
from semantic_stego.stego.payload import PayloadCapacityError, bits_to_text, random_bits, text_to_bits

LOGGER = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.payload_rng = np.random.default_rng(config.payload_seed)
        self.loader = CocoImageLoader(config.coco_root, config.split, config.max_images, config.seed)
        self.detector = YoloDetector(config.yolo_model, config.confidence_threshold, config.image_size)
        self.embedder = SvdEmbedder(config.payload_policy)
        self.extractor = SvdExtractor()
        self.writer = ResultWriter(config.output_dir)
        self.run_id = config.output_dir.name

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        self.writer.save_config(self.config)
        records = self.loader.iter_records()
        attacks = build_attack_grid(self.config)

        for record in tqdm(records, desc="images"):
            self._process_image(record.image_id, record.image_path, attacks)

        self.writer.close()
        self._write_summary()

    def _process_image(self, image_id: str, image_path: Path, attacks) -> None:
        image_start = perf_counter()
        image = read_image_rgb(image_path)
        try:
            if any(strategy != "full_image" for strategy in self.config.roi_strategies):
                detections, yolo_time_ms = self.detector.detect(image)
            else:
                detections, yolo_time_ms = [], 0.0
        except Exception as exc:
            LOGGER.error("YOLO failed on %s: %s", image_path, exc)
            self.writer.write_result(self._failure_row(image_id, image_path, image, "failed_unknown", str(exc), 0.0))
            return

        for roi_strategy in self.config.roi_strategies:
            roi = select_roi(image.shape, detections, roi_strategy, self.rng, self.config.min_roi_area)
            if roi is None:
                if self.config.skip_no_detection:
                    self.writer.write_result(
                        self._failure_row(
                            image_id, image_path, image, "failed_no_detection", "No valid detection for ROI strategy", yolo_time_ms, roi_strategy=roi_strategy
                        )
                    )
                    continue

            assert roi is not None
            if self.config.save_roi_debug:
                debug_path = self.config.output_dir / "roi_debug" / f"{image_id}_{roi.strategy}.jpg"
                save_image_rgb(debug_path, draw_roi(image, roi))

            payload_bits = self._build_payload_bits()
            for svd_band in self.config.svd_bands:
                try:
                    embed_result = self.embedder.embed(
                        image=image,
                        roi=roi,
                        payload_bits=payload_bits,
                        band=svd_band,
                        strength=self.config.embedding_strength,
                        mode="qim",
                    )
                except PayloadCapacityError as exc:
                    self.writer.write_result(
                        self._failure_row(
                            image_id, image_path, image, "failed_payload_too_large", str(exc), yolo_time_ms, roi_strategy=roi.strategy, roi=roi, svd_band=svd_band
                        )
                    )
                    continue
                except Exception as exc:
                    self.writer.write_result(
                        self._failure_row(
                            image_id, image_path, image, "failed_svd", str(exc), yolo_time_ms, roi_strategy=roi.strategy, roi=roi, svd_band=svd_band
                        )
                    )
                    continue

                clean_stego = embed_result.stego_image
                if self.config.save_images:
                    image_out = self.config.output_dir / "images" / f"{image_id}_{roi.strategy}_{svd_band}.png"
                    save_image_rgb(image_out, clean_stego)

                for attack in attacks:
                    attack_params = dict(attack.params)
                    attack_params["rng"] = self.rng
                    with Timer() as attack_timer:
                        attacked = apply_attack(clean_stego, attack.attack_type, attack_params)

                    for decoder_type in self.config.decoders:
                        original_for_decoder = image if decoder_type == "non_blind" else None
                        try:
                            extract_result = self.extractor.extract(
                                attacked,
                                metadata=embed_result.metadata,
                                original_image=original_for_decoder,
                                decoder_type=decoder_type,
                            )
                            row = self._success_row(
                                image_id=image_id,
                                image_path=image_path,
                                image=image,
                                roi=roi,
                                detections_count=len(detections),
                                yolo_time_ms=yolo_time_ms,
                                embed_result=embed_result,
                                attack=attack,
                                attack_time_ms=attack_timer.elapsed_ms,
                                attacked=attacked,
                                decoder_type=decoder_type,
                                extract_result=extract_result,
                                total_time_ms=(perf_counter() - image_start) * 1000.0,
                            )
                        except Exception as exc:
                            row = self._failure_row(
                                image_id,
                                image_path,
                                image,
                                "failed_decode",
                                str(exc),
                                yolo_time_ms,
                                roi_strategy=roi.strategy,
                                roi=roi,
                                svd_band=svd_band,
                                decoder_type=decoder_type,
                                attack_type=attack.attack_type,
                            )
                        self.writer.write_result(row)

    def _build_payload_bits(self) -> np.ndarray:
        if self.config.payload_text is not None:
            return text_to_bits(self.config.payload_text)
        return random_bits(self.config.payload_bits, self.payload_rng)

    def _success_row(self, *, image_id, image_path, image, roi: ROI, detections_count, yolo_time_ms, embed_result, attack, attack_time_ms, attacked, decoder_type, extract_result, total_time_ms):
        embedded_bits = embed_result.embedded_bits
        recovered_bits = extract_result.bits[: len(embedded_bits)]
        bit_err = bit_errors(embedded_bits, recovered_bits)
        total_bits = len(embedded_bits)
        payload_text = self.config.payload_text
        recovered_text = bits_to_text(recovered_bits) if payload_text is not None else ""
        image_metrics = compute_roi_metrics(image, attacked, roi)
        correct_bits = max(0, total_bits - bit_err)
        payload_requested = embed_result.requested_bits
        height, width = image.shape[:2]
        return {
            "run_id": self.run_id,
            "dataset": f"coco/{self.config.split}",
            "image_id": image_id,
            "image_path": str(image_path),
            "image_width": width,
            "image_height": height,
            "roi_strategy": roi.strategy,
            "roi_class_id": roi.class_id,
            "roi_class_name": roi.class_name,
            "roi_confidence": roi.confidence,
            "roi_x1": roi.x1,
            "roi_y1": roi.y1,
            "roi_x2": roi.x2,
            "roi_y2": roi.y2,
            "roi_width": roi.width,
            "roi_height": roi.height,
            "roi_area": roi.area,
            "roi_area_ratio": roi.area / float(width * height),
            "num_detections": detections_count,
            "svd_band": embed_result.metadata.band,
            "decoder_type": decoder_type,
            "embedding_strength": self.config.embedding_strength,
            "payload_bits": self.config.payload_bits,
            "payload_text": payload_text,
            "payload_seed": self.config.payload_seed,
            "payload_bits_requested": payload_requested,
            "payload_bits_capacity": embed_result.payload_bits_capacity,
            "payload_bits_embedded": total_bits,
            "payload_bits_dropped": embed_result.payload_bits_dropped,
            "payload_retention_ratio": total_bits / max(payload_requested, 1),
            "payload_truncated": embed_result.payload_truncated,
            "payload_success_ratio": correct_bits / max(payload_requested, 1),
            "bpp_roi": total_bits / max(roi.area, 1),
            "bpp_image": total_bits / max(width * height, 1),
            "attack_type": attack.attack_type,
            "attack_strength": attack.strength,
            "attack_param_sigma": attack.params.get("sigma"),
            "attack_param_kernel": attack.params.get("kernel_size"),
            "attack_param_quality": attack.params.get("quality"),
            "PSNR_full": compute_psnr(image, attacked),
            "PSNR_roi": image_metrics["PSNR_roi"],
            "SSIM_full": compute_ssim(image, attacked),
            "SSIM_roi": image_metrics["SSIM_roi"],
            "bit_errors": bit_err,
            "total_bits": total_bits,
            "BER": bit_error_rate(embedded_bits, recovered_bits),
            "exact_match": exact_match(embedded_bits, recovered_bits),
            "character_accuracy": character_accuracy(payload_text or "", recovered_text) if payload_text is not None else None,
            "yolo_time_ms": yolo_time_ms,
            "embedding_time_ms": embed_result.embedding_time_ms,
            "extraction_time_ms": extract_result.extraction_time_ms,
            "svd_time_ms": embed_result.svd_time_ms,
            "attack_time_ms": attack_time_ms,
            "total_time_ms": total_time_ms,
            "svd_reconstruction_error": embed_result.svd_reconstruction_error,
            "status": "success",
            "error_message": None,
        }

    def _failure_row(self, image_id, image_path, image, status, error_message, yolo_time_ms, **extras):
        height, width = image.shape[:2]
        roi = extras.get("roi")
        roi_strategy = extras.get("roi_strategy")
        return {
            "run_id": self.run_id,
            "dataset": f"coco/{self.config.split}",
            "image_id": image_id,
            "image_path": str(image_path),
            "image_width": width,
            "image_height": height,
            "roi_strategy": roi_strategy or (roi.strategy if roi else None),
            "roi_class_id": roi.class_id if roi else None,
            "roi_class_name": roi.class_name if roi else None,
            "roi_confidence": roi.confidence if roi else None,
            "roi_x1": roi.x1 if roi else None,
            "roi_y1": roi.y1 if roi else None,
            "roi_x2": roi.x2 if roi else None,
            "roi_y2": roi.y2 if roi else None,
            "roi_width": roi.width if roi else None,
            "roi_height": roi.height if roi else None,
            "roi_area": roi.area if roi else None,
            "roi_area_ratio": (roi.area / float(width * height)) if roi else None,
            "num_detections": roi.num_detections if roi else 0,
            "svd_band": extras.get("svd_band"),
            "decoder_type": extras.get("decoder_type"),
            "embedding_strength": self.config.embedding_strength,
            "payload_bits": self.config.payload_bits,
            "payload_text": self.config.payload_text,
            "payload_seed": self.config.payload_seed,
            "attack_type": extras.get("attack_type"),
            "yolo_time_ms": yolo_time_ms,
            "status": status,
            "error_message": error_message,
        }

    def _write_summary(self) -> None:
        results_path = self.config.output_dir / "results.csv"
        if not results_path.exists():
            return
        frame = pd.read_csv(results_path)
        success = frame[frame["status"] == "success"]
        if success.empty:
            return
        summary = success.groupby(["roi_strategy", "svd_band", "decoder_type", "attack_type"], dropna=False).agg(
            BER_mean=("BER", "mean"),
            BER_std=("BER", "std"),
            PSNR_roi_mean=("PSNR_roi", "mean"),
            SSIM_roi_mean=("SSIM_roi", "mean"),
            MRR=("exact_match", "mean"),
        )
        summary.to_csv(self.config.output_dir / "summary.csv")
