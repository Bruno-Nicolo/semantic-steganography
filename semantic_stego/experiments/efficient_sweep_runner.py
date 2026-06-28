from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from semantic_stego.attacks.attacks import apply_attack
from semantic_stego.config.schemas import AttackConfig, ExperimentConfig, ROI
from semantic_stego.data.coco_loader import CocoImageLoader
from semantic_stego.data.image_io import crop_roi, draw_roi, read_image_rgb, rgb_to_gray, save_image_rgb
from semantic_stego.detection.roi_selector import select_roi
from semantic_stego.detection.yolo_detector import YoloDetector
from semantic_stego.experiments.grid import build_attack_grid
from semantic_stego.experiments.result_writer import ResultWriter
from semantic_stego.metrics.image_metrics import compute_psnr, compute_roi_metrics, compute_ssim
from semantic_stego.metrics.message_metrics import bit_error_rate, bit_errors, exact_match
from semantic_stego.stego.embedder import SvdEmbedder
from semantic_stego.stego.extractor import SvdExtractor
from semantic_stego.stego.payload import PayloadCapacityError, random_bits
from semantic_stego.svd.svd_from_scratch import measure_numpy_svd_time_ms, svd_decompose
from semantic_stego.svd.svd_utils import compute_reconstruction_error, effective_singular_capacity

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StrengthSetting:
    value: float
    mode: str


@dataclass(slots=True)
class RoiSvdCache:
    gray_channel: np.ndarray
    U: np.ndarray
    S: np.ndarray
    Vt: np.ndarray
    svd_time_ms: float
    numpy_svd_time_ms: float
    reconstruction_error: float


class EfficientSweepRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        payload_bits_values: list[int],
        absolute_deltas: list[float],
        proportional_deltas: list[float],
        experiment_block: str = "comprehensive_grid",
    ):
        self.config = config
        self.payload_bits_values = payload_bits_values
        self.strength_settings = [
            *[StrengthSetting(value, "absolute") for value in absolute_deltas],
            *[StrengthSetting(value, "proportional_singular") for value in proportional_deltas],
        ]
        self.experiment_block = experiment_block
        self.rng = np.random.default_rng(config.seed)
        self.payload_rng = np.random.default_rng(config.payload_seed)
        self.loader = CocoImageLoader(config.coco_root, config.split, None, config.seed)
        self.detector = YoloDetector(config.yolo_model, config.confidence_threshold, config.image_size)
        self.embedder = SvdEmbedder(config.payload_policy, config.repetition_factor)
        self.extractor = SvdExtractor()
        self.writer = ResultWriter(config.output_dir)
        self.run_id = config.output_dir.name
        self.accepted_images = 0

    def run(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        self.writer.save_config(
            {
                "base_config": self.config,
                "payload_bits_values": self.payload_bits_values,
                "absolute_deltas": [setting.value for setting in self.strength_settings if setting.mode == "absolute"],
                "proportional_deltas": [setting.value for setting in self.strength_settings if setting.mode == "proportional_singular"],
                "experiment_block": self.experiment_block,
                "execution_order": "image_centric",
            }
        )
        attacks = build_attack_grid(self.config)
        records = self.loader.iter_records()

        for record in tqdm(records, desc="images"):
            if self.config.max_images is not None and self.accepted_images >= self.config.max_images:
                break
            self._process_image(record.image_id, record.image_path, attacks)

        self.writer.close()
        self._write_summary()

    def _process_image(self, image_id: str, image_path: Path, attacks: list[AttackConfig]) -> None:
        image_start = perf_counter()
        image = read_image_rgb(image_path)
        height, width = image.shape[:2]
        payloads = {payload_bits: random_bits(payload_bits, self.payload_rng) for payload_bits in self.payload_bits_values}

        try:
            if any(strategy != "full_image" for strategy in self.config.roi_strategies):
                detections, yolo_time_ms = self.detector.detect(image)
            else:
                detections, yolo_time_ms = [], 0.0
        except Exception as exc:
            LOGGER.error("YOLO failed on %s: %s", image_path, exc)
            self.writer.write_result(self._failure_row(image_id, image_path, image, "failed_unknown", str(exc), 0.0))
            return

        rois = self._select_rois(image, detections)
        if not rois:
            self.writer.write_result(
                self._failure_row(
                    image_id,
                    image_path,
                    image,
                    "failed_no_roi",
                    "No requested ROI strategy produced a valid ROI",
                    yolo_time_ms,
                    image_filter_reason="No requested ROI strategy produced a valid ROI",
                )
            )
            return

        self.accepted_images += 1
        roi_caches = self._build_roi_caches(image, rois)
        for roi_strategy, roi in rois.items():
            if self.config.save_roi_debug:
                debug_path = self.config.output_dir / "roi_debug" / f"{image_id}_{roi.strategy}.jpg"
                save_image_rgb(debug_path, draw_roi(image, roi))

            cache = roi_caches[roi_strategy]
            capacity_cache: dict[tuple[str, float, str], int] = {}
            for svd_band in self.config.svd_bands:
                for strength in self.strength_settings:
                    symbol_capacity = effective_singular_capacity(cache.S, svd_band, strength.value, strength.mode)
                    capacity_cache[(svd_band, strength.value, strength.mode)] = symbol_capacity // self.embedder.repetition_factor

            for payload_bits, payload in payloads.items():
                for strength in self.strength_settings:
                    for svd_band in self.config.svd_bands:
                        payload_capacity = capacity_cache[(svd_band, strength.value, strength.mode)]
                        if not self._capacity_is_usable(payload_capacity, len(payload)):
                            self.writer.write_result(
                                self._failure_row(
                                    image_id,
                                    image_path,
                                    image,
                                    "failed_payload_incompatible",
                                    self._capacity_error(roi.strategy, svd_band, payload_capacity, len(payload)),
                                    yolo_time_ms,
                                    roi=roi,
                                    roi_strategy=roi.strategy,
                                    svd_band=svd_band,
                                    payload_bits=payload_bits,
                                    embedding_strength=strength.value,
                                    embedding_strength_mode=strength.mode,
                                    image_filter_reason=self._capacity_error(roi.strategy, svd_band, payload_capacity, len(payload)),
                                )
                            )
                            continue

                        try:
                            embed_result = self.embedder.embed_from_decomposition(
                                image=image,
                                roi=roi,
                                payload_bits=payload,
                                band=svd_band,
                                strength=strength.value,
                                strength_mode=strength.mode,
                                mode="qim",
                                gray_channel=cache.gray_channel,
                                U=cache.U,
                                S=cache.S,
                                Vt=cache.Vt,
                                svd_time_ms=cache.svd_time_ms,
                                numpy_svd_time_ms=cache.numpy_svd_time_ms,
                                svd_reconstruction_error=cache.reconstruction_error,
                            )
                        except PayloadCapacityError as exc:
                            self.writer.write_result(
                                self._failure_row(
                                    image_id,
                                    image_path,
                                    image,
                                    "failed_payload_too_large",
                                    str(exc),
                                    yolo_time_ms,
                                    roi=roi,
                                    roi_strategy=roi.strategy,
                                    svd_band=svd_band,
                                    payload_bits=payload_bits,
                                    embedding_strength=strength.value,
                                    embedding_strength_mode=strength.mode,
                                )
                            )
                            continue
                        except Exception as exc:
                            self.writer.write_result(
                                self._failure_row(
                                    image_id,
                                    image_path,
                                    image,
                                    "failed_svd",
                                    str(exc),
                                    yolo_time_ms,
                                    roi=roi,
                                    roi_strategy=roi.strategy,
                                    svd_band=svd_band,
                                    payload_bits=payload_bits,
                                    embedding_strength=strength.value,
                                    embedding_strength_mode=strength.mode,
                                )
                            )
                            continue

                        for attack in attacks:
                            attack_params = dict(attack.params)
                            attack_params["rng"] = self.rng
                            attack_start = perf_counter()
                            attacked = apply_attack(embed_result.stego_image, attack.attack_type, attack_params)
                            attack_time_ms = (perf_counter() - attack_start) * 1000.0
                            image_metrics = compute_roi_metrics(image, attacked, roi)
                            psnr_full = compute_psnr(image, attacked)
                            ssim_full = compute_ssim(image, attacked)

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
                                        image_shape=(height, width),
                                        roi=roi,
                                        detections_count=len(detections),
                                        yolo_time_ms=yolo_time_ms,
                                        embed_result=embed_result,
                                        payload_bits=payload_bits,
                                        strength=strength,
                                        attack=attack,
                                        attack_time_ms=attack_time_ms,
                                        decoder_type=decoder_type,
                                        extract_result=extract_result,
                                        image_metrics=image_metrics,
                                        psnr_full=psnr_full,
                                        ssim_full=ssim_full,
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
                                        roi=roi,
                                        roi_strategy=roi.strategy,
                                        svd_band=svd_band,
                                        decoder_type=decoder_type,
                                        attack_type=attack.attack_type,
                                        payload_bits=payload_bits,
                                        embedding_strength=strength.value,
                                        embedding_strength_mode=strength.mode,
                                    )
                                self.writer.write_result(row)

    def _select_rois(self, image: np.ndarray, detections) -> dict[str, ROI]:
        selected: dict[str, ROI] = {}
        for strategy in self.config.roi_strategies:
            roi = select_roi(image.shape, detections, strategy, self.rng, self.config.min_roi_area)
            if roi is not None:
                selected[strategy] = roi
        return selected

    def _build_roi_caches(self, image: np.ndarray, rois: dict[str, ROI]) -> dict[str, RoiSvdCache]:
        caches: dict[str, RoiSvdCache] = {}
        for strategy, roi in rois.items():
            roi_patch = crop_roi(image, roi)
            gray_channel = rgb_to_gray(roi_patch).astype(np.float64)
            svd_start = perf_counter()
            U, S, Vt = svd_decompose(gray_channel)
            svd_time_ms = (perf_counter() - svd_start) * 1000.0
            numpy_svd_time_ms = measure_numpy_svd_time_ms(gray_channel)
            reconstruction_error = compute_reconstruction_error(gray_channel, U, S, Vt)
            caches[strategy] = RoiSvdCache(gray_channel, U, S, Vt, svd_time_ms, numpy_svd_time_ms, reconstruction_error)
        return caches

    def _capacity_is_usable(self, capacity: int, requested_payload_bits: int) -> bool:
        if self.config.payload_policy == "truncate_message":
            return capacity > 0
        return capacity >= requested_payload_bits

    def _capacity_error(self, strategy: str, band: str, capacity: int, requested_payload_bits: int) -> str:
        required_capacity = 1 if self.config.payload_policy == "truncate_message" else requested_payload_bits
        return f"ROI strategy '{strategy}' band '{band}' effective capacity {capacity} is below required payload length {required_capacity}"

    def _success_row(
        self,
        *,
        image_id,
        image_path,
        image_shape,
        roi: ROI,
        detections_count,
        yolo_time_ms,
        embed_result,
        payload_bits: int,
        strength: StrengthSetting,
        attack,
        attack_time_ms,
        decoder_type,
        extract_result,
        image_metrics,
        psnr_full,
        ssim_full,
        total_time_ms,
    ):
        height, width = image_shape
        embedded_bits = embed_result.embedded_bits
        recovered_bits = extract_result.bits[: len(embedded_bits)]
        bit_err = bit_errors(embedded_bits, recovered_bits)
        total_bits = len(embedded_bits)
        correct_bits = max(0, total_bits - bit_err)
        payload_requested = embed_result.requested_bits
        return {
            "run_id": self.run_id,
            "experiment_block": self.experiment_block,
            "dataset": f"coco/{self.config.split}",
            "image_id": image_id,
            "image_path": str(image_path),
            "image_width": width,
            "image_height": height,
            "image_accepted": True,
            "image_filter_reason": None,
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
            "embedding_strength": strength.value,
            "embedding_strength_mode": strength.mode,
            "embedding_delta_mean": embed_result.metadata.strength,
            "payload_bits": payload_bits,
            "payload_text": None,
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
            "PSNR_full": psnr_full,
            "PSNR_roi": image_metrics["PSNR_roi"],
            "SSIM_full": ssim_full,
            "SSIM_roi": image_metrics["SSIM_roi"],
            "bit_errors": bit_err,
            "total_bits": total_bits,
            "BER": bit_error_rate(embedded_bits, recovered_bits),
            "exact_match": exact_match(embedded_bits, recovered_bits),
            "character_accuracy": None,
            "yolo_time_ms": yolo_time_ms,
            "embedding_time_ms": embed_result.embedding_time_ms,
            "extraction_time_ms": extract_result.extraction_time_ms,
            "svd_time_ms": embed_result.svd_time_ms,
            "numpy_svd_time_ms": embed_result.numpy_svd_time_ms,
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
            "experiment_block": self.experiment_block,
            "dataset": f"coco/{self.config.split}",
            "image_id": image_id,
            "image_path": str(image_path),
            "image_width": width,
            "image_height": height,
            "image_accepted": roi is not None,
            "image_filter_reason": extras.get("image_filter_reason"),
            "roi_strategy": roi_strategy or (roi.strategy if roi else "not_reached"),
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
            "svd_band": extras.get("svd_band") or "not_reached",
            "decoder_type": extras.get("decoder_type") or "not_reached",
            "embedding_strength": extras.get("embedding_strength", self.config.embedding_strength),
            "embedding_strength_mode": extras.get("embedding_strength_mode", self.config.embedding_strength_mode),
            "payload_bits": extras.get("payload_bits", self.config.payload_bits),
            "payload_text": None,
            "payload_seed": self.config.payload_seed,
            "payload_bits_requested": extras.get("payload_bits", self.config.payload_bits),
            "attack_type": extras.get("attack_type") or "not_reached",
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
        summary = success.groupby(["roi_strategy", "svd_band", "decoder_type", "attack_type", "payload_bits", "embedding_strength", "embedding_strength_mode"], dropna=False).agg(
            BER_mean=("BER", "mean"),
            BER_std=("BER", "std"),
            PSNR_roi_mean=("PSNR_roi", "mean"),
            SSIM_roi_mean=("SSIM_roi", "mean"),
            MRR=("exact_match", "mean"),
        )
        summary.to_csv(self.config.output_dir / "summary.csv")
