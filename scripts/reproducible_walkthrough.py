from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from semantic_stego.config.schemas import Detection, ROI
from semantic_stego.data.image_io import crop_roi, draw_roi, read_image_rgb, rgb_to_gray, save_image_rgb
from semantic_stego.detection.roi_selector import select_roi
from semantic_stego.detection.yolo_detector import YoloDetector
from semantic_stego.metrics.image_metrics import compute_psnr, compute_roi_metrics, compute_ssim
from semantic_stego.metrics.message_metrics import bit_error_rate, bit_errors, exact_match
from semantic_stego.stego.embedder import SvdEmbedder, _nearest_qim_codeword
from semantic_stego.stego.extractor import SvdExtractor
from semantic_stego.stego.payload import bits_to_text, text_to_bits
from semantic_stego.svd.svd_from_scratch import svd_decompose


@dataclass(frozen=True)
class WalkthroughSetup:
    name: str
    roi_strategy: str
    svd_band: str
    decoder_type: str
    delta: float
    purpose: str


PRESENTATION_CASES = [
    {
        "slug": "baseline_success",
        "purpose": "Baseline: ROI grande, banda ad alta energia e decoder non-blind con estrazione corretta.",
        "roi_strategies": ["largest"],
        "svd_bands": ["high_energy"],
        "decoders": ["non_blind"],
        "deltas": [0.75, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 160.0, 200.0],
        "require_exact": True,
    },
    {
        "slug": "delta_low_probe",
        "purpose": "Probe: stesso tipo di ROI/banda ma Delta basso, per mostrare quando la modifica puo' non sopravvivere alla quantizzazione.",
        "roi_strategies": ["largest"],
        "svd_bands": ["high_energy"],
        "decoders": ["non_blind"],
        "deltas": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        "require_exact": False,
    },
    {
        "slug": "roi_change_success",
        "purpose": "Cambio ROI: stessa pipeline su una sezione piu' piccola per isolare l'effetto della ROI.",
        "roi_strategies": ["smallest"],
        "svd_bands": ["high_energy", "mid_energy", "low_energy"],
        "decoders": ["non_blind"],
        "deltas": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0],
        "require_exact": True,
    },
    {
        "slug": "band_change_probe",
        "purpose": "Cambio banda SVD: stessa ROI grande, ma valori singolari in una banda diversa; utile anche quando il segnale non sopravvive.",
        "roi_strategies": ["largest", "full_image"],
        "svd_bands": ["mid_energy", "low_energy"],
        "decoders": ["non_blind"],
        "deltas": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 160.0, 200.0],
        "require_exact": False,
    },
    {
        "slug": "blind_decoder_success",
        "purpose": "Decoder blind: estrazione senza immagine originale, per confrontare l'assunzione del decoder.",
        "roi_strategies": ["largest", "full_image"],
        "svd_bands": ["high_energy", "mid_energy"],
        "decoders": ["blind"],
        "deltas": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0, 40.0, 60.0, 80.0, 100.0, 160.0, 200.0],
        "require_exact": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a complete, reproducible SVD steganography walkthrough."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reproducible_walkthrough"))
    parser.add_argument("--payload-bits", default="10", help="Binary payload used by default, e.g. '1010'.")
    parser.add_argument("--message", default=None, help="Optional UTF-8 text payload. Overrides --payload-bits.")
    parser.add_argument("--input-image", type=Path, default=None, help="Optional RGB image path. If omitted, a deterministic synthetic image is generated.")
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLO model path/name used when --input-image is provided.")
    parser.add_argument("--confidence-threshold", type=float, default=0.25, help="YOLO confidence threshold used when --input-image is provided.")
    parser.add_argument("--image-size", type=int, default=640, help="YOLO inference image size used when --input-image is provided.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetition-factor", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image, detections = load_or_build_example_image(args, rng)
    payload_bits, payload_label = build_payload(args.payload_bits, args.message)
    embedder = SvdEmbedder(payload_policy="raise_error", repetition_factor=args.repetition_factor)
    extractor = SvdExtractor()
    setups = build_presentation_setups(image, detections, payload_bits, embedder, extractor, args.seed)

    save_image_rgb(output_dir / "00_original.png", image)
    save_image_rgb(output_dir / "00_detected_sections.png", draw_detections(image, detections))
    write_detections(output_dir / "detections.json", detections)

    summaries = []
    for setup in setups:
        setup_dir = output_dir / setup.name
        setup_dir.mkdir(parents=True, exist_ok=True)
        roi = select_required_roi(image.shape, detections, setup.roi_strategy, rng)
        save_image_rgb(setup_dir / "01_roi_selected.png", draw_roi(image, roi))

        embed_result = embedder.embed(
            image=image,
            roi=roi,
            payload_bits=payload_bits,
            band=setup.svd_band,
            strength=setup.delta,
            mode="qim",
        )
        stego_image = embed_result.stego_image
        save_image_rgb(setup_dir / "02_final_stego.png", stego_image)
        save_difference_heatmap(image, stego_image, roi, setup_dir / "03_roi_difference_heatmap.png")

        original_for_decoder = image if setup.decoder_type == "non_blind" else None
        extract_result = extractor.extract(
            stego_image,
            metadata=embed_result.metadata,
            original_image=original_for_decoder,
            decoder_type=setup.decoder_type,
        )
        recovered_bits = extract_result.bits[: len(embed_result.embedded_bits)]
        recovered_message = bits_to_display(recovered_bits)

        singular_rows = build_singular_value_rows(image, stego_image, embed_result, setup.delta)
        write_singular_values_csv(setup_dir / "04_singular_values_modified.csv", singular_rows)
        singular_summary = summarize_singular_rows(singular_rows)
        image_delta_summary = summarize_image_delta(image, stego_image, roi)
        save_setup_report(
            setup_dir / "README.md",
            setup,
            roi,
            payload_label,
            recovered_message,
            embed_result,
            recovered_bits,
            image,
            stego_image,
            singular_summary,
            image_delta_summary,
        )
        save_contact_sheet(setup_dir, setup, roi, image, stego_image)

        summaries.append(
            {
                "setup": asdict(setup),
                "roi": roi_to_dict(roi),
                "payload_requested_bits": int(len(payload_bits)),
                "payload_embedded_bits": int(len(embed_result.embedded_bits)),
                "payload_dropped_bits": int(embed_result.payload_bits_dropped),
                "payload": payload_label,
                "extracted_message": recovered_message,
                "exact_match": bool(exact_match(embed_result.embedded_bits, recovered_bits)),
                "bit_errors": int(bit_errors(embed_result.embedded_bits, recovered_bits)),
                "ber": float(bit_error_rate(embed_result.embedded_bits, recovered_bits)),
                "singular_values_modified": int(len(embed_result.metadata.indices)),
                **singular_summary,
                **image_delta_summary,
            }
        )

    write_comparison_summary(output_dir / "comparison_summary.csv", summaries)
    write_manifest(output_dir / "manifest.json", args, payload_label, summaries)
    write_index(output_dir / "README.md", payload_label, summaries)


def build_presentation_setups(
    image: np.ndarray,
    detections: list[Detection],
    payload_bits: np.ndarray,
    embedder: SvdEmbedder,
    extractor: SvdExtractor,
    seed: int,
) -> list[WalkthroughSetup]:
    setups: list[WalkthroughSetup] = []
    for case_index, case in enumerate(PRESENTATION_CASES, start=1):
        setup = find_case_setup(case, case_index, image, detections, payload_bits, embedder, extractor, seed)
        if setup is not None:
            setups.append(setup)
    if not setups:
        raise RuntimeError("No embeddable presentation setup found for this image/payload")
    return setups


def find_case_setup(
    case: dict[str, object],
    case_index: int,
    image: np.ndarray,
    detections: list[Detection],
    payload_bits: np.ndarray,
    embedder: SvdEmbedder,
    extractor: SvdExtractor,
    seed: int,
) -> WalkthroughSetup | None:
    fallback: WalkthroughSetup | None = None
    require_exact = bool(case["require_exact"])
    for roi_strategy in case["roi_strategies"]:
        roi = select_roi(image.shape, detections, str(roi_strategy), np.random.default_rng(seed), min_roi_area=1)
        if roi is None:
            continue
        for svd_band in case["svd_bands"]:
            for decoder_type in case["decoders"]:
                for delta in case["deltas"]:
                    setup = make_setup(case_index, str(case["slug"]), str(case["purpose"]), str(roi_strategy), str(svd_band), str(decoder_type), float(delta))
                    try:
                        embed_result = embedder.embed(image, roi, payload_bits, str(svd_band), float(delta), mode="qim")
                        original_for_decoder = image if decoder_type == "non_blind" else None
                        extract_result = extractor.extract(embed_result.stego_image, embed_result.metadata, original_for_decoder, str(decoder_type))
                    except Exception:
                        continue
                    recovered_bits = extract_result.bits[: len(embed_result.embedded_bits)]
                    is_exact = bool(exact_match(embed_result.embedded_bits, recovered_bits))
                    changed_pixels = count_changed_roi_pixels(image, embed_result.stego_image, roi)
                    if require_exact and is_exact and changed_pixels > 0:
                        return setup
                    if not require_exact and fallback is None:
                        fallback = setup
                    if not require_exact and (not is_exact or changed_pixels == 0):
                        return setup
    return fallback


def make_setup(index: int, slug: str, purpose: str, roi_strategy: str, svd_band: str, decoder_type: str, delta: float) -> WalkthroughSetup:
    delta_slug = str(delta).replace(".", "_")
    name = f"{index:02d}_{slug}_{roi_strategy}_{svd_band}_{decoder_type}_delta{delta_slug}"
    return WalkthroughSetup(name, roi_strategy, svd_band, decoder_type, delta, purpose)


def build_payload(payload_bits: str, message: str | None) -> tuple[np.ndarray, str]:
    if message is not None:
        return text_to_bits(message), message
    cleaned = payload_bits.strip().replace(" ", "")
    if not cleaned or any(bit not in "01" for bit in cleaned):
        raise ValueError("--payload-bits must contain only 0 and 1")
    return np.asarray([int(bit) for bit in cleaned], dtype=np.uint8), cleaned


def bits_to_display(bits: np.ndarray) -> str:
    bit_string = "".join(str(int(bit)) for bit in bits)
    if len(bits) > 0 and len(bits) % 8 == 0:
        text = bits_to_text(bits)
        return text if text else bit_string
    return bit_string


def count_changed_roi_pixels(original: np.ndarray, stego: np.ndarray, roi: ROI) -> int:
    diff = np.abs(crop_roi(stego, roi).astype(np.int16) - crop_roi(original, roi).astype(np.int16))
    return int(np.count_nonzero(np.any(diff > 0, axis=2)))


def load_or_build_example_image(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, list[Detection]]:
    if args.input_image is not None:
        image = read_image_rgb(args.input_image)
        detector = YoloDetector(args.yolo_model, args.confidence_threshold, args.image_size)
        detections, _ = detector.detect(image)
        if not detections:
            raise RuntimeError(
                f"YOLO did not find any detection in {args.input_image}. "
                "Lower --confidence-threshold or choose another image."
            )
        return image, detections

    image = build_synthetic_image(rng)
    detections = [
        Detection(24, 28, 184, 168, 0.99, 1, "synthetic_large_object"),
        Detection(114, 90, 190, 158, 0.96, 2, "synthetic_small_object"),
        Detection(30, 178, 92, 228, 0.94, 3, "synthetic_tiny_object"),
    ]
    return image, detections


def build_synthetic_image(rng: np.random.Generator) -> np.ndarray:
    height, width = 256, 256
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.zeros((height, width, 3), dtype=np.float64)
    base[:, :, 0] = 36 + 54 * xx / width
    base[:, :, 1] = 70 + 95 * yy / height
    base[:, :, 2] = 118 + 18 * np.sin((xx + yy) / 30.0)
    background_texture = rng.normal(0, 2.0, size=base.shape)
    image = np.clip(base + background_texture, 0, 255).astype(np.uint8)

    cv2.rectangle(image, (24, 28), (184, 168), (214, 94, 68), thickness=-1)
    add_patch_texture(image, (24, 28, 184, 168), rng, strength=5)
    for y in range(42, 158, 18):
        cv2.line(image, (36, y), (172, y + 6), (229, 126, 93), thickness=2)
    cv2.circle(image, (152, 124), 34, (34, 164, 99), thickness=-1)
    add_patch_texture(image, (118, 90, 186, 158), rng, strength=4)
    cv2.circle(image, (142, 112), 6, (76, 196, 131), thickness=-1)
    cv2.rectangle(image, (30, 178), (92, 228), (70, 101, 218), thickness=-1)
    add_patch_texture(image, (30, 178, 92, 228), rng, strength=4)
    cv2.rectangle(image, (42, 190), (80, 216), (95, 128, 236), thickness=2)
    cv2.line(image, (0, 245), (255, 190), (239, 218, 72), thickness=5)
    return image


def add_patch_texture(image: np.ndarray, box: tuple[int, int, int, int], rng: np.random.Generator, strength: int) -> None:
    x1, y1, x2, y2 = box
    patch = image[y1:y2, x1:x2].astype(np.int16)
    texture = rng.integers(-strength, strength + 1, size=patch.shape, dtype=np.int16)
    image[y1:y2, x1:x2] = np.clip(patch + texture, 0, 255).astype(np.uint8)


def select_required_roi(image_shape: tuple[int, int, int], detections: list[Detection], strategy: str, rng: np.random.Generator) -> ROI:
    roi = select_roi(image_shape, detections, strategy, rng, min_roi_area=1)
    if roi is None:
        raise RuntimeError(f"Unable to select ROI with strategy '{strategy}'")
    return roi


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    output = image.copy()
    colors = [(255, 40, 40), (40, 220, 90), (80, 130, 255), (245, 205, 45), (205, 80, 245)]
    for index, detection in enumerate(detections):
        color = colors[index % len(colors)]
        cv2.rectangle(output, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        label = f"{index + 1}: {detection.class_name}"
        label_y = max(14, detection.y1 - 6)
        cv2.putText(output, label, (detection.x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return output


def build_singular_value_rows(image: np.ndarray, stego_image: np.ndarray, embed_result, delta: float) -> list[dict[str, object]]:
    roi = embed_result.metadata.roi
    original_s = svd_decompose(rgb_to_gray(crop_roi(image, roi)).astype(np.float64))[1]
    actual_stego_s = svd_decompose(rgb_to_gray(crop_roi(stego_image, roi)).astype(np.float64))[1]
    coded_bits = np.repeat(embed_result.embedded_bits, embed_result.metadata.repetition_factor)

    rows = []
    for position, singular_index in enumerate(embed_result.metadata.indices):
        bit = int(coded_bits[position])
        original_sigma = float(original_s[singular_index])
        target_sigma = _nearest_qim_codeword(original_sigma, delta, bit)
        actual_sigma = float(actual_stego_s[singular_index])
        rows.append(
            {
                "position": position,
                "singular_index": int(singular_index),
                "payload_bit_index": position // embed_result.metadata.repetition_factor,
                "coded_bit": bit,
                "original_sigma": original_sigma,
                "target_sigma": target_sigma,
                "actual_stego_sigma": actual_sigma,
                "target_delta": target_sigma - original_sigma,
                "actual_delta": actual_sigma - original_sigma,
            }
        )
    return rows


def summarize_singular_rows(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {
            "target_delta_abs_mean": 0.0,
            "actual_delta_abs_mean": 0.0,
            "actual_delta_abs_max": 0.0,
            "singular_values_changed_after_quantization": 0,
        }
    target_abs = np.asarray([abs(float(row["target_delta"])) for row in rows], dtype=np.float64)
    actual_abs = np.asarray([abs(float(row["actual_delta"])) for row in rows], dtype=np.float64)
    return {
        "target_delta_abs_mean": float(target_abs.mean()),
        "actual_delta_abs_mean": float(actual_abs.mean()),
        "actual_delta_abs_max": float(actual_abs.max()),
        "singular_values_changed_after_quantization": int(np.count_nonzero(actual_abs > 1e-9)),
    }


def summarize_image_delta(original: np.ndarray, stego: np.ndarray, roi: ROI) -> dict[str, float | int]:
    roi_diff = np.abs(crop_roi(stego, roi).astype(np.int16) - crop_roi(original, roi).astype(np.int16))
    changed_pixels = int(np.count_nonzero(np.any(roi_diff > 0, axis=2)))
    total_pixels = max(1, roi.width * roi.height)
    return {
        "changed_pixels_roi": changed_pixels,
        "changed_pixels_roi_ratio": float(changed_pixels / total_pixels),
        "mean_abs_rgb_delta_roi": float(roi_diff.mean()),
        "max_abs_rgb_delta_roi": int(roi_diff.max()) if roi_diff.size else 0,
    }


def write_singular_values_csv(path: Path, rows: list[dict[str, object]]) -> None:
    header = [
        "position",
        "singular_index",
        "payload_bit_index",
        "coded_bit",
        "original_sigma",
        "target_sigma",
        "actual_stego_sigma",
        "target_delta",
        "actual_delta",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(row[column]) for column in header) + "\n")


def save_difference_heatmap(original: np.ndarray, stego: np.ndarray, roi: ROI, path: Path) -> None:
    diff = np.abs(crop_roi(stego, roi).astype(np.int16) - crop_roi(original, roi).astype(np.int16)).mean(axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.imshow(diff, cmap="magma")
    plt.colorbar(label="mean absolute RGB difference")
    plt.title("ROI difference heatmap")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_contact_sheet(setup_dir: Path, setup: WalkthroughSetup, roi: ROI, original: np.ndarray, stego: np.ndarray) -> None:
    roi_image = draw_roi(original, roi)
    diff = np.clip(np.abs(stego.astype(np.int16) - original.astype(np.int16)) * 8, 0, 255).astype(np.uint8)
    images = [("Original", original), ("ROI selected", roi_image), ("Final stego", stego), ("Amplified diff x8", diff)]
    plt.figure(figsize=(10, 7))
    for index, (title, image) in enumerate(images, start=1):
        plt.subplot(2, 2, index)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
    plt.suptitle(f"{setup.roi_strategy}, {setup.svd_band}, {setup.decoder_type}, Delta={setup.delta:g}")
    plt.tight_layout()
    plt.savefig(setup_dir / "05_contact_sheet.png", dpi=150)
    plt.close()


def save_setup_report(
    path: Path,
    setup: WalkthroughSetup,
    roi: ROI,
    original_message: str,
    recovered_text: str,
    embed_result,
    recovered_bits: np.ndarray,
    image: np.ndarray,
    stego_image: np.ndarray,
    singular_summary: dict[str, float | int],
    image_delta_summary: dict[str, float | int],
) -> None:
    metrics = compute_roi_metrics(image, stego_image, roi)
    psnr_full = compute_psnr(image, stego_image)
    ssim_full = compute_ssim(image, stego_image)
    bit_err = bit_errors(embed_result.embedded_bits, recovered_bits)
    ber = bit_error_rate(embed_result.embedded_bits, recovered_bits)
    content = f"""# {setup.name}

## Setup

- ROI strategy: `{setup.roi_strategy}`
- SVD band: `{setup.svd_band}`
- Decoder: `{setup.decoder_type}`
- Delta: `{setup.delta:g}`
- Repetition factor: `{embed_result.metadata.repetition_factor}`
- Scopo: {setup.purpose}

## Passi riproducibili

1. Immagine di partenza: `../00_original.png`
2. ROI selezionata: `01_roi_selected.png`
3. Valori singolari modificati: `04_singular_values_modified.csv`
4. Immagine finale: `02_final_stego.png`
5. Messaggio estratto: `{recovered_text}`

## ROI

- Box: `x1={roi.x1}, y1={roi.y1}, x2={roi.x2}, y2={roi.y2}`
- Dimensione: `{roi.width}x{roi.height}`
- Classe: `{roi.class_name}`

## Payload

- Messaggio originale: `{original_message}`
- Bit richiesti: `{embed_result.requested_bits}`
- Bit embedded: `{len(embed_result.embedded_bits)}`
- Bit scartati: `{embed_result.payload_bits_dropped}`
- Messaggio estratto: `{recovered_text}`
- Bit errors: `{bit_err}`
- BER: `{ber:.6f}`
- Exact match: `{exact_match(embed_result.embedded_bits, recovered_bits)}`

## Segnale effettivamente scritto

- Target delta medio sui valori singolari: `{singular_summary['target_delta_abs_mean']:.6f}`
- Actual delta medio dopo ricostruzione/quantizzazione: `{singular_summary['actual_delta_abs_mean']:.6f}`
- Valori singolari ancora diversi dopo quantizzazione: `{singular_summary['singular_values_changed_after_quantization']}`
- Pixel cambiati nella ROI: `{image_delta_summary['changed_pixels_roi']}` (`{image_delta_summary['changed_pixels_roi_ratio']:.4%}`)
- Differenza RGB media nella ROI: `{image_delta_summary['mean_abs_rgb_delta_roi']:.6f}`
- Differenza RGB massima nella ROI: `{image_delta_summary['max_abs_rgb_delta_roi']}`

## Qualita' immagine

- PSNR full: `{psnr_full:.4f}`
- PSNR ROI: `{metrics['PSNR_roi']:.4f}`
- SSIM full: `{ssim_full:.4f}`
- SSIM ROI: `{metrics['SSIM_roi']:.4f}`

## Artefatti

- `03_roi_difference_heatmap.png`
- `05_contact_sheet.png`
"""
    path.write_text(content, encoding="utf-8")


def write_detections(path: Path, detections: list[Detection]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(detection) for detection in detections], handle, indent=2)


def write_comparison_summary(path: Path, summaries: list[dict[str, object]]) -> None:
    header = [
        "setup",
        "purpose",
        "roi_strategy",
        "roi_width",
        "roi_height",
        "svd_band",
        "decoder_type",
        "delta",
        "payload_embedded_bits",
        "singular_values_modified",
        "target_delta_abs_mean",
        "actual_delta_abs_mean",
        "actual_delta_abs_max",
        "singular_values_changed_after_quantization",
        "changed_pixels_roi",
        "changed_pixels_roi_ratio",
        "mean_abs_rgb_delta_roi",
        "max_abs_rgb_delta_roi",
        "extracted_message",
        "bit_errors",
        "ber",
        "exact_match",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for summary in summaries:
            setup = summary["setup"]
            roi = summary["roi"]
            writer.writerow(
                {
                    "setup": setup["name"],
                    "purpose": setup["purpose"],
                    "roi_strategy": setup["roi_strategy"],
                    "roi_width": roi["width"],
                    "roi_height": roi["height"],
                    "svd_band": setup["svd_band"],
                    "decoder_type": setup["decoder_type"],
                    "delta": setup["delta"],
                    "payload_embedded_bits": summary["payload_embedded_bits"],
                    "singular_values_modified": summary["singular_values_modified"],
                    "target_delta_abs_mean": summary["target_delta_abs_mean"],
                    "actual_delta_abs_mean": summary["actual_delta_abs_mean"],
                    "actual_delta_abs_max": summary["actual_delta_abs_max"],
                    "singular_values_changed_after_quantization": summary["singular_values_changed_after_quantization"],
                    "changed_pixels_roi": summary["changed_pixels_roi"],
                    "changed_pixels_roi_ratio": summary["changed_pixels_roi_ratio"],
                    "mean_abs_rgb_delta_roi": summary["mean_abs_rgb_delta_roi"],
                    "max_abs_rgb_delta_roi": summary["max_abs_rgb_delta_roi"],
                    "extracted_message": summary["extracted_message"],
                    "bit_errors": summary["bit_errors"],
                    "ber": summary["ber"],
                    "exact_match": summary["exact_match"],
                }
            )


def write_manifest(path: Path, args: argparse.Namespace, payload_label: str, summaries: list[dict[str, object]]) -> None:
    manifest = {
        "seed": args.seed,
        "payload": payload_label,
        "payload_bits_argument": args.payload_bits,
        "message_argument": args.message,
        "input_image": str(args.input_image) if args.input_image else "synthetic",
        "detection_source": "yolo" if args.input_image else "synthetic_manual",
        "yolo_model": args.yolo_model if args.input_image else None,
        "confidence_threshold": args.confidence_threshold if args.input_image else None,
        "image_size": args.image_size if args.input_image else None,
        "setups": summaries,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def write_index(path: Path, message: str, summaries: list[dict[str, object]]) -> None:
    lines = [
        "# Reproducible SVD Walkthrough",
        "",
        f"Payload originale: `{message}`",
        "",
        "Artefatti comuni:",
        "",
        "- `00_original.png`: immagine di partenza",
        "- `00_detected_sections.png`: tutte le sezioni candidate prima della scelta della ROI",
        "- `detections.json`: detection candidate usate per selezionare le ROI",
        "- `comparison_summary.csv`: tabella comparativa pronta per slide/report",
        "- `manifest.json`: riepilogo machine-readable",
        "",
        "## Setup eseguiti",
        "",
        "| Setup | Scopo | ROI | Banda SVD | Decoder | Delta | Pixel ROI cambiati | Actual delta medio | Messaggio estratto | BER |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for summary in summaries:
        setup = summary["setup"]
        lines.append(
            "| "
            f"[{setup['name']}]({setup['name']}/README.md) | "
            f"{setup['purpose']} | "
            f"{setup['roi_strategy']} | "
            f"{setup['svd_band']} | "
            f"{setup['decoder_type']} | "
            f"{setup['delta']:g} | "
            f"{summary['changed_pixels_roi']} | "
            f"{summary['actual_delta_abs_mean']:.6f} | "
            f"`{summary['extracted_message']}` | "
            f"{summary['ber']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def roi_to_dict(roi: ROI) -> dict[str, object]:
    return {
        "x1": roi.x1,
        "y1": roi.y1,
        "x2": roi.x2,
        "y2": roi.y2,
        "width": roi.width,
        "height": roi.height,
        "strategy": roi.strategy,
        "class_id": roi.class_id,
        "class_name": roi.class_name,
        "confidence": roi.confidence,
        "num_detections": roi.num_detections,
    }


if __name__ == "__main__":
    main()
