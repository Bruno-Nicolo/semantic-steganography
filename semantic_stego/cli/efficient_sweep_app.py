from __future__ import annotations

import logging

from semantic_stego.config.cli_args import build_parser
from semantic_stego.config.schemas import ExperimentConfig

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = build_parser()
    parser.description = "Image-centric semantic steganography sweep."
    parser.add_argument("--payload-bits-values", nargs="+", type=int, default=[8, 64, 128, 512])
    parser.add_argument("--absolute-deltas", nargs="+", type=float, default=[0.5, 10, 20, 40, 80])
    parser.add_argument("--proportional-deltas", nargs="+", type=float, default=[0.05, 0.1])
    parser.add_argument("--experiment-block", default="comprehensive_grid")
    return parser.parse_args()


def parse_config(args) -> ExperimentConfig:
    if not args.payload_bits_values:
        raise SystemExit("At least one payload length is required.")
    if not args.absolute_deltas and not args.proportional_deltas:
        raise SystemExit("At least one absolute or proportional delta is required.")

    return ExperimentConfig(
        coco_root=args.coco_root,
        split=args.split,
        output_dir=args.output_dir,
        max_images=args.max_images,
        image_size=args.image_size,
        yolo_model=args.yolo_model,
        confidence_threshold=args.confidence_threshold,
        roi_strategies=args.roi_strategies,
        svd_bands=args.svd_bands,
        decoders=args.decoders,
        attacks=args.attacks,
        jpeg_qualities=args.jpeg_qualities,
        noise_sigmas=args.noise_sigmas,
        blur_kernels=args.blur_kernels,
        payload_text=None,
        payload_bits=args.payload_bits_values[0],
        payload_seed=args.payload_seed,
        embedding_strength=args.absolute_deltas[0] if args.absolute_deltas else args.proportional_deltas[0],
        embedding_strength_mode="absolute" if args.absolute_deltas else "proportional_singular",
        repetition_factor=args.repetition_factor,
        seed=args.seed,
        save_images=args.save_images,
        save_roi_debug=args.save_roi_debug,
        min_roi_area=args.min_roi_area,
        skip_no_detection=args.skip_no_detection,
        payload_policy=args.payload_policy,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    args = parse_args()
    config = parse_config(args)
    LOGGER.info(
        "Efficient sweep | split=%s | output=%s | max_images=%s | payloads=%s | absolute_deltas=%s | proportional_deltas=%s",
        config.split,
        config.output_dir,
        config.max_images,
        args.payload_bits_values,
        args.absolute_deltas,
        args.proportional_deltas,
    )

    from semantic_stego.experiments.efficient_sweep_runner import EfficientSweepRunner

    runner = EfficientSweepRunner(
        config=config,
        payload_bits_values=args.payload_bits_values,
        absolute_deltas=args.absolute_deltas,
        proportional_deltas=args.proportional_deltas,
        experiment_block=args.experiment_block,
    )
    runner.run()


if __name__ == "__main__":
    main()
