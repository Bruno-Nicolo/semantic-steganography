from __future__ import annotations

from semantic_stego.config.cli_args import build_parser
from semantic_stego.config.schemas import ExperimentConfig
from semantic_stego.experiments.runner import ExperimentRunner


def parse_config() -> ExperimentConfig:
    args = build_parser().parse_args()
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
        payload_text=args.payload_text,
        payload_bits=args.payload_bits,
        payload_seed=args.payload_seed,
        embedding_strength=args.embedding_strength,
        seed=args.seed,
        save_images=args.save_images,
        save_roi_debug=args.save_roi_debug,
        min_roi_area=args.min_roi_area,
        skip_no_detection=args.skip_no_detection,
        payload_policy=args.payload_policy,
    )


def main() -> None:
    config = parse_config()
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
