from __future__ import annotations

import argparse
from pathlib import Path

from semantic_stego.data.coco_loader import download_coco_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a small COCO subset from Kaggle.")
    parser.add_argument("--coco-root", type=Path, default=Path("data/coco"))
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--max-images", type=int, default=250)
    parser.add_argument("--keep-archive", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = download_coco_subset(
        coco_root=args.coco_root,
        split=args.split,
        max_images=args.max_images,
        keep_archive=args.keep_archive,
    )
    print(f"Saved {args.max_images} images to {output_dir}")


if __name__ == "__main__":
    main()
