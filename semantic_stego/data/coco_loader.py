from __future__ import annotations

from pathlib import Path

import numpy as np

from semantic_stego.config.schemas import ImageRecord


class CocoImageLoader:
    def __init__(self, coco_root: Path, split: str, max_images: int | None = None, seed: int = 42):
        self.split_dir = coco_root / split
        self.max_images = max_images
        self.seed = seed

    def iter_records(self) -> list[ImageRecord]:
        if not self.split_dir.exists():
            raise FileNotFoundError(f"COCO split directory not found: {self.split_dir}")

        image_paths = sorted(
            [path for path in self.split_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if self.max_images is not None and len(image_paths) > self.max_images:
            rng = np.random.default_rng(self.seed)
            indices = np.sort(rng.choice(len(image_paths), size=self.max_images, replace=False))
            image_paths = [image_paths[index] for index in indices]

        return [ImageRecord(image_id=path.stem, image_path=path) for path in image_paths]
