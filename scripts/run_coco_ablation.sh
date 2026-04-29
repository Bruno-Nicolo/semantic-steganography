#!/usr/bin/env bash
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/coco_ablation_small \
  --max-images 50 \
  --roi-strategies largest smallest random full_image \
  --svd-bands high_energy mid_energy low_energy \
  --decoders non_blind blind \
  --attacks none gaussian_noise gaussian_blur jpeg \
  --noise-sigmas 5 \
  --blur-kernels 3 \
  --jpeg-qualities 90 \
  --payload-bits 128 \
  --embedding-strength 10 \
  --seed 42
