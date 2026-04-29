#!/usr/bin/env bash

# Run evaluation on 200 images with full parameters
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/evaluation_200 \
  --max-images 200 \
  --roi-strategies largest smallest random full_image \
  --svd-bands high_energy mid_energy low_energy \
  --decoders non_blind blind \
  --attacks none gaussian_noise gaussian_blur jpeg_compression \
  --noise-sigmas 5 10 20 \
  --blur-kernels 3 5 7 \
  --jpeg-qualities 90 70 50 30 \
  --payload-bits 128 \
  --embedding-strength 10.0 \
  --seed 42 \
  --skip-no-detection \
  --save-images
