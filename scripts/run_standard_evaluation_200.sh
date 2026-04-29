#!/usr/bin/env bash

# Run a standard evaluation on 200 images with balanced parameters
# Faster than the full evaluation but covers the main scenarios.
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/standard_evaluation_200 \
  --max-images 200 \
  --roi-strategies largest full_image \
  --svd-bands mid_energy \
  --decoders non_blind blind \
  --attacks none gaussian_noise jpeg \
  --noise-sigmas 10 \
  --jpeg-qualities 70 \
  --payload-bits 128 \
  --embedding-strength 10.0 \
  --seed 42 \
  --skip-no-detection
