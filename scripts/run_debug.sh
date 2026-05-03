#!/usr/bin/env bash
# Default number of images
NUM_IMAGES=${1:-10}

python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir "outputs/debug_${NUM_IMAGES}" \
  --max-images "${NUM_IMAGES}" \
  --roi-strategies largest full_image \
  --svd-bands mid_energy \
  --decoders non_blind \
  --attacks none \
  --payload-bits 64 \
  --embedding-strength 10 \
  --seed 42 \
  --save-roi-debug
