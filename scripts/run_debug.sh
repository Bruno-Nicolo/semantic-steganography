#!/usr/bin/env bash
python -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir outputs/debug \
  --max-images 10 \
  --roi-strategies largest full_image \
  --svd-bands mid_energy \
  --decoders non_blind \
  --attacks none \
  --payload-bits 64 \
  --embedding-strength 10 \
  --seed 42 \
  --save-roi-debug
