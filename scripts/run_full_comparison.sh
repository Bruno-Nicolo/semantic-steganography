#!/usr/bin/env bash

set -euo pipefail

NUM_IMAGES=${1:-12}
PAYLOAD_TEXT=${2:-g}
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
EMBEDDING_STRENGTH=${EMBEDDING_STRENGTH:-20}
REPETITION_FACTOR=${REPETITION_FACTOR:-3}
OUTPUT_DIR="outputs/full_comparison_${NUM_IMAGES}"
ANALYSIS_DIR="${OUTPUT_DIR}/analysis"

# Full comparison grid:
# - YOLO-driven ROIs: largest, smallest, random
# - No-YOLO baseline: full_image
# - All SVD bands, decoders, and attacks
"${PYTHON_BIN}" -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir "${OUTPUT_DIR}" \
  --max-images "${NUM_IMAGES}" \
  --roi-strategies largest smallest random full_image \
  --svd-bands high_energy mid_energy low_energy \
  --decoders non_blind blind \
  --attacks none gaussian_noise gaussian_blur jpeg_compression \
  --noise-sigmas 5 10 20 \
  --blur-kernels 3 5 7 \
  --jpeg-qualities 90 70 50 30 \
  --payload-text "${PAYLOAD_TEXT}" \
  --embedding-strength "${EMBEDDING_STRENGTH}" \
  --repetition-factor "${REPETITION_FACTOR}" \
  --seed 42 \
  --skip-no-detection

"${PYTHON_BIN}" scripts/analyze_results.py "${OUTPUT_DIR}" --analysis-dir "${ANALYSIS_DIR}"

open "${OUTPUT_DIR}"
