#!/usr/bin/env bash

set -euo pipefail

NUM_IMAGES=${1:-12}
PAYLOAD_TEXT=${2:-g}
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
OUTPUT_ROOT="outputs/clean_sweep_${NUM_IMAGES}"
ANALYSIS_DIR="${OUTPUT_ROOT}/analysis"

STRENGTHS=(10 15 20 30)
REPETITIONS=(1 3 5 7)
ROI_STRATEGIES=(largest full_image)
SVD_BANDS=(high_energy)
DECODERS=(non_blind blind)

mkdir -p "${OUTPUT_ROOT}"

for strength in "${STRENGTHS[@]}"; do
  for repetition in "${REPETITIONS[@]}"; do
    RUN_DIR="${OUTPUT_ROOT}/strength_${strength}_rep_${repetition}"
    "${PYTHON_BIN}" -m semantic_stego.cli.app \
      --coco-root data/coco \
      --split val2017 \
      --output-dir "${RUN_DIR}" \
      --max-images "${NUM_IMAGES}" \
      --roi-strategies "${ROI_STRATEGIES[@]}" \
      --svd-bands "${SVD_BANDS[@]}" \
      --decoders "${DECODERS[@]}" \
      --attacks none \
      --payload-text "${PAYLOAD_TEXT}" \
      --embedding-strength "${strength}" \
      --repetition-factor "${repetition}" \
      --seed 42 \
      --skip-no-detection
  done
done

"${PYTHON_BIN}" scripts/analyze_results.py "${OUTPUT_ROOT}" --analysis-dir "${ANALYSIS_DIR}"

open "${OUTPUT_ROOT}"
