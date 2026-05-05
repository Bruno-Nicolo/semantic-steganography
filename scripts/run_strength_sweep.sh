#!/usr/bin/env bash

NUM_IMAGES=${1:-50}
PAYLOAD_TEXT=${2:-SEMANTIC_STEGO}
STRENGTHS=${3:-"5 10 15 20 30"}

for strength in ${STRENGTHS}; do
  python -m semantic_stego.cli.app \
    --coco-root data/coco \
    --split val2017 \
    --output-dir "outputs/strength_${strength}_${NUM_IMAGES}" \
    --max-images "${NUM_IMAGES}" \
    --roi-strategies largest smallest random full_image \
    --svd-bands high_energy mid_energy low_energy \
    --decoders non_blind blind \
    --attacks none \
    --payload-text "${PAYLOAD_TEXT}" \
    --embedding-strength "${strength}" \
    --seed 42 \
    --skip-no-detection
done

python3 scripts/analyze_results.py outputs/strength_*_${NUM_IMAGES} --analysis-dir "outputs/analysis_strength_${NUM_IMAGES}"
