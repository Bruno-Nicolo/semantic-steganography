#!/usr/bin/env bash

set -euo pipefail

SHARDS_ROOT="${SHARDS_ROOT:-data/coco/val2017_shards}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REPETITION_FACTOR="${REPETITION_FACTOR:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stego_sweep_shards}"
CSV_EXPORT_DIR="${OUTPUT_ROOT}/csv_exports"
SHARD_INPUTS_DIR="${OUTPUT_ROOT}/shard_inputs"
PAYLOAD_BITS_VALUES=(${PAYLOAD_BITS_VALUES:-8 64 128 512})
ABSOLUTE_DELTAS=(${ABSOLUTE_DELTAS:-0.5 10 20 40 80})
PROPORTIONAL_DELTAS=(${PROPORTIONAL_DELTAS:-0.05 0.1})

if [ ! -d "${SHARDS_ROOT}" ]; then
  echo "Shard directory not found: ${SHARDS_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${CSV_EXPORT_DIR}"
mkdir -p "${SHARD_INPUTS_DIR}"

shopt -s nullglob
SHARD_DIRS=("${SHARDS_ROOT}"/shard_*)

if [ ${#SHARD_DIRS[@]} -eq 0 ]; then
  echo "No shard directories found under: ${SHARDS_ROOT}" >&2
  exit 1
fi

for SHARD_DIR in "${SHARD_DIRS[@]}"; do
  if [ ! -d "${SHARD_DIR}" ]; then
    continue
  fi

  SHARD_NAME="$(basename "${SHARD_DIR}")"
  SHARD_OUTPUT_DIR="${OUTPUT_ROOT}/${SHARD_NAME}"
  SHARD_COCO_ROOT="${SHARD_INPUTS_DIR}/${SHARD_NAME}"
  SHARD_SPLIT_DIR="${SHARD_COCO_ROOT}/val2017"
  IMAGE_FILES=("${SHARD_DIR}"/*)
  IMAGE_COUNT=${#IMAGE_FILES[@]}

  if [ ${IMAGE_COUNT} -eq 0 ]; then
    echo "Skipping empty shard: ${SHARD_DIR}" >&2
    continue
  fi

  if [ -e "${SHARD_OUTPUT_DIR}" ]; then
    echo "Output directory already exists: ${SHARD_OUTPUT_DIR}" >&2
    exit 1
  fi

  if [ -e "${SHARD_SPLIT_DIR}" ] && [ ! -L "${SHARD_SPLIT_DIR}" ]; then
    echo "Shard split path already exists and is not a symlink: ${SHARD_SPLIT_DIR}" >&2
    exit 1
  fi

  mkdir -p "${SHARD_COCO_ROOT}"
  if [ ! -L "${SHARD_SPLIT_DIR}" ]; then
    SHARD_ABS_DIR="$(cd "${SHARD_DIR}" && pwd)"
    ln -s "${SHARD_ABS_DIR}" "${SHARD_SPLIT_DIR}"
  fi

  echo "Running image-centric sweep on ${SHARD_NAME} (${IMAGE_COUNT} images)"
  "${PYTHON_BIN}" -m semantic_stego.cli.efficient_sweep_app \
    --coco-root "${SHARD_COCO_ROOT}" \
    --split val2017 \
    --output-dir "${SHARD_OUTPUT_DIR}" \
    --max-images "${IMAGE_COUNT}" \
    --roi-strategies largest full_image smallest \
    --svd-bands mid_energy low_energy high_energy \
    --decoders non_blind blind \
    --attacks none gaussian_noise gaussian_blur jpeg_compression \
    --noise-sigmas 5 10 20 \
    --blur-kernels 3 5 7 \
    --jpeg-qualities 80 50 30 \
    --payload-bits-values "${PAYLOAD_BITS_VALUES[@]}" \
    --absolute-deltas "${ABSOLUTE_DELTAS[@]}" \
    --proportional-deltas "${PROPORTIONAL_DELTAS[@]}" \
    --repetition-factor "${REPETITION_FACTOR}" \
    --seed 42 \
    --skip-no-detection

  cp "${SHARD_OUTPUT_DIR}/results.csv" "${CSV_EXPORT_DIR}/${SHARD_NAME}_results.csv"
done

open "${CSV_EXPORT_DIR}"
