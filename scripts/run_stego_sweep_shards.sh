#!/usr/bin/env bash

set -euo pipefail

SHARDS_ROOT="${SHARDS_ROOT:-data/coco/val2017_shards}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REPETITION_FACTOR="${REPETITION_FACTOR:-3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/stego_sweep_shards}"
CSV_EXPORT_DIR="${OUTPUT_ROOT}/csv_exports"
SHARD_INPUTS_DIR="${OUTPUT_ROOT}/shard_inputs"
ANALYSIS_DIR="${OUTPUT_ROOT}/analysis"
PAYLOAD_BITS_VALUES=(${PAYLOAD_BITS_VALUES:-8 16 64 128 256 512 1024})
ABSOLUTE_DELTAS=(${ABSOLUTE_DELTAS:-0.25 5 10 20 40 60 80})
PROPORTIONAL_DELTA_FACTORS=(${PROPORTIONAL_DELTA_FACTORS:-0.05})

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

run_one_configuration() {
  local shard_name="$1"
  local shard_coco_root="$2"
  local image_count="$3"
  local payload_bits="$4"
  local strength="$5"
  local strength_mode="$6"
  local run_label="$7"
  local output_dir="${OUTPUT_ROOT}/${shard_name}/${run_label}"

  if [ -e "${output_dir}" ]; then
    echo "Output directory already exists: ${output_dir}" >&2
    exit 1
  fi

  echo "Running ${run_label} on ${shard_name} (${image_count} images)"
  "${PYTHON_BIN}" -m semantic_stego.cli.app \
    --coco-root "${shard_coco_root}" \
    --split val2017 \
    --output-dir "${output_dir}" \
    --max-images "${image_count}" \
    --roi-strategies largest smallest random full_image \
    --svd-bands high_energy mid_energy low_energy \
    --decoders non_blind blind \
    --attacks none gaussian_noise gaussian_blur jpeg_compression \
    --noise-sigmas 5 10 20 \
    --blur-kernels 3 5 7 \
    --jpeg-qualities 90 70 50 30 \
    --payload-bits "${payload_bits}" \
    --embedding-strength "${strength}" \
    --embedding-strength-mode "${strength_mode}" \
    --repetition-factor "${REPETITION_FACTOR}" \
    --seed 42 \
    --skip-no-detection

  cp "${output_dir}/results.csv" "${CSV_EXPORT_DIR}/${shard_name}_${run_label}_results.csv"
}

for SHARD_DIR in "${SHARD_DIRS[@]}"; do
  if [ ! -d "${SHARD_DIR}" ]; then
    continue
  fi

  SHARD_NAME="$(basename "${SHARD_DIR}")"
  SHARD_COCO_ROOT="${SHARD_INPUTS_DIR}/${SHARD_NAME}"
  SHARD_SPLIT_DIR="${SHARD_COCO_ROOT}/val2017"
  IMAGE_FILES=("${SHARD_DIR}"/*)
  IMAGE_COUNT=${#IMAGE_FILES[@]}

  if [ ${IMAGE_COUNT} -eq 0 ]; then
    echo "Skipping empty shard: ${SHARD_DIR}" >&2
    continue
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

  for PAYLOAD_BITS in "${PAYLOAD_BITS_VALUES[@]}"; do
    for DELTA in "${ABSOLUTE_DELTAS[@]}"; do
      DELTA_LABEL="$(printf '%s' "${DELTA}" | tr '.' 'p')"
      run_one_configuration "${SHARD_NAME}" "${SHARD_COCO_ROOT}" "${IMAGE_COUNT}" "${PAYLOAD_BITS}" "${DELTA}" "absolute" "bits${PAYLOAD_BITS}_delta${DELTA_LABEL}"
    done

    for FACTOR in "${PROPORTIONAL_DELTA_FACTORS[@]}"; do
      FACTOR_LABEL="$(printf '%s' "${FACTOR}" | tr '.' 'p')"
      run_one_configuration "${SHARD_NAME}" "${SHARD_COCO_ROOT}" "${IMAGE_COUNT}" "${PAYLOAD_BITS}" "${FACTOR}" "proportional_singular" "bits${PAYLOAD_BITS}_proportional${FACTOR_LABEL}"
    done
  done
done

"${PYTHON_BIN}" scripts/analyze_results.py "${OUTPUT_ROOT}" --analysis-dir "${ANALYSIS_DIR}"

open "${ANALYSIS_DIR}/plots"
