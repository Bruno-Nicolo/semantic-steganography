#!/usr/bin/env bash

set -euo pipefail

OUTPUT_DIR=${1:-outputs/smoke_test}
MAX_IMAGES=${2:-1}

if [[ ! -d "data/coco/val2017" ]]; then
  printf 'Smoke test failed: missing dataset directory data/coco/val2017\n' >&2
  exit 1
fi

python3 - <<'PY'
import importlib.util

required_modules = ["cv2", "numpy", "pandas", "matplotlib", "PIL", "skimage", "tqdm", "ultralytics"]
missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "Smoke test failed: missing Python dependencies: " + ", ".join(missing) + ". "
        "Install them with: pip install -r requirements.txt"
    )
PY

python3 -m semantic_stego.cli.app \
  --coco-root data/coco \
  --split val2017 \
  --output-dir "${OUTPUT_DIR}" \
  --max-images "${MAX_IMAGES}" \
  --roi-strategies full_image \
  --svd-bands mid_energy \
  --decoders non_blind \
  --attacks none \
  --payload-bits 8 \
  --embedding-strength 10 \
  --seed 42

RESULTS_CSV="${OUTPUT_DIR}/results.csv"

if [[ ! -f "${RESULTS_CSV}" ]]; then
  printf 'Smoke test failed: missing %s\n' "${RESULTS_CSV}" >&2
  exit 1
fi

python3 - "${RESULTS_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
with results_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))

if not rows:
    raise SystemExit(f"Smoke test failed: {results_path} is empty")

success_rows = [row for row in rows if row.get("status") == "success"]
if not success_rows:
    statuses = sorted({row.get("status", "") for row in rows})
    raise SystemExit(
        "Smoke test failed: no successful rows found. "
        f"Observed statuses: {', '.join(statuses)}"
    )

row = success_rows[0]
required_fields = [
    "roi_strategy",
    "svd_band",
    "decoder_type",
    "attack_type",
    "BER",
    "exact_match",
    "payload_bits_embedded",
    "PSNR_full",
    "SSIM_full",
]
missing = [field for field in required_fields if row.get(field) in {None, ""}]
if missing:
    raise SystemExit(
        "Smoke test failed: successful row is missing required fields: "
        + ", ".join(missing)
    )

print("Smoke test passed")
print(f"results_csv={results_path}")
print(f"rows={len(rows)}")
print(f"success_rows={len(success_rows)}")
print(f"first_success_status={row['status']}")
print(f"first_success_exact_match={row['exact_match']}")
print(f"first_success_ber={row['BER']}")
PY
