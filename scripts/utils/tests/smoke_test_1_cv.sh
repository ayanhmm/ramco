#!/usr/bin/env bash
set -euo pipefail

# === directories & files ===
WRAP_DIR="data/raw/wrap_images"
UNWRAP_DIR="data/raw/nowrap_images"
CAL_CSV="data/calibration_results.csv"

# === 1. POC edge-based CV ===
echo "✔️  sheet_counter_poc.py"
python3 scripts/1_cv/sheet_counter_poc.py \
  --wrap_dir  "$WRAP_DIR" \
  --unwrap_dir "$UNWRAP_DIR"

# === 2. Hough-based CV ===
echo "✔️  sheet_counter_hough.py"
python3 scripts/1_cv/sheet_counter_ridges.py \
  --wrap_dir  "$WRAP_DIR" \
  --unwrap_dir "$UNWRAP_DIR"

# === 3. Calibrated CV (linear) ===
echo "✔️  sheet_counter_calibrated.py"
python3 scripts/1_cv/sheet_counter_calibrated.py \
  --wrap_dir    "$WRAP_DIR" \
  --unwrap_dir  "$UNWRAP_DIR" \
  --calibration "$CAL_CSV"

echo
echo "✅  1-CV smoke tests passed!"

