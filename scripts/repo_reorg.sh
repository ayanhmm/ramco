#!/usr/bin/env bash
set -euo pipefail

# 1) Create the target subdirectories
for dir in 1_cv 2_calibration 3_ml 4_multimodal 5_unified utils; do
  mkdir -p scripts/$dir
done

# 2) Move scripts into their new homes
mv scripts/sheet_counter_poc.py                  scripts/1_cv/
mv scripts/sheet_counter_ridges.py               scripts/1_cv/
mv scripts/sheet_counter_calibrated.py           scripts/1_cv/

mv scripts/full_calibration.py                   scripts/2_calibration/
mv scripts/full_calibration_dbscan.py            scripts/2_calibration/
mv scripts/calibrate_counts.py                   scripts/2_calibration/
mv scripts/sheet_counter_calibrated_dbscan.py    scripts/2_calibration/

mv scripts/train_feature_regressors.py           scripts/3_ml/
mv scripts/sheet_counter_ml.py                   scripts/3_ml/
mv scripts/retrain_cv_only_models.py             scripts/3_ml/
mv scripts/sheet_counter_inference.py            scripts/3_ml/

mv scripts/generate_captions.py                  scripts/4_multimodal/
mv scripts/train_multimodal.py                   scripts/4_multimodal/
mv scripts/train_multimodal_cached.py            scripts/4_multimodal/
mv scripts/train_multimodal_pca.py               scripts/4_multimodal/
mv scripts/train_multimodal_pca_with_layers.py   scripts/4_multimodal/
mv scripts/sheet_counter_multimodal.py           scripts/4_multimodal/
mv scripts/sheet_counter_multimodal_with_layers.py scripts/4_multimodal/

mv scripts/sheet_counter_unified.py              scripts/5_unified/

# Shared utilities
mv scripts/list_slides_and_captions.py           scripts/utils/
# (if you have any other helpers, move them here)

# 3) Fix up README.md and docs/usage.md to point to the new paths
#    This assumes those files refer to “scripts/xyz.py” — we’ll prefix with the folder:
for doc in README.md docs/usage.md; do
  sed -i '' \
    -e 's|scripts/sheet_counter_poc.py|scripts/1_cv/sheet_counter_poc.py|g' \
    -e 's|scripts/sheet_counter_ridges.py|scripts/1_cv/sheet_counter_ridges.py|g' \
    -e 's|scripts/sheet_counter_calibrated.py|scripts/1_cv/sheet_counter_calibrated.py|g' \
    -e 's|scripts/full_calibration.py|scripts/2_calibration/full_calibration.py|g' \
    -e 's|scripts/full_calibration_dbscan.py|scripts/2_calibration/full_calibration_dbscan.py|g' \
    -e 's|scripts/calibrate_counts.py|scripts/2_calibration/calibrate_counts.py|g' \
    -e 's|scripts/sheet_counter_calibrated_dbscan.py|scripts/2_calibration/sheet_counter_calibrated_dbscan.py|g' \
    -e 's|scripts/train_feature_regressors.py|scripts/3_ml/train_feature_regressors.py|g' \
    -e 's|scripts/sheet_counter_ml.py|scripts/3_ml/sheet_counter_ml.py|g' \
    -e 's|scripts/retrain_cv_only_models.py|scripts/3_ml/retrain_cv_only_models.py|g' \
    -e 's|scripts/sheet_counter_inference.py|scripts/3_ml/sheet_counter_inference.py|g' \
    -e 's|scripts/generate_captions.py|scripts/4_multimodal/generate_captions.py|g' \
    -e 's|scripts/train_multimodal.py|scripts/4_multimodal/train_multimodal.py|g' \
    -e 's|scripts/train_multimodal_cached.py|scripts/4_multimodal/train_multimodal_cached.py|g' \
    -e 's|scripts/train_multimodal_pca.py|scripts/4_multimodal/train_multimodal_pca.py|g' \
    -e 's|scripts/train_multimodal_pca_with_layers.py|scripts/4_multimodal/train_multimodal_pca_with_layers.py|g' \
    -e 's|scripts/sheet_counter_multimodal.py|scripts/4_multimodal/sheet_counter_multimodal.py|g' \
    -e 's|scripts/sheet_counter_multimodal_with_layers.py|scripts/4_multimodal/sheet_counter_multimodal_with_layers.py|g' \
    -e 's|scripts/sheet_counter_unified.py|scripts/5_unified/sheet_counter_unified.py|g' \
    -e 's|scripts/list_slides_and_captions.py|scripts/utils/list_slides_and_captions.py|g' \
    $doc
done

echo "✅ scripts/ reorganized into 1_cv, 2_calibration, 3_ml, 4_multimodal, 5_unified, utils and docs updated."

