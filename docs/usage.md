# Ramco Sheet-Counter

Automated pipeline for counting stacked fiber-cement sheets from photos.

---

## Repository Layout

.
├── data/
│ ├── raw/
│ │ ├── wrap_images/ ← your wrapped-image set
│ │ ├── nowrap_images/ ← your unwrapped-image set
│ │ ├── "Sheet stack with stretch wrap.pptx"
│ │ ├── "Sheet stack without stretch wrap.pptx"
│ │ └── calibration_results.csv ← ground-truth calibration parameters
│ └── extracted/ ← optional: extracted features, etc.
├── models/ ← trained RF pipelines (*.pkl)
├── scripts/ ← all runnable scripts
│ ├── 1_cv/ ← classic computer-vision scripts
│ ├── 2_ml/ ← CV+ML and calibration scripts
│ ├── 3_multimodal/ ← BLIP/CLIP based experiments
│ └── sheet_counter_unified.py ← unified hybrid entry point
├── notebooks/ ← exploration & EDA
└── docs/
├── usage.md ← this file
└── testing.md ← how to run smoke tests

yaml
Copy
Edit

---

## Quickstart

1. **Create & activate your virtualenv**  
   ```bash
   python3 -m venv ramco_env
   source ramco_env/bin/activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the unified counter

bash
Copy
Edit
python scripts/sheet_counter_unified.py \
  --wrap-pptx  data/raw/"Sheet stack with stretch wrap.pptx" \
  --nowrap-pptx data/raw/"Sheet stack without stretch wrap.pptx" \
  --wrap-dir   data/raw/wrap_images \
  --nowrap-dir data/raw/nowrap_images \
  --calibration data/raw/calibration_results.csv
Parses exact counts from any slide titles in your PPTs

Falls back to a calibrated CV + RF model on raw images

Smoke-Test Suite
We include shell scripts under scripts/utils/tests/ to verify each pivot layer:

1. CV smoke tests
bash
Copy
Edit
bash scripts/utils/tests/smoke_test_1_cv.sh
Runs:

1_cv/sheet_counter_poc.py

1_cv/sheet_counter_ridges.py

1_cv/sheet_counter_calibrated.py

2. ML smoke tests
bash
Copy
Edit
bash scripts/utils/tests/smoke_test_2_ml.sh
Runs:

2_ml/train_feature_regressors.py

2_ml/sheet_counter_ml.py

2_ml/retrain_cv_only_models.py

3. Multimodal smoke tests
bash
Copy
Edit
bash scripts/utils/tests/smoke_test_3_multimodal.sh
Runs:

3_multimodal/train_multimodal.py

3_multimodal/sheet_counter_multimodal.py

Script Catalog & Intent
Script	Intent / Pivot
scripts/1_cv/sheet_counter_poc.py	POC CV: simple edge-based line counting; undercounts in scenes
scripts/1_cv/sheet_counter_ridges.py	Ridge detection + Hough; still noisy
scripts/1_cv/sheet_counter_calibrated.py	CV + per-mode linear calibration (final = a·raw + b)
scripts/2_ml/train_feature_regressors.py	RF on CV features: [raw_count, length, edges, layers]
scripts/2_ml/sheet_counter_ml.py	Inference: apply CV-trained RF to new images
scripts/2_ml/retrain_cv_only_models.py	Retrain CV-only RF on updated ground truth
scripts/3_multimodal/train_multimodal.py	BLIP+CLIP multimodal regressor
scripts/3_multimodal/sheet_counter_multimodal.py	Inference using multimodal RF models
scripts/sheet_counter_unified.py	Unified Hybrid: exact PPT parse → CV+RF fallback with calibration

Evolution Roadmap
Primitive CV → under-counts due to occlusions & noise

Ridge & DBSCAN → better grouping but still high variance

Linear Calibration → corrects systematic bias (a·raw + b)

Random Forest (CV features) → major MSE reduction

Multimodal (BLIP+CLIP) → marginal gains; embeddings too coarse

Dimensionality Reduction (PCA) → limited by sample size

Layer-count feature → captures vertical stacks; risk of over-fitting

Unified Hybrid → combines exact PPT parsing + robust ML fallback


