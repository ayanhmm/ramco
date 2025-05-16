Ramco Sheet-Counter

Automated pipeline for counting stacked fiber-cement sheets from photos.

Repository Layout

.
├── data/
│   ├── raw/
│   │   ├── wrap_images/          ← your wrapped-image set
│   │   ├── nowrap_images/        ← your unwrapped-image set
│   │   ├── "Sheet stack with stretch wrap.pptx"
│   │   ├── "Sheet stack without stretch wrap.pptx"
│   │   └── calibration_results.csv (ground-truth a,b)
│   └── extracted/                ← optional: extracted features, etc.
├── models/                       ← trained RF pipelines (`*.pkl`)
├── scripts/                      ← all runnable scripts
│   ├── 1_cv/                     ← classic computer-vision scripts
│   ├── 2_ml/                     ← CV+ML and calibration scripts
│   ├── 3_multimodal/             ← BLIP/CLIP based experiments
│   └── sheet_counter_unified.py  ← unified hybrid entry point
├── notebooks/                    ← exploration & EDA
└── docs/
    ├── usage.md                  ← this file
    └── testing.md                ← how to run smoke tests

Quickstart

# 1. Create & activate virtualenv
python3 -m venv ramco_env
source ramco_env/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Unified counter (parses PPT titles + fallback CV+RF)
python scripts/sheet_counter_unified.py \
  --wrap-pptx  data/raw/"Sheet stack with stretch wrap.pptx" \
  --nowrap-pptx data/raw/"Sheet stack without stretch wrap.pptx" \
  --wrap-dir   data/raw/wrap_images \
  --nowrap-dir data/raw/nowrap_images \
  --calibration data/raw/calibration_results.csv

Smoke-Test Suite

We include a small set of shell scripts under scripts/utils/tests/ to verify each pivot:

1_cv/smoke_test_1_cv.shRuns:

sheet_counter_poc.py

sheet_counter_ridges.py

sheet_counter_calibrated.py

2_ml/smoke_test_2_ml.shRuns your CV+ML and multimodal inference scripts.

…and so on.

Running the CV smoke tests

# from project root
bash scripts/utils/tests/smoke_test_1_cv.sh

It will look under data/raw/wrap_images and data/raw/nowrap_images.You should see console ✅ passes or detailed errors.

Script Catalog & Intent

Script

Intent / Pivot

scripts/1_cv/sheet_counter_poc.py

POC CV: simple edge-counting; undercounts in complex scenes

scripts/1_cv/sheet_counter_ridges.py

Ridge detection + Hough; still noisy

scripts/1_cv/sheet_counter_calibrated.py

CV + per-mode linear calibration (final = a·raw + b)

scripts/2_ml/train_feature_regressors.py

RF on CV features: [raw, length, edges, layers]

scripts/2_ml/sheet_counter_ml.py

Inference CV+RF

scripts/2_ml/retrain_cv_only_models.py

Retrain RF on updated GT

scripts/3_multimodal/train_multimodal.py

BLIP+CLIP multimodal regressor

scripts/3_multimodal/sheet_counter_multimodal.py

Inference multimodal

scripts/sheet_counter_unified.py

Hybrid: exact PPT parse → CV+RF fallback with calibration

Next Steps & Testing

Add smoke tests under scripts/utils/tests/ for each layer (ML, multimodal, unified).

Validate on your CI by invoking those scripts on a subset of data/raw/.

Document any new script arguments or dependencies back in this guide.

Happy counting!


