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


---

## Quickstart

1. **Create & activate your virtualenv**  
   ```bash
   python3 -m venv ramco_env
   source ramco_env/bin/activate

## Install dependencies
   ```bash
   pip install -r requirements.txt

## Run the unified counter
    ```bash
    python scripts/sheet_counter_unified.py \
   --wrap-pptx  data/raw/"Sheet stack with stretch wrap.pptx" \
   --nowrap-pptx data/raw/"Sheet stack without stretch wrap.pptx" \
   --wrap-dir   data/raw/wrap_images \
   --nowrap-dir data/raw/nowrap_images \
   --calibration data/raw/calibration_results.csv



pip install -r requirements.txt

