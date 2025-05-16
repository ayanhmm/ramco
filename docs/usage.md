# Usage

This repository provides a set of scripts and models to automate counting stacked fiber-cement sheets from images (wrapped and unwrapped). Follow these steps to get started:

---

## Quickstart

1. **Create & activate your virtual environment**  
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
  --nowrap-dir data/raw/nowrap_images
This will:

Parse exact counts from any slide titles when available

Fall back to a trained regression model on CV features for “raw” photos

Script Catalog & Intent
Script	Intent / Pivot Description
scripts/sheet_counter_poc.py	Proof-of-concept CV: simple edge-based line counting; undercounts in complex scenes.
scripts/sheet_counter_ridges.py	Alternate CV: ridge detection to capture edges; still noisy in real images.
scripts/full_calibration.py	Linear Calibration: apply analytic bias correction (final = a·raw + b).
scripts/full_calibration_dbscan.py	DBSCAN Clustering: cluster detected line midpoints to group per-sheet detections.
scripts/calibrate_counts.py	Calibration Utility: interactive fitting of (a, b) against PPT ground truth.
scripts/list_slides_and_captions.py	Slide Metadata Extraction: map PPT slides → image filenames & extract title text.
scripts/generate_captions.py	BLIP Captioning: auto-annotate images to bootstrap multimodal features.
scripts/train_feature_regressors.py	RF on CV Features: train RandomForest on [raw_count, length, edges, layers].
scripts/sheet_counter_ml.py	Inference CV-ML: apply the CV-trained RF models to new images.
scripts/train_multimodal.py	Multimodal (BLIP+CLIP): combine BLIP captions + CLIP embeddings for regression.
scripts/train_multimodal_cached.py	Cached Features: extract & cache BLIP/CLIP embeddings to speed up iterations.
scripts/train_multimodal_pca.py	PCA on CLIP: reduce 512-dim embeddings to top components (limited by sample size).
scripts/train_multimodal_pca_with_layers.py	Add Layer Count: include vertical-Hough sheet layers into multimodal PCA pipeline.
scripts/sheet_counter_multimodal.py	Inference Multimodal: run the best multimodal RF model on new images.
scripts/sheet_counter_multimodal_with_layers.py	Inference + Layers: include layer count in inference.
scripts/retrain_cv_only_models.py	Retrain CV-only RF: rebuild RF models on CV features using latest ground truth splits.
scripts/sheet_counter_unified.py	Unified Hybrid: parse exact PPT counts when available; else fallback to CV+RF(+cal).

Evolution Roadmap
Primitive CV → undercounts due to occlusions and noise

Ridge & DBSCAN → better grouping but still high variance

Linear Calibration → reduce systematic bias (a·raw + b)

Random Forest on CV features → major MSE reduction

Multimodal (BLIP+CLIP) → marginal gains from text embeddings

Dimensionality Reduction (PCA) → limited by sample size

Layer Count Feature → captures vertical stacks; risk of overfitting

Unified Hybrid → combine exact parsing + robust ML fallback

For full details and options, see each script’s docstring or the deeper design notes in docs/architecture.md.
