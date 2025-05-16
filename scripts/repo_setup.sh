#!/usr/bin/env bash
set -euo pipefail

# === 1. Define your repo settings ===
REMOTE_URL="git@github.com:spandaai/ramco.git"
MAIN_BRANCH="main"

# === 2. Create the new directory layout ===
mkdir -p \
  data/raw \
  data/raw/wrap_images \
  data/raw/nowrap_images \
  data/extracted \
  models \
  scripts \
  notebooks \
  docs

# === 3. Move existing artifacts into the new structure ===
# Python scripts
mv sheet_counter_*.py        scripts/ 2>/dev/null || true
mv full_calibration*.py      scripts/ 2>/dev/null || true
mv train_*.py                scripts/ 2>/dev/null || true
mv calibrate_counts.py       scripts/ 2>/dev/null || true
mv list_slides_and_captions.py scripts/ 2>/dev/null || true
mv generate_captions.py      scripts/ 2>/dev/null || true
mv retrain_cv_only_models.py scripts/ 2>/dev/null || true

# Raw PPTs
mv *.pptx            data/raw/          2>/dev/null || true

# Raw images
mv ramco_images/wrap_images   data/raw/wrap_images   2>/dev/null || true
mv ramco_images/nowrap_images data/raw/nowrap_images 2>/dev/null || true
# remove empty parent
rmdir ramco_images 2>/dev/null || true

# Extracted images
mv extracted_images/ data/extracted/    2>/dev/null || true

# Captions
mv captions.json     data/raw/          2>/dev/null || true

# Model pickles
mv rf_*.pkl           models/ 2>/dev/null || true

# Notebooks & docs
mv *.ipynb            notebooks/ 2>/dev/null || true

# === 4. Initialize git (if needed) ===
if [ ! -d .git ]; then
  git init
  git remote add origin "$REMOTE_URL"
fi

# === 5. Create requirements.txt ===
pip freeze > requirements.txt

# === 6. Write README.md ===
cat > README.md <<'EOF'
# Ramco Sheet-Counter

Automated pipeline for counting stacked fiber-cement sheets.

## Directory layout

- **data/raw/**      Raw PPTs, wrap & nowrap images, captions.json  
- **data/extracted/** Derived image crops / annotations  
- **models/**        Trained regression bundles (.pkl)  
- **scripts/**       All runnable Python scripts  
- **notebooks/**     Exploration & EDA (.ipynb)  
- **docs/**          Architecture & usage docs  
- **README.md**  
- **requirements.txt**  
- **.gitignore**

## Quickstart

```bash
# create & activate your venv
python3 -m venv ramco_env && source ramco_env/bin/activate

# install deps
pip install -r requirements.txt

# run unified counter
python scripts/sheet_counter_unified.py \
  --wrap-pptx  data/raw/"Sheet stack with stretch wrap.pptx" \
  --nowrap-pptx data/raw/"Sheet stack without stretch wrap.pptx" \
  --wrap-dir   data/raw/wrap_images \
  --nowrap-dir data/raw/nowrap_images
```

See docs/usage.md for full details.
EOF

# === 7. Write .gitignore ===
cat > .gitignore <<'EOF'

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Virtual environment
ramco_env/

# Models: track .pkl files, ignore any new artifacts
models/
!models/*.pkl

# Data
data/extracted/
# allow tracking raw images and PPTs in data/raw

# OS artifacts
.DS_Store
EOF

# === 8. Stage & commit everything ===
git add .
git commit -m "Reorganize repo: scripts/, data/, models/, notebooks/, docs/ + README + .gitignore + requirements"

# === 9. Push (first time) ===
git branch -M "$MAIN_BRANCH"
git push -u origin "$MAIN_BRANCH"

echo
 echo "🎉 Repo structure initialized and pushed to $REMOTE_URL on '$MAIN_BRANCH'!"

