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
python scripts/5_unified/sheet_counter_unified.py \
  --wrap-pptx  data/raw/"Sheet stack with stretch wrap.pptx" \
  --nowrap-pptx data/raw/"Sheet stack without stretch wrap.pptx" \
  --wrap-dir   data/raw/wrap_images \
  --nowrap-dir data/raw/nowrap_images
```

See docs/usage.md for full details.
