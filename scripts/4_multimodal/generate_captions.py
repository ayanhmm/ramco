#!/usr/bin/env python3
"""
generate_captions.py

Auto-caption all wrapped/unwrapped images with BLIP (once),
saving results to captions.json for later reuse.
"""

import os, cv2, json
from tqdm import tqdm
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ─── CONFIG ───────────────────────────────────────────────
WRAP_DIR      = "data/raw/wrap_images"
NOWRAP_DIR    = "data/raw/nowrap_images"
OUT_JSON    = "data/previews/captions.json"
IMAGE_EXTS  = (".jpg", ".jpeg", ".png")
# ──────────────────────────────────────────────────────────

# Load BLIP once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.eval()

def caption_image(path):
    img = cv2.imread(path)[..., ::-1]
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True)

if __name__=="__main__":
    # gather all image paths
    all_paths = []
    for d in (WRAP_DIR, NOWRAP_DIR):
        for fn in os.listdir(d):
            if fn.lower().endswith(IMAGE_EXTS):
                all_paths.append(os.path.join(d, fn))
    captions = {}
    for p in tqdm(all_paths, desc="Captioning images"):
        captions[os.path.basename(p)] = caption_image(p)
    # write to disk
    with open(OUT_JSON, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"Saved {len(captions)} captions to {OUT_JSON}")


'''
python scripts/4_multimodal/generate_captions.py 
'''