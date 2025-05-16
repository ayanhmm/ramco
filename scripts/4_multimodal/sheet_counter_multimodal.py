#!/usr/bin/env python3
"""
sheet_counter_multimodal.py

Loads RF models trained on [CV features + CLIP text embeddings].
Applies them to new Wrapped/Unwrapped images and prints counts.
"""

import os, glob, pickle, cv2, numpy as np, json, torch
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from transformers import CLIPProcessor, CLIPModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RF_WRAP   = "rf_wrapped_multi.pkl"
RF_NOWRAP = "rf_unwrapped_multi.pkl"
CAP_JSON  = "captions.json"
# Folders to process via CLI
# ────────────────────────────────────────────────────────────────────────────────

# Load models & embeddings
rf_wrap   = pickle.load(open(RF_WRAP, "rb"))
rf_nowrap = pickle.load(open(RF_NOWRAP, "rb"))
captions  = json.load(open(CAP_JSON, "r"))
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model= CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# Preproc constants
CANNY1, CANNY2, KSOB = 50, 150, 3
PHT_PARAMS = dict(rho=1, theta=np.pi/180, threshold=150,
                  minLineLength=70, maxLineGap=8)
DB_PARAMS  = dict(eps=5, min_samples=1)

def extract_features(path):
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)
    sob  = cv2.Sobel(gray, cv2.CV_64F, 0,1,KSOB)
    norm = np.uint8(np.abs(sob)/(np.abs(sob).max()+1e-6)*255)
    edges= cv2.Canny(norm, CANNY1, CANNY2)
    total_edges = edges.sum()/255

    segs = cv2.HoughLinesP(edges, **PHT_PARAMS)
    raw=length=0; mids=[]
    if segs is not None:
        for x1,y1,x2,y2 in segs[:,0]:
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if ang<1 or abs(ang-180)<1:
                length += np.hypot(x2-x1,y2-y1)
                mids.append([(y1+y2)/2.])
        if mids:
            labels = DBSCAN(**DB_PARAMS).fit(np.array(mids)).labels_
            raw = len([l for l in set(labels) if l!=-1])

    fn = os.path.basename(path)
    txt = captions.get(fn, "")
    inp = clip_proc(text=[txt], return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_emb = clip_model.get_text_features(**inp)[0].cpu().numpy()

    return np.concatenate([[raw, length, total_edges], txt_emb])

def process_folder(model, folder, label):
    imgs = []
    for ext in ("*.jpg","*.jpeg","*.png"):
        imgs += glob.glob(os.path.join(folder, ext))
    imgs = sorted(imgs)
    total = 0
    print(f"\nProcessing {label} ({len(imgs)} images)…")
    for p in tqdm(imgs, desc=label):
        feat = extract_features(p)
        pred = model.predict([feat])[0]
        total += pred
        print(f"  {os.path.basename(p):25s} → {int(round(pred))}")
    print(f"Subtotal {label}: {total}")
    return total

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--wrap_dir",   required=True)
    parser.add_argument("--unwrap_dir", required=True)
    args = parser.parse_args()

    tw = process_folder(rf_wrap,   args.wrap_dir,   "Wrapped")
    tu = process_folder(rf_nowrap, args.unwrap_dir, "Unwrapped")
    print("\n" + "="*40)
    print(f"GRAND TOTAL: {tw + tu}")
    print("="*40)

