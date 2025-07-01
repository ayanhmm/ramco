#!/usr/bin/env python3
"""
sheet_counter_inference.py

Loads the two trained regressors and applies them to new wrapped/unwrapped images.
"""

import os, glob, pickle, cv2, numpy as np
import pandas as pd

# ─── EDIT THESE PATHS ──────────────────────────────────────────────────────────
# RF_WRAPPED_PKL   = "rf_wrapped.pkl"
# RF_UNWRAPPED_PKL = "rf_unwrapped.pkl"
RF_WRAPPED_PKL   = "rf_wrapped_multi.pkl"
RF_UNWRAPPED_PKL = "rf_unwrapped_multi.pkl"
# ──────────────────────────────────────────────────────────────────────────────

# Preprocessing & feature extraction (same as training)
def extract_features_for_image(path):
    img    = cv2.imread(path)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray   = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)
    sob    = cv2.Sobel(gray, cv2.CV_64F, 0,1,3)
    abs_s  = np.abs(sob)
    norm   = np.uint8(abs_s / (abs_s.max()+1e-6) * 255)
    edges  = cv2.Canny(norm, 50, 150)
    total_edges = edges.sum()/255

    segs = cv2.HoughLinesP(edges,1,np.pi/180,150,minLineLength=70,maxLineGap=8)
    raw = length = 0
    mids = []
    if segs is not None:
        for x1,y1,x2,y2 in segs[:,0]:
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if ang<1 or abs(ang-180)<1:
                length += np.hypot(x2-x1,y2-y1)
                mids.append([(y1+y2)/2.])
        if mids:
            from sklearn.cluster import DBSCAN
            labels = DBSCAN(eps=5, min_samples=1).fit(np.array(mids)).labels_
            raw = len([l for l in set(labels) if l!=-1])

    return [raw, length, total_edges]

def process_folder(model, folder):
    imgs = sorted(glob.glob(os.path.join(folder, "*.jpg")) +
                  glob.glob(os.path.join(folder, "*.jpeg")) +
                  glob.glob(os.path.join(folder, "*.png")))
    results = []
    for p in imgs:
        feat = extract_features_for_image(p)
        pred = model.predict([feat])[0]
        results.append((os.path.basename(p), int(round(pred))))
    return results

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--wrap_dir",   required=True)
    p.add_argument("--unwrap_dir", required=True)
    args = p.parse_args()

    rf_wrap   = pickle.load(open(RF_WRAPPED_PKL,   "rb"))
    rf_unwrap = pickle.load(open(RF_UNWRAPPED_PKL, "rb"))

    print("\nWrapped Counts:")
    for fn, cnt in process_folder(rf_wrap, args.wrap_dir):
        print(f"  {fn:25s} → {cnt}")

    print("\nUnwrapped Counts:")
    for fn, cnt in process_folder(rf_unwrap, args.unwrap_dir):
        print(f"  {fn:25s} → {cnt}")  
#     ```

# **Run inference** on any image sets:

# ```bash
# python sheet_counter_inference.py \
#   --wrap_dir ramco_images/wrap_images \
#   --unwrap_dir ramco_images/nowrap_images

'''
python scripts/3_ml/sheet_counter_inference.py \
    --wrap_dir   data/raw/wrap_images \
    --unwrap_dir data/raw/nowrap_images
'''

