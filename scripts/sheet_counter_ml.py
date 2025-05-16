#!/usr/bin/env python3
"""
sheet_counter_ml.py

Loads the two RandomForest models trained on
[raw_count, total_length, total_edges] features,
applies them to new Wrapped / Unwrapped images,
and prints out predicted sheet counts.
"""

import os, glob, pickle, cv2, numpy as np
import pandas as pd
import argparse
from sklearn.cluster import DBSCAN

# ─── PATHS TO YOUR TRAINED MODELS ───────────────────────────────────────────────
RF_WRAPPED_PKL   = "rf_wrapped.pkl"
RF_UNWRAPPED_PKL = "rf_unwrapped.pkl"
# ────────────────────────────────────────────────────────────────────────────────

# Preprocessing & feature‐extraction constants (same as training)
IMAGE_EXTS    = ("*.jpg","*.jpeg","*.png")
CANNY_THRESH1 = 50
CANNY_THRESH2 = 150
SOBEL_KSIZE   = 3

# Hough/DBSCAN params (same as training)
PHT_THRESH     = 150
MIN_LINE_LEN   = 70
MAX_LINE_GAP   = 8
DBSCAN_EPS     = 5
DBSCAN_MIN_SMP = 1

def extract_features(path):
    # 1. Read & Sobel → Canny
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)
    sob  = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=SOBEL_KSIZE)
    abs_sob = np.absolute(sob)
    norm    = np.uint8(abs_sob / (abs_sob.max()+1e-6) * 255)
    edges   = cv2.Canny(norm, CANNY_THRESH1, CANNY_THRESH2)
    total_edges = edges.sum() / 255

    # 2. Probabilistic Hough + DBSCAN
    segs = cv2.HoughLinesP(edges, 1, np.pi/180, PHT_THRESH,
                           minLineLength=MIN_LINE_LEN,
                           maxLineGap=MAX_LINE_GAP)
    raw_count = 0
    total_length = 0
    mids = []
    if segs is not None:
        for x1,y1,x2,y2 in segs[:,0]:
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if ang < 1 or abs(ang-180) < 1:
                total_length += np.hypot(x2-x1, y2-y1)
                mids.append([(y1+y2)/2.0])
        if mids:
            labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SMP).fit(np.array(mids)).labels_
            raw_count = len([l for l in set(labels) if l != -1])

    return [raw_count, total_length, total_edges]

def process_folder(model, folder, label):
    paths = []
    for ext in IMAGE_EXTS:
        paths += glob.glob(os.path.join(folder, ext))
    paths = sorted(paths)

    print(f"\n--- {label} ({len(paths)} images) ---")
    total = 0
    for p in paths:
        feat = extract_features(p)
        pred = model.predict([feat])[0]
        total += pred
        print(f"{os.path.basename(p):25s} → {int(round(pred))}")
    print(f"Subtotal {label}: {total}\n")
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--wrap_dir",   required=True, help="Folder of wrapped images")
    parser.add_argument("--unwrap_dir", required=True, help="Folder of unwrapped images")
    args = parser.parse_args()

    # Load models
    rf_wrap   = pickle.load(open(RF_WRAPPED_PKL,   "rb"))
    rf_unwrap = pickle.load(open(RF_UNWRAPPED_PKL, "rb"))

    # Predict
    tot_wrap   = process_folder(rf_wrap,   args.wrap_dir, "Wrapped")
    tot_unwrap = process_folder(rf_unwrap, args.unwrap_dir, "Unwrapped")

    print("="*40)
    print(f"GRAND TOTAL: {tot_wrap + tot_unwrap}")
    print("="*40)

