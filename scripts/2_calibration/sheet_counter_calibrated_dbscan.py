#!/usr/bin/env python3
"""
sheet_counter_calibrated_dbscan.py

Counts corrugated sheets using:
 1) Sobel → Canny
 2) Probabilistic Hough (PHT=150, minLen=70, maxGap=8)
 3) DBSCAN(eps=5) on segment mid-y’s
 4) Linear correction from calibration_results_dbscan.csv
"""

import cv2, numpy as np, glob, os, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression

# ─── PATHS ─────────────────────────────────────────────────────────────────────
CAL_CSV    = "calibration_results_dbscan.csv"
WRAP_DIR   = "ramco_images/wrap_images"
UNWRAP_DIR = "ramco_images/nowrap_images"
# ────────────────────────────────────────────────────────────────────────────────

# Best DBSCAN‐PHT params from grid search
PHT_THRESH   = 150
MIN_LINE_LEN = 70
MAX_LINE_GAP = 8
DBSCAN_EPS   = 5

# Preprocessing parameters
CANNY_THRESH1 = 50
CANNY_THRESH2 = 150
SOBEL_KSIZE   = 3

# Load calibration and fit regression
df_cal = pd.read_csv(CAL_CSV)
reg    = LinearRegression().fit(df_cal[["pred_raw"]], df_cal["gt_count"])
SLOPE, INTERCEPT = reg.coef_[0], reg.intercept_

def preprocess(path):
    """Read image, Sobel on Y-axis, normalize, Canny edge."""
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)
    sob  = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=SOBEL_KSIZE)
    abs_sob = np.uint8(np.absolute(sob) / (np.abs(sob).max()+1e-6) * 255)
    return cv2.Canny(abs_sob, CANNY_THRESH1, CANNY_THRESH2)

def raw_count_dbscan(bw):
    segs = cv2.HoughLinesP(
        bw, 1, np.pi/180,
        PHT_THRESH,
        minLineLength=MIN_LINE_LEN,
        maxLineGap=MAX_LINE_GAP
    )
    if segs is None:
        return 0
    mids = []
    for x1,y1,x2,y2 in segs[:,0]:
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if ang < 1 or abs(ang-180) < 1:
            mids.append([(y1+y2)/2.0])
    if not mids:
        return 0
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=1).fit(np.array(mids)).labels_
    return len([l for l in set(labels) if l != -1])

def process(label, directory):
    print(f"\nProcessing {label} in: {directory}")
    total_raw = total_cal = 0
    for ext in ("*.jpg","*.jpeg","*.png"):
        for p in sorted(glob.glob(os.path.join(directory, ext))):
            bw  = preprocess(p)
            r   = raw_count_dbscan(bw)
            c   = int(SLOPE * r + INTERCEPT)
            total_raw += r
            total_cal += c
            print(f"  {os.path.basename(p):25} raw={r:4d} → cal={c:4d}")
    print(f"Subtotal {label}: raw={total_raw:4d}, cal={total_cal:4d}")
    return total_raw, total_cal

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--wrap_dir",   default=WRAP_DIR,   help="Wrapped images folder")
    p.add_argument("--unwrap_dir", default=UNWRAP_DIR, help="Unwrapped images folder")
    args = p.parse_args()

    wr_raw, wr_cal   = process("Wrapped",   args.wrap_dir)
    un_raw, un_cal   = process("Unwrapped", args.unwrap_dir)
    grand_raw = wr_raw + un_raw
    grand_cal = wr_cal + un_cal

    print("\n" + "="*40)
    print(f"TOTAL WRAPPED   : raw={wr_raw:4d}, cal={wr_cal:4d}")
    print(f"TOTAL UNWRAPPED : raw={un_raw:4d}, cal={un_cal:4d}")
    print(f"GRAND TOTAL     : raw={grand_raw:4d}, cal={grand_cal:4d}")
    print("="*40)

