#!/usr/bin/env python3
"""
sheet_counter_calibrated.py

POC to count sheets using Sobel+Hough with calibrated thresholds
and a linear correction learned from calibration_results.csv.
"""

import cv2, numpy as np, glob, os
import pandas as pd
from sklearn.linear_model import LinearRegression

# --- CALIBRATION & POC PARAMETERS ---
SOBEL_KSIZE    = 3
HOUGH_THRESH   = 200      # from calibration
MIN_LINE_GAP   = 15       # from calibration
ANGLE_TOLERANCE= np.pi/180 * 5
IMG_EXTS       = ('*.jpg','*.jpeg','*.png')

# Load calibration table and fit linear correction
cal_df = pd.read_csv("calibration_results.csv")
reg    = LinearRegression().fit(cal_df[['pred_raw']], cal_df['gt_count'])
SLOPE, INTERCEPT = reg.coef_[0], reg.intercept_

def preprocess(path):
    """Read image → Sobel on Y → Otsu binarization."""
    img  = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sob  = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=SOBEL_KSIZE)
    abs_sob = np.uint8(cv2.normalize(np.abs(sob), None, 0,255, cv2.NORM_MINMAX))
    _, bw   = cv2.threshold(abs_sob, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw

def raw_count(bw):
    """Detect horizontal lines via Hough and cluster by y0."""
    lines = cv2.HoughLines(bw,1,np.pi/180,HOUGH_THRESH)
    if lines is None: return 0
    y0s = [rho/(np.sin(theta)+1e-6)
           for rho,theta in lines[:,0]
           if abs(theta) < ANGLE_TOLERANCE or abs(theta-np.pi)<ANGLE_TOLERANCE]
    if not y0s: return 0
    y0s.sort()
    clusters = [y0s[0]]
    for y in y0s[1:]:
        if abs(y-clusters[-1]) > MIN_LINE_GAP:
            clusters.append(y)
    return len(clusters)

def process_dir(label, directory):
    print(f"\nProcessing {label} images in: {directory}")
    total_raw = 0
    total_cal = 0
    paths = sorted(sum((glob.glob(os.path.join(directory, ext)) for ext in IMG_EXTS), []))
    for p in paths:
        bw  = preprocess(p)
        r   = raw_count(bw)
        c   = int(SLOPE * r + INTERCEPT)
        total_raw += r
        total_cal += c
        print(f"  {os.path.basename(p):25} raw={r:3d} → cal={c:3d}")
    print(f"Subtotal {label}: raw={total_raw:4d}, cal={total_cal:4d}")
    return total_raw, total_cal

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--wrap_dir",   required=True, help="Folder with wrapped images")
    p.add_argument("--unwrap_dir", required=True, help="Folder with unwrapped images")
    args = p.parse_args()

    wr_raw, wr_cal = process_dir("Wrapped",   args.wrap_dir)
    un_raw, un_cal = process_dir("Unwrapped", args.unwrap_dir)

    print("\n" + "="*40)
    print(f"TOTAL WRAPPED : raw={wr_raw:4d}, cal={wr_cal:4d}")
    print(f"TOTAL UNWRAPPED: raw={un_raw:4d}, cal={un_cal:4d}")
    print(f"GRAND TOTAL    : raw={wr_raw+un_raw:4d}, cal={wr_cal+un_cal:4d}")
    print("="*40)

