#!/usr/bin/env python3
"""
sheet_counter_hough.py

Adaptive POC: count corrugated sheets by detecting horizontal lines via Hough Transform.
"""

import cv2, numpy as np, glob, os

# --- PARAMETERS you can still tweak slightly ---
SOBEL_KSIZE   = 3
HOUGH_THRESH  = 150  # accumulator threshold for HoughLines
ANGLE_TOLERANCE = np.pi/180 * 5  # ±5° around horizontal
MIN_LINE_GAP = 10  # px to consider two lines distinct
IMG_EXTS = ('*.jpg','*.jpeg','*.png')

def preprocess(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sobel for horizontal edges
    sob = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=SOBEL_KSIZE)
    abs_sob = np.uint8(cv2.normalize(np.absolute(sob), None, 0,255, cv2.NORM_MINMAX))
    # Otsu threshold
    _, bw = cv2.threshold(abs_sob, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw

def count_sheets_via_hough(bw):
    # detect lines in (rho,theta)
    lines = cv2.HoughLines(bw, 1, np.pi/180, HOUGH_THRESH)
    if lines is None:
        return 0

    # keep only near-horizontal lines (theta≈0 or π)
    horzs = []
    for rho,theta in lines[:,0]:
        if abs(theta) < ANGLE_TOLERANCE or abs(theta-np.pi) < ANGLE_TOLERANCE:
            # compute y-intercept: rho = x*cosθ + y*sinθ ⇒ y = (rho - x cosθ)/sinθ
            # for x=0: y0 = rho/sinθ
            y0 = rho/np.sin(theta+1e-6)
            horzs.append(y0)
    if not horzs:
        return 0

    # cluster by y0: sort and count distinct clusters > MIN_LINE_GAP apart
    horzs = sorted(horzs)
    clusters = [horzs[0]]
    for y in horzs[1:]:
        if abs(y - clusters[-1]) > MIN_LINE_GAP:
            clusters.append(y)
    return len(clusters)

def process_dir(name, d):
    paths = sorted(sum((glob.glob(os.path.join(d,ext)) for ext in IMG_EXTS), []))
    print(f"\nProcessing {len(paths)} images in {name}")
    total = 0
    for p in paths:
        bw = preprocess(p)
        cnt = count_sheets_via_hough(bw)
        total += cnt
        print(f"  {os.path.basename(p):30} → {cnt}")
    print(f"Subtotal {name}: {total}")
    return total

if __name__=="__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--wrap_dir",   required=True)
    p.add_argument("--unwrap_dir", required=True)
    args = p.parse_args()

    w = process_dir("Wrapped",   args.wrap_dir)
    u = process_dir("Unwrapped", args.unwrap_dir)
    print("\nGRAND TOTAL:", w+u)

