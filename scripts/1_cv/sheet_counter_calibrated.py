#!/usr/bin/env python3
"""
sheet_counter_calibrated.py

Count sheets by simple CV line-detection, then apply a linear calibration
(final = a·raw + b) per mode (“wrap” vs “nowrap”).
"""
import os
import cv2
import numpy as np
import argparse
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

def compute_raw_count(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)

    sob = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sob = np.uint8(np.abs(sob) / (np.abs(sob).max() + 1e-6) * 255)
    edges = cv2.Canny(sob, 50, 150)

    segs = cv2.HoughLinesP(edges, 1, np.pi/180,
                           threshold=150, minLineLength=70, maxLineGap=8)
    if segs is None:
        return 0

    mids = []
    for x1, y1, x2, y2 in segs[:,0]:
        angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if angle < 2 or angle > 178:
            mids.append([(y1 + y2) / 2.])
    if not mids:
        return 0

    labels = DBSCAN(eps=5, min_samples=1).fit(np.vstack(mids)).labels_
    return len(set(labels))

def load_calibration(csv_path):
    df = pd.read_csv(csv_path)
    # normalize column names to mode,a,b
    first3 = list(df.columns[:3])
    if first3 != ['mode','a','b']:
        df = df.rename(columns={
            first3[0]: 'mode',
            first3[1]: 'a',
            first3[2]: 'b',
        })
    return {row['mode']: (float(row['a']), float(row['b']))
            for _, row in df.iterrows()}

def process_folder(folder, mode, ab):
    a, b = ab
    imgs = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(('.jpg','.jpeg','.png'))
    )
    subtotal = 0
    print(f"\n--- {mode.title()} ({len(imgs)} images) ---")
    for fn in tqdm(imgs, desc=mode):
        raw = compute_raw_count(os.path.join(folder, fn))
        cal = int(round(a * raw + b))
        print(f"{fn:25s} → raw={raw:3d}  cal={cal:3d}")
        subtotal += cal
    print(f"Subtotal {mode.title()}: {subtotal}")
    return subtotal

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrap_dir",    required=True, help="folder of wrapped images")
    p.add_argument("--unwrap_dir",  required=True, help="folder of unwrapped images")
    p.add_argument("--calibration", required=True,
                   help="CSV file with columns [mode, a, b] per row")
    args = p.parse_args()

    calib = load_calibration(args.calibration)
    wrap_ab   = calib.get('wrap',   (1.0, 0.0))
    nowrap_ab = calib.get('nowrap', (1.0, 0.0))

    tot_w = process_folder(args.wrap_dir,   'wrap',   wrap_ab)
    tot_u = process_folder(args.unwrap_dir, 'nowrap', nowrap_ab)

    print("\n" + "="*40)
    print(f"GRAND TOTAL: {tot_w + tot_u}")
    print("="*40)

