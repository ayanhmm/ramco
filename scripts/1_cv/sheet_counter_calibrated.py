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
import matplotlib.pyplot as plt

import cv2
import numpy as np
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def compute_raw_count(image_path, preview_dir="data/previews/1_sheet_counter_calibrated"):
    img = cv2.imread(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

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
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle < 2 or angle > 178:
            mids.append([(y1 + y2) / 2.])

    if not mids:
        return 0

    # Cluster midpoints with DBSCAN
    mids_np = np.array(mids)
    labels = DBSCAN(eps=5, min_samples=1).fit(mids_np).labels_
    raw_count = len(set(labels))

    # Optional: visualize and save clustered lines
    img_lines = img.copy()
    cmap = plt.colormaps.get_cmap("tab10")
    print(labels)
    for i, y in enumerate(mids_np[:, 0]):
        label = labels[i]
        color = (150, 150, 150) if label == -1 else tuple(
            int(c * 255) for c in cmap(label % 10)[:3][::-1]
        )
        y_int = int(round(y))
        cv2.line(img_lines, (0, y_int), (img.shape[1], y_int), color, 2)

    # Save preview
    os.makedirs(preview_dir, exist_ok=True)
    out_path = os.path.join(preview_dir, f"{basename}_hline_clusters.jpg")
    cv2.imwrite(out_path, img_lines)

    return raw_count

def load_calibration(csv_path):
    df = pd.read_csv(csv_path)
    # normalize column names to mode,a,b
    # first3 = list(df.columns[:3])
    # if first3 != ['mode','a','b']:
    #     df = df.rename(columns={
    #         first3[0]: 'mode',
    #         first3[1]: 'a',
    #         first3[2]: 'b',
    #     })
    first5 = list(df.columns[:5])
    if first5 != ['image','mode','actual','a','b']:
        df = df.rename(columns={
            first5[1]: 'mode',
            first5[3]: 'a',
            first5[4]: 'b',
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
    p.add_argument("--nowrap_dir",  required=True, help="folder of unwrapped images")
    p.add_argument("--calibration", required=True, help="CSV file with columns [mode, a, b] per row")
    args = p.parse_args()

    calib = load_calibration(args.calibration)
    print(f"Calibration: {calib}")
    wrap_ab   = calib.get('wrap',   (1.0, 0.0))
    nowrap_ab = calib.get('nowrap', (1.0, 0.0))
    print(f"Wrap:   a={wrap_ab[0]:.3f}, b={wrap_ab[1]:.3f}")

    tot_w = 0
    # tot_w = process_folder(args.wrap_dir,   'wrap',   wrap_ab)
    tot_u = process_folder(args.nowrap_dir, 'nowrap', nowrap_ab)

    print("\n" + "="*40)
    print(f"GRAND TOTAL: {tot_w + tot_u}")
    print("="*40)

'''
python scripts/1_cv/sheet_counter_calibrated.py \
    --wrap_dir   data/raw/wrap_images \
    --nowrap_dir data/raw/nowrap_images \
    --calibration data/calibration_results.csv
'''