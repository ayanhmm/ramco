#!/usr/bin/env python3
"""
sheet_counter_poc.py

POC to count corrugated sheets in two sets of images (wrapped vs. unwrapped)
via edge detection + contour analysis.
"""

import cv2
import glob
import os

# --- PARAMETERS (tune these per your lighting/stack conditions) ---
CANNY_THRESH1     = 50
CANNY_THRESH2     = 150
MORPH_KERNEL_SIZE = 3      # for closing gaps
MIN_CONTOUR_AREA  = 5000   # ignore tiny noise
IMG_EXTENSIONS    = ('*.jpg', '*.jpeg', '*.png')

# --- HELPERS ---
def preprocess(img):
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq    = clahe.apply(gray)
    return cv2.GaussianBlur(eq, (5,5), 0)

def count_sheets(img):
    edges   = cv2.Canny(img, CANNY_THRESH1, CANNY_THRESH2)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    closed  = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    # filter out small artifacts
    sheets = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    return len(sheets)

def process_dir(name, directory):
    """Count sheets in all images under `directory`."""
    # gather image paths
    paths = []
    for ext in IMG_EXTENSIONS:
        paths += glob.glob(os.path.join(directory, ext))
    paths = sorted(paths)

    if not paths:
        print(f"  [!] No images found in {directory}")
        return 0

    subtotal = 0
    print(f"\nProcessing {len(paths)} images in {name}: {directory}")
    print("-" * 40)
    for img_path in paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [skipped] {os.path.basename(img_path)}")
            continue

        pre = preprocess(img)
        cnt = count_sheets(pre)
        subtotal += cnt
        print(f"  {os.path.basename(img_path):30} → {cnt} sheets")

    print("-" * 40)
    print(f"Subtotal for {name}: {subtotal} sheets\n")
    return subtotal

# --- MAIN POC ---
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--wrap_dir",   required=True,
                   help="Folder containing *wrapped* stock-yard images")
    p.add_argument("--unwrap_dir", required=True,
                   help="Folder containing *unwrapped* stock-yard images")
    args = p.parse_args()

    total_wrapped   = process_dir("Wrapped",   args.wrap_dir)
    total_unwrapped = process_dir("Unwrapped", args.unwrap_dir)
    grand_total     = total_wrapped + total_unwrapped

    print("=" * 40)
    print(f"TOTAL wrap_images:     {total_wrapped} sheets")
    print(f"TOTAL unwrap_images:   {total_unwrapped} sheets")
    print(f"GRAND TOTAL:           {grand_total} sheets")
    print("=" * 40)

