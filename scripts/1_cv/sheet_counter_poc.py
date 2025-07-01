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
    output_path="data/previews/1_sheet_counter_poc"
    yo = cv2.GaussianBlur(eq, (5,5), 0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_path = os.path.join(output_path, f"preprocessed.jpg")
    cv2.imwrite(output_path, yo)
    return yo

def count_sheets(img, output_path="data/previews/1_sheet_counter_poc"):
    # Step 1: Edge detection
    edges = cv2.Canny(img, CANNY_THRESH1, CANNY_THRESH2)

    # Step 2: Morphological closing to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Step 3: Find external contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Filter out small noise
    sheets = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    # Visualization
    vis = img.copy()
    cv2.drawContours(vis, sheets, -1, (0, 255, 0), 5)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_path = os.path.join(output_path, f"hline_clusters.jpg")
        cv2.imwrite(output_path, vis)
    else:
        # Optional: show in a window (good for debugging)
        cv2.imshow("Detected Sheets", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wrap_dir",   required=True,
                   help="Folder containing *wrapped* stock-yard images")
    p.add_argument("--nowrap_dir", required=True,
                   help="Folder containing *unwrapped* stock-yard images")
    args = p.parse_args()
    print(args)

    total_wrapped = 0
    total_unwrapped = 0
    # total_wrapped   = process_dir("Wrapped",   args.wrap_dir)
    total_unwrapped = process_dir("Unwrapped", args.nowrap_dir)
    grand_total     = total_wrapped + total_unwrapped

    print("=" * 40)
    print(f"TOTAL wrap_images:     {total_wrapped} sheets")
    print(f"TOTAL unwrap_images:   {total_unwrapped} sheets")
    print(f"GRAND TOTAL:           {grand_total} sheets")
    print("=" * 40)


'''
python scripts/1_cv/sheet_counter_poc.py \
    --wrap_dir   data/raw/wrap_images \
    --nowrap_dir data/raw/nowrap_images 
'''