import os
import cv2
import numpy as np

def get_stack_tilt_angle(gray,
                         canny_thresh1=50,
                         canny_thresh2=150,
                         hough_thresh=80,
                         min_line_len=150,
                         max_line_gap=10,
                         angle_tol=20):
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)
    lines = cv2.HoughLinesP(edges,
                            rho=1,
                            theta=np.pi/180,
                            threshold=hough_thresh,
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    if lines is None:
        return 0.0

    angles = []
    for x1, y1, x2, y2 in lines[:,0]:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0: continue
        ang = np.degrees(np.arctan2(dy, dx))
        if abs(ang) < angle_tol:
            angles.append(ang)
    return float(np.median(angles)) if angles else 0.0

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT
    )

def unsharp_mask(img, blur_ksize=(9,9), sigma=10.0, amount=1.5, threshold=0):
    """
    Classic unsharp mask: sharpen = original + amount*(original - blurred)
    threshold optionally skips low-contrast areas.
    """
    blurred = cv2.GaussianBlur(img, blur_ksize, sigma)
    diff = cv2.subtract(img, blurred)
    # optionally threshold
    if threshold > 0:
        low_contrast = np.abs(diff) < threshold
        diff[low_contrast] = 0
    sharpened = cv2.addWeighted(img, 1.0, diff, amount, 0)
    return sharpened

def crop_pure_black_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows = np.any(gray>0, axis=1)
    cols = np.any(gray>0, axis=0)
    if not rows.any() or not cols.any():
        return img
    y0, y1 = np.argmax(rows), h - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), w - np.argmax(cols[::-1])
    return img[y0:y1, x0:x1]

def align_and_save(path, out_dir="final_aligned", debug=False):
    img = cv2.imread(path)
    if img is None:
        print(f"[!] Cannot read {path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tilt = get_stack_tilt_angle(gray)
    print(f"{os.path.basename(path)} → tilt = {tilt:.2f}°")

    # 1) rotate
    rotated = rotate_image(img, tilt)

    # 2) sharpen
    sharpened = unsharp_mask(rotated,
                             blur_ksize=(9,9),
                             sigma=10.0,
                             amount=1.5,
                             threshold=5)

    # 3) crop pure-black
    final = crop_pure_black_border(sharpened)

    # 4) save
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(path))
    cv2.imwrite(out_path, final)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        for i, im in enumerate([img, rotated, final]):
            plt.subplot(1,3,i+1)
            title = ["Original", f"Rotated {tilt:.2f}°", "Final"][i]
            plt.title(title)
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def main(input_folder, output_folder="final_aligned"):
    if not os.path.isdir(input_folder):
        print("[!] Invalid input folder.")
        return
    for fn in os.listdir(input_folder):
        if not fn.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
            continue
        align_and_save(os.path.join(input_folder, fn),
                       out_dir=output_folder,
                       debug=False)

if __name__ == "__main__":
    # input_folder  = "data/raw/nowrap_images"
    input_folder  = "data/png"
    
    output_folder = "data/png/final_aligned"
    main(input_folder, output_folder)
