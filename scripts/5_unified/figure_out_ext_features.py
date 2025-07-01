import os, glob, pickle, cv2, numpy as np, torch
from pptx import Presentation
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from transformers import CLIPProcessor, CLIPModel
import cv2

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RF_WRAP_BUNDLE   = "./models/rf_wrapped_multi_pca_layers.pkl"
RF_NOWRAP_BUNDLE = "./models/rf_unwrapped_multi_pca_layers.pkl"
CLIP_MODEL_NAME  = "openai/clip-vit-base-patch32"
# ───────────────────────────────────────────────────────────────────────────────

# Load trained bundles
wrap_bundle   = pickle.load(open(RF_WRAP_BUNDLE,   "rb"))
nowrap_bundle = pickle.load(open(RF_NOWRAP_BUNDLE, "rb"))
pca_wrap,   model_wrap   = wrap_bundle["pca"],   wrap_bundle["model"]
pca_nowrap, model_nowrap = nowrap_bundle["pca"], nowrap_bundle["model"]

# Initialize CLIP text encoder
clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
clip_model.eval()


path = "data/raw/image1_2.jpeg"
txt = "2.50m - Up lap- 52 No's"
save_images = 1


# Base output folder (could already have other contents)
base_output_dir = "data"
preview_dir = os.path.join(base_output_dir, "previews/5_figure_out_ext_features/")
os.makedirs(preview_dir, exist_ok=True)
basename = os.path.splitext(os.path.basename(path))[0]
preview_path = os.path.join(preview_dir, f"{basename}_whatever.jpg")


"""Compute CV feats + CLIP-text-PCA feats; return combined feature vector."""
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)
cv2.imwrite(os.path.join(preview_dir, f"1_original.jpg"), img)

# ── CV horizontal edges → raw_count & length
# Sobel edge detection (to highlight horizontal gradients),
# Canny edge detection (to extract edges clearly),
# Hough Line Transform (to detect straight horizontal line segments).
sob_y = cv2.Sobel(gray, cv2.CV_64F, 0,1,3)
norm_y = np.uint8(np.abs(sob_y)/(np.abs(sob_y).max()+1e-6)*255)
if(save_images):
    cv2.imwrite(os.path.join(preview_dir, f"2_y_sobel.jpg"), norm_y)

ed_y = cv2.Canny(norm_y, 50,150)
if(save_images):
    cv2.imwrite(os.path.join(preview_dir, f"3_y_canny.jpg"), ed_y)


segs_h = cv2.HoughLinesP(ed_y,0.2,np.pi/180,0,0,8)
img_lines = img.copy()
if segs_h is not None:
    for x1, y1, x2, y2 in segs_h[:, 0]:
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
if(save_images):
    cv2.imwrite(os.path.join(preview_dir, f"4_y_hough_lines.jpg"), img_lines)




raw = length = 0
mids = []
least_samples = 1
epsilon_eps = 2
if segs_h is not None:
    for x1,y1,x2,y2 in segs_h[:,0]:
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if ang<1 or abs(ang-180)<1:
            length += np.hypot(x2-x1, y2-y1)
            mids.append([(y1+y2)/2.])
    if mids:
        # labels_h = DBSCAN(eps=5, min_samples=1).fit(np.array(mids)).labels_
        labels_h = DBSCAN(eps=epsilon_eps, min_samples=least_samples).fit(np.array(mids)).labels_
        
        raw = len([l for l in set(labels_h) if l!=-1])
total_edges = ed_y.sum()/255
print(f"Found {len(mids)} midlines, clustering into {len(set(labels_h))} groups.")






import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Only if mids are found and clustered
noise = 0
if mids:
    mids_np = np.array(mids)
    labels_h = DBSCAN(eps=epsilon_eps, min_samples=least_samples).fit(mids_np).labels_
    raw = len([l for l in set(labels_h) if l != -1])

    midline_img = img.copy()  # Use correct image source
    cmap = plt.colormaps.get_cmap("tab10")

    for i, y in enumerate(mids_np[:, 0]):
        label = labels_h[i]
        if label == -1:
            color = (0, 0, 0)
            noise = noise + 1
        else:
            rgba = cmap(label % 10)
            color = tuple(int(255 * c) for c in rgba[:3])[::-1]

        y_int = int(round(y))
        cv2.line(midline_img, (0, y_int), (midline_img.shape[1], y_int), color, 2)

    out_path = os.path.abspath(os.path.join(preview_dir, "5_y_clustered_midlines.jpg"))
    success = cv2.imwrite(out_path, midline_img)
    print("Saved to:", out_path if success else "Failed to save image!")
print("noise: ",noise)








# ── CV vertical edges → layer_count
sob_x = cv2.Sobel(gray, cv2.CV_64F, 1,0,3)
norm_x = np.uint8(np.abs(sob_x)/(np.abs(sob_x).max()+1e-6)*255)
if(save_images):
    cv2.imwrite(os.path.join(preview_dir, f"6_x_sobel.jpg"), norm_x)

ed_x = cv2.Canny(norm_x, 50,150)
if(save_images):
    cv2.imwrite(os.path.join(preview_dir, f"7_x_canny.jpg"), ed_x)

segs_v = cv2.HoughLinesP(ed_x,1,np.pi/180,100,30,5)
img_lines_2 = img.copy()
if segs_v is not None:
    for x1, y1, x2, y2 in segs_v[:, 0]:
        cv2.line(img_lines_2, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue vertical lines

if(save_images):
    cv2.imwrite(os.path.join(preview_dir, f"8_x_hough_lines.jpg"), img_lines_2)

layer_count = 0
midsx = []
if segs_v is not None:
    for x1,y1,x2,y2 in segs_v[:,0]:
        ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
        if abs(ang-90)<5:
            midsx.append([(x1+x2)/2.])
    if midsx:
        labels_v = DBSCAN(eps=10, min_samples=1).fit(np.array(midsx)).labels_
        layer_count = len([l for l in set(labels_v) if l!=-1])





if midsx:
    midsx_np = np.array(midsx)
    labels_v = DBSCAN(eps=10, min_samples=1).fit(midsx_np).labels_
    layer_count = len([l for l in set(labels_v) if l != -1])

    img_vlines = img.copy()
    cmap = plt.colormaps.get_cmap("tab10")
    
    for i, x in enumerate(midsx_np[:, 0]):
        label = labels_v[i]
        if label == -1:
            color = (120, 120, 120)  # gray for noise
        else:
            rgba = cmap(label % 10)
            color = tuple(int(255 * c) for c in rgba[:3])[::-1]  # RGB → BGR

        x_int = int(round(x))
        cv2.line(img_vlines, (x_int, 0), (x_int, img.shape[0]), color, 2)
    out_path = os.path.join("previews", f"9_x_clustered_midlines.jpg")
    if(save_images):
        cv2.imwrite(out_path, img_vlines)








cv_feats = [raw, length, total_edges, layer_count]
print(f"CV Features: {cv_feats}")

# ── CLIP text → embed → PCA
# fn = os.path.basename(path)
# txt = title_map.get(fn, "")
inputs = clip_proc(text=[txt], return_tensors="pt", padding=True)
print(inputs)
with torch.no_grad():
    text_emb = clip_model.get_text_features(**inputs)[0].cpu().numpy()
# print(f"Text embedding shape: {text_emb.shape}")

# choose correct PCA by folder name
folder = os.path.basename(os.path.dirname(path))
if folder == "wrap_images":
    text_pca = pca_wrap.transform([text_emb])[0]
else:
    text_pca = pca_nowrap.transform([text_emb])[0]

# return np.hstack([cv_feats, text_pca])
'''
python scripts/5_unified/figure_out_ext_features.py 
'''
