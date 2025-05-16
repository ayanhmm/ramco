#!/usr/bin/env python3
"""
sheet_counter_multimodal_with_layers.py

Loads rf_{wrapped,unwrapped}_multi_pca_layers.pkl (PCA + RF models)
Extracts CV features + layer_count + CLIP text-PCA embeddings
Predicts per-image and subtotals for Wrapped and Unwrapped folders.
"""

import os, glob, pickle, cv2, numpy as np, json, torch
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from transformers import CLIPProcessor, CLIPModel

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RF_WRAP       = "rf_wrapped_multi_pca_layers.pkl"
RF_NOWRAP     = "rf_unwrapped_multi_pca_layers.pkl"
CAP_JSON      = "captions.json"
# ────────────────────────────────────────────────────────────────────────────────

# load models
wrap_bundle   = pickle.load(open(RF_WRAP,   "rb"))
unwrap_bundle = pickle.load(open(RF_NOWRAP, "rb"))
pca_wrap,   model_wrap   = wrap_bundle["pca"],   wrap_bundle["model"]
pca_nowrap, model_nowrap = unwrap_bundle["pca"], unwrap_bundle["model"]

captions  = json.load(open(CAP_JSON))
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model= CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# constants
IMAGE_EXTS = ("*.jpg","*.jpeg","*.png")
CANNY1, CANNY2, KSOB = 50, 150, 3

def extract_feat(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.25, beta=0)
    # horizontal edges
    sob_y = cv2.Sobel(gray, cv2.CV_64F, 0,1,KSOB)
    norm_y= np.uint8(np.abs(sob_y)/(np.abs(sob_y).max()+1e-6)*255)
    edges_y = cv2.Canny(norm_y, CANNY1, CANNY2)
    segs_h = cv2.HoughLinesP(edges_y,1,np.pi/180,150,minLineLength=70,maxLineGap=8)
    raw=length=0; mids_h=[]
    if segs_h is not None:
        for x1,y1,x2,y2 in segs_h[:,0]:
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if ang<1 or abs(ang-180)<1:
                length += np.hypot(x2-x1, y2-y1)
                mids_h.append([(y1+y2)/2.])
        if mids_h:
            lbl = DBSCAN(eps=5, min_samples=1).fit(np.array(mids_h)).labels_
            raw = len([l for l in set(lbl) if l!=-1])
    total_edges = edges_y.sum()/255

    # vertical edges → layer_count
    sob_x = cv2.Sobel(gray, cv2.CV_64F, 1,0,KSOB)
    norm_x= np.uint8(np.abs(sob_x)/(np.abs(sob_x).max()+1e-6)*255)
    edges_x = cv2.Canny(norm_x, CANNY1, CANNY2)
    segs_v = cv2.HoughLinesP(edges_x,1,np.pi/180,100,minLineLength=30,maxLineGap=5)
    layer_count=0; mids_v=[]
    if segs_v is not None:
        for x1,y1,x2,y2 in segs_v[:,0]:
            ang = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
            if abs(ang-90)<5:
                mids_v.append([(x1+x2)/2.])
        if mids_v:
            lblv = DBSCAN(eps=10, min_samples=1).fit(np.array(mids_v)).labels_
            layer_count = len([l for l in set(lblv) if l!=-1])

    # text embed + PCA
    fn = os.path.basename(path)
    inp= clip_proc(text=[captions.get(fn,"")], return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_emb = clip_model.get_text_features(**inp)[0].cpu().numpy()
    txt_pca = pca_wrap.transform([txt_emb])[0] if path.startswith("ramco_images/wrap_images") else pca_nowrap.transform([txt_emb])[0]

    return np.concatenate([[raw, length, total_edges, layer_count], txt_pca])

def process(model, folder, label):
    imgs=[] 
    for ext in IMAGE_EXTS: imgs+=glob.glob(os.path.join(folder,ext))
    imgs=sorted(imgs)
    total=0
    print(f"\nProcessing {label} ({len(imgs)} images)…")
    for p in tqdm(imgs, desc=label):
        feat= extract_feat(p)
        pred= model.predict([feat])[0]
        total += pred
        print(f"  {os.path.basename(p):25s} → {int(round(pred))}")
    print(f"Subtotal {label}: {total}")
    return total

if __name__=="__main__":
    import argparse
    p= argparse.ArgumentParser(__doc__)
    p.add_argument("--wrap_dir",   required=True)
    p.add_argument("--unwrap_dir", required=True)
    args= p.parse_args()

    tw = process(model_wrap,   args.wrap_dir,   "Wrapped")
    tn = process(model_nowrap, args.unwrap_dir, "Unwrapped")
    print("\n" + "="*40)
    print(f"GRAND TOTAL: {tw+tn}")
    print("="*40)

