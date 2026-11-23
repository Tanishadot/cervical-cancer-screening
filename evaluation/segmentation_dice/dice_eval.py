"""Dice / IoU of the repo's nucleus segmentation (backend/feature_extraction.py,
branch feature/performance-optimization) against Herlev ground-truth nucleus masks.

Herlev mask colours: (0,0,255) nucleus, (0,0,128) cytoplasm, (255,0,0) background,
(128,128,128) unlabelled.
Usage: copy backend/feature_extraction.py (branch feature/performance-optimization) next to this
file, then: python dice_eval.py <Dataset-Herlev dir> <out.json>
"""
import glob, json, os, sys, logging
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import CytologyFeatureExtractor
logging.disable(logging.CRITICAL)

root, out = sys.argv[1], sys.argv[2]
ext = CytologyFeatureExtractor()
NUC = (0, 0, 255)


def scores(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    ps, gs = pred.sum(), gt.sum()
    dice = 2 * inter / (ps + gs) if (ps + gs) else 1.0
    union = np.logical_or(pred, gt).sum()
    iou = inter / union if union else 1.0
    prec = inter / ps if ps else 0.0
    rec = inter / gs if gs else 1.0
    return dice, iou, prec, rec, int(inter), int(ps), int(gs)


rows = []
for mpath in sorted(glob.glob(os.path.join(root, "*", "*-d.bmp"))):
    ipath = mpath[:-6] + ".BMP"
    if not os.path.exists(ipath):
        ipath = mpath[:-6] + "_2.BMP"   # 4 carcinoma_in_situ images carry a _2 suffix
    if not os.path.exists(ipath):
        continue
    img_check = Image.open(ipath)
    if img_check.size != Image.open(mpath).size:
        continue
    img = np.array(Image.open(ipath).convert("RGB"))
    gt_full = (np.array(Image.open(mpath).convert("RGB")) == NUC).all(2)
    cls = os.path.basename(os.path.dirname(mpath))
    r = {"image": os.path.basename(ipath), "class": cls}
    # (a) native resolution
    pred = ext._segment_regions(img)[0] > 0
    r["native"] = scores(pred, gt_full)
    # (b) 224x224, as the dashboard feeds images (LANCZOS resize), GT resized nearest
    img224 = np.array(Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS))
    gt224 = np.array(Image.fromarray(gt_full.astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST)) > 0
    pred224 = ext._segment_regions(img224)[0] > 0
    r["r224"] = scores(pred224, gt224)
    rows.append(r)

json.dump(rows, open(out, "w"))
print("evaluated", len(rows), "images")
