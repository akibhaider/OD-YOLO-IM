"""Evaluate YOLO12x-1280 best.pt on test set at multiple conf/iou thresholds"""
import os, json
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch, glob, yaml, numpy as np
from ultralytics import YOLO

PROJECT = '/mnt/Data/AKIB/YOLO-OD-IM'
MODEL = os.path.join(PROJECT, 'runs', 'yolo12x_1280', 'weights', 'best.pt')
DATA = os.path.join(PROJECT, 'dataset', 'data_abs.yaml')
TEST_IMG = os.path.join(PROJECT, 'dataset', 'test', 'images')
TEST_LBL = os.path.join(PROJECT, 'dataset', 'test', 'labels')

with open(DATA) as f:
    cfg = yaml.safe_load(f)
class_names = cfg['names']
NC = len(class_names)

test_images = sorted(glob.glob(os.path.join(TEST_IMG, '*')))
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Model: {MODEL}")
print(f"Test images: {len(test_images)}, Classes: {NC}")

# Load model
model = YOLO(MODEL)

# --- Helpers ---
def load_gt(lbl_path, w, h):
    boxes, classes = [], []
    if not os.path.exists(lbl_path):
        return boxes, classes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
    return boxes, classes

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / (union + 1e-8)

def evaluate(model, images, conf, iou_thr, match_iou=0.5, imgsz=1280):
    tp_total, fp_total, fn_total = 0, 0, 0
    for img_path in images:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(TEST_LBL, fname + '.txt')
        
        results = model.predict(img_path, conf=conf, iou=iou_thr, imgsz=imgsz, 
                                device=0, verbose=False, max_det=300)
        r = results[0]
        h, w = r.orig_shape
        
        gt_boxes, gt_classes = load_gt(lbl_path, w, h)
        
        pred_boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.empty((0,4))
        pred_classes = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) > 0 else np.array([])
        
        matched_gt = set()
        tp, fp = 0, 0
        for pi in range(len(pred_boxes)):
            best_iou, best_gi = 0, -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt:
                    continue
                if pred_classes[pi] != gt_classes[gi]:
                    continue
                iou_val = compute_iou(pred_boxes[pi], gt_boxes[gi])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi
            if best_iou >= match_iou and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    
    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {'precision': precision, 'recall': recall, 'f1': f1, 
            'tp': tp_total, 'fp': fp_total, 'fn': fn_total}

# Skip official val (OOM at 1280 batch), use per-image evaluation instead

# Sweep configs
print("\n" + "="*70)
print("📊 Test Set Evaluation — YOLO12x-1280")
print("="*70)
configs = [
    (0.05, 0.5), (0.05, 0.7),
    (0.10, 0.5), (0.10, 0.7),
    (0.15, 0.5), (0.15, 0.7),
    (0.20, 0.5), (0.20, 0.7),
    (0.25, 0.5), (0.25, 0.7),
    (0.30, 0.5), (0.30, 0.7),
]

results_all = {}
print(f"\n{'Config':30s} {'P':>8s} {'R':>8s} {'F1':>8s} {'TP':>6s} {'FP':>6s} {'FN':>6s}")
print("-" * 76)

for conf, iou in configs:
    key = f"conf={conf:.2f}_iou={iou:.1f}"
    res = evaluate(model, test_images, conf, iou, imgsz=1280)
    results_all[key] = res
    marker = ""
    if res['precision'] >= 0.76 and res['recall'] >= 0.85:
        marker = " 🎯 TARGET MET!"
    elif res['precision'] >= 0.76:
        marker = " ✅P"
    elif res['recall'] >= 0.85:
        marker = " ✅R"
    print(f"{key:30s} {res['precision']:8.4f} {res['recall']:8.4f} {res['f1']:8.4f} {res['tp']:6d} {res['fp']:6d} {res['fn']:6d}{marker}")

# Also try TTA
print("\n" + "="*70)
print("📊 With TTA (Test-Time Augmentation)")
print("="*70)

tta_configs = [(0.10, 0.5), (0.10, 0.7), (0.15, 0.5), (0.15, 0.7), (0.20, 0.5)]
print(f"\n{'Config':30s} {'P':>8s} {'R':>8s} {'F1':>8s}")
print("-" * 60)

for conf, iou in tta_configs:
    key = f"TTA_conf={conf:.2f}_iou={iou:.1f}"
    tp_t, fp_t, fn_t = 0, 0, 0
    for img_path in test_images:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(TEST_LBL, fname + '.txt')
        results = model.predict(img_path, conf=conf, iou=iou, imgsz=1280,
                                device=0, verbose=False, augment=True, max_det=300)
        r = results[0]
        h, w = r.orig_shape
        gt_boxes, gt_classes = load_gt(lbl_path, w, h)
        pred_boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.empty((0,4))
        pred_classes = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) > 0 else np.array([])
        matched_gt = set()
        tp, fp = 0, 0
        for pi in range(len(pred_boxes)):
            best_iou, best_gi = 0, -1
            for gi in range(len(gt_boxes)):
                if gi in matched_gt or pred_classes[pi] != gt_classes[gi]:
                    continue
                iou_val = compute_iou(pred_boxes[pi], gt_boxes[gi])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_gi = gi
            if best_iou >= 0.5 and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1
        fn_t += len(gt_boxes) - len(matched_gt)
        tp_t += tp
        fp_t += fp
    p = tp_t / (tp_t + fp_t + 1e-8)
    r_val = tp_t / (tp_t + fn_t + 1e-8)
    f1 = 2*p*r_val / (p + r_val + 1e-8)
    results_all[key] = {'precision': p, 'recall': r_val, 'f1': f1}
    marker = ""
    if p >= 0.76 and r_val >= 0.85:
        marker = " 🎯 TARGET MET!"
    print(f"{key:30s} {p:8.4f} {r_val:8.4f} {f1:8.4f}{marker}")

# Save
with open(os.path.join(PROJECT, 'yolo12x_1280_results.json'), 'w') as f:
    json.dump(results_all, f, indent=2)
print("\n✅ Results saved to yolo12x_1280_results.json")
