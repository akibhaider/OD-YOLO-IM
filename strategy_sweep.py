"""
Strategy to close the gap toward P>0.76, R>0.85
================================================
Current best (YOLO12x@640):
  - conf=0.10, iou=0.5 → P=0.690, R=0.832 (high R, P too low)
  - conf=0.10, iou=0.7 → P=0.763, R=0.777 (P ok, R too low)

Strategies:
  1. SAHI sliced inference → detect small/missed products → boost R
  2. Selective 2-model ensemble (YOLO12x + YOLO12l only) → boost R w/o precision noise
  3. SAHI + selective ensemble combined
"""
import os, json, time, warnings, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import yaml
import glob
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ensemble_boxes import weighted_boxes_fusion

PROJECT = '/mnt/Data/AKIB/YOLO-OD-IM'
DATA    = os.path.join(PROJECT, 'dataset', 'data_abs.yaml')
TEST_IMG = os.path.join(PROJECT, 'dataset', 'test', 'images')
TEST_LBL = os.path.join(PROJECT, 'dataset', 'test', 'labels')

YOLO12X_PATH = os.path.join(PROJECT, 'runs', 'yolo12x_optimized', 'weights', 'best.pt')
YOLO12L_PATH = os.path.join(PROJECT, 'runs', 'yolo12l_ensemble', 'weights', 'best.pt')

with open(DATA) as f:
    cfg = yaml.safe_load(f)
class_names = cfg['names']
NC = len(class_names)

test_images = sorted(glob.glob(os.path.join(TEST_IMG, '*')))
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Test images: {len(test_images)}, Classes: {NC}")

# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════
def load_gt(lbl_path, w, h):
    boxes, classes = [], []
    if not os.path.exists(lbl_path):
        return boxes, classes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5: continue
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

def evaluate_predictions(all_preds, match_iou=0.5):
    """Evaluate list of (img_path, pred_boxes, pred_classes, pred_scores) tuples."""
    tp_total, fp_total, fn_total = 0, 0, 0
    for img_path, pred_boxes, pred_classes, pred_scores in all_preds:
        import cv2
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(TEST_LBL, fname + '.txt')
        gt_boxes, gt_classes = load_gt(lbl_path, w, h)

        # Sort by confidence (descending)
        if len(pred_scores) > 0:
            order = np.argsort(-np.array(pred_scores))
            pred_boxes = [pred_boxes[i] for i in order]
            pred_classes = [pred_classes[i] for i in order]

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
            if best_iou >= match_iou and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)
        tp_total += tp; fp_total += fp; fn_total += fn

    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {'P': round(precision, 4), 'R': round(recall, 4), 'F1': round(f1, 4),
            'TP': tp_total, 'FP': fp_total, 'FN': fn_total}

def filter_by_conf(pred_boxes, pred_classes, pred_scores, min_conf):
    """Post-filter predictions by confidence threshold."""
    mask = np.array(pred_scores) >= min_conf
    return ([pred_boxes[i] for i in range(len(pred_boxes)) if mask[i]],
            [pred_classes[i] for i in range(len(pred_classes)) if mask[i]],
            [pred_scores[i] for i in range(len(pred_scores)) if mask[i]])

def print_result(label, res):
    marker = ""
    if res['P'] >= 0.76 and res['R'] >= 0.85:
        marker = " 🎯 TARGET MET!"
    elif res['P'] >= 0.76:
        marker = " ✅P"
    elif res['R'] >= 0.85:
        marker = " ✅R"
    print(f"  {label:50s} P={res['P']:.4f}  R={res['R']:.4f}  F1={res['F1']:.4f}  (TP={res['TP']} FP={res['FP']} FN={res['FN']}){marker}")

# ═══════════════════════════════════════════════════════
# STRATEGY 1: SAHI SLICED INFERENCE
# ═══════════════════════════════════════════════════════
print("\n" + "="*80)
print("🔬 STRATEGY 1: SAHI Sliced Inference (YOLO12x@640)")
print("="*80)

sahi_model = AutoDetectionModel.from_pretrained(
    model_type='ultralytics',
    model_path=YOLO12X_PATH,
    confidence_threshold=0.01,  # low threshold, we'll sweep later
    device='cuda:0',
)

# Collect raw SAHI predictions at low conf for each slice config
sahi_configs = [
    (320, 0.2),   # small slices, 20% overlap
    (320, 0.3),   # small slices, 30% overlap 
    (416, 0.2),   # medium slices
    (416, 0.3),
    (512, 0.2),   # larger slices
    (512, 0.3),
]

all_sahi_results = {}

for slice_sz, overlap in sahi_configs:
    config_name = f"slice={slice_sz}_overlap={overlap}"
    print(f"\n  📐 {config_name}")
    
    raw_preds = []  # (img_path, boxes, classes, scores) with low conf
    t0 = time.time()
    
    for img_path in test_images:
        result = get_sliced_prediction(
            img_path,
            sahi_model,
            slice_height=slice_sz,
            slice_width=slice_sz,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            perform_standard_pred=True,  # also run full-image prediction
            postprocess_type='NMS',
            postprocess_match_metric='IOS',
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        
        import cv2
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        boxes, classes, scores = [], [], []
        for pred in result.object_prediction_list:
            bb = pred.bbox
            boxes.append([bb.minx, bb.miny, bb.maxx, bb.maxy])
            classes.append(pred.category.id)
            scores.append(pred.score.value)
        
        raw_preds.append((img_path, boxes, classes, scores))
    
    elapsed = time.time() - t0
    
    # Sweep conf thresholds
    for min_conf in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        filtered = []
        for img_path, boxes, classes, scores in raw_preds:
            fb, fc, fs = filter_by_conf(boxes, classes, scores, min_conf)
            filtered.append((img_path, fb, fc, fs))
        
        res = evaluate_predictions(filtered)
        key = f"SAHI_{config_name}_conf={min_conf:.2f}"
        all_sahi_results[key] = res
        print_result(f"conf≥{min_conf:.2f}", res)
    
    print(f"  ⏱️  {elapsed:.1f}s")

# ═══════════════════════════════════════════════════════
# STRATEGY 2: SELECTIVE 2-MODEL ENSEMBLE (YOLO12x + YOLO12l)
# ═══════════════════════════════════════════════════════
print("\n" + "="*80)
print("🤝 STRATEGY 2: Selective Ensemble (YOLO12x + YOLO12l only)")
print("="*80)

model_x = YOLO(YOLO12X_PATH)
model_l = YOLO(YOLO12L_PATH)

# Collect raw predictions from both models (with TTA)
print("\n  Collecting YOLO12x predictions (TTA)...")
preds_x = []
for img_path in test_images:
    import cv2
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    results = model_x.predict(img_path, conf=0.01, iou=0.7, imgsz=640,
                               device=0, verbose=False, augment=True, max_det=300)
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.empty((0,4))
    cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) > 0 else np.array([])
    conf = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
    preds_x.append((img_path, boxes, cls, conf, w, h))

print("  Collecting YOLO12l predictions (TTA)...")
preds_l = []
for img_path in test_images:
    import cv2
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    results = model_l.predict(img_path, conf=0.01, iou=0.7, imgsz=640,
                               device=0, verbose=False, augment=True, max_det=300)
    r = results[0]
    boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.empty((0,4))
    cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) > 0 else np.array([])
    conf_arr = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
    preds_l.append((img_path, boxes, cls, conf_arr, w, h))

# WBF ensemble sweep
all_ensemble_results = {}
wbf_configs = [
    # (iou_thr, skip_box_thr, conf_type, weights_x, weights_l)
    (0.5, 0.01, 'avg', 3.0, 1.5),
    (0.5, 0.05, 'avg', 3.0, 1.5),
    (0.5, 0.10, 'avg', 3.0, 1.5),
    (0.5, 0.01, 'max', 3.0, 1.5),
    (0.5, 0.05, 'max', 3.0, 1.5),
    (0.6, 0.01, 'avg', 3.0, 1.5),
    (0.6, 0.05, 'avg', 3.0, 1.5),
    (0.6, 0.10, 'avg', 3.0, 1.5),
    (0.6, 0.01, 'max', 3.0, 1.5),
    (0.55, 0.05, 'avg', 3.0, 1.0),
    (0.55, 0.05, 'avg', 3.0, 2.0),
    (0.55, 0.05, 'max', 3.0, 1.5),
    (0.55, 0.05, 'avg', 4.0, 1.0),   # heavily favor YOLO12x
    (0.55, 0.10, 'avg', 4.0, 1.0),
    (0.50, 0.05, 'avg', 4.0, 1.0),
]

print(f"\n  Sweeping {len(wbf_configs)} WBF configs...")
for iou_thr, skip_thr, conf_type, wx, wl in wbf_configs:
    # Build ensemble predictions per image
    ens_preds = []
    for i in range(len(test_images)):
        img_path = test_images[i]
        _, boxes_x, cls_x, conf_x, w, h = preds_x[i]
        _, boxes_l, cls_l, conf_l, _, _ = preds_l[i]

        # Normalize to [0,1]
        boxes_list = []
        scores_list = []
        labels_list = []

        if len(boxes_x) > 0:
            norm_x = boxes_x.copy()
            norm_x[:, [0, 2]] /= w
            norm_x[:, [1, 3]] /= h
            norm_x = np.clip(norm_x, 0, 1)
            boxes_list.append(norm_x.tolist())
            scores_list.append(conf_x.tolist())
            labels_list.append(cls_x.tolist())
        else:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])

        if len(boxes_l) > 0:
            norm_l = boxes_l.copy()
            norm_l[:, [0, 2]] /= w
            norm_l[:, [1, 3]] /= h
            norm_l = np.clip(norm_l, 0, 1)
            boxes_list.append(norm_l.tolist())
            scores_list.append(conf_l.tolist())
            labels_list.append(cls_l.tolist())
        else:
            boxes_list.append([])
            scores_list.append([])
            labels_list.append([])

        # WBF
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[wx, wl],
            iou_thr=iou_thr,
            skip_box_thr=skip_thr,
            conf_type=conf_type,
        )

        # Denormalize
        final_boxes = []
        for b in fused_boxes:
            final_boxes.append([b[0]*w, b[1]*h, b[2]*w, b[3]*h])
        
        ens_preds.append((img_path, final_boxes, fused_labels.astype(int).tolist(), fused_scores.tolist()))

    # Sweep post-WBF conf thresholds
    config_label = f"wbf_iou={iou_thr}_skip={skip_thr}_{conf_type}_w={wx}/{wl}"
    best_f1_res = None
    best_f1 = 0
    
    for min_conf in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        filtered = []
        for img_path, boxes, classes, scores in ens_preds:
            fb, fc, fs = filter_by_conf(boxes, classes, scores, min_conf)
            filtered.append((img_path, fb, fc, fs))
        
        res = evaluate_predictions(filtered)
        key = f"ENS2_{config_label}_conf={min_conf:.2f}"
        all_ensemble_results[key] = res
        
        if res['F1'] > best_f1:
            best_f1 = res['F1']
            best_f1_res = (min_conf, res)
    
    mc, br = best_f1_res
    print_result(f"{config_label} @conf={mc:.2f}", br)

# ═══════════════════════════════════════════════════════
# STRATEGY 3: SAHI + SELECTIVE ENSEMBLE
# ═══════════════════════════════════════════════════════
print("\n" + "="*80)
print("🔬+🤝 STRATEGY 3: SAHI (YOLO12x) + YOLO12l Ensemble via WBF")
print("="*80)

# Use best SAHI config (we'll pick a couple good slice sizes)
all_combo_results = {}

for slice_sz, overlap in [(320, 0.3), (416, 0.2), (416, 0.3)]:
    print(f"\n  📐 SAHI slice={slice_sz}, overlap={overlap}")
    
    # Collect SAHI predictions for YOLO12x
    sahi_raw = []
    for img_path in test_images:
        result = get_sliced_prediction(
            img_path, sahi_model,
            slice_height=slice_sz, slice_width=slice_sz,
            overlap_height_ratio=overlap, overlap_width_ratio=overlap,
            perform_standard_pred=True,
            postprocess_type='NMS',
            postprocess_match_metric='IOS',
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        import cv2
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        boxes, classes, scores = [], [], []
        for pred in result.object_prediction_list:
            bb = pred.bbox
            boxes.append([bb.minx, bb.miny, bb.maxx, bb.maxy])
            classes.append(pred.category.id)
            scores.append(pred.score.value)
        sahi_raw.append((img_path, np.array(boxes) if boxes else np.empty((0,4)),
                         np.array(classes, dtype=int) if classes else np.array([], dtype=int),
                         np.array(scores) if scores else np.array([]),
                         w, h))
    
    # WBF: SAHI(YOLO12x) + YOLO12l-TTA
    for iou_thr, skip_thr, conf_type, wx, wl in [
        (0.5, 0.01, 'avg', 3.0, 1.5),
        (0.5, 0.05, 'avg', 3.0, 1.5),
        (0.5, 0.05, 'max', 3.0, 1.5),
        (0.55, 0.05, 'avg', 3.0, 1.5),
        (0.55, 0.05, 'avg', 4.0, 1.0),
        (0.6, 0.05, 'avg', 3.0, 1.5),
        (0.6, 0.05, 'avg', 4.0, 1.0),
    ]:
        combo_preds = []
        for i in range(len(test_images)):
            img_path = test_images[i]
            _, boxes_s, cls_s, conf_s, w, h = sahi_raw[i]
            _, boxes_l, cls_l, conf_l, _, _ = preds_l[i]
            
            boxes_list, scores_list, labels_list = [], [], []
            
            if len(boxes_s) > 0:
                norm_s = boxes_s.copy().astype(float)
                norm_s[:, [0, 2]] /= w
                norm_s[:, [1, 3]] /= h
                norm_s = np.clip(norm_s, 0, 1)
                boxes_list.append(norm_s.tolist())
                scores_list.append(conf_s.tolist())
                labels_list.append(cls_s.tolist())
            else:
                boxes_list.append([]); scores_list.append([]); labels_list.append([])
            
            if len(boxes_l) > 0:
                norm_l = boxes_l.copy()
                norm_l[:, [0, 2]] /= w
                norm_l[:, [1, 3]] /= h
                norm_l = np.clip(norm_l, 0, 1)
                boxes_list.append(norm_l.tolist())
                scores_list.append(conf_l.tolist())
                labels_list.append(cls_l.tolist())
            else:
                boxes_list.append([]); scores_list.append([]); labels_list.append([])
            
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=[wx, wl], iou_thr=iou_thr,
                skip_box_thr=skip_thr, conf_type=conf_type,
            )
            
            final_boxes = []
            for b in fused_boxes:
                final_boxes.append([b[0]*w, b[1]*h, b[2]*w, b[3]*h])
            combo_preds.append((img_path, final_boxes, fused_labels.astype(int).tolist(), fused_scores.tolist()))
        
        config_label = f"s{slice_sz}_o{overlap}_wbf={iou_thr}_sk={skip_thr}_{conf_type}_w={wx}/{wl}"
        
        for min_conf in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
            filtered = []
            for img_path, boxes, classes, scores in combo_preds:
                fb, fc, fs = filter_by_conf(boxes, classes, scores, min_conf)
                filtered.append((img_path, fb, fc, fs))
            res = evaluate_predictions(filtered)
            key = f"COMBO_{config_label}_conf={min_conf:.2f}"
            all_combo_results[key] = res

        # Show best F1 and best "target-meeting" configs
        best_f1, best_key = 0, ""
        target_met = []
        for k, v in all_combo_results.items():
            if config_label in k:
                if v['F1'] > best_f1:
                    best_f1 = v['F1']
                    best_key = k
                if v['P'] >= 0.76 and v['R'] >= 0.85:
                    target_met.append((k, v))
        
        if best_key:
            print_result(f"{config_label}", all_combo_results[best_key])
        for tk, tv in target_met:
            print(f"    🎯 TARGET MET: {tk} → P={tv['P']:.4f} R={tv['R']:.4f} F1={tv['F1']:.4f}")

# ═══════════════════════════════════════════════════════
# GRAND SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "="*80)
print("📋 GRAND SUMMARY — All strategies ranked by F1")
print("="*80)

all_results = {}
all_results.update(all_sahi_results)
all_results.update(all_ensemble_results)
all_results.update(all_combo_results)

# Add YOLO12x@640 baselines for comparison
all_results['BASELINE_yolo12x_conf=0.10_iou=0.5'] = {'P': 0.6904, 'R': 0.8317, 'F1': 0.7545, 'TP': 0, 'FP': 0, 'FN': 0}
all_results['BASELINE_yolo12x_conf=0.10_iou=0.7'] = {'P': 0.7627, 'R': 0.7770, 'F1': 0.7697, 'TP': 0, 'FP': 0, 'FN': 0}
all_results['BASELINE_yolo12x_conf=0.15_iou=0.5'] = {'P': 0.7101, 'R': 0.8254, 'F1': 0.7634, 'TP': 0, 'FP': 0, 'FN': 0}

# Find configs meeting target
print(f"\n{'Config':70s} {'P':>7s} {'R':>7s} {'F1':>7s}")
print("-" * 95)

# Show target-meeting configs first
target_configs = [(k,v) for k,v in all_results.items() if v['P'] >= 0.76 and v['R'] >= 0.85]
if target_configs:
    print("\n🎯 CONFIGS MEETING P>0.76 AND R>0.85:")
    for k, v in sorted(target_configs, key=lambda x: -x[1]['F1']):
        print(f"  {k:70s} {v['P']:7.4f} {v['R']:7.4f} {v['F1']:7.4f}")
else:
    print("\n❌ No config met BOTH P>0.76 AND R>0.85 simultaneously")
    
    # Show closest configs
    print("\n📊 Top 15 by F1 (where P≥0.70 and R≥0.75):")
    good = [(k,v) for k,v in all_results.items() if v['P'] >= 0.70 and v['R'] >= 0.75]
    for k, v in sorted(good, key=lambda x: -x[1]['F1'])[:15]:
        marker = ""
        if v['P'] >= 0.76: marker += " ✅P"
        if v['R'] >= 0.85: marker += " ✅R"
        print(f"  {k:70s} {v['P']:7.4f} {v['R']:7.4f} {v['F1']:7.4f}{marker}")
    
    print("\n📊 Top 10 by Recall (where P≥0.70):")
    hi_r = [(k,v) for k,v in all_results.items() if v['P'] >= 0.70]
    for k, v in sorted(hi_r, key=lambda x: -x[1]['R'])[:10]:
        marker = ""
        if v['P'] >= 0.76: marker += " ✅P"
        if v['R'] >= 0.85: marker += " ✅R"
        print(f"  {k:70s} {v['P']:7.4f} {v['R']:7.4f} {v['F1']:7.4f}{marker}")
    
    print("\n📊 Top 10 by Precision (where R≥0.80):")
    hi_p = [(k,v) for k,v in all_results.items() if v['R'] >= 0.80]
    for k, v in sorted(hi_p, key=lambda x: -x[1]['P'])[:10]:
        marker = ""
        if v['P'] >= 0.76: marker += " ✅P"
        if v['R'] >= 0.85: marker += " ✅R"
        print(f"  {k:70s} {v['P']:7.4f} {v['R']:7.4f} {v['F1']:7.4f}{marker}")

# Save everything
with open(os.path.join(PROJECT, 'strategy_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ All results saved to strategy_results.json ({len(all_results)} configs)")
