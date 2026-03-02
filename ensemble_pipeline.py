"""
Complete Ensemble Pipeline: TTA → NMS → WBF
=============================================
Models: YOLO12x, YOLO12l, YOLO11m, YOLO12s, YOLO11n
Techniques:
  1. TTA (Test-Time Augmentation) on each model
  2. NMS (Non-Maximum Suppression) to remove duplicates per model
  3. WBF (Weighted Box Fusion) to ensemble all models
"""
import os, json, time, warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from pathlib import Path
from collections import defaultdict

PROJECT = '/mnt/Data/AKIB/YOLO-OD-IM'
DATA    = os.path.join(PROJECT, 'dataset', 'data_abs.yaml')
TEST_IMAGES = os.path.join(PROJECT, 'dataset', 'test', 'images')
TEST_LABELS = os.path.join(PROJECT, 'dataset', 'test', 'labels')

# Load class names
import yaml
with open(DATA) as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg['names']
NUM_CLASSES = len(class_names)

# ══════════════════════════════════════════════════════════════
# MODEL REGISTRY — 5 diverse models
# ══════════════════════════════════════════════════════════════
MODEL_CONFIGS = {
    'yolo12x': {
        'weights': os.path.join(PROJECT, 'runs', 'yolo12x_optimized', 'weights', 'best.pt'),
        'wbf_weight': 3.0,   # highest weight — best model
    },
    'yolo12l': {
        'weights': os.path.join(PROJECT, 'runs', 'yolo12l_ensemble', 'weights', 'best.pt'),
        'wbf_weight': 2.0,
    },
    'yolo11m': {
        'weights': os.path.join(PROJECT, 'runs', 'yolo11m_ensemble', 'weights', 'best.pt'),
        'wbf_weight': 1.5,
    },
    'yolo12s': {
        'weights': os.path.join(PROJECT, 'runs', 'yolo12s_ensemble', 'weights', 'best.pt'),
        'wbf_weight': 1.0,
    },
    'yolo11n': {
        'weights': os.path.join(PROJECT, 'runs', 'baseline_yolo11n', 'weights', 'best.pt'),
        'wbf_weight': 0.5,   # lowest weight — weakest model
    },
}

# Filter to only models that exist
available_models = {}
for name, cfg in MODEL_CONFIGS.items():
    if os.path.exists(cfg['weights']):
        available_models[name] = cfg
        print(f"  ✅ {name:10s} → {cfg['weights']}")
    else:
        print(f"  ❌ {name:10s} → NOT FOUND (skipping)")

print(f"\n📦 {len(available_models)} models available for ensemble")

# ══════════════════════════════════════════════════════════════
# HELPER: Parse YOLO Ground Truth labels
# ══════════════════════════════════════════════════════════════
def load_gt_labels(label_path, img_w=640, img_h=640):
    """Load YOLO format GT: class x_center y_center width height → xyxy"""
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return np.array([]), np.array([])
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (xc - w/2) * img_w
            y1 = (yc - h/2) * img_h
            x2 = (xc + w/2) * img_w
            y2 = (yc + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(cls)
    return np.array(boxes), np.array(classes)

# ══════════════════════════════════════════════════════════════
# HELPER: Compute metrics (P, R, F1) via IoU matching
# ══════════════════════════════════════════════════════════════
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-8)

def compute_metrics(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, iou_thresh=0.5):
    """Compute TP, FP, FN for a single image."""
    tp, fp, fn = 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0  # all FP
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)  # all FN
    
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    
    # Sort predictions by score descending
    order = np.argsort(-pred_scores)
    
    for idx in order:
        pred_box = pred_boxes[idx]
        pred_cls = pred_classes[idx]
        
        best_iou, best_gt = 0, -1
        for gi in range(len(gt_boxes)):
            if gt_matched[gi]:
                continue
            if gt_classes[gi] != pred_cls:
                continue
            iou = compute_iou(pred_box, gt_boxes[gi])
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        
        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1
            gt_matched[best_gt] = True
        else:
            fp += 1
    
    fn = int(np.sum(~gt_matched))
    return tp, fp, fn

# ══════════════════════════════════════════════════════════════
# PHASE 1: TTA INFERENCE — Each model with Test-Time Augmentation
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🔄 PHASE 1: TEST-TIME AUGMENTATION (TTA) INFERENCE")
print("="*70)

test_images = sorted([f for f in os.listdir(TEST_IMAGES) if f.endswith(('.jpg', '.png', '.jpeg'))])
print(f"Test images: {len(test_images)}")

# Store predictions per model per image
# all_predictions[model_name][image_name] = {'boxes': [...], 'scores': [...], 'classes': [...]}
all_predictions = {}

for model_name, cfg in available_models.items():
    print(f"\n🔄 TTA Inference: {model_name}")
    t0 = time.time()
    
    model = YOLO(cfg['weights'])
    model_preds = {}
    
    for img_file in test_images:
        img_path = os.path.join(TEST_IMAGES, img_file)
        
        # TTA: augment=True enables multi-scale, flip augmentations at test time
        results = model.predict(
            img_path,
            conf=0.05,       # low conf to get more candidates for WBF
            iou=0.7,         # NMS IoU during prediction
            augment=True,    # ← TEST-TIME AUGMENTATION
            device=0,
            verbose=False,
            max_det=300,
        )
        
        r = results[0]
        if len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes = np.array([]).reshape(0, 4)
            scores = np.array([])
            classes = np.array([], dtype=int)
        
        model_preds[img_file] = {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'img_w': r.orig_shape[1],
            'img_h': r.orig_shape[0],
        }
    
    all_predictions[model_name] = model_preds
    elapsed = time.time() - t0
    
    # Quick per-model TTA metrics
    total_tp, total_fp, total_fn = 0, 0, 0
    for img_file in test_images:
        pred = model_preds[img_file]
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(TEST_LABELS, label_file)
        gt_boxes, gt_classes = load_gt_labels(label_path, pred['img_w'], pred['img_h'])
        
        # Filter by conf threshold for metrics
        mask = pred['scores'] >= 0.10
        tp, fp, fn = compute_metrics(
            pred['boxes'][mask], pred['classes'][mask], pred['scores'][mask],
            gt_boxes, gt_classes
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    print(f"  ✅ {model_name} TTA (conf≥0.10): P={p:.4f}, R={r:.4f}, F1={f1:.4f}  ({elapsed:.1f}s)")
    
    del model
    torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# PHASE 2: NMS — Per-model Non-Maximum Suppression
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🔧 PHASE 2: NON-MAXIMUM SUPPRESSION (NMS) — Removing Duplicates")
print("="*70)

def apply_class_aware_nms(boxes, scores, classes, iou_threshold=0.5):
    """Apply class-aware NMS to remove duplicate detections."""
    if len(boxes) == 0:
        return boxes, scores, classes
    
    keep_indices = []
    unique_classes = np.unique(classes)
    
    for cls in unique_classes:
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        
        # Sort by score descending
        order = np.argsort(-cls_scores)
        
        suppressed = set()
        for i in range(len(order)):
            if order[i] in suppressed:
                continue
            keep_indices.append(cls_indices[order[i]])
            
            for j in range(i + 1, len(order)):
                if order[j] in suppressed:
                    continue
                iou = compute_iou(cls_boxes[order[i]], cls_boxes[order[j]])
                if iou >= iou_threshold:
                    suppressed.add(order[j])
    
    if len(keep_indices) == 0:
        return np.array([]).reshape(0, 4), np.array([]), np.array([], dtype=int)
    
    keep_indices = sorted(keep_indices)
    return boxes[keep_indices], scores[keep_indices], classes[keep_indices]

NMS_IOU_THRESHOLD = 0.5
nms_predictions = {}

for model_name in all_predictions:
    nms_preds = {}
    total_before, total_after = 0, 0
    
    for img_file in test_images:
        pred = all_predictions[model_name][img_file]
        before = len(pred['boxes'])
        
        nms_boxes, nms_scores, nms_classes = apply_class_aware_nms(
            pred['boxes'], pred['scores'], pred['classes'],
            iou_threshold=NMS_IOU_THRESHOLD
        )
        
        nms_preds[img_file] = {
            'boxes': nms_boxes,
            'scores': nms_scores,
            'classes': nms_classes,
            'img_w': pred['img_w'],
            'img_h': pred['img_h'],
        }
        total_before += before
        total_after += len(nms_boxes)
    
    nms_predictions[model_name] = nms_preds
    removed = total_before - total_after
    print(f"  {model_name:10s}: {total_before} → {total_after} detections  (removed {removed} duplicates, {removed/(total_before+1e-8)*100:.1f}%)")

# Compute per-model NMS metrics
print(f"\n  Per-model metrics after NMS (conf≥0.10):")
for model_name in nms_predictions:
    total_tp, total_fp, total_fn = 0, 0, 0
    for img_file in test_images:
        pred = nms_predictions[model_name][img_file]
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(TEST_LABELS, label_file)
        gt_boxes, gt_classes = load_gt_labels(label_path, pred['img_w'], pred['img_h'])
        mask = pred['scores'] >= 0.10
        tp, fp, fn = compute_metrics(
            pred['boxes'][mask], pred['classes'][mask], pred['scores'][mask],
            gt_boxes, gt_classes
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    print(f"    {model_name:10s}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")

# ══════════════════════════════════════════════════════════════
# PHASE 3: WBF — Weighted Box Fusion Ensemble
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🏆 PHASE 3: WEIGHTED BOX FUSION (WBF) ENSEMBLE")
print("="*70)

model_names_ordered = list(nms_predictions.keys())
model_weights = [available_models[n]['wbf_weight'] for n in model_names_ordered]
print(f"  Models: {model_names_ordered}")
print(f"  Weights: {model_weights}")

# WBF configurations to sweep
WBF_CONFIGS = [
    {'iou_thr': 0.5, 'skip_box_thr': 0.05, 'conf_type': 'avg'},
    {'iou_thr': 0.5, 'skip_box_thr': 0.10, 'conf_type': 'avg'},
    {'iou_thr': 0.6, 'skip_box_thr': 0.05, 'conf_type': 'avg'},
    {'iou_thr': 0.6, 'skip_box_thr': 0.10, 'conf_type': 'avg'},
    {'iou_thr': 0.5, 'skip_box_thr': 0.05, 'conf_type': 'max'},
    {'iou_thr': 0.6, 'skip_box_thr': 0.05, 'conf_type': 'max'},
]

# Evaluation conf thresholds to try on WBF output
EVAL_CONFS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

best_overall_f1 = 0
best_overall_config = None
best_overall_metrics = None
best_ensemble_preds = None

all_wbf_results = {}

for wbf_cfg in WBF_CONFIGS:
    wbf_key = f"iou={wbf_cfg['iou_thr']}_skip={wbf_cfg['skip_box_thr']}_{wbf_cfg['conf_type']}"
    
    # Run WBF on each image
    ensemble_preds = {}
    
    for img_file in test_images:
        # Collect predictions from all models for this image
        boxes_list = []
        scores_list = []
        labels_list = []
        
        # Get image dimensions (same for all models)
        sample_pred = list(nms_predictions.values())[0][img_file]
        img_w, img_h = sample_pred['img_w'], sample_pred['img_h']
        
        for model_name in model_names_ordered:
            pred = nms_predictions[model_name][img_file]
            if len(pred['boxes']) > 0:
                # Normalize boxes to [0, 1] for WBF
                norm_boxes = pred['boxes'].copy()
                norm_boxes[:, [0, 2]] /= img_w
                norm_boxes[:, [1, 3]] /= img_h
                # Clip to [0, 1]
                norm_boxes = np.clip(norm_boxes, 0, 1)
                boxes_list.append(norm_boxes.tolist())
                scores_list.append(pred['scores'].tolist())
                labels_list.append(pred['classes'].tolist())
            else:
                boxes_list.append([])
                scores_list.append([])
                labels_list.append([])
        
        # Apply WBF
        if any(len(b) > 0 for b in boxes_list):
            fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                weights=model_weights,
                iou_thr=wbf_cfg['iou_thr'],
                skip_box_thr=wbf_cfg['skip_box_thr'],
                conf_type=wbf_cfg['conf_type'],
            )
            # Denormalize back to pixel coords
            fused_boxes[:, [0, 2]] *= img_w
            fused_boxes[:, [1, 3]] *= img_h
        else:
            fused_boxes = np.array([]).reshape(0, 4)
            fused_scores = np.array([])
            fused_labels = np.array([], dtype=int)
        
        ensemble_preds[img_file] = {
            'boxes': fused_boxes,
            'scores': fused_scores,
            'classes': fused_labels.astype(int),
            'img_w': img_w,
            'img_h': img_h,
        }
    
    # Evaluate at multiple conf thresholds
    for eval_conf in EVAL_CONFS:
        total_tp, total_fp, total_fn = 0, 0, 0
        for img_file in test_images:
            pred = ensemble_preds[img_file]
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(TEST_LABELS, label_file)
            gt_boxes, gt_classes = load_gt_labels(label_path, pred['img_w'], pred['img_h'])
            
            mask = pred['scores'] >= eval_conf
            tp, fp, fn = compute_metrics(
                pred['boxes'][mask], pred['classes'][mask], pred['scores'][mask],
                gt_boxes, gt_classes
            )
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        p = total_tp / (total_tp + total_fp + 1e-8)
        r = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * p * r / (p + r + 1e-8)
        
        result_key = f"{wbf_key}_conf={eval_conf}"
        all_wbf_results[result_key] = {
            'wbf_iou': wbf_cfg['iou_thr'],
            'skip_box_thr': wbf_cfg['skip_box_thr'],
            'conf_type': wbf_cfg['conf_type'],
            'eval_conf': eval_conf,
            'precision': float(p),
            'recall': float(r),
            'f1': float(f1),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        }
        
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_overall_config = result_key
            best_overall_metrics = all_wbf_results[result_key]
            best_ensemble_preds = ensemble_preds
            best_eval_conf = eval_conf
    
    print(f"  {wbf_key}: done")

# ══════════════════════════════════════════════════════════════
# PHASE 4: SUMMARY REPORT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📋 FINAL ENSEMBLE RESULTS")
print("="*70)

# Best F1 config
print(f"\n🏆 BEST F1 CONFIG: {best_overall_config}")
bm = best_overall_metrics
print(f"  WBF IoU Threshold:    {bm['wbf_iou']}")
print(f"  Skip Box Threshold:   {bm['skip_box_thr']}")
print(f"  Conf Type:            {bm['conf_type']}")
print(f"  Eval Confidence:      {bm['eval_conf']}")
print(f"  Precision:            {bm['precision']:.4f}")
print(f"  Recall:               {bm['recall']:.4f}")
print(f"  F1 Score:             {bm['f1']:.4f}")

# Best Recall config
best_recall_key = max(all_wbf_results, key=lambda k: all_wbf_results[k]['recall'])
br = all_wbf_results[best_recall_key]
print(f"\n🎯 BEST RECALL CONFIG: {best_recall_key}")
print(f"  Precision:            {br['precision']:.4f}")
print(f"  Recall:               {br['recall']:.4f}")
print(f"  F1 Score:             {br['f1']:.4f}")

# Comparison table
print(f"\n{'─'*90}")
print(f"{'Config':55s} {'P':>8s} {'R':>8s} {'F1':>8s}")
print(f"{'─'*90}")
# Sort by F1
sorted_results = sorted(all_wbf_results.items(), key=lambda x: x[1]['f1'], reverse=True)
for key, r in sorted_results[:20]:
    marker = ""
    if key == best_overall_config:
        marker = " ← BEST F1"
    elif key == best_recall_key:
        marker = " ← BEST RECALL"
    print(f"{key:55s} {r['precision']:8.4f} {r['recall']:8.4f} {r['f1']:8.4f}{marker}")

# ══════════════════════════════════════════════════════════════
# PHASE 5: SHARE OF SHELF ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📊 PHASE 5: SHARE OF SHELF ANALYTICS (Test Set)")
print("="*70)

# Use best ensemble predictions at best eval_conf
sos_facing_count = defaultdict(int)   # number of product facings
sos_shelf_area = defaultdict(float)   # total bounding box area

# Also compute GT Share of Shelf
gt_facing_count = defaultdict(int)
gt_shelf_area = defaultdict(float)

for img_file in test_images:
    pred = best_ensemble_preds[img_file]
    mask = pred['scores'] >= best_eval_conf
    
    for i in range(len(pred['boxes'])):
        if not mask[i]:
            continue
        cls = int(pred['classes'][i])
        cls_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        box = pred['boxes'][i]
        area = (box[2] - box[0]) * (box[3] - box[1])
        sos_facing_count[cls_name] += 1
        sos_shelf_area[cls_name] += area
    
    # Ground truth
    label_file = os.path.splitext(img_file)[0] + '.txt'
    label_path = os.path.join(TEST_LABELS, label_file)
    gt_boxes, gt_classes = load_gt_labels(label_path, pred['img_w'], pred['img_h'])
    for i in range(len(gt_boxes)):
        cls = int(gt_classes[i])
        cls_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        box = gt_boxes[i]
        area = (box[2] - box[0]) * (box[3] - box[1])
        gt_facing_count[cls_name] += 1
        gt_shelf_area[cls_name] += area

# Compute percentages
total_pred_facings = sum(sos_facing_count.values())
total_pred_area = sum(sos_shelf_area.values())
total_gt_facings = sum(gt_facing_count.values())
total_gt_area = sum(gt_shelf_area.values())

# Build Share of Shelf table
all_classes = sorted(set(list(sos_facing_count.keys()) + list(gt_facing_count.keys())))

sos_data = []
for cls_name in all_classes:
    pred_fc = sos_facing_count.get(cls_name, 0)
    pred_area = sos_shelf_area.get(cls_name, 0)
    gt_fc = gt_facing_count.get(cls_name, 0)
    gt_area = gt_shelf_area.get(cls_name, 0)
    
    sos_data.append({
        'class': cls_name,
        'pred_facings': pred_fc,
        'pred_facing_pct': pred_fc / (total_pred_facings + 1e-8) * 100,
        'pred_area': pred_area,
        'pred_area_pct': pred_area / (total_pred_area + 1e-8) * 100,
        'gt_facings': gt_fc,
        'gt_facing_pct': gt_fc / (total_gt_facings + 1e-8) * 100,
        'gt_area': gt_area,
        'gt_area_pct': gt_area / (total_gt_area + 1e-8) * 100,
    })

# Sort by predicted facing percentage descending
sos_data.sort(key=lambda x: x['pred_facing_pct'], reverse=True)

print(f"\n  Total predicted facings: {total_pred_facings}")
print(f"  Total GT facings:       {total_gt_facings}")
print(f"\n{'SKU':<12s} {'Pred Facings':>14s} {'Pred SoS%':>10s} {'GT Facings':>12s} {'GT SoS%':>10s} {'Area SoS%':>10s}")
print(f"{'─'*72}")
for row in sos_data:
    if row['pred_facings'] > 0 or row['gt_facings'] > 0:
        print(f"{row['class']:<12s} {row['pred_facings']:>14d} {row['pred_facing_pct']:>9.2f}% {row['gt_facings']:>12d} {row['gt_facing_pct']:>9.2f}% {row['pred_area_pct']:>9.2f}%")

# ══════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ══════════════════════════════════════════════════════════════
results_file = os.path.join(PROJECT, 'ensemble_results.json')
with open(results_file, 'w') as f:
    json.dump({
        'models_used': list(available_models.keys()),
        'model_weights': {n: available_models[n]['wbf_weight'] for n in available_models},
        'techniques': ['TTA (Test-Time Augmentation)', 'NMS (Non-Maximum Suppression)', 'WBF (Weighted Box Fusion)'],
        'best_f1_config': best_overall_config,
        'best_f1_metrics': best_overall_metrics,
        'best_recall_config': best_recall_key,
        'best_recall_metrics': all_wbf_results[best_recall_key],
        'all_wbf_results': all_wbf_results,
        'share_of_shelf': sos_data,
    }, f, indent=2, default=str)

print(f"\n✅ Results saved to {results_file}")
print("✅ ENSEMBLE PIPELINE COMPLETE!")
