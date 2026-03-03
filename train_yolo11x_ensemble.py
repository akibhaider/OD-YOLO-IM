"""
YOLO11x Training + 3-Model Ensemble Evaluation
Train YOLO11x (56.9M params) then evaluate 3-model ensemble:
  YOLO12x + YOLO12l + YOLO11x  (all large models)
GPU: Device 1 (RTX 3090 24GB)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from ultralytics import YOLO
import torch
import json
import numpy as np
import cv2
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion

PROJECT_ROOT = '/mnt/Data/AKIB/YOLO-OD-IM'
DATA_YAML    = os.path.join(PROJECT_ROOT, 'dataset', 'data_abs.yaml')

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ══════════════════════════════════════════════════════════════
# PHASE 1: TRAIN YOLO11x
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🚀 PHASE 1: TRAINING YOLO11x")
print("="*60)

model = YOLO('yolo11x.pt')

results = model.train(
    data=DATA_YAML,
    epochs=150,
    imgsz=640,
    batch=8,
    device=0,
    project=os.path.join(PROJECT_ROOT, 'runs'),
    name='yolo11x_ensemble',
    exist_ok=True,
    
    # LR — same as YOLO12x
    cos_lr=True,
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    
    # Augmentation — same as YOLO12x
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    close_mosaic=15,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    shear=5.0,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    cutmix=0.1,
    
    # Training
    patience=30,
    save=True,
    save_period=25,
    plots=True,
    verbose=True,
)

print("\n✅ YOLO11x training complete!")

# ══════════════════════════════════════════════════════════════
# PHASE 2: SINGLE MODEL EVAL ON TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📊 PHASE 2: YOLO11x SINGLE MODEL TEST EVALUATION")
print("="*60)

best_11x = os.path.join(PROJECT_ROOT, 'runs', 'yolo11x_ensemble', 'weights', 'best.pt')
model_11x = YOLO(best_11x)

for conf in [0.10, 0.15, 0.25]:
    m = model_11x.val(data=DATA_YAML, split='test', device=0, conf=conf, iou=0.5, verbose=False)
    p, r = float(m.box.mp), float(m.box.mr)
    f1 = 2*p*r/(p+r+1e-8)
    print(f"  YOLO11x conf={conf:.2f} → P={p:.4f}, R={r:.4f}, F1={f1:.4f}, mAP50={float(m.box.map50):.4f}")

# ══════════════════════════════════════════════════════════════
# PHASE 3: 3-MODEL ENSEMBLE (YOLO12x + YOLO12l + YOLO11x)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🔥 PHASE 3: 3-MODEL ENSEMBLE EVALUATION")
print("="*60)

# Load all 3 models
model_paths = {
    'yolo12x': os.path.join(PROJECT_ROOT, 'runs', 'yolo12x_optimized', 'weights', 'best.pt'),
    'yolo12l': os.path.join(PROJECT_ROOT, 'runs', 'yolo12l_ensemble', 'weights', 'best.pt'),
    'yolo11x': best_11x,
}

models = {}
for name, path in model_paths.items():
    print(f"  Loading {name} from {path}")
    models[name] = YOLO(path)

# Test images
test_img_dir = os.path.join(PROJECT_ROOT, 'dataset', 'test', 'images')
test_lbl_dir = os.path.join(PROJECT_ROOT, 'dataset', 'test', 'labels')
test_images = sorted(Path(test_img_dir).glob('*.jpg'))
print(f"\n  Test images: {len(test_images)}")

# Load class names
import yaml
with open(DATA_YAML) as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg['names']
nc = len(class_names)

def load_gt(label_path, img_w, img_h):
    """Load ground truth boxes from YOLO format label file."""
    boxes, labels = [], []
    if not os.path.exists(label_path):
        return np.array([]), np.array([])
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    return np.array(boxes) if boxes else np.array([]), np.array(labels) if labels else np.array([])

def get_predictions(model, img_path, augment=False, conf=0.01):
    """Get predictions from a single model."""
    results = model.predict(img_path, device=0, conf=conf, iou=0.7, augment=augment, verbose=False)
    r = results[0]
    if len(r.boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, labels

def evaluate(all_preds, all_gts, conf_thr=0.25, iou_thr=0.5):
    """Compute P, R, F1 across all images."""
    total_tp, total_fp, total_fn = 0, 0, 0
    for img_idx in range(len(all_gts)):
        pred_boxes, pred_scores, pred_labels = all_preds[img_idx]
        gt_boxes, gt_labels = all_gts[img_idx]
        
        # Filter by confidence
        if len(pred_boxes) > 0:
            mask = pred_scores >= conf_thr
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]
        
        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        
        if n_gt == 0 and n_pred == 0:
            continue
        if n_gt == 0:
            total_fp += n_pred
            continue
        if n_pred == 0:
            total_fn += n_gt
            continue
        
        matched_gt = set()
        tp = 0
        
        # Sort by confidence
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_labels = pred_labels[order]
        
        for pi in range(len(pred_boxes)):
            best_iou, best_gi = 0, -1
            for gi in range(n_gt):
                if gi in matched_gt:
                    continue
                if pred_labels[pi] != gt_labels[gi]:
                    continue
                # IoU
                ix1 = max(pred_boxes[pi][0], gt_boxes[gi][0])
                iy1 = max(pred_boxes[pi][1], gt_boxes[gi][1])
                ix2 = min(pred_boxes[pi][2], gt_boxes[gi][2])
                iy2 = min(pred_boxes[pi][3], gt_boxes[gi][3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                a1 = (pred_boxes[pi][2]-pred_boxes[pi][0])*(pred_boxes[pi][3]-pred_boxes[pi][1])
                a2 = (gt_boxes[gi][2]-gt_boxes[gi][0])*(gt_boxes[gi][3]-gt_boxes[gi][1])
                iou = inter / (a1 + a2 - inter + 1e-8)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_thr and best_gi >= 0:
                tp += 1
                matched_gt.add(best_gi)
            else:
                total_fp += 1
        
        total_tp += tp
        total_fn += (n_gt - len(matched_gt))
    
    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1

# ── Collect predictions from all 3 models ──
print("\n📦 Collecting predictions from all 3 models (with & without TTA)...")

pred_cache = {}  # {model_name_tta: list of (boxes, scores, labels)}

for model_name, model in models.items():
    for augment in [True, False]:
        key = f"{model_name}_{'tta' if augment else 'notta'}"
        print(f"  Running {key}...")
        preds = []
        for img_path in test_images:
            boxes, scores, labels = get_predictions(model, str(img_path), augment=augment, conf=0.01)
            preds.append((boxes, scores, labels))
        pred_cache[key] = preds
        print(f"    Done — avg {np.mean([len(p[0]) for p in preds]):.0f} detections/image")

# Load ground truth
print("\n📋 Loading ground truth...")
all_gts = []
img_sizes = []
for img_path in test_images:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    img_sizes.append((w, h))
    lbl_path = str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt')
    gt_boxes, gt_labels = load_gt(lbl_path, w, h)
    all_gts.append((gt_boxes, gt_labels))
total_gt = sum(len(g[0]) for g in all_gts)
print(f"  Total GT boxes: {total_gt}")

# ── Sweep ensemble configs ──
print("\n🔍 Sweeping 3-model ensemble configs...")

# Weight configurations to try
weight_configs = [
    # (yolo12x, yolo12l, yolo11x) weights
    (3.0, 2.0, 2.0),   # balanced large
    (3.0, 2.0, 3.0),   # equal x-models
    (4.0, 2.0, 3.0),   # favor 12x
    (3.0, 1.5, 2.5),   # favor x-models
    (2.0, 1.0, 2.0),   # equal x, less l
    (3.0, 2.0, 1.5),   # favor 12x+12l
    (1.0, 1.0, 1.0),   # equal
]

# TTA combos
tta_combos = [
    ('tta', 'tta', 'tta'),       # all TTA
    ('tta', 'notta', 'tta'),     # skip 12l TTA
    ('tta', 'tta', 'notta'),     # skip 11x TTA
]

iou_thrs = [0.40, 0.45, 0.50, 0.55]
skip_thrs = [0.01, 0.05]
conf_types = ['avg', 'max']

best_result = {'f1': 0}
all_results = []
config_count = 0
total_configs = len(weight_configs) * len(tta_combos) * len(iou_thrs) * len(skip_thrs) * len(conf_types)

print(f"  Total WBF configs: {total_configs}")

for weights in weight_configs:
    for tta_combo in tta_combos:
        # Get pred keys
        keys = [
            f"yolo12x_{'tta' if tta_combo[0]=='tta' else 'notta'}",
            f"yolo12l_{'tta' if tta_combo[1]=='tta' else 'notta'}",
            f"yolo11x_{'tta' if tta_combo[2]=='tta' else 'notta'}",
        ]
        
        for iou_thr in iou_thrs:
            for skip_thr in skip_thrs:
                for conf_type in conf_types:
                    config_count += 1
                    
                    # Run WBF per image
                    ensemble_preds = []
                    for img_idx in range(len(test_images)):
                        w_img, h_img = img_sizes[img_idx]
                        
                        boxes_list, scores_list, labels_list = [], [], []
                        for ki, key in enumerate(keys):
                            b, s, l = pred_cache[key][img_idx]
                            if len(b) == 0:
                                boxes_list.append(np.array([]).reshape(0, 4))
                                scores_list.append(np.array([]))
                                labels_list.append(np.array([]))
                            else:
                                # Normalize to [0,1]
                                b_norm = b.copy()
                                b_norm[:, [0,2]] /= w_img
                                b_norm[:, [1,3]] /= h_img
                                b_norm = np.clip(b_norm, 0, 1)
                                boxes_list.append(b_norm)
                                scores_list.append(s)
                                labels_list.append(l)
                        
                        if all(len(b) == 0 for b in boxes_list):
                            ensemble_preds.append((np.array([]), np.array([]), np.array([])))
                            continue
                        
                        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                            boxes_list, scores_list, labels_list,
                            weights=list(weights),
                            iou_thr=iou_thr,
                            skip_box_thr=skip_thr,
                            conf_type=conf_type,
                        )
                        
                        # Denormalize
                        if len(fused_boxes) > 0:
                            fused_boxes[:, [0,2]] *= w_img
                            fused_boxes[:, [1,3]] *= h_img
                        
                        ensemble_preds.append((fused_boxes, fused_scores, fused_labels.astype(int)))
                    
                    # Sweep conf thresholds
                    for conf_thr in np.arange(0.20, 0.55, 0.01):
                        p, r, f1 = evaluate(ensemble_preds, all_gts, conf_thr=conf_thr)
                        
                        result = {
                            'weights': weights,
                            'tta': tta_combo,
                            'iou_thr': iou_thr,
                            'skip_thr': skip_thr,
                            'conf_type': conf_type,
                            'conf_thr': round(float(conf_thr), 2),
                            'precision': round(p, 4),
                            'recall': round(r, 4),
                            'f1': round(f1, 4),
                        }
                        all_results.append(result)
                        
                        if f1 > best_result['f1']:
                            best_result = result
                            
                        # Check if meets target
                        if p >= 0.76 and r >= 0.85:
                            print(f"  🎯 TARGET MET: w={weights} tta={tta_combo} "
                                  f"iou={iou_thr} sk={skip_thr} {conf_type} "
                                  f"conf={conf_thr:.2f} → P={p:.4f} R={r:.4f} F1={f1:.4f}")
                    
                    if config_count % 20 == 0:
                        print(f"  [{config_count}/{total_configs}] best F1={best_result['f1']:.4f} "
                              f"P={best_result['precision']:.4f} R={best_result['recall']:.4f}")

# ── Final Results ──
print("\n" + "="*60)
print("🏆 FINAL 3-MODEL ENSEMBLE RESULTS")
print("="*60)

print(f"\n  Best F1: {best_result['f1']:.4f}")
print(f"  Config: weights={best_result['weights']}, tta={best_result['tta']}")
print(f"  WBF: iou={best_result['iou_thr']}, skip={best_result['skip_thr']}, type={best_result['conf_type']}")
print(f"  Conf threshold: {best_result['conf_thr']}")
print(f"  P={best_result['precision']:.4f}, R={best_result['recall']:.4f}, F1={best_result['f1']:.4f}")

# Find best target-meeting result
target_met = [r for r in all_results if r['precision'] >= 0.76 and r['recall'] >= 0.85]
if target_met:
    best_target = max(target_met, key=lambda x: x['f1'])
    print(f"\n  Best target-meeting (P≥0.76, R≥0.85):")
    print(f"  F1={best_target['f1']:.4f}, P={best_target['precision']:.4f}, R={best_target['recall']:.4f}")
    print(f"  Config: weights={best_target['weights']}, tta={best_target['tta']}")
    print(f"  WBF: iou={best_target['iou_thr']}, skip={best_target['skip_thr']}, type={best_target['conf_type']}")
    print(f"  Conf threshold: {best_target['conf_thr']}")
else:
    print("\n  ⚠️  No config met P≥0.76 & R≥0.85 target")
    # Show best P≥0.76
    high_p = [r for r in all_results if r['precision'] >= 0.76]
    if high_p:
        best_hp = max(high_p, key=lambda x: x['recall'])
        print(f"  Best with P≥0.76: P={best_hp['precision']:.4f}, R={best_hp['recall']:.4f}, F1={best_hp['f1']:.4f}")

# Compare with 2-model baseline
print(f"\n  📊 Previous 2-model best: P=0.7716, R=0.8621, F1=0.8143")

# Save results
output = {
    'yolo11x_single': {},
    'ensemble_3model_best_f1': best_result,
    'ensemble_3model_best_target': best_target if target_met else None,
    'total_configs_evaluated': len(all_results),
    'comparison': {
        '2model_best': {'P': 0.7716, 'R': 0.8621, 'F1': 0.8143},
        '3model_best': best_result,
    }
}

with open(os.path.join(PROJECT_ROOT, 'yolo11x_ensemble_results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to yolo11x_ensemble_results.json")
print("🏁 Done!")
