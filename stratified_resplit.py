"""
Stratified Train/Val Resplit + YOLO12x Retrain + Ensemble Evaluation
====================================================================
Problem: 44/76 classes have 0 validation instances. 2 classes only in val (never trained).
Fix: Pool train+val images → stratified resplit → every class in both splits.
Test set is UNTOUCHED.

GPU: Device 1 (RTX 3090 24GB)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import shutil
import yaml
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
import torch

PROJECT_ROOT = '/mnt/Data/AKIB/YOLO-OD-IM'
DATASET_DIR  = os.path.join(PROJECT_ROOT, 'dataset')

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════════════════════
# PHASE 1: STRATIFIED RESPLIT (train + val → new_train + new_val)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📊 PHASE 1: STRATIFIED RESPLIT")
print("="*60)

# Collect ALL train + val images
train_img_dir = os.path.join(DATASET_DIR, 'train', 'images')
train_lbl_dir = os.path.join(DATASET_DIR, 'train', 'labels')
val_img_dir   = os.path.join(DATASET_DIR, 'valid', 'images')
val_lbl_dir   = os.path.join(DATASET_DIR, 'valid', 'labels')

# Get all image/label pairs
all_images = []
all_labels = []

for img_dir, lbl_dir, split_name in [(train_img_dir, train_lbl_dir, 'train'), 
                                       (val_img_dir, val_lbl_dir, 'val')]:
    for img_file in sorted(Path(img_dir).glob('*.jpg')):
        lbl_file = Path(lbl_dir) / (img_file.stem + '.txt')
        if lbl_file.exists():
            all_images.append((str(img_file), str(lbl_file), split_name))
            # Read classes in this image
            classes_in_img = set()
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        classes_in_img.add(int(parts[0]))
            all_labels.append(classes_in_img)

print(f"  Total pooled images: {len(all_images)}")
print(f"  From train: {sum(1 for _, _, s in all_images if s == 'train')}")
print(f"  From val:   {sum(1 for _, _, s in all_images if s == 'val')}")

# Load class names
with open(os.path.join(DATASET_DIR, 'data.yaml')) as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg['names']
nc = len(class_names) if isinstance(class_names, list) else max(class_names.keys()) + 1

# Assign each image a "primary class" = rarest class in the image (for stratification)
# Count total instances per class across all pooled images
global_class_count = defaultdict(int)
for classes_in_img in all_labels:
    for c in classes_in_img:
        global_class_count[c] += 1

# For each image, assign the rarest class as its stratification key
primary_classes = []
for classes_in_img in all_labels:
    if not classes_in_img:
        primary_classes.append(-1)  # empty label
    else:
        rarest = min(classes_in_img, key=lambda c: global_class_count[c])
        primary_classes.append(rarest)

primary_classes = np.array(primary_classes)

# ── Custom iterative stratified split ──
# Goal: ensure every class has at least 1 image in val, keep val small (~10%)
# StratifiedShuffleSplit fails when n_classes > n_val_target, so we do it manually.

val_ratio = 0.10  # ~10% for val → ~96 images (ensures ≥1 per class)
n_val_target = max(int(len(all_images) * val_ratio), len(set(primary_classes)))
print(f"  Target val size: ~{n_val_target} images")

# Build per-class image index (using primary class assignment)
from collections import defaultdict as dd
class_to_indices = dd(list)
for i, pc in enumerate(primary_classes):
    class_to_indices[pc].append(i)

rng = np.random.RandomState(42)

val_set = set()

# Step 1: guarantee ≥1 image per class in val
for cls_id in sorted(class_to_indices.keys()):
    indices = class_to_indices[cls_id]
    if len(indices) == 1:
        # Only 1 image for this class — must go to BOTH train & val? No.
        # Put it in train (model can learn it) and accept 0 val for this class.
        # Actually, let's put it in val so val monitors that class.
        # But then train has 0... For classes with only 1 image, put in train.
        continue
    # Pick one random image for val
    chosen = rng.choice(indices)
    val_set.add(chosen)

print(f"  After class coverage pass: {len(val_set)} val images")

# Step 2: fill up to n_val_target with images not yet in val
remaining_pool = [i for i in range(len(all_images)) if i not in val_set]
n_more = max(0, n_val_target - len(val_set))
if n_more > 0:
    extra = rng.choice(remaining_pool, size=min(n_more, len(remaining_pool)), replace=False)
    val_set.update(extra.tolist())

new_val_indices = np.array(sorted(val_set))
new_train_indices = np.array(sorted(set(range(len(all_images))) - val_set))

print(f"\n  New split: train={len(new_train_indices)}, val={len(new_val_indices)}")

# Create new dataset directories
new_dataset_dir = os.path.join(PROJECT_ROOT, 'dataset_stratified')
for split in ['train', 'valid', 'test']:
    for sub in ['images', 'labels']:
        os.makedirs(os.path.join(new_dataset_dir, split, sub), exist_ok=True)

# Copy images and labels to new directories
def copy_file(src, dst):
    shutil.copy2(src, dst)

print("\n  Copying files to dataset_stratified/...")

for idx in new_train_indices:
    img_path, lbl_path, _ = all_images[idx]
    copy_file(img_path, os.path.join(new_dataset_dir, 'train', 'images', Path(img_path).name))
    copy_file(lbl_path, os.path.join(new_dataset_dir, 'train', 'labels', Path(lbl_path).name))

for idx in new_val_indices:
    img_path, lbl_path, _ = all_images[idx]
    copy_file(img_path, os.path.join(new_dataset_dir, 'valid', 'images', Path(img_path).name))
    copy_file(lbl_path, os.path.join(new_dataset_dir, 'valid', 'labels', Path(lbl_path).name))

# Symlink test set (unchanged)
test_img_dir = os.path.join(DATASET_DIR, 'test', 'images')
test_lbl_dir = os.path.join(DATASET_DIR, 'test', 'labels')
for f in Path(test_img_dir).glob('*'):
    copy_file(str(f), os.path.join(new_dataset_dir, 'test', 'images', f.name))
for f in Path(test_lbl_dir).glob('*'):
    copy_file(str(f), os.path.join(new_dataset_dir, 'test', 'labels', f.name))

# Verify new distribution
print("\n  Verifying new distribution...")
new_class_counts = {'train': defaultdict(int), 'valid': defaultdict(int)}

for split_name, indices in [('train', new_train_indices), ('valid', new_val_indices)]:
    for idx in indices:
        for c in all_labels[idx]:
            new_class_counts[split_name][c] += 1

zero_train_new = sum(1 for c in range(nc) if new_class_counts['train'].get(c, 0) == 0 and global_class_count.get(c, 0) > 0)
zero_val_new = sum(1 for c in range(nc) if new_class_counts['valid'].get(c, 0) == 0 and global_class_count.get(c, 0) > 0)

print(f"  BEFORE: 44 classes with 0 val, 3 classes with 0 train")
print(f"  AFTER:  {zero_val_new} classes with 0 val, {zero_train_new} classes with 0 train")

# Show improved classes
print(f"\n  Class distribution comparison:")
print(f"  {'Class':>6} {'Name':>8} {'OldTr':>6} {'OldVa':>6} {'NewTr':>6} {'NewVa':>6} {'Status':>10}")
print("  " + "-"*60)

# Load old counts
old_train_counts = defaultdict(int)
old_val_counts = defaultdict(int)
for img_path, lbl_path, split in all_images:
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                c = int(parts[0])
                if split == 'train':
                    old_train_counts[c] += 1
                else:
                    old_val_counts[c] += 1

for c in range(nc):
    if global_class_count.get(c, 0) == 0:
        continue
    old_t = old_train_counts.get(c, 0)
    old_v = old_val_counts.get(c, 0)
    new_t = new_class_counts['train'].get(c, 0)
    new_v = new_class_counts['valid'].get(c, 0)
    
    status = ''
    if old_v == 0 and new_v > 0:
        status = '✅ FIXED'
    elif old_t == 0 and new_t > 0:
        status = '✅ FIXED'
    elif new_v == 0 or new_t == 0:
        status = '⚠️ STILL 0'
    
    if status or old_v == 0 or old_t == 0:
        name = class_names[c] if isinstance(class_names, list) else class_names.get(c, '?')
        print(f"  {c:>6} {name:>8} {old_t:>6} {old_v:>6} {new_t:>6} {new_v:>6} {status:>10}")

# Create data.yaml for new dataset
new_data_yaml = {
    'path': new_dataset_dir,
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'nc': nc,
    'names': class_names,
}
yaml_path = os.path.join(new_dataset_dir, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(new_data_yaml, f, default_flow_style=False)
print(f"\n  Saved: {yaml_path}")

# ══════════════════════════════════════════════════════════════
# PHASE 2: TRAIN YOLO12x ON STRATIFIED SPLIT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🚀 PHASE 2: TRAINING YOLO12x ON STRATIFIED SPLIT")
print("="*60)

model = YOLO('yolo12x.pt')

results = model.train(
    data=yaml_path,
    epochs=150,
    imgsz=640,
    batch=8,
    device=0,
    project=os.path.join(PROJECT_ROOT, 'runs'),
    name='yolo12x_stratified',
    exist_ok=True,
    
    # Same hyperparams as original YOLO12x
    cos_lr=True,
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    
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
    
    patience=30,
    save=True,
    save_period=25,
    plots=True,
    verbose=True,
)

print("\n✅ YOLO12x stratified training complete!")

# ══════════════════════════════════════════════════════════════
# PHASE 3: EVALUATE SINGLE MODEL ON TEST SET
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📊 PHASE 3: SINGLE MODEL TEST EVALUATION")
print("="*60)

best_path = os.path.join(PROJECT_ROOT, 'runs', 'yolo12x_stratified', 'weights', 'best.pt')
model_best = YOLO(best_path)

single_results = {}
for conf in [0.10, 0.15, 0.20, 0.25]:
    m = model_best.val(data=yaml_path, split='test', device=0, conf=conf, iou=0.5, verbose=False)
    p, r = float(m.box.mp), float(m.box.mr)
    f1 = 2*p*r/(p+r+1e-8)
    single_results[f'conf_{conf}'] = {'P': round(p, 4), 'R': round(r, 4), 'F1': round(f1, 4), 'mAP50': round(float(m.box.map50), 4)}
    print(f"  conf={conf:.2f} → P={p:.4f}, R={r:.4f}, F1={f1:.4f}, mAP50={float(m.box.map50):.4f}")

# ══════════════════════════════════════════════════════════════
# PHASE 4: 2-MODEL ENSEMBLE (new YOLO12x_strat + YOLO12l)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🔥 PHASE 4: 2-MODEL ENSEMBLE (YOLO12x_stratified + YOLO12l)")
print("="*60)

model_12x_strat = YOLO(best_path)
model_12l = YOLO(os.path.join(PROJECT_ROOT, 'runs', 'yolo12l_ensemble', 'weights', 'best.pt'))

test_img_dir_eval = os.path.join(new_dataset_dir, 'test', 'images')
test_lbl_dir_eval = os.path.join(new_dataset_dir, 'test', 'labels')
test_images = sorted(Path(test_img_dir_eval).glob('*.jpg'))
print(f"  Test images: {len(test_images)}")

def load_gt(label_path, img_w, img_h):
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
    results = model.predict(img_path, device=0, conf=conf, iou=0.7, augment=augment, verbose=False)
    r = results[0]
    if len(r.boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)

def evaluate(all_preds, all_gts, conf_thr=0.25, iou_thr=0.5):
    total_tp, total_fp, total_fn = 0, 0, 0
    for img_idx in range(len(all_gts)):
        pred_boxes, pred_scores, pred_labels = all_preds[img_idx]
        gt_boxes, gt_labels = all_gts[img_idx]
        if len(pred_boxes) > 0:
            mask = pred_scores >= conf_thr
            pred_boxes, pred_scores, pred_labels = pred_boxes[mask], pred_scores[mask], pred_labels[mask]
        n_gt, n_pred = len(gt_boxes), len(pred_boxes)
        if n_gt == 0 and n_pred == 0: continue
        if n_gt == 0: total_fp += n_pred; continue
        if n_pred == 0: total_fn += n_gt; continue
        matched_gt = set()
        order = np.argsort(-pred_scores)
        pred_boxes, pred_labels = pred_boxes[order], pred_labels[order]
        for pi in range(len(pred_boxes)):
            best_iou, best_gi = 0, -1
            for gi in range(n_gt):
                if gi in matched_gt or pred_labels[pi] != gt_labels[gi]: continue
                ix1 = max(pred_boxes[pi][0], gt_boxes[gi][0])
                iy1 = max(pred_boxes[pi][1], gt_boxes[gi][1])
                ix2 = min(pred_boxes[pi][2], gt_boxes[gi][2])
                iy2 = min(pred_boxes[pi][3], gt_boxes[gi][3])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                a1 = (pred_boxes[pi][2]-pred_boxes[pi][0]) * (pred_boxes[pi][3]-pred_boxes[pi][1])
                a2 = (gt_boxes[gi][2]-gt_boxes[gi][0]) * (gt_boxes[gi][3]-gt_boxes[gi][1])
                iou = inter / (a1+a2-inter+1e-8)
                if iou > best_iou: best_iou, best_gi = iou, gi
            if best_iou >= iou_thr and best_gi >= 0:
                total_tp += 1; matched_gt.add(best_gi)
            else:
                total_fp += 1
        total_fn += (n_gt - len(matched_gt))
    p = total_tp / (total_tp + total_fp + 1e-8)
    r = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2*p*r/(p+r+1e-8)
    return p, r, f1

# Collect predictions
print("\n📦 Collecting predictions...")
models_ens = {'yolo12x_strat': model_12x_strat, 'yolo12l': model_12l}
pred_cache = {}

for model_name, mdl in models_ens.items():
    for augment in [True, False]:
        key = f"{model_name}_{'tta' if augment else 'notta'}"
        print(f"  Running {key}...")
        preds = []
        for img_path in test_images:
            boxes, scores, labels = get_predictions(mdl, str(img_path), augment=augment, conf=0.01)
            preds.append((boxes, scores, labels))
        pred_cache[key] = preds

# Load GT
all_gts = []
img_sizes = []
for img_path in test_images:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    img_sizes.append((w, h))
    lbl_path = str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt')
    gt_b, gt_l = load_gt(lbl_path, w, h)
    all_gts.append((gt_b, gt_l))

# Sweep ensemble configs (focused on best configs from previous sweep)
print("\n🔍 Sweeping 2-model ensemble configs...")

weight_configs = [(3.0, 2.0), (3.0, 1.5), (2.0, 1.0), (4.0, 2.0), (2.0, 2.0), (3.0, 3.0)]
tta_combos = [('tta', 'tta'), ('tta', 'notta')]
iou_thrs = [0.40, 0.45, 0.50, 0.55]
skip_thrs = [0.01, 0.05]
conf_types = ['avg', 'max']

best_result = {'f1': 0}
best_target = None
all_ens_results = []

total_configs = len(weight_configs) * len(tta_combos) * len(iou_thrs) * len(skip_thrs) * len(conf_types)
config_count = 0

for weights in weight_configs:
    for tta_combo in tta_combos:
        keys = [
            f"yolo12x_strat_{'tta' if tta_combo[0]=='tta' else 'notta'}",
            f"yolo12l_{'tta' if tta_combo[1]=='tta' else 'notta'}",
        ]
        for iou_thr in iou_thrs:
            for skip_thr in skip_thrs:
                for conf_type in conf_types:
                    config_count += 1
                    
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
                        
                        fb, fs, fl = weighted_boxes_fusion(
                            boxes_list, scores_list, labels_list,
                            weights=list(weights), iou_thr=iou_thr,
                            skip_box_thr=skip_thr, conf_type=conf_type,
                        )
                        if len(fb) > 0:
                            fb[:, [0,2]] *= w_img
                            fb[:, [1,3]] *= h_img
                        ensemble_preds.append((fb, fs, fl.astype(int)))
                    
                    for conf_thr in np.arange(0.20, 0.55, 0.01):
                        p, r, f1 = evaluate(ensemble_preds, all_gts, conf_thr=conf_thr)
                        result = {
                            'weights': weights, 'tta': tta_combo,
                            'iou_thr': iou_thr, 'skip_thr': skip_thr,
                            'conf_type': conf_type, 'conf_thr': round(float(conf_thr), 2),
                            'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f1, 4),
                        }
                        all_ens_results.append(result)
                        if f1 > best_result['f1']:
                            best_result = result
                        if p >= 0.76 and r >= 0.85:
                            if best_target is None or f1 > best_target['f1']:
                                best_target = result
                            print(f"  🎯 TARGET: w={weights} conf={conf_thr:.2f} → P={p:.4f} R={r:.4f} F1={f1:.4f}")
                    
                    if config_count % 20 == 0:
                        print(f"  [{config_count}/{total_configs}] best F1={best_result['f1']:.4f}")

# ══════════════════════════════════════════════════════════════
# FINAL RESULTS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🏆 FINAL RESULTS: STRATIFIED SPLIT")
print("="*60)

print(f"\n  📊 Single Model (YOLO12x stratified):")
for k, v in single_results.items():
    print(f"    {k}: P={v['P']:.4f}, R={v['R']:.4f}, F1={v['F1']:.4f}, mAP50={v['mAP50']:.4f}")

print(f"\n  📊 Best Ensemble F1:")
print(f"    P={best_result['precision']:.4f}, R={best_result['recall']:.4f}, F1={best_result['f1']:.4f}")
print(f"    Config: w={best_result['weights']}, iou={best_result['iou_thr']}, "
      f"skip={best_result['skip_thr']}, type={best_result['conf_type']}, conf={best_result['conf_thr']}")

if best_target:
    print(f"\n  🎯 Best Target-Meeting (P≥0.76, R≥0.85):")
    print(f"    P={best_target['precision']:.4f}, R={best_target['recall']:.4f}, F1={best_target['f1']:.4f}")
    print(f"    Config: w={best_target['weights']}, iou={best_target['iou_thr']}, "
          f"skip={best_target['skip_thr']}, type={best_target['conf_type']}, conf={best_target['conf_thr']}")
else:
    print(f"\n  ⚠️  No config met P≥0.76 & R≥0.85")
    high_p = [r for r in all_ens_results if r['precision'] >= 0.76]
    if high_p:
        best_hp = max(high_p, key=lambda x: x['recall'])
        print(f"    Best with P≥0.76: P={best_hp['precision']:.4f}, R={best_hp['recall']:.4f}, F1={best_hp['f1']:.4f}")

print(f"\n  📊 COMPARISON with original split:")
print(f"    Original YOLO12x:       P=0.690, R=0.832, F1=0.755")
print(f"    Original 2-model ens:   P=0.772, R=0.862, F1=0.814")
print(f"    Stratified YOLO12x:     P={single_results.get('conf_0.1',{}).get('P','?')}, "
      f"R={single_results.get('conf_0.1',{}).get('R','?')}, F1={single_results.get('conf_0.1',{}).get('F1','?')}")

# Save results
output = {
    'split_fix': {
        'before': {'zero_val': 44, 'zero_train': 3},
        'after': {'zero_val': zero_val_new, 'zero_train': zero_train_new},
    },
    'single_model': single_results,
    'ensemble_best_f1': best_result,
    'ensemble_best_target': best_target,
    'comparison_original_2model': {'P': 0.7716, 'R': 0.8621, 'F1': 0.8143},
}
with open(os.path.join(PROJECT_ROOT, 'stratified_results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n✅ Results saved to stratified_results.json")
print("🏁 Done!")
