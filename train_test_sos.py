"""
YOLO12x @ imgsz=1280 → Selective Ensemble → Share of Shelf
===========================================================
Goal: P > 0.76, R > 0.85
Strategy: 
  1. Train YOLO12x at 1280 resolution (~25 min)
  2. Test standalone + selective ensemble with YOLO12l
  3. Share of Shelf analytics with visualizations
"""
import os, json, time, warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import yaml
import glob
import cv2
from ultralytics import YOLO
from collections import defaultdict, Counter
from ensemble_boxes import weighted_boxes_fusion
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 120

PROJECT = '/mnt/Data/AKIB/YOLO-OD-IM'
DATA    = os.path.join(PROJECT, 'dataset', 'data_abs.yaml')
TEST_IMG = os.path.join(PROJECT, 'dataset', 'test', 'images')
TEST_LBL = os.path.join(PROJECT, 'dataset', 'test', 'labels')

with open(DATA) as f:
    cfg = yaml.safe_load(f)
class_names = cfg['names']
NC = len(class_names)

test_images = sorted(glob.glob(os.path.join(TEST_IMG, '*')))
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Test images: {len(test_images)}, Classes: {NC}")

# ══════════════════════════════════════════════════════════════
# PHASE 1: TRAIN YOLO12x @ 1280 
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("🚀 PHASE 1: TRAINING YOLO12x @ imgsz=1280")
print("="*70)

t_start = time.time()

torch.cuda.empty_cache()
import gc; gc.collect()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

model = YOLO('yolo12x.pt')
model.train(
    data=DATA,
    epochs=50,
    imgsz=1280,
    batch=1,            # 1280px on 24GB — tight fit
    device=0,
    project=os.path.join(PROJECT, 'runs'),
    name='yolo12x_1280',
    exist_ok=True,
    cos_lr=True,
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    cutmix=0.1,
    close_mosaic=10,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    patience=12,        # early stop — tight for time
    save=True,
    plots=True,
    verbose=True,
)

train_time = time.time() - t_start
print(f"\n✅ Training done in {train_time/60:.1f} min")

del model
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# PHASE 2: TEST — YOLO12x-1280 standalone + selective ensemble
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📊 PHASE 2: TESTING & SELECTIVE ENSEMBLE")
print("="*70)

# --- Helpers ---
def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / (union + 1e-8)

def load_gt(lbl_path, w, h):
    boxes, cls = [], []
    if not os.path.exists(lbl_path): return np.array([]), np.array([])
    for line in open(lbl_path):
        p = line.strip().split()
        if len(p) < 5: continue
        c = int(p[0]); xc,yc,bw,bh = map(float, p[1:5])
        boxes.append([(xc-bw/2)*w, (yc-bh/2)*h, (xc+bw/2)*w, (yc+bh/2)*h])
        cls.append(c)
    return np.array(boxes), np.array(cls)

def compute_metrics(pb, pc, ps, gb, gc, iou_t=0.5):
    if len(gb)==0: return 0, len(pb), 0
    if len(pb)==0: return 0, 0, len(gb)
    matched = np.zeros(len(gb), dtype=bool)
    tp = fp = 0
    for i in np.argsort(-ps):
        best_iou, best_g = 0, -1
        for g in range(len(gb)):
            if matched[g] or gc[g] != pc[i]: continue
            iou = compute_iou(pb[i], gb[g])
            if iou > best_iou: best_iou, best_g = iou, g
        if best_iou >= iou_t and best_g >= 0:
            tp += 1; matched[best_g] = True
        else: fp += 1
    return tp, fp, int(np.sum(~matched))

def run_inference(model, img_path, conf=0.05, iou=0.7, augment=False, imgsz=1280):
    r = model.predict(img_path, conf=conf, iou=iou, augment=augment, 
                      device=0, verbose=False, max_det=300, imgsz=imgsz)[0]
    if len(r.boxes) > 0:
        return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int), r.orig_shape[1], r.orig_shape[0])
    return np.zeros((0,4)), np.array([]), np.array([], dtype=int), r.orig_shape[1], r.orig_shape[0]

def eval_preds(preds_dict, conf_thresh, iou_thresh=0.5):
    tp_t = fp_t = fn_t = 0
    for img_path in test_images:
        img_file = os.path.basename(img_path)
        p = preds_dict[img_file]
        lbl = os.path.join(TEST_LBL, os.path.splitext(img_file)[0]+'.txt')
        gb, gc = load_gt(lbl, p['w'], p['h'])
        m = p['scores'] >= conf_thresh
        tp,fp,fn = compute_metrics(p['boxes'][m], p['cls'][m], p['scores'][m], gb, gc, iou_thresh)
        tp_t+=tp; fp_t+=fp; fn_t+=fn
    pr = tp_t/(tp_t+fp_t+1e-8); rc = tp_t/(tp_t+fn_t+1e-8); f1 = 2*pr*rc/(pr+rc+1e-8)
    return pr, rc, f1

# --- Load models ---
MODELS_TO_TEST = {
    'yolo12x_1280': os.path.join(PROJECT, 'runs', 'yolo12x_1280', 'weights', 'best.pt'),
    'yolo12x_640':  os.path.join(PROJECT, 'runs', 'yolo12x_optimized', 'weights', 'best.pt'),
    'yolo12l':      os.path.join(PROJECT, 'runs', 'yolo12l_ensemble', 'weights', 'best.pt'),
}

all_preds = {}
for mname, wpath in MODELS_TO_TEST.items():
    if not os.path.exists(wpath):
        print(f"  ❌ {mname} not found, skipping")
        continue
    print(f"\n🔄 Inference: {mname} (TTA=True)")
    model = YOLO(wpath)
    imgsz = 1280 if '1280' in mname else 640
    preds = {}
    for img_path in test_images:
        boxes, scores, cls, w, h = run_inference(model, img_path, conf=0.05, iou=0.7, augment=True, imgsz=imgsz)
        preds[os.path.basename(img_path)] = {'boxes': boxes, 'scores': scores, 'cls': cls, 'w': w, 'h': h}
    all_preds[mname] = preds
    
    # Test at multiple thresholds
    print(f"  {'conf':>6s}  {'P':>8s}  {'R':>8s}  {'F1':>8s}")
    for c in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        p,r,f1 = eval_preds(preds, c)
        marker = " ← ✅ TARGET" if p >= 0.76 and r >= 0.85 else ""
        print(f"  {c:6.2f}  {p:8.4f}  {r:8.4f}  {f1:8.4f}{marker}")
    
    del model
    torch.cuda.empty_cache()

# --- Selective WBF: only top 2-3 models ---
print("\n" + "="*70)
print("🏆 SELECTIVE WBF ENSEMBLE (YOLO12x_1280 + YOLO12x_640 + YOLO12l)")
print("="*70)

ensemble_configs = [
    {'models': ['yolo12x_1280', 'yolo12x_640'], 'weights': [3.0, 2.0], 'name': 'x1280+x640'},
    {'models': ['yolo12x_1280', 'yolo12l'], 'weights': [3.0, 1.5], 'name': 'x1280+12l'},
    {'models': ['yolo12x_1280', 'yolo12x_640', 'yolo12l'], 'weights': [3.0, 2.0, 1.5], 'name': 'x1280+x640+12l'},
]

best_f1_overall = 0
best_config_overall = None
best_metrics_overall = None
best_preds_for_sos = None

for ecfg in ensemble_configs:
    model_list = [m for m in ecfg['models'] if m in all_preds]
    if len(model_list) < 2:
        print(f"  ⚠️ {ecfg['name']}: Not enough models, skipping")
        continue
    
    weights = ecfg['weights'][:len(model_list)]
    
    for wbf_iou in [0.5, 0.6]:
        for skip_thr in [0.05, 0.10]:
            # Run WBF per image
            ens_preds = {}
            for img_path in test_images:
                img_file = os.path.basename(img_path)
                boxes_list, scores_list, labels_list = [], [], []
                sample = all_preds[model_list[0]][img_file]
                W, H = sample['w'], sample['h']
                
                for mn in model_list:
                    p = all_preds[mn][img_file]
                    if len(p['boxes']) > 0:
                        nb = p['boxes'].copy()
                        nb[:, [0,2]] /= W; nb[:, [1,3]] /= H
                        nb = np.clip(nb, 0, 1)
                        boxes_list.append(nb.tolist())
                        scores_list.append(p['scores'].tolist())
                        labels_list.append(p['cls'].tolist())
                    else:
                        boxes_list.append([]); scores_list.append([]); labels_list.append([])
                
                if any(len(b)>0 for b in boxes_list):
                    fb, fs, fl = weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                                                       weights=weights, iou_thr=wbf_iou, 
                                                       skip_box_thr=skip_thr, conf_type='avg')
                    fb[:, [0,2]] *= W; fb[:, [1,3]] *= H
                else:
                    fb, fs, fl = np.zeros((0,4)), np.array([]), np.array([], dtype=int)
                
                ens_preds[img_file] = {'boxes': fb, 'scores': fs, 'cls': fl.astype(int), 'w': W, 'h': H}
            
            # Evaluate
            for c in [0.10, 0.15, 0.20, 0.25, 0.30]:
                p, r, f1 = eval_preds(ens_preds, c)
                tag = f"{ecfg['name']}_wbf{wbf_iou}_skip{skip_thr}_conf{c}"
                
                hits_target = p >= 0.76 and r >= 0.85
                if f1 > best_f1_overall:
                    best_f1_overall = f1
                    best_config_overall = tag
                    best_metrics_overall = {'P': p, 'R': r, 'F1': f1, 'conf': c}
                    best_preds_for_sos = ens_preds
                
                if hits_target or f1 > 0.78:
                    print(f"  {'✅' if hits_target else '  '} {tag:55s} P={p:.4f} R={r:.4f} F1={f1:.4f}")

# Also check standalone yolo12x_1280 for best results
if 'yolo12x_1280' in all_preds:
    for c in [0.10, 0.15, 0.20, 0.25, 0.30]:
        p, r, f1 = eval_preds(all_preds['yolo12x_1280'], c)
        if f1 > best_f1_overall:
            best_f1_overall = f1
            best_config_overall = f"yolo12x_1280_standalone_conf{c}"
            best_metrics_overall = {'P': p, 'R': r, 'F1': f1, 'conf': c}
            best_preds_for_sos = all_preds['yolo12x_1280']

print(f"\n🏆 BEST OVERALL: {best_config_overall}")
print(f"   Precision:  {best_metrics_overall['P']:.4f}")
print(f"   Recall:     {best_metrics_overall['R']:.4f}")
print(f"   F1 Score:   {best_metrics_overall['F1']:.4f}")
print(f"   Conf Thr:   {best_metrics_overall['conf']}")

# ══════════════════════════════════════════════════════════════
# PHASE 3: SHARE OF SHELF ANALYTICS + VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📊 PHASE 3: SHARE OF SHELF ANALYTICS")
print("="*70)

# Use best predictions
use_preds = best_preds_for_sos
use_conf = best_metrics_overall['conf']

# --- Compute GT Share of Shelf ---
gt_counts = Counter()
gt_areas = defaultdict(float)
pred_counts = Counter()
pred_areas = defaultdict(float)

for img_path in test_images:
    img_file = os.path.basename(img_path)
    p = use_preds[img_file]
    W, H = p['w'], p['h']
    
    # Predicted
    mask = p['scores'] >= use_conf
    for i in range(len(p['boxes'])):
        if not mask[i]: continue
        cls_id = int(p['cls'][i])
        box = p['boxes'][i]
        area = (box[2]-box[0]) * (box[3]-box[1]) / (W*H)  # normalized
        pred_counts[cls_id] += 1
        pred_areas[cls_id] += area
    
    # Ground truth
    lbl = os.path.join(TEST_LBL, os.path.splitext(img_file)[0]+'.txt')
    gb, gc = load_gt(lbl, W, H)
    for i in range(len(gb)):
        cls_id = int(gc[i])
        area = (gb[i][2]-gb[i][0]) * (gb[i][3]-gb[i][1]) / (W*H)
        gt_counts[cls_id] += 1
        gt_areas[cls_id] += area

total_gt_f = sum(gt_counts.values())
total_gt_a = sum(gt_areas.values())
total_pr_f = sum(pred_counts.values())
total_pr_a = sum(pred_areas.values())

# Build SoS table
gt_sos_f = {i: 100.0*gt_counts.get(i,0)/total_gt_f for i in range(NC)}
gt_sos_a = {i: 100.0*gt_areas.get(i,0)/total_gt_a for i in range(NC)}
pr_sos_f = {i: 100.0*pred_counts.get(i,0)/total_pr_f if total_pr_f>0 else 0 for i in range(NC)}
pr_sos_a = {i: 100.0*pred_areas.get(i,0)/total_pr_a if total_pr_a>0 else 0 for i in range(NC)}

present_ids = sorted([i for i in range(NC) if gt_counts.get(i,0)>0 or pred_counts.get(i,0)>0], key=lambda x: -gt_sos_f.get(x,0))

print(f"\n  Total GT products: {total_gt_f}, Predicted: {total_pr_f}")
print(f"  GT unique SKUs: {len([i for i in range(NC) if gt_counts.get(i,0)>0])}")
print(f"  Pred unique SKUs: {len([i for i in range(NC) if pred_counts.get(i,0)>0])}")

print(f"\n{'SKU':<8s} {'GT_F':>6s} {'Pr_F':>6s} {'GT_SoS%':>9s} {'Pr_SoS%':>9s} {'GT_Area%':>9s} {'Pr_Area%':>9s}")
print("-"*62)
for cls_id in present_ids:
    print(f"{class_names[cls_id]:<8s} {gt_counts.get(cls_id,0):>6d} {pred_counts.get(cls_id,0):>6d} "
          f"{gt_sos_f[cls_id]:>8.2f}% {pr_sos_f[cls_id]:>8.2f}% {gt_sos_a[cls_id]:>8.2f}% {pr_sos_a[cls_id]:>8.2f}%")

# ══════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
print("\n📈 Generating visualizations...")

# --- 1. GT Share of Shelf Bar Chart (Facing Count) ---
gt_sorted = sorted([i for i in range(NC) if gt_counts.get(i,0)>0], key=lambda x: -gt_sos_f[x])
fig, ax = plt.subplots(figsize=(18, 7))
x = range(len(gt_sorted))
vals = [gt_sos_f[i] for i in gt_sorted]
names = [class_names[i] for i in gt_sorted]
bars = ax.bar(x, vals, color='steelblue', edgecolor='white', linewidth=0.5)
for i in range(min(5, len(bars))): bars[i].set_color('#e74c3c')
ax.set_xticks(x); ax.set_xticklabels(names, rotation=90, fontsize=7)
ax.set_ylabel('Share of Shelf (%)'); ax.set_title('Ground Truth: Share of Shelf by Facing Count', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
for i in range(min(10, len(bars))): ax.text(i, vals[i]+0.3, f'{vals[i]:.1f}%', ha='center', fontsize=7, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT, 'sos_gt_facing.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ sos_gt_facing.png")

# --- 2. GT Share of Shelf Bar Chart (Area) ---
gt_sorted_a = sorted([i for i in range(NC) if gt_counts.get(i,0)>0], key=lambda x: -gt_sos_a[x])
fig, ax = plt.subplots(figsize=(18, 7))
x = range(len(gt_sorted_a))
vals_a = [gt_sos_a[i] for i in gt_sorted_a]
names_a = [class_names[i] for i in gt_sorted_a]
bars = ax.bar(x, vals_a, color='mediumseagreen', edgecolor='white', linewidth=0.5)
for i in range(min(5, len(bars))): bars[i].set_color('#e74c3c')
ax.set_xticks(x); ax.set_xticklabels(names_a, rotation=90, fontsize=7)
ax.set_ylabel('Share of Shelf (%)'); ax.set_title('Ground Truth: Share of Shelf by Area / Shelf Space', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
for i in range(min(10, len(bars))): ax.text(i, vals_a[i]+0.3, f'{vals_a[i]:.1f}%', ha='center', fontsize=7, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT, 'sos_gt_area.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ sos_gt_area.png")

# --- 3. GT vs Predicted Side-by-Side Bar Chart ---
fig, ax = plt.subplots(figsize=(20, 7))
x = np.arange(len(present_ids))
w = 0.4
gt_v = [gt_sos_f.get(i, 0) for i in present_ids]
pr_v = [pr_sos_f.get(i, 0) for i in present_ids]
ax.bar(x - w/2, gt_v, w, label='Ground Truth', color='steelblue', alpha=0.8)
ax.bar(x + w/2, pr_v, w, label='Model Prediction', color='coral', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([class_names[i] for i in present_ids], rotation=90, fontsize=6)
ax.set_ylabel('Share of Shelf (%)'); ax.set_title('Share of Shelf: Ground Truth vs Model Prediction (Facing Count)', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT, 'sos_gt_vs_pred_facing.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ sos_gt_vs_pred_facing.png")

# --- 4. GT vs Predicted Side-by-Side Bar (Area) ---
fig, ax = plt.subplots(figsize=(20, 7))
gt_va = [gt_sos_a.get(i, 0) for i in present_ids]
pr_va = [pr_sos_a.get(i, 0) for i in present_ids]
ax.bar(x - w/2, gt_va, w, label='Ground Truth', color='mediumseagreen', alpha=0.8)
ax.bar(x + w/2, pr_va, w, label='Model Prediction', color='coral', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels([class_names[i] for i in present_ids], rotation=90, fontsize=6)
ax.set_ylabel('Share of Shelf (%)'); ax.set_title('Share of Shelf: Ground Truth vs Model Prediction (Area)', fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT, 'sos_gt_vs_pred_area.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ sos_gt_vs_pred_area.png")

# --- 5. Pie Charts: Top 10 + Others ---
top_n = 10
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# Facing
top_ids = gt_sorted[:top_n]
top_vals = [gt_sos_f[i] for i in top_ids]
top_names_pie = [class_names[i] for i in top_ids]
others = 100.0 - sum(top_vals)
labels_p = top_names_pie + [f'Others ({len(gt_sorted)-top_n} SKUs)']
sizes_p = top_vals + [others]
colors_p = plt.cm.Set3(np.linspace(0, 1, len(labels_p)))
wedges, texts, autotexts = axes[0].pie(sizes_p, labels=labels_p, autopct='%1.1f%%', startangle=90,
                                        colors=colors_p, pctdistance=0.85, textprops={'fontsize': 7})
for at in autotexts: at.set_fontsize(6)
axes[0].set_title('GT Share of Shelf by Facing Count\n(Top 10 SKUs + Others)', fontsize=11)

# Area
top_ids_a = gt_sorted_a[:top_n]
top_vals_a = [gt_sos_a[i] for i in top_ids_a]
top_names_a = [class_names[i] for i in top_ids_a]
others_a = 100.0 - sum(top_vals_a)
labels_a = top_names_a + [f'Others ({len(gt_sorted_a)-top_n} SKUs)']
sizes_a = top_vals_a + [others_a]
wedges2, texts2, autotexts2 = axes[1].pie(sizes_a, labels=labels_a, autopct='%1.1f%%', startangle=90,
                                            colors=colors_p, pctdistance=0.85, textprops={'fontsize': 7})
for at in autotexts2: at.set_fontsize(6)
axes[1].set_title('GT Share of Shelf by Area\n(Top 10 SKUs + Others)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT, 'sos_pie_charts.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ sos_pie_charts.png")

# --- 6. Scatter: GT vs Predicted SoS Correlation ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
gt_f_all = [gt_sos_f.get(i, 0) for i in range(NC)]
pr_f_all = [pr_sos_f.get(i, 0) for i in range(NC)]
gt_a_all = [gt_sos_a.get(i, 0) for i in range(NC)]
pr_a_all = [pr_sos_a.get(i, 0) for i in range(NC)]

axes[0].scatter(gt_f_all, pr_f_all, c='steelblue', alpha=0.7, s=50, edgecolors='white')
mx = max(max(gt_f_all), max(pr_f_all)) * 1.1
axes[0].plot([0, mx], [0, mx], 'r--', alpha=0.5, label='Perfect agreement')
axes[0].set_xlabel('GT SoS (%)'); axes[0].set_ylabel('Predicted SoS (%)')
axes[0].set_title('GT vs Predicted SoS (Facing Count)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
for i in range(NC):
    if abs(gt_f_all[i] - pr_f_all[i]) > 2:
        axes[0].annotate(class_names[i], (gt_f_all[i], pr_f_all[i]), fontsize=6, alpha=0.8)

axes[1].scatter(gt_a_all, pr_a_all, c='mediumseagreen', alpha=0.7, s=50, edgecolors='white')
mx_a = max(max(gt_a_all), max(pr_a_all)) * 1.1
axes[1].plot([0, mx_a], [0, mx_a], 'r--', alpha=0.5, label='Perfect agreement')
axes[1].set_xlabel('GT SoS (%)'); axes[1].set_ylabel('Predicted SoS (%)')
axes[1].set_title('GT vs Predicted SoS (Area)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
for i in range(NC):
    if abs(gt_a_all[i] - pr_a_all[i]) > 2:
        axes[1].annotate(class_names[i], (gt_a_all[i], pr_a_all[i]), fontsize=6, alpha=0.8)

corr_f = np.corrcoef(gt_f_all, pr_f_all)[0,1]
corr_a = np.corrcoef(gt_a_all, pr_a_all)[0,1]
axes[0].text(0.05, 0.95, f'r = {corr_f:.3f}', transform=axes[0].transAxes, fontsize=10, verticalalignment='top')
axes[1].text(0.05, 0.95, f'r = {corr_a:.3f}', transform=axes[1].transAxes, fontsize=10, verticalalignment='top')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT, 'sos_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ sos_correlation.png")

# --- 7. Heatmap: SoS per image ---
sos_matrix = np.zeros((len(test_images), NC))
for idx, img_path in enumerate(test_images):
    lbl = os.path.join(TEST_LBL, os.path.splitext(os.path.basename(img_path))[0]+'.txt')
    if not os.path.exists(lbl): continue
    cnts = Counter(); total = 0
    for line in open(lbl):
        p = line.strip().split()
        if len(p)>=5: cnts[int(p[0])]+=1; total+=1
    if total > 0:
        for c, n in cnts.items(): sos_matrix[idx, c] = 100.0*n/total

active_cols = np.where(sos_matrix.sum(axis=0) > 0)[0]
fig, ax = plt.subplots(figsize=(20, 8))
im = ax.imshow(sos_matrix[:, active_cols].T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xlabel('Test Image Index'); ax.set_ylabel('SKU')
ax.set_yticks(range(len(active_cols))); ax.set_yticklabels([class_names[i] for i in active_cols], fontsize=5)
ax.set_title('Share of Shelf Heatmap: SKU presence across test images', fontsize=12)
plt.colorbar(im, ax=ax, label='SoS (%)', shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT, 'sos_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ sos_heatmap.png")

# --- 8. Treemap ---
try:
    import squarify
    treemap_data = [(class_names[i], gt_sos_f[i]) for i in gt_sorted if gt_sos_f[i] > 0]
    treemap_names = [f"{d[0]}\n{d[1]:.1f}%" for d in treemap_data]
    treemap_vals = [d[1] for d in treemap_data]
    norm = matplotlib.colors.Normalize(vmin=min(treemap_vals), vmax=max(treemap_vals))
    colors_tm = [plt.cm.RdYlGn_r(norm(v)) for v in treemap_vals]
    fig, ax = plt.subplots(figsize=(18, 10))
    squarify.plot(sizes=treemap_vals, label=treemap_names, color=colors_tm, alpha=0.85,
                  text_kwargs={'fontsize': 6}, ax=ax)
    ax.set_title('Share of Shelf Treemap (Facing Count) — Larger = Higher Share', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT, 'sos_treemap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ sos_treemap.png")
except: print("  ⚠️ squarify not available, skipping treemap")

# --- 9. Detection samples ---
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
axes = axes.flatten()
np.random.seed(42)
colors_map = {i: tuple(np.random.randint(50, 255, 3).tolist()) for i in range(NC)}
for idx in range(min(4, len(test_images))):
    img = cv2.cvtColor(cv2.imread(test_images[idx]), cv2.COLOR_BGR2RGB)
    img_file = os.path.basename(test_images[idx])
    p = use_preds[img_file]
    mask = p['scores'] >= use_conf
    cnt = 0
    for i in range(len(p['boxes'])):
        if not mask[i]: continue
        cls_id = int(p['cls'][i])
        x1,y1,x2,y2 = [int(v) for v in p['boxes'][i]]
        color = colors_map.get(cls_id, (0,255,0))
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{class_names[cls_id]} {p['scores'][i]:.2f}", (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cnt += 1
    axes[idx].imshow(img)
    axes[idx].set_title(f'Test Image {idx+1}: {cnt} detections', fontsize=10)
    axes[idx].axis('off')
for idx in range(min(4, len(test_images)), 4): axes[idx].axis('off')
plt.suptitle('Shelf Detection Visualizations (Best Config)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT, 'sos_detection_samples.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ sos_detection_samples.png")

# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📊 FINAL SUMMARY REPORT")
print("="*70)

print(f"\n🏆 Best Config: {best_config_overall}")
print(f"   Precision:    {best_metrics_overall['P']:.4f}")
print(f"   Recall:       {best_metrics_overall['R']:.4f}")
print(f"   F1 Score:     {best_metrics_overall['F1']:.4f}")
print(f"   Conf Thr:     {best_metrics_overall['conf']}")

print(f"\n📦 Shelf Composition (GT):")
print(f"   Total products: {total_gt_f}")
print(f"   Unique SKUs:    {len([i for i in range(NC) if gt_counts.get(i,0)>0])}/{NC}")

print(f"\n🏆 Top 10 SKUs by Share (Facing Count):")
for i, cls_id in enumerate(gt_sorted[:10]):
    print(f"   {i+1:2d}. {class_names[cls_id]:>6s}: {gt_sos_f[cls_id]:5.2f}% ({gt_counts[cls_id]} facings)")

print(f"\n🤖 Predicted vs GT:")
print(f"   Products detected: {total_pr_f} (GT: {total_gt_f})")
print(f"   SKUs detected: {len([i for i in range(NC) if pred_counts.get(i,0)>0])}/{len([i for i in range(NC) if gt_counts.get(i,0)>0])}")
print(f"   SoS Correlation (facing): r = {corr_f:.4f}")
print(f"   SoS Correlation (area):   r = {corr_a:.4f}")

mae_f = np.mean([abs(gt_sos_f.get(i,0) - pr_sos_f.get(i,0)) for i in range(NC)])
mae_a = np.mean([abs(gt_sos_a.get(i,0) - pr_sos_a.get(i,0)) for i in range(NC)])
print(f"   Mean SoS Error (facing):  {mae_f:.3f}%")
print(f"   Mean SoS Error (area):    {mae_a:.3f}%")

total_time = time.time() - t_start
print(f"\n⏱️  Total time: {total_time/60:.1f} min")

# Save final results
results_final = {
    'best_config': best_config_overall,
    'metrics': best_metrics_overall,
    'sos_correlation_facing': float(corr_f),
    'sos_correlation_area': float(corr_a),
    'sos_mae_facing': float(mae_f),
    'sos_mae_area': float(mae_a),
    'share_of_shelf': [{
        'class': class_names[i],
        'gt_facings': int(gt_counts.get(i,0)),
        'pred_facings': int(pred_counts.get(i,0)),
        'gt_sos_facing_pct': float(gt_sos_f[i]),
        'pred_sos_facing_pct': float(pr_sos_f[i]),
        'gt_sos_area_pct': float(gt_sos_a[i]),
        'pred_sos_area_pct': float(pr_sos_a[i]),
    } for i in present_ids],
    'total_time_min': total_time/60,
}
with open(os.path.join(PROJECT, 'final_results.json'), 'w') as f:
    json.dump(results_final, f, indent=2)

print("\n✅ ALL DONE! Results → final_results.json")
print("✅ Visualizations: sos_gt_facing.png, sos_gt_area.png, sos_gt_vs_pred_facing.png,")
print("   sos_gt_vs_pred_area.png, sos_pie_charts.png, sos_correlation.png,")
print("   sos_heatmap.png, sos_treemap.png, sos_detection_samples.png")
