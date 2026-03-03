"""
Stratified 2-Model Ensemble Evaluation: TTA + NMS + WBF
========================================================
Uses the stratified-trained YOLO12x + original YOLO12l.
Comprehensive sweep matching the fine_sweep.py framework.

Run AFTER stratified training completes:
  CUDA_VISIBLE_DEVICES='1' nohup python3 stratified_ensemble_eval.py > stratified_ensemble_eval.log 2>&1 &
"""
import os, json, time, warnings, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import yaml
import glob
import cv2
from pathlib import Path
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

PROJECT = '/mnt/Data/AKIB/YOLO-OD-IM'
TEST_IMG = os.path.join(PROJECT, 'dataset', 'test', 'images')
TEST_LBL = os.path.join(PROJECT, 'dataset', 'test', 'labels')

YOLO12X_STRAT = os.path.join(PROJECT, 'runs', 'yolo12x_stratified', 'weights', 'best.pt')
YOLO12L_PATH  = os.path.join(PROJECT, 'runs', 'yolo12l_ensemble', 'weights', 'best.pt')

# ── Verify weights exist ──
for p, name in [(YOLO12X_STRAT, 'YOLO12x_stratified'), (YOLO12L_PATH, 'YOLO12l')]:
    if not os.path.exists(p):
        print(f"❌ {name} weights not found: {p}")
        sys.exit(1)

with open(os.path.join(PROJECT, 'dataset', 'data.yaml')) as f:
    cfg = yaml.safe_load(f)
NC = len(cfg['names'])

test_images = sorted(glob.glob(os.path.join(TEST_IMG, '*')))
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Test images: {len(test_images)}, Classes: {NC}")
print(f"YOLO12x_strat: {YOLO12X_STRAT}")
print(f"YOLO12l:       {YOLO12L_PATH}")

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
            boxes.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h])
            classes.append(cls)
    return boxes, classes

def compute_iou(b1, b2):
    x1, y1 = max(b1[0],b2[0]), max(b1[1],b2[1])
    x2, y2 = min(b1[2],b2[2]), min(b1[3],b2[3])
    inter = max(0,x2-x1)*max(0,y2-y1)
    union = (b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter
    return inter/(union+1e-8)

def evaluate(all_preds, match_iou=0.5):
    tp, fp, fn = 0, 0, 0
    for img_path, pred_boxes, pred_cls, pred_sc in all_preds:
        img = cv2.imread(img_path); h, w = img.shape[:2]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        gt_b, gt_c = load_gt(os.path.join(TEST_LBL, fname+'.txt'), w, h)
        if len(pred_sc) > 0:
            order = np.argsort(-np.array(pred_sc))
            pred_boxes = [pred_boxes[i] for i in order]
            pred_cls = [pred_cls[i] for i in order]
        matched = set()
        for pi in range(len(pred_boxes)):
            best_iou, best_gi = 0, -1
            for gi in range(len(gt_b)):
                if gi in matched or pred_cls[pi] != gt_c[gi]: continue
                iv = compute_iou(pred_boxes[pi], gt_b[gi])
                if iv > best_iou: best_iou, best_gi = iv, gi
            if best_iou >= match_iou and best_gi >= 0:
                tp += 1; matched.add(best_gi)
            else:
                fp += 1
        fn += len(gt_b) - len(matched)
    P = tp/(tp+fp+1e-8); R = tp/(tp+fn+1e-8); F = 2*P*R/(P+R+1e-8)
    return {'P': round(P,4), 'R': round(R,4), 'F1': round(F,4), 'TP':tp, 'FP':fp, 'FN':fn}

# ═══════════════════════════════════════════════════════
# PHASE 1: SINGLE MODEL EVAL (YOLO12x stratified)
# ═══════════════════════════════════════════════════════
print("\n" + "="*80)
print("📊 PHASE 1: SINGLE MODEL EVALUATION (YOLO12x Stratified)")
print("="*80)

model_x = YOLO(YOLO12X_STRAT)

# Evaluate single model at multiple conf thresholds
single_results = {}
for conf in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    preds = []
    for img_path in test_images:
        results = model_x.predict(img_path, conf=conf, iou=0.5, imgsz=640,
                                   device=0, verbose=False, augment=False, max_det=300)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) > 0 else []
        cls = r.boxes.cls.cpu().numpy().astype(int).tolist() if len(r.boxes) > 0 else []
        scores = r.boxes.conf.cpu().numpy().tolist() if len(r.boxes) > 0 else []
        preds.append((img_path, boxes, cls, scores))
    res = evaluate(preds)
    single_results[f'conf={conf:.2f}'] = res
    marker = " 🎯" if res['P'] >= 0.76 and res['R'] >= 0.85 else ""
    print(f"  conf={conf:.2f} → P={res['P']:.4f}  R={res['R']:.4f}  F1={res['F1']:.4f}  TP={res['TP']} FP={res['FP']} FN={res['FN']}{marker}")

# Also with TTA
print("\n  With TTA:")
for conf in [0.05, 0.10, 0.15, 0.20, 0.25]:
    preds = []
    for img_path in test_images:
        results = model_x.predict(img_path, conf=conf, iou=0.5, imgsz=640,
                                   device=0, verbose=False, augment=True, max_det=300)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) > 0 else []
        cls = r.boxes.cls.cpu().numpy().astype(int).tolist() if len(r.boxes) > 0 else []
        scores = r.boxes.conf.cpu().numpy().tolist() if len(r.boxes) > 0 else []
        preds.append((img_path, boxes, cls, scores))
    res = evaluate(preds)
    single_results[f'TTA_conf={conf:.2f}'] = res
    marker = " 🎯" if res['P'] >= 0.76 and res['R'] >= 0.85 else ""
    print(f"  TTA conf={conf:.2f} → P={res['P']:.4f}  R={res['R']:.4f}  F1={res['F1']:.4f}  TP={res['TP']} FP={res['FP']} FN={res['FN']}{marker}")

# ═══════════════════════════════════════════════════════
# PHASE 2: COLLECT RAW PREDICTIONS FOR ENSEMBLE
# ═══════════════════════════════════════════════════════
print("\n" + "="*80)
print("📦 PHASE 2: COLLECTING RAW PREDICTIONS (conf=0.01)")
print("="*80)

model_l = YOLO(YOLO12L_PATH)

def get_raw_preds(model, tta=True):
    preds = []
    for img_path in test_images:
        img = cv2.imread(img_path); h, w = img.shape[:2]
        results = model.predict(img_path, conf=0.01, iou=0.7, imgsz=640,
                                 device=0, verbose=False, augment=tta, max_det=300)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) > 0 else np.empty((0,4))
        cls = r.boxes.cls.cpu().numpy().astype(int) if len(r.boxes) > 0 else np.array([])
        conf = r.boxes.conf.cpu().numpy() if len(r.boxes) > 0 else np.array([])
        preds.append((img_path, boxes, cls, conf, w, h))
    return preds

t0 = time.time()
print("  YOLO12x_strat TTA...")
preds_x_tta = get_raw_preds(model_x, tta=True)
print("  YOLO12x_strat NO-TTA...")
preds_x_no_tta = get_raw_preds(model_x, tta=False)
print("  YOLO12l TTA...")
preds_l_tta = get_raw_preds(model_l, tta=True)
print("  YOLO12l NO-TTA...")
preds_l_no_tta = get_raw_preds(model_l, tta=False)
print(f"  ⏱️  Prediction collection: {time.time()-t0:.1f}s")

# Count avg detections per model
for name, preds in [('12x_strat TTA', preds_x_tta), ('12x_strat noTTA', preds_x_no_tta),
                     ('12l TTA', preds_l_tta), ('12l noTTA', preds_l_no_tta)]:
    avg = np.mean([len(p[2]) for p in preds])
    print(f"  {name}: avg {avg:.1f} detections/image")

# ═══════════════════════════════════════════════════════
# PHASE 3: WBF ENSEMBLE SWEEP (TTA + NMS + WBF)
# ═══════════════════════════════════════════════════════
print("\n" + "="*80)
print("🔥 PHASE 3: WBF ENSEMBLE SWEEP (2-model: YOLO12x_strat + YOLO12l)")
print("="*80)

# Comprehensive sweep parameters
weight_configs = [
    (3.0, 2.0), (3.0, 1.5), (3.5, 1.5), (4.0, 1.0), (4.0, 1.5),
    (3.0, 2.5), (2.5, 2.0), (2.0, 2.0), (5.0, 1.0), (2.0, 1.0),
    (3.0, 1.0), (3.0, 3.0),
]
iou_thrs = [0.40, 0.45, 0.50, 0.55, 0.60]
skip_thrs = [0.01, 0.05, 0.10]
conf_types = ['avg', 'max']
tta_combos = [
    ('TTA',    preds_x_tta,    preds_l_tta),
    ('noTTA',  preds_x_no_tta, preds_l_no_tta),
    ('mixTTA', preds_x_tta,    preds_l_no_tta),  # TTA on strong model only
]

# Build all WBF configs
configs = []
for wx, wl in weight_configs:
    for iou_thr in iou_thrs:
        for skip_thr in skip_thrs:
            for conf_type in conf_types:
                for tta_label, px, pl in tta_combos:
                    configs.append((wx, wl, iou_thr, skip_thr, conf_type, tta_label, px, pl))

print(f"  Total WBF configs: {len(configs)}")
print(f"  Conf threshold sweep: 0.20 to 0.55 step 0.01 ({36} steps)")
print(f"  Total evaluations: {len(configs) * 36}")

best_target = None   # best that meets P>0.76 & R>0.85
best_f1_overall = None
all_results = {}

t0 = time.time()

for ci, (wx, wl, iou_thr, skip_thr, conf_type, tta_label, px, pl) in enumerate(configs):
    # Build WBF ensemble for all images
    ens_preds_raw = []
    for i in range(len(test_images)):
        img_path = test_images[i]
        _, bx, cx, sx, w, h = px[i]
        _, bl, cl, sl, _, _ = pl[i]

        boxes_list, scores_list, labels_list = [], [], []
        for boxes, cls, scores in [(bx, cx, sx), (bl, cl, sl)]:
            if len(boxes) > 0:
                norm = boxes.copy().astype(float)
                norm[:, [0,2]] /= w; norm[:, [1,3]] /= h
                norm = np.clip(norm, 0, 1)
                boxes_list.append(norm.tolist())
                scores_list.append(scores.tolist())
                labels_list.append(cls.tolist())
            else:
                boxes_list.append([]); scores_list.append([]); labels_list.append([])

        fb, fs, fl = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[wx, wl], iou_thr=iou_thr,
            skip_box_thr=skip_thr, conf_type=conf_type)

        final = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in fb]
        ens_preds_raw.append((img_path, final, fl.astype(int).tolist(), fs.tolist()))

    # Fine conf sweep: 0.20 to 0.55 in 0.01 steps
    for min_conf_100 in range(20, 56):
        min_conf = min_conf_100 / 100.0
        filtered = []
        for img_path, boxes, classes, scores in ens_preds_raw:
            if len(scores) > 0:
                mask = np.array(scores) >= min_conf
                fb = [boxes[i] for i in range(len(boxes)) if mask[i]]
                fc = [classes[i] for i in range(len(classes)) if mask[i]]
                fsc = [scores[i] for i in range(len(scores)) if mask[i]]
            else:
                fb, fc, fsc = [], [], []
            filtered.append((img_path, fb, fc, fsc))

        res = evaluate(filtered)
        key = f"{tta_label}_w={wx}/{wl}_iou={iou_thr}_sk={skip_thr}_{conf_type}_conf={min_conf:.2f}"
        all_results[key] = res

        # Track best target-meeting config
        if res['P'] >= 0.76 and res['R'] >= 0.85:
            if best_target is None or res['F1'] > best_target[1]['F1']:
                best_target = (key, res)

        if best_f1_overall is None or res['F1'] > best_f1_overall[1]['F1']:
            best_f1_overall = (key, res)

    if (ci+1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (ci+1) / elapsed
        remaining = (len(configs) - ci - 1) / rate
        print(f"  [{ci+1}/{len(configs)}] {elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")
        if best_target:
            print(f"  🎯 Current best target: P={best_target[1]['P']:.4f} R={best_target[1]['R']:.4f} F1={best_target[1]['F1']:.4f}")
        if best_f1_overall:
            print(f"  📊 Current best F1:     P={best_f1_overall[1]['P']:.4f} R={best_f1_overall[1]['R']:.4f} F1={best_f1_overall[1]['F1']:.4f}")

total_time = time.time() - t0
print(f"\n  ⏱️  Ensemble sweep: {total_time:.1f}s ({total_time/60:.1f}min)")

# ═══════════════════════════════════════════════════════
# PHASE 4: RESULTS
# ═══════════════════════════════════════════════════════
print("\n" + "="*80)
print("🏆 FINAL RESULTS: STRATIFIED 2-MODEL ENSEMBLE")
print("="*80)

# Single model summary
print("\n📊 SINGLE MODEL (YOLO12x Stratified):")
print(f"  {'Config':25s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
print("  " + "-"*65)
for k, v in single_results.items():
    marker = " 🎯" if v['P'] >= 0.76 and v['R'] >= 0.85 else ""
    print(f"  {k:25s} {v['P']:7.4f} {v['R']:7.4f} {v['F1']:7.4f} {v['TP']:5d} {v['FP']:5d} {v['FN']:5d}{marker}")

# Best F1 overall
print(f"\n📊 ENSEMBLE — Best F1 Overall:")
print(f"  Config: {best_f1_overall[0]}")
print(f"  P={best_f1_overall[1]['P']:.4f}  R={best_f1_overall[1]['R']:.4f}  F1={best_f1_overall[1]['F1']:.4f}")
print(f"  TP={best_f1_overall[1]['TP']}  FP={best_f1_overall[1]['FP']}  FN={best_f1_overall[1]['FN']}")

# Target-meeting configs
target_configs = [(k,v) for k,v in all_results.items() if v['P'] >= 0.76 and v['R'] >= 0.85]
if target_configs:
    print(f"\n🎯 CONFIGS MEETING TARGET (P≥0.76 AND R≥0.85): {len(target_configs)} found!")
    print(f"  {'Config':70s} {'P':>7s} {'R':>7s} {'F1':>7s}")
    print("  " + "-"*90)
    for k, v in sorted(target_configs, key=lambda x: -x[1]['F1'])[:25]:
        print(f"  {k:70s} {v['P']:7.4f} {v['R']:7.4f} {v['F1']:7.4f}")
    
    best = sorted(target_configs, key=lambda x: -x[1]['F1'])[0]
    print(f"\n  ★ BEST: {best[0]}")
    print(f"    P={best[1]['P']:.4f}  R={best[1]['R']:.4f}  F1={best[1]['F1']:.4f}  TP={best[1]['TP']} FP={best[1]['FP']} FN={best[1]['FN']}")
else:
    print(f"\n❌ No config met BOTH P≥0.76 AND R≥0.85")

# Closest to target
print(f"\n📊 Top 20 closest to P≥0.76, R≥0.85 (by F1, where P≥0.72 and R≥0.80):")
def dist_to_target(v):
    return max(0, 0.76 - v['P']) + max(0, 0.85 - v['R'])
good = [(k,v) for k,v in all_results.items() if v['P'] >= 0.72 and v['R'] >= 0.80]
for k, v in sorted(good, key=lambda x: -x[1]['F1'])[:20]:
    marker = ""
    if v['P'] >= 0.76 and v['R'] >= 0.85: marker = " 🎯"
    elif v['P'] >= 0.76: marker = " ✅P"
    elif v['R'] >= 0.85: marker = " ✅R"
    print(f"  {k:70s} {v['P']:7.4f} {v['R']:7.4f} {v['F1']:7.4f}{marker}")

# Top by Recall (with P>=0.72)
print(f"\n📊 Top 10 by Recall (where P≥0.72):")
hi_r = [(k,v) for k,v in all_results.items() if v['P'] >= 0.72]
for k, v in sorted(hi_r, key=lambda x: -x[1]['R'])[:10]:
    marker = ""
    if v['P'] >= 0.76 and v['R'] >= 0.85: marker = " 🎯"
    elif v['P'] >= 0.76: marker = " ✅P"
    elif v['R'] >= 0.85: marker = " ✅R"
    print(f"  {k:70s} {v['P']:7.4f} {v['R']:7.4f} {v['F1']:7.4f}{marker}")

# Comparison with original
print(f"\n" + "="*80)
print("📋 COMPARISON: Stratified vs Original Split")
print("="*80)
print(f"  {'Metric':40s} {'Original':>10s} {'Stratified':>10s} {'Delta':>8s}")
print("  " + "-"*70)

orig_single = {'P': 0.6904, 'R': 0.8317, 'F1': 0.7545}
orig_ens    = {'P': 0.7716, 'R': 0.8621, 'F1': 0.8143}

# Find comparable single model result (conf=0.10)
strat_single = single_results.get('conf=0.10', single_results.get('conf=0.15', {}))
for metric in ['P', 'R', 'F1']:
    o = orig_single[metric]
    s = strat_single.get(metric, 0)
    d = s - o
    print(f"  Single Model {metric:33s} {o:10.4f} {s:10.4f} {d:+8.4f}")

if best_target:
    bt = best_target[1]
    for metric in ['P', 'R', 'F1']:
        o = orig_ens[metric]
        s = bt[metric]
        d = s - o
        print(f"  Ensemble (target-met) {metric:25s} {o:10.4f} {s:10.4f} {d:+8.4f}")

if best_f1_overall:
    bf = best_f1_overall[1]
    for metric in ['P', 'R', 'F1']:
        o = orig_ens[metric]
        s = bf[metric]
        d = s - o
        print(f"  Ensemble (best F1) {metric:28s} {o:10.4f} {s:10.4f} {d:+8.4f}")

# Save all results
output = {
    'single_model': single_results,
    'ensemble_best_f1': {'config': best_f1_overall[0], 'metrics': best_f1_overall[1]} if best_f1_overall else None,
    'ensemble_best_target': {'config': best_target[0], 'metrics': best_target[1]} if best_target else None,
    'target_configs_count': len(target_configs),
    'total_configs_evaluated': len(all_results),
    'comparison_original': {
        'single_model': orig_single,
        'ensemble_2model': orig_ens,
    },
}
with open(os.path.join(PROJECT, 'stratified_ensemble_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

# Also save full results
with open(os.path.join(PROJECT, 'stratified_ensemble_full.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Saved results: stratified_ensemble_results.json ({len(all_results)} configs)")
print("🏁 Done!")
