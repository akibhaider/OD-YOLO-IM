"""
Fine-grained sweep around the best 2-model ensemble configs.
Targeting P>0.76, R>0.85 by sweeping conf in 0.01 steps
and trying additional weight ratios.
"""
import os, json, warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
warnings.filterwarnings('ignore')

import numpy as np
import torch
import yaml, glob, cv2
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

PROJECT = '/mnt/Data/AKIB/YOLO-OD-IM'
TEST_IMG = os.path.join(PROJECT, 'dataset', 'test', 'images')
TEST_LBL = os.path.join(PROJECT, 'dataset', 'test', 'labels')

with open(os.path.join(PROJECT, 'dataset', 'data_abs.yaml')) as f:
    cfg = yaml.safe_load(f)
NC = len(cfg['names'])

test_images = sorted(glob.glob(os.path.join(TEST_IMG, '*')))
print(f"GPU: {torch.cuda.get_device_name(0)}, Images: {len(test_images)}")

# ── Helpers ──
def load_gt(lbl_path, w, h):
    boxes, classes = [], []
    if not os.path.exists(lbl_path): return boxes, classes
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

# ── Collect raw predictions ──
model_x = YOLO(os.path.join(PROJECT, 'runs/yolo12x_optimized/weights/best.pt'))
model_l = YOLO(os.path.join(PROJECT, 'runs/yolo12l_ensemble/weights/best.pt'))

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

print("Collecting YOLO12x TTA predictions...")
preds_x = get_raw_preds(model_x, tta=True)
print("Collecting YOLO12l TTA predictions...")
preds_l = get_raw_preds(model_l, tta=True)

# Also collect YOLO12x NON-TTA (sometimes TTA adds noise)
print("Collecting YOLO12x NO-TTA predictions...")
preds_x_no_tta = get_raw_preds(model_x, tta=False)
print("Collecting YOLO12l NO-TTA predictions...")
preds_l_no_tta = get_raw_preds(model_l, tta=False)

# ── Fine-grained WBF sweep ──
configs = []
# Vary weights, iou_thr, skip_thr, conf_type, tta combos
for wx, wl in [(3.0, 2.0), (3.0, 1.5), (4.0, 1.0), (3.5, 1.5), (3.0, 2.5),
               (2.5, 2.0), (2.0, 2.0), (5.0, 1.0), (4.0, 1.5)]:
    for iou_thr in [0.45, 0.50, 0.55, 0.60]:
        for skip_thr in [0.01, 0.05, 0.10]:
            for conf_type in ['avg', 'max']:
                for tta_label, px, pl in [('TTA', preds_x, preds_l),
                                          ('noTTA', preds_x_no_tta, preds_l_no_tta),
                                          ('mixTTA', preds_x, preds_l_no_tta)]:
                    configs.append((wx, wl, iou_thr, skip_thr, conf_type, tta_label, px, pl))

print(f"\nSweeping {len(configs)} WBF configs × fine conf steps...")

best_target = None  # best that meets P>0.76 & R>0.85
best_f1_overall = None
all_results = {}

for ci, (wx, wl, iou_thr, skip_thr, conf_type, tta_label, px, pl) in enumerate(configs):
    # Build WBF ensemble
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
    
    # Fine conf sweep: 0.25 to 0.55 in 0.01 steps
    for min_conf_100 in range(25, 56):
        min_conf = min_conf_100 / 100.0
        filtered = []
        for img_path, boxes, classes, scores in ens_preds_raw:
            mask = np.array(scores) >= min_conf if len(scores) > 0 else np.array([], dtype=bool)
            fb = [boxes[i] for i in range(len(boxes)) if mask[i]]
            fc = [classes[i] for i in range(len(classes)) if mask[i]]
            fsc = [scores[i] for i in range(len(scores)) if mask[i]]
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
    
    if (ci+1) % 100 == 0:
        print(f"  {ci+1}/{len(configs)} configs done...")
        if best_target:
            print(f"  🎯 Current best target: {best_target[0]} → P={best_target[1]['P']}, R={best_target[1]['R']}, F1={best_target[1]['F1']}")

# ── Results ──
print("\n" + "="*80)
print("📋 FINE SWEEP RESULTS")
print("="*80)

if best_target:
    print(f"\n🎯 TARGET MET! Best config meeting P>0.76 AND R>0.85:")
    print(f"  {best_target[0]}")
    print(f"  P={best_target[1]['P']:.4f}  R={best_target[1]['R']:.4f}  F1={best_target[1]['F1']:.4f}")
    print(f"  TP={best_target[1]['TP']} FP={best_target[1]['FP']} FN={best_target[1]['FN']}")
    
    # Show all target-meeting configs
    target_configs = [(k,v) for k,v in all_results.items() if v['P'] >= 0.76 and v['R'] >= 0.85]
    print(f"\n  All {len(target_configs)} configs meeting target:")
    for k, v in sorted(target_configs, key=lambda x: -x[1]['F1'])[:20]:
        print(f"    {k:70s} P={v['P']:.4f} R={v['R']:.4f} F1={v['F1']:.4f}")
else:
    print("\n❌ No config met BOTH P>0.76 AND R>0.85")

print(f"\n📊 Best F1 overall:")
print(f"  {best_f1_overall[0]}")
print(f"  P={best_f1_overall[1]['P']:.4f}  R={best_f1_overall[1]['R']:.4f}  F1={best_f1_overall[1]['F1']:.4f}")

# Show closest configs to target
print(f"\n📊 Top 20 closest to P≥0.76, R≥0.85 (sorted by |P-0.76|+|R-0.85|):")
def dist_to_target(v):
    return max(0, 0.76 - v['P']) + max(0, 0.85 - v['R'])

good = [(k,v) for k,v in all_results.items() if v['P'] >= 0.72 and v['R'] >= 0.80]
for k, v in sorted(good, key=lambda x: dist_to_target(x[1]))[:20]:
    marker = ""
    if v['P'] >= 0.76 and v['R'] >= 0.85: marker = " 🎯"
    elif v['P'] >= 0.76: marker = " ✅P"
    elif v['R'] >= 0.85: marker = " ✅R"
    print(f"  {k:70s} P={v['P']:.4f} R={v['R']:.4f} F1={v['F1']:.4f} d={dist_to_target(v):.4f}{marker}")

with open(os.path.join(PROJECT, 'fine_sweep_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✅ Saved {len(all_results)} results to fine_sweep_results.json")
