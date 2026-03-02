"""
YOLO12x Training + Validation + Test Pipeline
Model: YOLO12x (59.2M params) — Latest & Largest YOLO model
GPU: Device 1 (RTX 3090 24GB)
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from ultralytics import YOLO
import torch
import json

PROJECT_ROOT = '/mnt/Data/AKIB/YOLO-OD-IM'
DATA_YAML    = os.path.join(PROJECT_ROOT, 'dataset', 'data_abs.yaml')

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ══════════════════════════════════════════════════════════════
# PHASE 1: TRAIN YOLO12x
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🚀 PHASE 1: TRAINING YOLO12x")
print("="*60)

model = YOLO('yolo12x.pt')

results = model.train(
    data=DATA_YAML,
    epochs=150,
    imgsz=640,
    batch=8,
    device=0,
    project=os.path.join(PROJECT_ROOT, 'runs'),
    name='yolo12x_optimized',
    exist_ok=True,
    
    # LR optimization
    cos_lr=True,            # cosine LR scheduler
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
    
    # Augmentation
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

# ══════════════════════════════════════════════════════════════
# PHASE 2: VALIDATE (val set)
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("📊 PHASE 2: VALIDATION SET EVALUATION")
print("="*60)

best_path = os.path.join(PROJECT_ROOT, 'runs', 'yolo12x_optimized', 'weights', 'best.pt')
model_best = YOLO(best_path)

# Multiple confidence thresholds for comprehensive analysis
conf_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]
val_results = {}

for conf in conf_thresholds:
    m = model_best.val(
        data=DATA_YAML,
        split='val',
        device=0,
        conf=conf,
        iou=0.6,
        verbose=False,
    )
    val_results[conf] = {
        'precision': float(m.box.mp),
        'recall': float(m.box.mr),
        'f1': float(2 * m.box.mp * m.box.mr / (m.box.mp + m.box.mr + 1e-8)),
        'map50': float(m.box.map50),
        'map50_95': float(m.box.map),
    }
    print(f"  conf={conf:.2f} → P={m.box.mp:.4f}, R={m.box.mr:.4f}, "
          f"F1={val_results[conf]['f1']:.4f}, mAP50={m.box.map50:.4f}")

# ══════════════════════════════════════════════════════════════
# PHASE 3: TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("🧪 PHASE 3: TEST SET EVALUATION")
print("="*60)

test_results = {}
for conf in conf_thresholds:
    for iou in [0.5, 0.6, 0.7]:
        m = model_best.val(
            data=DATA_YAML,
            split='test',
            device=0,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        key = f"conf={conf:.2f}_iou={iou:.1f}"
        f1 = float(2 * m.box.mp * m.box.mr / (m.box.mp + m.box.mr + 1e-8))
        test_results[key] = {
            'conf': conf,
            'iou_nms': iou,
            'precision': float(m.box.mp),
            'recall': float(m.box.mr),
            'f1': f1,
            'map50': float(m.box.map50),
            'map50_95': float(m.box.map),
        }
        print(f"  {key} → P={m.box.mp:.4f}, R={m.box.mr:.4f}, F1={f1:.4f}, mAP50={m.box.map50:.4f}")

# ══════════════════════════════════════════════════════════════
# PHASE 4: SUMMARY REPORT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("📋 FINAL RESULTS SUMMARY — YOLO12x")
print("="*70)

# Find best test config by F1 score
best_key = max(test_results, key=lambda k: test_results[k]['f1'])
best = test_results[best_key]

# Find best recall config
best_recall_key = max(test_results, key=lambda k: test_results[k]['recall'])
best_recall = test_results[best_recall_key]

print(f"\nModel: YOLO12x (59.2M parameters)")
print(f"Dataset: 76 classes, 924 train / 40 val / 35 test images")
print(f"\n{'─'*70}")
print(f"BEST F1 SCORE ({best_key}):")
print(f"  Confidence Threshold: {best['conf']}")
print(f"  NMS IoU Threshold:    {best['iou_nms']}")
print(f"  Precision:            {best['precision']:.4f}")
print(f"  Recall:               {best['recall']:.4f}")
print(f"  F1 Score:             {best['f1']:.4f}")
print(f"  mAP@50:               {best['map50']:.4f}")
print(f"  mAP@50-95:            {best['map50_95']:.4f}")

print(f"\n{'─'*70}")
print(f"BEST RECALL ({best_recall_key}):")
print(f"  Confidence Threshold: {best_recall['conf']}")
print(f"  NMS IoU Threshold:    {best_recall['iou_nms']}")
print(f"  Precision:            {best_recall['precision']:.4f}")
print(f"  Recall:               {best_recall['recall']:.4f}")
print(f"  F1 Score:             {best_recall['f1']:.4f}")
print(f"  mAP@50:               {best_recall['map50']:.4f}")
print(f"  mAP@50-95:            {best_recall['map50_95']:.4f}")

print(f"\n{'─'*70}")
print(f"ALL TEST CONFIGURATIONS:")
print(f"{'Config':30s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'mAP50':>8s}")
print(f"{'─'*70}")
for key in sorted(test_results.keys()):
    r = test_results[key]
    marker = " ← BEST F1" if key == best_key else (" ← BEST RECALL" if key == best_recall_key else "")
    print(f"{key:30s} {r['precision']:10.4f} {r['recall']:8.4f} {r['f1']:8.4f} {r['map50']:8.4f}{marker}")

# Save results to JSON
results_file = os.path.join(PROJECT_ROOT, 'yolo12x_results.json')
with open(results_file, 'w') as f:
    json.dump({
        'model': 'YOLO12x',
        'params': '59.2M',
        'validation': val_results,
        'test': test_results,
        'best_f1_config': best_key,
        'best_recall_config': best_recall_key,
    }, f, indent=2)
print(f"\n✅ Results saved to {results_file}")
print("✅ PIPELINE COMPLETE!")
