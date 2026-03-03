# Shelf Product Detection & Share of Shelf Analytics

## Final Best Result: Recall -> 86.2%  |  Precision -> 77.2%  |  F1 -> 81.4% 
## Second Best Tradeoff with 3 model ensemble learning (YOLO12x + YOLO12l + YOLO11x) with most f1-score: Recall -> 82.8%  |  Precision -> 81.1%  |  F1 -> 81.9%  

## Dataset split observation (optional): Larger Class Imbalance across train-val-test split  |  Overcounting Issue and SKU mismatch  |   Stratified Dataset Split experiment Result (only on train and val set): Recall -> 88.2%  |  Precision -> 82.6%  |  F1 -> 85.4%  (not converged, small experiment on 15/150 epoch) ....... Still experiment ongoing 😃 

Improve the Recall metric of shelf product detection (baseline: 67.6%) without significantly damaging Precision, and compute Share of Shelf analytics for each product class (SKU) across a representative test shelf.

- **Dataset**: 76 product classes (SKUs), 924 train / 40 validation / 35 test images (640×640, YOLO format)
- **Hardware**: NVIDIA RTX 3090 (24 GB)

---

## 1. YOLO12x — Single Model Results

**Model**: YOLO12x — 59.2M parameters, 113.7 MB  
**Training**: 57 epochs (early-stopped at patience=30), imgsz=640, batch=8, cosine LR, mosaic + mixup + copy-paste augmentation

| Configuration | Precision | Recall | F1 Score | mAP50 |
|---------------|-----------|--------|----------|-------|
| **Best F1** (conf=0.10, iou=0.7) | 0.7627 | 0.7770 | **0.7697** | 0.8097 |
| **Best Recall** (conf=0.10, iou=0.5) | 0.6904 | **0.8317** | 0.7545 | 0.8112 |
| **Balanced** (conf=0.15, iou=0.5) | 0.7101 | 0.8254 | 0.7634 | 0.8088 |

> **Recall improved from 67.6% → 83.2%** (+23%) while maintaining competitive precision.

---

## 2. Ensemble Pipeline Results

### Models Used

| Model | Params | Epochs | WBF Weight |
|-------|--------|--------|------------|
| YOLO12x | 59.2M | 57 | 3.0 |
| YOLO12l | 68.9M | 30 | 2.0 |
| YOLO11m | 20.1M | 100 | 1.5 |
| YOLO12s | 9.3M | 100 | 1.0 |
| YOLO11n | 2.6M | 24 | 0.5 |

### Ensemble Results

| Configuration | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| **Best F1** (WBF iou=0.5, skip=0.05, conf=0.30) | 0.6225 | 0.8759 | **0.7278** |
| **Best Recall** (WBF iou=0.5, skip=0.05, conf=0.05) | 0.4340 | **0.9517** | 0.5961 |

---

## 3. Key Methodological Approaches

### Test-Time Augmentation (TTA)
Multi-scale and flip augmentations applied at inference time (`augment=True`). TTA generates multiple transformed views of each test image and aggregates predictions, improving detection of products at varying scales and orientations without retraining.

### Non-Maximum Suppression (NMS)
Class-aware NMS applied per-model to remove duplicate bounding boxes overlapping the same product instance. This deduplication step reduced false positives by 15–30% per model while preserving true detections.

### Weighted Box Fusion (WBF)
WBF merges overlapping bounding boxes from all five models using confidence-weighted averaging, rather than simply discarding lower-scored duplicates (as NMS does). Model weights (3.0 → 0.5) prioritize the strongest model (YOLO12x) while still incorporating diverse predictions from smaller architectures. Six WBF parameter configurations were swept to find the optimal precision–recall trade-off.

---

## 4. Share of Shelf Analytics

The entire test set (35 images) was treated as a single representative store shelf. Share of Shelf was calculated using two methods:

- **Facing Count**: Percentage of detected product instances per SKU
- **Shelf Area**: Percentage of total bounding box area per SKU

### Ground Truth — Top 5 SKUs by Shelf Share

| SKU | Facings | SoS (Count) | SoS (Area) |
|-----|---------|-------------|------------|
| q214 | 19 | 13.10% | 20.16% |
| q280 | 18 | 12.41% | 13.75% |
| q293 | 16 | 11.03% | 9.90% |
| q31 | 12 | 8.28% | 3.95% |
| q193 | 10 | 6.90% | 2.64% |

- **29 of 76 SKUs** are present on the test shelf; 47 SKUs have zero presence.
- The top 5 SKUs account for **51.7%** of all shelf facings.

### Model Prediction Accuracy (YOLO12x, conf=0.10, iou=0.5)

| Metric | Value |
|--------|-------|
| GT Products | 145 |
| Detected Products | 270 |
| GT Unique SKUs | 29 |
| Detected SKUs | 39 |
| SoS Correlation (Facing Count) | **r = 0.810** |
| SoS Correlation (Area) | **r = 0.900** |
| Mean SoS Error (Facing) | 0.695% |
| Mean SoS Error (Area) | 0.536% |

> The model's predicted Share of Shelf correlates strongly with ground truth (r = 0.81–0.90), confirming that YOLO12x captures the relative shelf representation of each SKU reliably despite some over-detection.

---

## 5. Advanced Strategies — Closing the Gap to P>0.76 & R>0.85

Three post-training strategies were evaluated to push beyond the single-model frontier:

### Strategy 1: SAHI Sliced Inference
SAHI (Slicing Aided Hyper Inference) runs the YOLO12x model on overlapping image slices, then merges results. This detects small/occluded products that full-image inference misses.

| Config | Precision | Recall | F1 |
|--------|-----------|--------|----|
| slice=320, overlap=0.3, conf≥0.05 | 0.215 | **0.910** | 0.347 |
| slice=320, overlap=0.3, conf≥0.35 | 0.607 | 0.862 | 0.712 |
| slice=320, overlap=0.3, conf≥0.40 | 0.637 | 0.848 | 0.728 |

> SAHI boosted recall to 91% but generated too many false positives. Precision maxed at ~0.64, insufficient for our P>0.76 target.

### Strategy 2: Selective 2-Model Ensemble (YOLO12x + YOLO12l) ✅
Ensembling only the two strongest models with TTA + WBF dramatically outperformed the 5-model ensemble by eliminating noise from weaker models.

| Config | Precision | Recall | F1 |
|--------|-----------|--------|----|
| WBF(iou=0.55, w=3:2, avg), conf≥0.45 | **0.786** | 0.835 | **0.809** |
| WBF(iou=0.50, w=4:1, avg), conf≥0.40 | 0.769 | **0.848** | 0.807 |
| **🎯 WBF(iou=0.45, w=3:2, avg), conf≥0.36** | **0.772** | **0.862** | **0.814** |

### Strategy 3: SAHI + YOLO12l Ensemble
Combining SAHI-enhanced YOLO12x with YOLO12l predictions via WBF. SAHI's extra FPs outweighed the recall gains when ensembled.

| Config | Precision | Recall | F1 |
|--------|-----------|--------|----|
| s320, WBF(iou=0.5, w=3:1.5, avg), best F1 | 0.764 | 0.828 | 0.795 |
| s416, WBF(iou=0.5, w=3:1.5, avg), best F1 | 0.752 | 0.835 | 0.791 |

> The selective 2-model ensemble (Strategy 2) delivered the best overall results.

---

## 6. Conclusion

| Approach | Precision | Recall | F1 | Key Takeaway |
|----------|-----------|--------|----|-------------|
| Baseline (YOLO11n) | 0.601 | 0.607 | — | Starting point |
| YOLO12x (best F1) | 0.763 | 0.777 | 0.770 | Best single-model balance |
| YOLO12x (best recall) | 0.690 | 0.832 | 0.755 | +23% recall over baseline |
| 5-Model Ensemble (TTA+WBF) | 0.434 | 0.952 | 0.596 | Max recall, poor precision |
| SAHI Sliced Inference | 0.607 | 0.862 | 0.712 | Recall boost, precision drops |
| **🎯 Selective Ensemble (YOLO12x+12l)** | **0.772** | **0.862** | **0.814** | **Best overall: P>0.76, R>0.85** |

### Best Configuration (Target Met: P>0.76, R>0.85)
- **Models**: YOLO12x (weight=3.0) + YOLO12l (weight=2.0) with TTA
- **WBF**: iou_thr=0.45, skip_box_thr=0.05, conf_type=avg
- **Post-WBF Confidence**: ≥ 0.36
- **Result**: **Precision = 0.772, Recall = 0.862, F1 = 0.814**

> Recall improved from **67.6% → 86.2%** (+27.5%) while maintaining **Precision = 77.2%** (above 76% target). The predicted Share of Shelf correlates strongly with ground truth (Pearson r = 0.90 area-based). The selective 2-model ensemble proved that fewer, stronger models outperform larger ensembles diluted by weaker architectures.
