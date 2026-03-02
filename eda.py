"""
Comprehensive EDA for the Shelf Product Detection Dataset
"""
import os
import glob
import yaml
import numpy as np
from collections import Counter, defaultdict
import json

# ============================================================
# 1. Load Dataset Config
# ============================================================
DATASET_ROOT = "/mnt/Data/AKIB/YOLO-OD-IM/dataset"

with open(os.path.join(DATASET_ROOT, "data.yaml"), "r") as f:
    config = yaml.safe_load(f)

class_names = config['names']
num_classes = config['nc']
print(f"Number of classes (SKUs): {num_classes}")
print(f"Class names: {class_names[:10]}... (showing first 10)")

# ============================================================
# 2. Count images and labels per split
# ============================================================
splits = ['train', 'valid', 'test']
split_stats = {}

for split in splits:
    img_dir = os.path.join(DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(DATASET_ROOT, split, "labels")
    
    images = glob.glob(os.path.join(img_dir, "*"))
    labels = glob.glob(os.path.join(lbl_dir, "*.txt"))
    
    split_stats[split] = {
        'num_images': len(images),
        'num_labels': len(labels),
    }
    print(f"\n{split.upper()}: {len(images)} images, {len(labels)} label files")

# ============================================================
# 3. Parse ALL annotations
# ============================================================
def parse_yolo_labels(label_dir):
    """Parse all YOLO format label files in a directory."""
    all_annotations = []
    per_image_counts = []
    class_counter = Counter()
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    class_bboxes = defaultdict(list)  # class_id -> list of (w, h)
    
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
    
    for lf in label_files:
        with open(lf, 'r') as f:
            lines = f.readlines()
        
        count = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                all_annotations.append({
                    'class_id': cls_id,
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                    'file': os.path.basename(lf)
                })
                
                class_counter[cls_id] += 1
                bbox_widths.append(w)
                bbox_heights.append(h)
                bbox_areas.append(w * h)
                class_bboxes[cls_id].append((w, h))
                count += 1
        
        per_image_counts.append(count)
    
    return {
        'annotations': all_annotations,
        'per_image_counts': per_image_counts,
        'class_counter': class_counter,
        'bbox_widths': np.array(bbox_widths),
        'bbox_heights': np.array(bbox_heights),
        'bbox_areas': np.array(bbox_areas),
        'class_bboxes': class_bboxes,
        'num_files': len(label_files)
    }

print("\n" + "="*60)
print("PARSING ANNOTATIONS...")
print("="*60)

all_split_data = {}
for split in splits:
    lbl_dir = os.path.join(DATASET_ROOT, split, "labels")
    all_split_data[split] = parse_yolo_labels(lbl_dir)
    total_ann = len(all_split_data[split]['annotations'])
    print(f"{split.upper()}: {total_ann} total annotations across {all_split_data[split]['num_files']} images")

# ============================================================
# 4. Overall Statistics
# ============================================================
print("\n" + "="*60)
print("OVERALL STATISTICS")
print("="*60)

for split in splits:
    data = all_split_data[split]
    counts = data['per_image_counts']
    print(f"\n--- {split.upper()} ---")
    print(f"  Total annotations: {len(data['annotations'])}")
    print(f"  Objects per image: min={min(counts)}, max={max(counts)}, "
          f"mean={np.mean(counts):.1f}, median={np.median(counts):.1f}, std={np.std(counts):.1f}")
    print(f"  Bbox width  (normalized): min={data['bbox_widths'].min():.4f}, max={data['bbox_widths'].max():.4f}, "
          f"mean={data['bbox_widths'].mean():.4f}, median={np.median(data['bbox_widths']):.4f}")
    print(f"  Bbox height (normalized): min={data['bbox_heights'].min():.4f}, max={data['bbox_heights'].max():.4f}, "
          f"mean={data['bbox_heights'].mean():.4f}, median={np.median(data['bbox_heights']):.4f}")
    print(f"  Bbox area   (normalized): min={data['bbox_areas'].min():.6f}, max={data['bbox_areas'].max():.6f}, "
          f"mean={data['bbox_areas'].mean():.6f}")

# ============================================================
# 5. Class Distribution Analysis
# ============================================================
print("\n" + "="*60)
print("CLASS DISTRIBUTION (TRAIN)")
print("="*60)

train_counter = all_split_data['train']['class_counter']

# Sort by frequency
sorted_classes = train_counter.most_common()
print(f"\nTotal unique classes in train: {len(train_counter)}")
print(f"Total annotations in train: {sum(train_counter.values())}")

print("\nTop 15 most frequent classes:")
for cls_id, count in sorted_classes[:15]:
    print(f"  Class {cls_id:3d} ({class_names[cls_id]:>6s}): {count:5d} instances")

print("\nBottom 15 least frequent classes:")
for cls_id, count in sorted_classes[-15:]:
    print(f"  Class {cls_id:3d} ({class_names[cls_id]:>6s}): {count:5d} instances")

# Check for missing classes
all_class_ids_in_train = set(train_counter.keys())
missing_in_train = set(range(num_classes)) - all_class_ids_in_train
if missing_in_train:
    print(f"\n⚠️  Classes MISSING from training data: {sorted(missing_in_train)}")
    print(f"   That's {len(missing_in_train)} classes with ZERO training samples!")

# Class frequency distribution
counts_array = np.array([train_counter.get(i, 0) for i in range(num_classes)])
print(f"\nClass frequency statistics:")
print(f"  Min instances per class: {counts_array.min()}")
print(f"  Max instances per class: {counts_array.max()}")
print(f"  Mean instances per class: {counts_array.mean():.1f}")
print(f"  Median instances per class: {np.median(counts_array):.1f}")
print(f"  Std dev: {np.std(counts_array):.1f}")

# Imbalance ratio
print(f"\n  Imbalance ratio (max/min non-zero): {counts_array[counts_array > 0].max() / counts_array[counts_array > 0].min():.1f}x")

# Classes with very few samples
thresholds = [5, 10, 20, 50]
for t in thresholds:
    count_below = (counts_array < t).sum()
    print(f"  Classes with < {t} samples: {count_below}")

# ============================================================
# 6. Test Set Class Distribution
# ============================================================
print("\n" + "="*60)
print("CLASS DISTRIBUTION (TEST)")
print("="*60)

test_counter = all_split_data['test']['class_counter']
print(f"Total unique classes in test: {len(test_counter)}")
print(f"Total annotations in test: {sum(test_counter.values())}")

# Check test classes that don't appear in train
test_only_classes = set(test_counter.keys()) - all_class_ids_in_train
if test_only_classes:
    print(f"\n⚠️  Classes in TEST but NOT in TRAIN: {sorted(test_only_classes)}")
    for c in sorted(test_only_classes):
        print(f"   Class {c} ({class_names[c]}): {test_counter[c]} test instances")

# ============================================================
# 7. Bounding Box Size Distribution
# ============================================================
print("\n" + "="*60)
print("BOUNDING BOX SIZE ANALYSIS (TRAIN)")
print("="*60)

train_data = all_split_data['train']
areas = train_data['bbox_areas']

# COCO-style size categories (adapted for normalized coords, image is 640x640)
# small: area < 32^2/640^2 = 0.0025
# medium: 32^2/640^2 <= area < 96^2/640^2 = 0.0225
# large: area >= 96^2/640^2 = 0.0225
pixel_areas = areas * (640 * 640)  # convert to pixel areas
small = (pixel_areas < 32*32).sum()
medium = ((pixel_areas >= 32*32) & (pixel_areas < 96*96)).sum()
large = (pixel_areas >= 96*96).sum()

total = len(areas)
print(f"  Small objects  (<32x32 px): {small:5d} ({100*small/total:.1f}%)")
print(f"  Medium objects (32-96 px):   {medium:5d} ({100*medium/total:.1f}%)")
print(f"  Large objects  (>96x96 px):  {large:5d} ({100*large/total:.1f}%)")

# Aspect ratio analysis
aspect_ratios = train_data['bbox_widths'] / (train_data['bbox_heights'] + 1e-8)
print(f"\n  Aspect ratio (w/h): min={aspect_ratios.min():.2f}, max={aspect_ratios.max():.2f}, "
      f"mean={aspect_ratios.mean():.2f}, median={np.median(aspect_ratios):.2f}")

# ============================================================
# 8. Overlap / Density Analysis
# ============================================================
print("\n" + "="*60)
print("OBJECT DENSITY ANALYSIS")
print("="*60)

for split in splits:
    counts = all_split_data[split]['per_image_counts']
    print(f"\n{split.upper()} - Objects per image:")
    hist, bin_edges = np.histogram(counts, bins=[0, 5, 10, 20, 30, 50, 100, 200])
    for i in range(len(hist)):
        print(f"  {int(bin_edges[i]):3d}-{int(bin_edges[i+1]):3d}: {hist[i]:4d} images")

# ============================================================
# 9. Share of Shelf Analysis (Test set)
# ============================================================
print("\n" + "="*60)
print("SHARE OF SHELF ANALYSIS (TEST SET)")
print("="*60)

test_data = all_split_data['test']
test_annotations = test_data['annotations']

# Method 1: By count (number of facings)
total_test_objects = len(test_annotations)
print(f"\nTotal products on shelf (test): {total_test_objects}")

print("\nShare of Shelf by FACING COUNT:")
for cls_id, count in sorted(test_counter.items(), key=lambda x: -x[1]):
    pct = 100.0 * count / total_test_objects
    print(f"  {class_names[cls_id]:>6s}: {count:4d} facings ({pct:5.2f}%)")

# Method 2: By area (shelf space occupied)
class_areas = defaultdict(float)
total_area = 0.0
for ann in test_annotations:
    area = ann['w'] * ann['h']
    class_areas[ann['class_id']] += area
    total_area += area

print("\nShare of Shelf by AREA (shelf space):")
for cls_id in sorted(class_areas.keys(), key=lambda x: -class_areas[x]):
    pct = 100.0 * class_areas[cls_id] / total_area
    print(f"  {class_names[cls_id]:>6s}: area={class_areas[cls_id]:.4f} ({pct:5.2f}%)")

# ============================================================
# 10. Data Quality Checks
# ============================================================
print("\n" + "="*60)
print("DATA QUALITY CHECKS")
print("="*60)

for split in splits:
    annotations = all_split_data[split]['annotations']
    issues = 0
    for ann in annotations:
        # Check for out-of-bound coordinates
        if ann['cx'] < 0 or ann['cx'] > 1 or ann['cy'] < 0 or ann['cy'] > 1:
            issues += 1
        if ann['w'] <= 0 or ann['h'] <= 0 or ann['w'] > 1 or ann['h'] > 1:
            issues += 1
        if ann['class_id'] < 0 or ann['class_id'] >= num_classes:
            issues += 1
    print(f"  {split.upper()}: {issues} annotation issues found")

# ============================================================
# 11. Augmentation Analysis
# ============================================================
print("\n" + "="*60)
print("AUGMENTATION ANALYSIS")
print("="*60)

# Check if filenames suggest augmentation
train_files = [os.path.basename(f) for f in glob.glob(os.path.join(DATASET_ROOT, "train", "labels", "*.txt"))]
unique_bases = set()
for f in train_files:
    # Extract base image name (before the roboflow hash)
    base = f.split("_jpg.rf.")[0] if "_jpg.rf." in f else f
    unique_bases.add(base)

print(f"  Total training label files: {len(train_files)}")
print(f"  Unique base images: {len(unique_bases)}")
print(f"  Augmentation factor: {len(train_files)/len(unique_bases):.1f}x")

# Also check valid and test
for split in ['valid', 'test']:
    files = [os.path.basename(f) for f in glob.glob(os.path.join(DATASET_ROOT, split, "labels", "*.txt"))]
    bases = set()
    for f in files:
        base = f.split("_jpg.rf.")[0] if "_jpg.rf." in f else f
        bases.add(base)
    print(f"  {split}: {len(files)} files from {len(bases)} unique images ({len(files)/max(len(bases),1):.1f}x)")

print("\n✅ EDA Complete!")
