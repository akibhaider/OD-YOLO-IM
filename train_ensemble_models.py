"""
Quick-train 3 diverse models for WBF ensemble.
Models: YOLO12l, YOLO11m, YOLO12s  (diverse architectures + scales)
Already have: YOLO12x (57ep), YOLO11n (24ep)
Total ensemble: 5 models
"""
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from ultralytics import YOLO
import torch

PROJECT = '/mnt/Data/AKIB/YOLO-OD-IM'
DATA    = os.path.join(PROJECT, 'dataset', 'data_abs.yaml')

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MODELS = [
    ('yolo12l.pt', 'yolo12l_ensemble', 30, 8),    # large, 30 epochs, batch 8
    ('yolo11m.pt', 'yolo11m_ensemble', 30, 16),   # medium, 30 epochs, batch 16
    ('yolo12s.pt', 'yolo12s_ensemble', 30, 16),   # small, 30 epochs, batch 16
]

for weights, name, epochs, batch in MODELS:
    print(f"\n{'='*60}")
    print(f"🚀 Training {weights} → {name} ({epochs} epochs, batch={batch})")
    print(f"{'='*60}")
    t0 = time.time()
    
    model = YOLO(weights)
    model.train(
        data=DATA,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=0,
        project=os.path.join(PROJECT, 'runs'),
        name=name,
        exist_ok=True,
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        close_mosaic=10,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=15,
        save=True,
        plots=True,
        verbose=True,
    )
    
    elapsed = time.time() - t0
    print(f"✅ {name} done in {elapsed/60:.1f} min")
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()

print("\n✅ ALL ENSEMBLE MODELS TRAINED!")
