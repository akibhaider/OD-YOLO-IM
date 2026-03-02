"""
Train YOLO11m and YOLO12s with 100 epochs each for stronger ensemble.
Already have: YOLO12x (57ep), YOLO12l (30ep), YOLO11n (24ep)
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
    ('yolo11m.pt', 'yolo11m_ensemble', 100, 16),   # medium, 100 epochs, batch 16
    ('yolo12s.pt', 'yolo12s_ensemble', 100, 16),    # small, 100 epochs, batch 16
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
        warmup_epochs=5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        cutmix=0.1,
        close_mosaic=15,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=30,
        save=True,
        plots=True,
        verbose=True,
    )
    
    elapsed = time.time() - t0
    print(f"✅ {name} done in {elapsed/60:.1f} min")
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()

print("\n✅ YOLO11m + YOLO12s TRAINING COMPLETE!")
