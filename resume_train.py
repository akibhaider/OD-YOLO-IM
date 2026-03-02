"""Resume YOLO12x@1280 training from last.pt checkpoint (epoch 8)"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch, gc
torch.cuda.empty_cache()
gc.collect()

from ultralytics import YOLO

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

LAST_PT = '/mnt/Data/AKIB/YOLO-OD-IM/runs/yolo12x_1280/weights/last.pt'
print(f"\n🔄 Resuming training from: {LAST_PT}")

model = YOLO(LAST_PT)
model.train(resume=True)

print("\n✅ Training complete!")
