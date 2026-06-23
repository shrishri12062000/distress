"""
YOLOv8n knife detection training — run on Kaggle GPU.

Usage (inside Kaggle notebook cell):
    !python train_yolo.py
"""

import os
import sys
import yaml
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/distress_detection')
KNIFE_DATA = DRIVE_ROOT / 'processed' / 'knife' / 'knife_merged'
RUN_DIR    = DRIVE_ROOT / 'yolo_runs'
RUN_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, '/kaggle/working/distress-gesture-detection')

os.system('pip install -q ultralytics')
from ultralytics import YOLO

CONFIG_PATH = '/kaggle/working/distress-gesture-detection/configs/yolo.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

DATA_YAML   = str(KNIFE_DATA / 'data.yaml')
EPOCHS      = cfg['training']['epochs']
BATCH_SIZE  = cfg['training']['batch_size']
IMAGE_SIZE  = cfg['training']['image_size']
LR          = cfg['training']['learning_rate']
LR_FINAL    = cfg['training']['lr_final']
PATIENCE    = cfg['training']['patience']
DEVICE      = cfg['training']['device']
WORKERS     = cfg['training']['workers']
SEED        = cfg['training']['seed']
OPTIMIZER   = cfg['training']['optimizer']
MOMENTUM    = cfg['training']['momentum']
WEIGHT_DECAY = cfg['training']['weight_decay']
WARMUP_EPOCHS = cfg['training']['warmup_epochs']


def train():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f'Knife data.yaml not found at {DATA_YAML}. '
            'Run prepare_data.py first.'
        )

    print(f'── Training YOLOv8n knife detector ─────────────────')
    print(f'   Data:    {DATA_YAML}')
    print(f'   Epochs:  {EPOCHS}')
    print(f'   Batch:   {BATCH_SIZE}')
    print(f'   Device:  {DEVICE}')

    model = YOLO(f"{cfg['model']['architecture']}.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LR,
        lrf=LR_FINAL,
        patience=PATIENCE,
        device=DEVICE,
        workers=WORKERS,
        seed=SEED,
        optimizer=OPTIMIZER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        project=str(RUN_DIR),
        name='knife_yolov8n',
        exist_ok=True,
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )

    best_pt = RUN_DIR / 'knife_yolov8n' / 'weights' / 'best.pt'
    print(f'\n── Training complete ────────────────────────────────')
    print(f'   Best weights: {best_pt}')

    metrics = results.results_dict
    print(f'   mAP50:    {metrics.get("metrics/mAP50(B)", "N/A"):.4f}')
    print(f'   mAP50-95: {metrics.get("metrics/mAP50-95(B)", "N/A"):.4f}')


if __name__ == '__main__':
    train()
