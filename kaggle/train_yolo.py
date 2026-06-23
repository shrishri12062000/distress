"""
Step 4 — Train YOLOv8n knife detector on Kaggle GPU.
Run after train_stgcn.py completes.
"""

import os
import sys
import yaml
from pathlib import Path

REPO_DIR = Path('/kaggle/working/distress-gesture-detection')
WORKING  = Path('/kaggle/working/distress_detection')
KNIFE_DATA = WORKING / 'processed' / 'knife' / 'knife_merged'
RUN_DIR    = WORKING / 'yolo_runs'
RUN_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_DIR))

os.system('pip install -q ultralytics')
from ultralytics import YOLO

with open(REPO_DIR / 'configs' / 'yolo.yaml') as f:
    cfg = yaml.safe_load(f)

DATA_YAML    = str(KNIFE_DATA / 'data.yaml')
EPOCHS       = cfg['training']['epochs']
BATCH_SIZE   = cfg['training']['batch_size']
IMAGE_SIZE   = cfg['training']['image_size']
LR           = cfg['training']['learning_rate']
LR_FINAL     = cfg['training']['lr_final']
PATIENCE     = cfg['training']['patience']
WORKERS      = cfg['training']['workers']
SEED         = cfg['training']['seed']
OPTIMIZER    = cfg['training']['optimizer']
MOMENTUM     = cfg['training']['momentum']
WEIGHT_DECAY = cfg['training']['weight_decay']
WARMUP       = cfg['training']['warmup_epochs']


def train():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f'Knife data.yaml not found: {DATA_YAML}\n'
            'Run prepare_data.py first.'
        )

    print('── Training YOLOv8n knife detector ─────────────────')
    print(f'   Data:   {DATA_YAML}')
    print(f'   Epochs: {EPOCHS}  Batch: {BATCH_SIZE}')

    model = YOLO(f"{cfg['model']['architecture']}.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        lr0=LR,
        lrf=LR_FINAL,
        patience=PATIENCE,
        device=0,
        workers=WORKERS,
        seed=SEED,
        optimizer=OPTIMIZER,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP,
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
    print(f'\n── Done ─────────────────────────────────────────────')
    print(f'   Best weights: {best_pt}')
    m = results.results_dict
    print(f'   mAP50:    {m.get("metrics/mAP50(B)", "N/A")}')
    print(f'   mAP50-95: {m.get("metrics/mAP50-95(B)", "N/A")}')


if __name__ == '__main__':
    train()
