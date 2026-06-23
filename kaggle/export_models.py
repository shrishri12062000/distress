"""
Export trained ST-GCN and YOLOv8 models to ONNX format.
Run after training is complete.

Usage (inside Kaggle notebook cell):
    !python export_models.py
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

import torch

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/distress_detection')
CKPT_DIR   = DRIVE_ROOT / 'checkpoints' / 'stgcn'
YOLO_DIR   = DRIVE_ROOT / 'yolo_runs' / 'knife_yolov8n' / 'weights'
MODELS_DIR = DRIVE_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, '/kaggle/working/distress-gesture-detection')

from src.models.stgcn import STGCN

CONFIG_PATH = '/kaggle/working/distress-gesture-detection/configs/stgcn.yaml'
with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


def export_stgcn():
    print('── Exporting ST-GCN to ONNX ─────────────────────────')
    ckpt_path = CKPT_DIR / 'best.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'ST-GCN checkpoint not found: {ckpt_path}')

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    model = STGCN(
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes'],
        dropout=0.0,  # disable dropout for inference
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Dummy input: (batch=1, channels=3, frames=30, joints=17)
    dummy = torch.zeros(1, 3, cfg['data']['window_size'],
                        cfg['model']['num_joints'])
    out_path = str(MODELS_DIR / 'stgcn.onnx')

    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['skeleton'],
        output_names=['logits'],
        dynamic_axes={
            'skeleton': {0: 'batch_size'},
            'logits':   {0: 'batch_size'},
        },
    )

    # Verify
    import onnx
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    print(f'  ✓ ST-GCN exported → {out_path}')
    print(f'    Val acc at export: {ckpt["val_acc"]:.2f}%')
    print(f'    Trained epochs:    {ckpt["epoch"]}')


def export_yolo():
    print('\n── Exporting YOLOv8n to ONNX ────────────────────────')
    os.system('pip install -q ultralytics')
    from ultralytics import YOLO

    best_pt = YOLO_DIR / 'best.pt'
    if not best_pt.exists():
        raise FileNotFoundError(f'YOLOv8 weights not found: {best_pt}')

    model = YOLO(str(best_pt))
    out_path = str(MODELS_DIR / 'yolo_knife.onnx')

    model.export(
        format='onnx',
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
    )

    # YOLO exports next to the .pt file — move to our models dir
    yolo_onnx_src = str(YOLO_DIR / 'best.onnx')
    if os.path.exists(yolo_onnx_src):
        import shutil
        shutil.move(yolo_onnx_src, out_path)

    print(f'  ✓ YOLOv8n exported → {out_path}')


def print_model_summary():
    print('\n── Model Summary ────────────────────────────────────')
    for name in ['stgcn.onnx', 'yolo_knife.onnx']:
        p = MODELS_DIR / name
        if p.exists():
            size_mb = p.stat().st_size / (1024 ** 2)
            print(f'  {name}: {size_mb:.1f} MB')
        else:
            print(f'  {name}: NOT FOUND')

    print(f'\n  Download both files from Google Drive:')
    print(f'  {MODELS_DIR}')
    print(f'\n  Place them in your local project:')
    print(f'  distress-gesture-detection/models/')


if __name__ == '__main__':
    export_stgcn()
    export_yolo()
    print_model_summary()
