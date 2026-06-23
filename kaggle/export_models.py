"""
Step 5 — Export trained models to ONNX for CPU inference on your laptop.
Run after both training scripts complete.
"""

import os
import sys
import yaml
import shutil
import numpy as np
from pathlib import Path

import torch

REPO_DIR   = Path('/kaggle/working/distress-gesture-detection')
WORKING    = Path('/kaggle/working/distress_detection')
CKPT_DIR   = WORKING / 'checkpoints' / 'stgcn'
YOLO_DIR   = WORKING / 'yolo_runs' / 'knife_yolov8n' / 'weights'
MODELS_DIR = WORKING / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Also copy to /kaggle/working/ root so they appear in Kaggle output tab
OUTPUT_DIR = Path('/kaggle/working')

sys.path.insert(0, str(REPO_DIR))
from src.models.stgcn import STGCN

with open(REPO_DIR / 'configs' / 'stgcn.yaml') as f:
    cfg = yaml.safe_load(f)


def export_stgcn():
    print('── Exporting ST-GCN → ONNX ──────────────────────────')
    ckpt_path = CKPT_DIR / 'best.pth'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    ckpt  = torch.load(str(ckpt_path), map_location='cpu')
    model = STGCN(
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes'],
        dropout=0.0,
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    dummy    = torch.zeros(1, 3, cfg['data']['window_size'],
                           cfg['model']['num_joints'])
    out_path = str(MODELS_DIR / 'stgcn.onnx')

    torch.onnx.export(
        model, dummy, out_path,
        export_params=True, opset_version=17,
        do_constant_folding=True,
        input_names=['skeleton'], output_names=['logits'],
        dynamic_axes={'skeleton': {0: 'batch'}, 'logits': {0: 'batch'}},
    )

    import onnx
    onnx.checker.check_model(onnx.load(out_path))

    # Copy to output tab for easy download
    shutil.copy2(out_path, str(OUTPUT_DIR / 'stgcn.onnx'))

    size_mb = Path(out_path).stat().st_size / (1024 ** 2)
    print(f'  ✓ stgcn.onnx  ({size_mb:.1f} MB)')
    print(f'    Val acc: {ckpt["val_acc"]:.2f}%  |  Epochs: {ckpt["epoch"]}')


def export_yolo():
    print('\n── Exporting YOLOv8n → ONNX ─────────────────────────')
    os.system('pip install -q ultralytics')
    from ultralytics import YOLO

    best_pt = YOLO_DIR / 'best.pt'
    if not best_pt.exists():
        raise FileNotFoundError(f'YOLO weights not found: {best_pt}')

    model = YOLO(str(best_pt))
    model.export(format='onnx', imgsz=640, opset=17, simplify=True)

    # YOLO exports next to .pt — move to models dir
    yolo_src = YOLO_DIR / 'best.onnx'
    out_path = str(MODELS_DIR / 'yolo_knife.onnx')
    if yolo_src.exists():
        shutil.move(str(yolo_src), out_path)

    # Copy to output tab
    shutil.copy2(out_path, str(OUTPUT_DIR / 'yolo_knife.onnx'))

    size_mb = Path(out_path).stat().st_size / (1024 ** 2)
    print(f'  ✓ yolo_knife.onnx  ({size_mb:.1f} MB)')


def print_download_instructions():
    print('\n═' * 54)
    print('  Export complete!')
    print('\n  HOW TO DOWNLOAD YOUR MODELS:')
    print('  1. Click the Output tab (right panel in Kaggle)')
    print('  2. Download: stgcn.onnx  and  yolo_knife.onnx')
    print('  3. Place both files in your local project:')
    print('       distress-gesture-detection/models/')
    print('\n  Then run on your laptop:')
    print('       python scripts/run_camera.py')
    print('═' * 54)


if __name__ == '__main__':
    export_stgcn()
    export_yolo()
    print_download_instructions()
