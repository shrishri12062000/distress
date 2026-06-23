"""
Step 1 — Data Preparation (run on Kaggle before training).

Mounts Google Drive, processes ALL datasets, saves preprocessed
skeleton arrays (.npy) and merged knife dataset to Google Drive.

Run order:
    1. prepare_data.py   ← this file
    2. validate_data.py
    3. train_stgcn.py
    4. train_yolo.py
    5. export_models.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT  = Path('/content/drive/MyDrive/distress_detection')
KAGGLE_ROOT = Path('/kaggle/input')
CACHE_ROOT  = DRIVE_ROOT / 'skeleton_cache'
OUTPUT_ROOT = DRIVE_ROOT / 'processed'

for d in [DRIVE_ROOT, CACHE_ROOT, OUTPUT_ROOT,
          OUTPUT_ROOT / 'stgcn', OUTPUT_ROOT / 'knife']:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, '/kaggle/working/distress-gesture-detection')

os.system('pip install -q mediapipe datasets huggingface-hub')

from src.data.ntu_dataset    import NTUDataset
from src.data.fall_dataset   import (URFDDataset, Le2iDataset,
                                     SignalForHelpDataset)
from src.data.omnifall_dataset import OmniFallDataset
from src.data.knife_dataset  import merge_knife_datasets


def _save_dataset(dataset, out_dir, name):
    """Extract all (X, y) pairs from a dataset and save as .npy."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    xs, ys = [], []
    for x, y in tqdm(dataset, desc=name):
        xs.append(x)
        ys.append(y)

    if not xs:
        print(f'  [WARN] {name}: no samples found, skipping.')
        return 0

    np.save(out / 'X.npy', np.array(xs, dtype=np.float32))
    np.save(out / 'y.npy', np.array(ys, dtype=np.int64))
    print(f'  {name}: {len(xs)} samples saved → {out}')
    return len(xs)


def process_ntu():
    print('\n[1/6] NTU RGB+D 120 skeletons')
    skeleton_dir = str(KAGGLE_ROOT / 'ntu-rgbd-120-skeleton' / 'skeletons')
    if not os.path.exists(skeleton_dir):
        # Fallback: check for NTU RGB+D 60
        skeleton_dir = str(KAGGLE_ROOT / 'skeleton-data-of-ntu-rgbd-60-dataset')
        if not os.path.exists(skeleton_dir):
            print(f'  [SKIP] NTU skeleton dir not found.')
            return
    dataset = NTUDataset(skeleton_dir, window_size=30, augment=False)
    _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'ntu'), 'NTU')


def process_urfd():
    print('\n[2/6] UR Fall Detection Dataset')
    root = str(KAGGLE_ROOT / 'ur-fall-detection-dataset')
    if not os.path.exists(root):
        print('  [SKIP] URFD not found in /kaggle/input')
        return
    cache = str(CACHE_ROOT / 'urfd')
    dataset = URFDDataset(root, window_size=30, augment=False,
                          skeleton_cache_dir=cache)
    _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'urfd'), 'URFD')


def process_le2i():
    print('\n[3/6] Le2i / ImViA Fall Dataset')
    root = str(KAGGLE_ROOT / 'falldataset-imvia')
    if not os.path.exists(root):
        print('  [SKIP] Le2i not found in /kaggle/input')
        return
    cache = str(CACHE_ROOT / 'le2i')
    dataset = Le2iDataset(root, window_size=30, augment=False,
                          skeleton_cache_dir=cache)
    _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'le2i'), 'Le2i')


def process_signal_for_help():
    print('\n[4/6] Signal for Help Dataset')
    # This dataset is uploaded manually to Google Drive after obtaining from authors
    root = str(DRIVE_ROOT / 'datasets' / 'signal_for_help')
    if not os.path.exists(root):
        print('  [SKIP] Signal for Help dataset not found.')
        print(f'  Place it at: {root}')
        print('  Folder structure: signal_for_help/signal/ and signal_for_help/no_signal/')
        return
    cache = str(CACHE_ROOT / 'signal_for_help')
    dataset = SignalForHelpDataset(root, window_size=30, augment=False,
                                   skeleton_cache_dir=cache)
    _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'signal_for_help'),
                  'SignalForHelp')


def process_omnifall():
    print('\n[5/6] OmniFall 2025 Dataset (HuggingFace)')
    cache = str(CACHE_ROOT / 'omnifall')
    try:
        dataset = OmniFallDataset(
            skeleton_cache_dir=cache,
            splits=['of_staged', 'of_synthetic'],
            window_size=30,
            augment=False,
        )
        _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'omnifall'),
                      'OmniFall')
    except Exception as e:
        print(f'  [SKIP] OmniFall failed to load: {e}')


def process_knife():
    print('\n[6/6] Knife Detection Datasets')
    knife_dirs = [
        str(KAGGLE_ROOT / 'od-weapon-detection-knife-detection'),
        str(KAGGLE_ROOT / 'cctv-weapon-dataset'),
        str(KAGGLE_ROOT / 'cctv-atm-robbery-detection-dataset-gun-and-knife'),
        str(KAGGLE_ROOT / 'surveillance-vlm-weapon-and-knife-detection-dataset'),
    ]
    existing = [d for d in knife_dirs if os.path.exists(d)]
    if not existing:
        print('  [SKIP] No knife datasets found. Add them as Kaggle datasets.')
        return
    print(f'  Found {len(existing)} knife dataset(s)')
    knife_out = str(OUTPUT_ROOT / 'knife' / 'knife_merged')
    merge_knife_datasets(existing, knife_out, val_split=0.15)


def print_summary():
    print('\n── Preparation Summary ──────────────────────────────')
    stgcn_dir = OUTPUT_ROOT / 'stgcn'
    total = 0
    for src in ['ntu', 'urfd', 'le2i', 'signal_for_help', 'omnifall']:
        x_path = stgcn_dir / src / 'X.npy'
        if x_path.exists():
            n = np.load(str(x_path)).shape[0]
            print(f'  {src:<20}: {n:>5} samples')
            total += n
        else:
            print(f'  {src:<20}:  SKIPPED')
    print(f'  {"TOTAL":<20}: {total:>5} action samples')

    knife_yaml = OUTPUT_ROOT / 'knife' / 'knife_merged' / 'data.yaml'
    print(f'  knife data.yaml  : {"OK" if knife_yaml.exists() else "MISSING"}')
    print(f'\n  Next step: run validate_data.py')


if __name__ == '__main__':
    process_ntu()
    process_urfd()
    process_le2i()
    process_signal_for_help()
    process_omnifall()
    process_knife()
    print_summary()
