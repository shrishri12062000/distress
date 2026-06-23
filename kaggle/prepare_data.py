"""
Step 1 — Data Preparation (run on Kaggle before training).

Processes ALL datasets and saves preprocessed skeleton arrays
and merged knife dataset to /kaggle/working/.

Run order on Kaggle:
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

REPO_DIR    = Path('/kaggle/working/distress-gesture-detection')
KAGGLE_ROOT = Path('/kaggle/input')
WORKING     = Path('/kaggle/working/distress_detection')
CACHE_ROOT  = WORKING / 'skeleton_cache'
OUTPUT_ROOT = WORKING / 'processed'

for d in [WORKING, CACHE_ROOT, OUTPUT_ROOT,
          OUTPUT_ROOT / 'stgcn', OUTPUT_ROOT / 'knife']:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_DIR))

from src.data.ntu_dataset      import NTUDataset
from src.data.fall_dataset     import (URFDDataset, Le2iDataset,
                                       SignalForHelpDataset)
from src.data.omnifall_dataset import OmniFallDataset
from src.data.knife_dataset    import merge_knife_datasets


def _save_dataset(dataset, out_dir, name):
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
    print('\n[1/6] NTU RGB+D skeleton data')
    # Try NTU 120 first, fall back to NTU 60
    candidates = [
        KAGGLE_ROOT / 'ntu-rgbd-120-skeleton' / 'skeletons',
        KAGGLE_ROOT / 'skeleton-data-of-ntu-rgbd-60-dataset',
        KAGGLE_ROOT / 'ntu-rgbd-60-skeleton',
    ]
    skeleton_dir = next((str(p) for p in candidates if p.exists()), None)
    if skeleton_dir is None:
        print('  [SKIP] NTU skeleton data not found in /kaggle/input')
        return
    dataset = NTUDataset(skeleton_dir, window_size=30, augment=False)
    _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'ntu'), 'NTU')


def process_urfd():
    print('\n[2/6] UR Fall Detection Dataset')
    candidates = [
        KAGGLE_ROOT / 'ur-fall-detection-dataset',
        KAGGLE_ROOT / 'shahliza27' / 'ur-fall-detection-dataset',
    ]
    root = next((str(p) for p in candidates if p.exists()), None)
    if root is None:
        print('  [SKIP] URFD not found in /kaggle/input')
        return
    dataset = URFDDataset(root, window_size=30, augment=False,
                          skeleton_cache_dir=str(CACHE_ROOT / 'urfd'))
    _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'urfd'), 'URFD')


def process_le2i():
    print('\n[3/6] Le2i / ImViA Fall Dataset')
    candidates = [
        KAGGLE_ROOT / 'falldataset-imvia',
        KAGGLE_ROOT / 'tuyenldvn' / 'falldataset-imvia',
    ]
    root = next((str(p) for p in candidates if p.exists()), None)
    if root is None:
        print('  [SKIP] Le2i not found in /kaggle/input')
        return
    dataset = Le2iDataset(root, window_size=30, augment=False,
                          skeleton_cache_dir=str(CACHE_ROOT / 'le2i'))
    _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'le2i'), 'Le2i')


def process_signal_for_help():
    print('\n[4/6] Signal for Help Dataset')
    # Upload this dataset manually to Kaggle as a private dataset
    # then attach it to this notebook as "signal-for-help"
    candidates = [
        KAGGLE_ROOT / 'signal-for-help',
        KAGGLE_ROOT / 'signal-for-help-dataset',
    ]
    root = next((str(p) for p in candidates if p.exists()), None)
    if root is None:
        print('  [SKIP] Signal for Help not found.')
        print('  Upload it to Kaggle as a private dataset and attach it.')
        return
    dataset = SignalForHelpDataset(root, window_size=30, augment=False,
                                   skeleton_cache_dir=str(CACHE_ROOT / 'signal'))
    _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'signal_for_help'),
                  'SignalForHelp')


def process_omnifall():
    print('\n[5/6] OmniFall 2025 (HuggingFace)')
    try:
        dataset = OmniFallDataset(
            skeleton_cache_dir=str(CACHE_ROOT / 'omnifall'),
            splits=['of_staged', 'of_synthetic'],
            window_size=30,
            augment=False,
        )
        _save_dataset(dataset, str(OUTPUT_ROOT / 'stgcn' / 'omnifall'),
                      'OmniFall')
    except Exception as e:
        print(f'  [SKIP] OmniFall failed: {e}')


def process_knife():
    print('\n[6/6] Knife Detection Datasets')
    knife_dirs = [
        str(KAGGLE_ROOT / 'cctv-weapon-dataset'),
        str(KAGGLE_ROOT / 'cctv-atm-robbery-detection-dataset-gun-and-knife'),
        str(KAGGLE_ROOT / 'surveillance-vlm-weapon-and-knife-detection-dataset'),
        str(KAGGLE_ROOT / 'od-weapon-detection-knife-detection'),
    ]
    existing = [d for d in knife_dirs if os.path.exists(d)]
    if not existing:
        print('  [SKIP] No knife datasets found.')
        return
    print(f'  Found {len(existing)} knife dataset(s)')
    merge_knife_datasets(existing, str(OUTPUT_ROOT / 'knife' / 'knife_merged'),
                         val_split=0.15)


def print_summary():
    print('\n── Preparation Summary ──────────────────────────────')
    total = 0
    for src in ['ntu', 'urfd', 'le2i', 'signal_for_help', 'omnifall']:
        p = OUTPUT_ROOT / 'stgcn' / src / 'X.npy'
        if p.exists():
            n = np.load(str(p)).shape[0]
            print(f'  {src:<20}: {n:>5} samples')
            total += n
        else:
            print(f'  {src:<20}:  SKIPPED')
    print(f'  {"TOTAL":<20}: {total:>5} action samples')
    knife_yaml = OUTPUT_ROOT / 'knife' / 'knife_merged' / 'data.yaml'
    print(f'  knife data.yaml      : {"OK" if knife_yaml.exists() else "MISSING"}')
    print(f'\n  All data saved to: {WORKING}')
    print(f'  Next step: run validate_data.py')


if __name__ == '__main__':
    process_ntu()
    process_urfd()
    process_le2i()
    process_signal_for_help()
    process_omnifall()
    process_knife()
    print_summary()
