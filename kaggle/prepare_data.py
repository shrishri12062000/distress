"""
Run this first on Kaggle before training.
Mounts Google Drive, processes all datasets, saves preprocessed data to Drive.

Usage (inside Kaggle notebook cell):
    !python prepare_data.py
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Mount Google Drive
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

# ── Install deps ─────────────────────────────────────────────────────────────
os.system('pip install -q mediapipe')

from src.data.ntu_dataset   import NTUDataset
from src.data.fall_dataset  import URFDDataset, Le2iDataset, SignalForHelpDataset
from src.data.knife_dataset import merge_knife_datasets


def process_ntu(skeleton_dir, out_dir):
    """
    Process NTU RGB+D 120 skeleton files → .npy arrays saved per sample.
    skeleton_dir: folder containing .skeleton files
    """
    print('\n[1/4] Processing NTU RGB+D 120 skeletons...')
    dataset = NTUDataset(skeleton_dir, window_size=30, augment=False)
    out = Path(out_dir) / 'ntu'
    out.mkdir(parents=True, exist_ok=True)

    xs, ys = [], []
    for i, (x, y) in enumerate(tqdm(dataset, desc='NTU')):
        xs.append(x)
        ys.append(y)

    np.save(out / 'X.npy', np.array(xs, dtype=np.float32))
    np.save(out / 'y.npy', np.array(ys, dtype=np.int64))
    print(f'  NTU: {len(xs)} samples saved → {out}')


def process_urfd(urfd_root, out_dir):
    print('\n[2/4] Processing URFD dataset...')
    cache = str(CACHE_ROOT / 'urfd')
    dataset = URFDDataset(urfd_root, window_size=30, augment=False,
                          skeleton_cache_dir=cache)
    out = Path(out_dir) / 'urfd'
    out.mkdir(parents=True, exist_ok=True)

    xs, ys = [], []
    for x, y in tqdm(dataset, desc='URFD'):
        xs.append(x)
        ys.append(y)

    np.save(out / 'X.npy', np.array(xs, dtype=np.float32))
    np.save(out / 'y.npy', np.array(ys, dtype=np.int64))
    print(f'  URFD: {len(xs)} samples saved → {out}')


def process_le2i(le2i_root, out_dir):
    print('\n[3/4] Processing Le2i dataset...')
    cache = str(CACHE_ROOT / 'le2i')
    dataset = Le2iDataset(le2i_root, window_size=30, augment=False,
                          skeleton_cache_dir=cache)
    out = Path(out_dir) / 'le2i'
    out.mkdir(parents=True, exist_ok=True)

    xs, ys = [], []
    for x, y in tqdm(dataset, desc='Le2i'):
        xs.append(x)
        ys.append(y)

    np.save(out / 'X.npy', np.array(xs, dtype=np.float32))
    np.save(out / 'y.npy', np.array(ys, dtype=np.int64))
    print(f'  Le2i: {len(xs)} samples saved → {out}')


def process_knife(out_dir):
    print('\n[4/4] Merging knife datasets...')
    knife_dirs = [
        str(KAGGLE_ROOT / 'od-weapon-detection-knife-detection'),
        str(KAGGLE_ROOT / 'cctv-weapon-dataset'),
        str(KAGGLE_ROOT / 'cctv-atm-robbery-detection-dataset-gun-and-knife'),
    ]
    existing = [d for d in knife_dirs if os.path.exists(d)]
    if not existing:
        print('  No knife datasets found in /kaggle/input. '
              'Add them as Kaggle datasets first.')
        return

    knife_out = str(Path(out_dir) / 'knife_merged')
    merge_knife_datasets(existing, knife_out, val_split=0.15)


if __name__ == '__main__':
    # ── Paths — update if your Kaggle dataset names differ ───────────────────
    NTU_SKELETON_DIR = str(KAGGLE_ROOT / 'ntu-rgbd-120-skeleton' / 'skeletons')
    URFD_ROOT        = str(KAGGLE_ROOT / 'ur-fall-detection-dataset')
    LE2I_ROOT        = str(KAGGLE_ROOT / 'falldataset-imvia')

    if os.path.exists(NTU_SKELETON_DIR):
        process_ntu(NTU_SKELETON_DIR, str(OUTPUT_ROOT / 'stgcn'))
    else:
        print(f'[SKIP] NTU skeleton dir not found: {NTU_SKELETON_DIR}')

    if os.path.exists(URFD_ROOT):
        process_urfd(URFD_ROOT, str(OUTPUT_ROOT / 'stgcn'))
    else:
        print(f'[SKIP] URFD not found: {URFD_ROOT}')

    if os.path.exists(LE2I_ROOT):
        process_le2i(LE2I_ROOT, str(OUTPUT_ROOT / 'stgcn'))
    else:
        print(f'[SKIP] Le2i not found: {LE2I_ROOT}')

    process_knife(str(OUTPUT_ROOT / 'knife'))

    print('\n✓ Data preparation complete.')
    print(f'  Processed data saved to: {OUTPUT_ROOT}')
