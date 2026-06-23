"""
Step 2 — Data Validation (run after prepare_data.py, before training).

Checks that all processed datasets are correctly structured,
verifies class distributions, and catches problems before
a long training run fails silently.
"""

import os
import sys
import numpy as np
import yaml
from pathlib import Path
from collections import Counter

from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT  = Path('/content/drive/MyDrive/distress_detection')
OUTPUT_ROOT = DRIVE_ROOT / 'processed'
STGCN_DIR   = OUTPUT_ROOT / 'stgcn'
KNIFE_DIR   = OUTPUT_ROOT / 'knife' / 'knife_merged'

CLASS_NAMES = {0: 'normal', 1: 'help_signal',
               2: 'collapse_falling', 3: 'fall_down'}

PASS = '  [OK]  '
WARN = '  [WARN]'
FAIL = '  [FAIL]'


def validate_stgcn_source(name):
    src_dir = STGCN_DIR / name
    x_path  = src_dir / 'X.npy'
    y_path  = src_dir / 'y.npy'

    if not src_dir.exists():
        print(f'{WARN} {name}: directory not found (dataset was skipped)')
        return False, 0

    if not x_path.exists() or not y_path.exists():
        print(f'{FAIL} {name}: X.npy or y.npy missing — re-run prepare_data.py')
        return False, 0

    X = np.load(str(x_path))
    y = np.load(str(y_path))

    # Shape check: (N, 3, 30, 17)
    if X.ndim != 4 or X.shape[1] != 3 or X.shape[3] != 17:
        print(f'{FAIL} {name}: unexpected shape {X.shape} '
              f'(expected N×3×T×17)')
        return False, 0

    if len(X) != len(y):
        print(f'{FAIL} {name}: X/y length mismatch ({len(X)} vs {len(y)})')
        return False, 0

    if np.isnan(X).any():
        pct = np.isnan(X).mean() * 100
        print(f'{WARN} {name}: {pct:.1f}% NaN values in skeleton data')

    counts  = Counter(y.tolist())
    dist_str = '  '.join(
        f'{CLASS_NAMES.get(k, k)}: {v}'
        for k, v in sorted(counts.items())
    )
    print(f'{PASS} {name}: {len(X)} samples  [{dist_str}]')
    return True, len(X)


def validate_knife_dataset():
    if not KNIFE_DIR.exists():
        print(f'{FAIL} Knife dataset not found at {KNIFE_DIR}')
        print(f'       Re-run prepare_data.py with knife datasets added.')
        return False

    yaml_path = KNIFE_DIR / 'data.yaml'
    if not yaml_path.exists():
        print(f'{FAIL} Knife data.yaml missing')
        return False

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    train_img_dir = KNIFE_DIR / 'images' / 'train'
    val_img_dir   = KNIFE_DIR / 'images' / 'val'
    train_lbl_dir = KNIFE_DIR / 'labels' / 'train'
    val_lbl_dir   = KNIFE_DIR / 'labels' / 'val'

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if not d.exists():
            print(f'{FAIL} Knife: missing directory {d}')
            return False

    n_train = len(list(train_img_dir.glob('*.jpg'))) + \
              len(list(train_img_dir.glob('*.png')))
    n_val   = len(list(val_img_dir.glob('*.jpg'))) + \
              len(list(val_img_dir.glob('*.png')))
    n_lbl   = len(list(train_lbl_dir.glob('*.txt')))

    if n_train == 0:
        print(f'{FAIL} Knife: no training images found')
        return False

    if n_lbl != n_train:
        print(f'{WARN} Knife: image/label count mismatch '
              f'({n_train} images vs {n_lbl} labels)')

    # Spot-check one label file
    sample_lbl = next(train_lbl_dir.glob('*.txt'), None)
    if sample_lbl:
        with open(sample_lbl) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                print(f'{WARN} Knife: unexpected label format in {sample_lbl}')
                break
            if not parts[0] == '0':
                print(f'{WARN} Knife: non-knife class found ({parts[0]})')
                break

    print(f'{PASS} Knife dataset: {n_train} train / {n_val} val images')
    return True


def check_class_balance(total_counts):
    """Warn if any class is severely underrepresented."""
    if not total_counts:
        return
    total  = sum(total_counts.values())
    min_cls = min(total_counts, key=total_counts.get)
    max_cls = max(total_counts, key=total_counts.get)
    ratio   = total_counts[max_cls] / max(total_counts[min_cls], 1)

    print('\n── Overall Class Distribution ───────────────────────')
    for k in sorted(total_counts):
        n   = total_counts[k]
        pct = n / total * 100
        bar = '█' * int(pct / 2)
        print(f'  {CLASS_NAMES[k]:<20} {n:>5}  ({pct:5.1f}%)  {bar}')

    if ratio > 10:
        print(f'\n{WARN} Severe imbalance detected (ratio {ratio:.1f}x). '
              f'Class weights in stgcn.yaml will compensate.')
    elif ratio > 5:
        print(f'\n{WARN} Moderate imbalance (ratio {ratio:.1f}x). '
              f'Class weights applied during training.')
    else:
        print(f'\n{PASS} Class balance is acceptable (ratio {ratio:.1f}x)')


def validate_all():
    print('═' * 54)
    print('  Distress Detection — Data Validation')
    print('═' * 54)

    print('\n── Action (ST-GCN) Datasets ─────────────────────────')
    sources   = ['ntu', 'urfd', 'le2i', 'signal_for_help', 'omnifall']
    total_ok  = 0
    all_counts: Counter = Counter()

    for src in sources:
        ok, n = validate_stgcn_source(src)
        if ok:
            total_ok += n
            y_path = STGCN_DIR / src / 'y.npy'
            if y_path.exists():
                all_counts += Counter(np.load(str(y_path)).tolist())

    print(f'\n  Total action samples: {total_ok}')

    if total_ok < 500:
        print(f'{FAIL} Too few samples ({total_ok}). '
              f'Training will underfit. Add more datasets.')
    elif total_ok < 2000:
        print(f'{WARN} Low sample count ({total_ok}). '
              f'Results may be limited — try to add more data.')
    else:
        print(f'{PASS} Sample count is sufficient for training.')

    check_class_balance(dict(all_counts))

    print('\n── Knife (YOLOv8) Dataset ───────────────────────────')
    knife_ok = validate_knife_dataset()

    print('\n═' * 54)
    if total_ok > 0 and knife_ok:
        print('  ✓ All checks passed — ready to train.')
        print('  Next: run train_stgcn.py then train_yolo.py')
    else:
        missing = []
        if total_ok == 0:
            missing.append('action datasets')
        if not knife_ok:
            missing.append('knife dataset')
        print(f'  ✗ Issues found with: {", ".join(missing)}')
        print('  Fix the issues above before running training.')
    print('═' * 54)


if __name__ == '__main__':
    validate_all()
