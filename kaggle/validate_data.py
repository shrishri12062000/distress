"""
Step 2 — Data Validation (run after prepare_data.py, before training).
"""

import os
import sys
import numpy as np
import yaml
from pathlib import Path
from collections import Counter

WORKING     = Path('/kaggle/working/distress_detection')
OUTPUT_ROOT = WORKING / 'processed'
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
        print(f'{WARN} {name}: skipped during preparation')
        return False, 0

    if not x_path.exists() or not y_path.exists():
        print(f'{FAIL} {name}: X.npy or y.npy missing — re-run prepare_data.py')
        return False, 0

    X = np.load(str(x_path))
    y = np.load(str(y_path))

    if X.ndim != 4 or X.shape[1] != 3 or X.shape[3] != 17:
        print(f'{FAIL} {name}: unexpected shape {X.shape} (expected N×3×T×17)')
        return False, 0

    if len(X) != len(y):
        print(f'{FAIL} {name}: X/y mismatch ({len(X)} vs {len(y)})')
        return False, 0

    nan_pct = np.isnan(X).mean() * 100
    if nan_pct > 0:
        print(f'{WARN} {name}: {nan_pct:.1f}% NaN values in skeleton data')

    counts  = Counter(y.tolist())
    dist    = '  '.join(f'{CLASS_NAMES.get(k,k)}: {v}'
                        for k, v in sorted(counts.items()))
    print(f'{PASS} {name}: {len(X)} samples  [{dist}]')
    return True, len(X)


def validate_knife():
    if not KNIFE_DIR.exists():
        print(f'{FAIL} Knife dataset missing — re-run prepare_data.py')
        return False

    yaml_path = KNIFE_DIR / 'data.yaml'
    if not yaml_path.exists():
        print(f'{FAIL} Knife data.yaml missing')
        return False

    n_train = len(list((KNIFE_DIR / 'images' / 'train').glob('*.jpg'))) + \
              len(list((KNIFE_DIR / 'images' / 'train').glob('*.png')))
    n_val   = len(list((KNIFE_DIR / 'images' / 'val').glob('*.jpg'))) + \
              len(list((KNIFE_DIR / 'images' / 'val').glob('*.png')))

    if n_train == 0:
        print(f'{FAIL} Knife: no training images found')
        return False

    print(f'{PASS} Knife: {n_train} train / {n_val} val images')
    return True


def check_balance(counts):
    if not counts:
        return
    total   = sum(counts.values())
    min_cls = min(counts, key=counts.get)
    max_cls = max(counts, key=counts.get)
    ratio   = counts[max_cls] / max(counts[min_cls], 1)

    print('\n── Overall Class Distribution ───────────────────────')
    for k in sorted(counts):
        n   = counts[k]
        pct = n / total * 100
        bar = '█' * int(pct / 2)
        print(f'  {CLASS_NAMES[k]:<20} {n:>5}  ({pct:5.1f}%)  {bar}')

    if ratio > 10:
        print(f'\n{WARN} Severe imbalance ({ratio:.1f}x) — class weights will compensate.')
    elif ratio > 5:
        print(f'\n{WARN} Moderate imbalance ({ratio:.1f}x).')
    else:
        print(f'\n{PASS} Class balance acceptable ({ratio:.1f}x).')


def validate_all():
    print('═' * 54)
    print('  Distress Detection — Data Validation')
    print('═' * 54)
    print('\n── Action (ST-GCN) Datasets ─────────────────────────')

    total    = 0
    all_counts: Counter = Counter()
    for src in ['ntu', 'urfd', 'le2i', 'signal_for_help', 'omnifall']:
        ok, n = validate_stgcn_source(src)
        if ok:
            total += n
            y = np.load(str(STGCN_DIR / src / 'y.npy'))
            all_counts += Counter(y.tolist())

    print(f'\n  Total action samples: {total}')
    if total < 500:
        print(f'{FAIL} Too few samples — add more datasets.')
    elif total < 2000:
        print(f'{WARN} Low sample count — results may be limited.')
    else:
        print(f'{PASS} Sample count sufficient for training.')

    check_balance(dict(all_counts))

    print('\n── Knife (YOLOv8) Dataset ───────────────────────────')
    knife_ok = validate_knife()

    print('\n' + '═' * 54)
    if total > 0 and knife_ok:
        print('  ✓ All checks passed — ready to train.')
        print('  Next: run train_stgcn.py')
    else:
        print('  ✗ Fix issues above before training.')
    print('═' * 54)


if __name__ == '__main__':
    validate_all()
