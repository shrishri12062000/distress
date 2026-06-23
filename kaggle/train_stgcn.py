"""
Step 3 — Train ST-GCN action classifier on Kaggle GPU.
Run after prepare_data.py and validate_data.py.
"""

import os
import sys
import yaml
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

REPO_DIR    = Path('/kaggle/working/distress-gesture-detection')
WORKING     = Path('/kaggle/working/distress_detection')
PROCESSED   = WORKING / 'processed' / 'stgcn'
CKPT_DIR    = WORKING / 'checkpoints' / 'stgcn'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_DIR))

from src.models.stgcn import STGCN

with open(REPO_DIR / 'configs' / 'stgcn.yaml') as f:
    cfg = yaml.safe_load(f)

SEED         = cfg['training']['seed']
EPOCHS       = cfg['training']['epochs']
BATCH_SIZE   = cfg['training']['batch_size']
LR           = cfg['training']['learning_rate']
LR_STEPS     = cfg['training']['lr_decay_step']
LR_GAMMA     = cfg['training']['lr_decay_gamma']
WEIGHT_DECAY = cfg['training']['weight_decay']
MOMENTUM     = cfg['training']['momentum']
VAL_SPLIT    = cfg['training']['val_split']
PATIENCE     = cfg['training']['early_stopping_patience']
CLASS_WEIGHTS = list(cfg['training']['class_weights'].values())

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')


class NpyDataset(Dataset):
    def __init__(self, x_path, y_path, augment=False):
        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            if np.random.rand() < 0.5:
                x[0] *= -1
            x += np.random.normal(0, 0.01, x.shape).astype(np.float32)
        return torch.from_numpy(x), int(self.y[idx])


def load_all_datasets():
    train_sets, val_sets = [], []
    for src in ['ntu', 'urfd', 'le2i', 'signal_for_help', 'omnifall']:
        x_path = PROCESSED / src / 'X.npy'
        y_path = PROCESSED / src / 'y.npy'
        if not (x_path.exists() and y_path.exists()):
            print(f'  [SKIP] {src}')
            continue
        full  = NpyDataset(str(x_path), str(y_path))
        n_val = max(1, int(len(full) * VAL_SPLIT))
        n_tr  = len(full) - n_val
        tr, vl = random_split(full, [n_tr, n_val],
                               generator=torch.Generator().manual_seed(SEED))
        # Augment training split only
        tr_aug = NpyDataset(str(x_path), str(y_path), augment=True)
        tr.dataset = tr_aug
        train_sets.append(tr)
        val_sets.append(vl)
        print(f'  {src}: {n_tr} train / {n_val} val')

    if not train_sets:
        raise RuntimeError('No processed data found. Run prepare_data.py first.')
    return ConcatDataset(train_sets), ConcatDataset(val_sets)


def train():
    print('\n── Loading datasets ─────────────────────────────────')
    train_ds, val_ds = load_all_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f'  Train: {len(train_ds)}  Val: {len(val_ds)}')

    print('\n── Building ST-GCN model ────────────────────────────')
    model = STGCN(
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes'],
        dropout=cfg['model']['dropout'],
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {total_params:,}')

    weights   = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                milestones=LR_STEPS,
                                                gamma=LR_GAMMA)

    best_val_acc = 0.0
    no_improve   = 0

    print('\n── Training ─────────────────────────────────────────')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = tr_correct = tr_total = 0
        for x, y in tqdm(train_loader, desc=f'Ep {epoch}/{EPOCHS}', leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss    += loss.item() * x.size(0)
            tr_correct += (model(x).argmax(1) == y).sum().item()
            tr_total   += x.size(0)
        scheduler.step()

        model.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                vl_loss    += criterion(out, y).item() * x.size(0)
                vl_correct += (out.argmax(1) == y).sum().item()
                vl_total   += x.size(0)

        t_acc = tr_correct / tr_total * 100
        v_acc = vl_correct / vl_total * 100
        print(f'Ep {epoch:03d} | train {tr_loss/tr_total:.4f} {t_acc:.1f}% '
              f'| val {vl_loss/vl_total:.4f} {v_acc:.1f}%')

        ckpt = {'epoch': epoch, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': v_acc, 'cfg': cfg}
        torch.save(ckpt, str(CKPT_DIR / 'last.pth'))

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            no_improve   = 0
            torch.save(ckpt, str(CKPT_DIR / 'best.pth'))
            print(f'  ✓ Best model saved (val_acc={v_acc:.2f}%)')
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'  Early stopping at epoch {epoch}')
                break

    print(f'\n── Done. Best val acc: {best_val_acc:.2f}% ─────────')
    print(f'   Checkpoint: {CKPT_DIR / "best.pth"}')


if __name__ == '__main__':
    train()
