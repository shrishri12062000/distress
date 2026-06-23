"""
ST-GCN training script — run on Kaggle GPU after prepare_data.py completes.

Usage (inside Kaggle notebook cell):
    !python train_stgcn.py
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

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT  = Path('/content/drive/MyDrive/distress_detection')
PROCESSED   = DRIVE_ROOT / 'processed' / 'stgcn'
CKPT_DIR    = DRIVE_ROOT / 'checkpoints' / 'stgcn'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, '/kaggle/working/distress-gesture-detection')

from src.models.stgcn import STGCN

CONFIG_PATH = '/kaggle/working/distress-gesture-detection/configs/stgcn.yaml'
with open(CONFIG_PATH) as f:
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


class NpyDataset(Dataset):
    """Load pre-processed (X.npy, y.npy) pairs."""

    def __init__(self, x_path, y_path, augment=False):
        self.X = np.load(x_path).astype(np.float32)  # (N, C, T, V)
        self.y = np.load(y_path).astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            if np.random.rand() < 0.5:
                x = x.copy()
                x[0] *= -1  # flip x-channel
            x = x + np.random.normal(0, 0.01, x.shape).astype(np.float32)
        return torch.from_numpy(x), int(self.y[idx])


def load_all_datasets():
    """Load all processed .npy datasets and concatenate."""
    sources = ['ntu', 'urfd', 'le2i']
    train_sets, val_sets = [], []

    for src in sources:
        x_path = PROCESSED / src / 'X.npy'
        y_path = PROCESSED / src / 'y.npy'
        if not (x_path.exists() and y_path.exists()):
            print(f'[SKIP] {src} processed data not found')
            continue

        full = NpyDataset(str(x_path), str(y_path), augment=False)
        n_val = max(1, int(len(full) * VAL_SPLIT))
        n_train = len(full) - n_val
        train_ds, val_ds = random_split(
            full, [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED)
        )
        train_ds.dataset = NpyDataset(str(x_path), str(y_path), augment=True)
        train_sets.append(train_ds)
        val_sets.append(val_ds)
        print(f'  {src}: {n_train} train / {n_val} val')

    if not train_sets:
        raise RuntimeError('No processed datasets found. Run prepare_data.py first.')

    return ConcatDataset(train_sets), ConcatDataset(val_sets)


def train():
    print('\n── Loading datasets ─────────────────────────────────')
    train_ds, val_ds = load_all_datasets()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)}')

    print('\n── Building model ───────────────────────────────────')
    model = STGCN(
        in_channels=cfg['model']['in_channels'],
        num_classes=cfg['model']['num_classes'],
        dropout=cfg['model']['dropout'],
    ).to(DEVICE)

    class_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=LR,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                          nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_STEPS, gamma=LR_GAMMA
    )

    best_val_acc  = 0.0
    no_improve    = 0

    print('\n── Training ─────────────────────────────────────────')
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch}/{EPOCHS} [train]',
                         leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss    += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total   += x.size(0)

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_loss    += criterion(logits, y).item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total   += x.size(0)

        t_loss = train_loss / train_total
        t_acc  = train_correct / train_total * 100
        v_loss = val_loss / val_total
        v_acc  = val_correct / val_total * 100

        print(f'Epoch {epoch:03d} | '
              f'Train loss: {t_loss:.4f} acc: {t_acc:.2f}% | '
              f'Val loss: {v_loss:.4f} acc: {v_acc:.2f}%')

        # ── Checkpoint ──
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_acc': v_acc,
            'cfg': cfg,
        }
        torch.save(ckpt, str(CKPT_DIR / 'last.pth'))

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            no_improve = 0
            torch.save(ckpt, str(CKPT_DIR / 'best.pth'))
            print(f'  ✓ New best model saved (val_acc={v_acc:.2f}%)')
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f'  Early stopping at epoch {epoch}')
                break

    print(f'\n── Training complete. Best val acc: {best_val_acc:.2f}% ──')
    print(f'   Best checkpoint: {CKPT_DIR / "best.pth"}')


if __name__ == '__main__':
    train()
