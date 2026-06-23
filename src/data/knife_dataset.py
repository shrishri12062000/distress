import os
import yaml
import shutil
import random
from pathlib import Path


def merge_knife_datasets(dataset_dirs, output_dir, val_split=0.15, seed=42):
    """
    Merge multiple YOLO-format knife datasets into a single unified dataset.

    Each dataset_dir must follow YOLO layout:
        dataset_dir/
            images/train/   or   images/
            labels/train/   or   labels/

    Only keeps annotations where class == 0 (knife).
    Outputs a unified dataset with data.yaml for YOLOv8.
    """
    random.seed(seed)
    out = Path(output_dir)
    (out / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (out / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    all_pairs = []  # [(img_path, label_path)]

    for ds_dir in dataset_dirs:
        ds = Path(ds_dir)
        img_dirs = [
            ds / 'images' / 'train',
            ds / 'images' / 'valid',
            ds / 'images',
        ]
        lbl_dirs = [
            ds / 'labels' / 'train',
            ds / 'labels' / 'valid',
            ds / 'labels',
        ]

        img_dir = next((d for d in img_dirs if d.exists()), None)
        lbl_dir = next((d for d in lbl_dirs if d.exists()), None)
        if img_dir is None or lbl_dir is None:
            print(f"[knife_dataset] Skipping {ds_dir} — unexpected layout")
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                continue
            # Validate label contains a knife annotation
            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            knife_lines = [l for l in lines if l.startswith('0 ')]
            if not knife_lines:
                continue
            all_pairs.append((img_path, lbl_path, knife_lines))

    random.shuffle(all_pairs)
    val_count = max(1, int(len(all_pairs) * val_split))
    val_pairs   = all_pairs[:val_count]
    train_pairs = all_pairs[val_count:]

    def copy_pairs(pairs, split):
        for i, (img_path, _, knife_lines) in enumerate(pairs):
            dst_img = out / 'images' / split / f"{split}_{i:05d}{img_path.suffix}"
            dst_lbl = out / 'labels' / split / f"{split}_{i:05d}.txt"
            shutil.copy2(img_path, dst_img)
            with open(dst_lbl, 'w') as f:
                f.write('\n'.join(knife_lines) + '\n')

    copy_pairs(train_pairs, 'train')
    copy_pairs(val_pairs,   'val')

    data_yaml = {
        'path': str(out.resolve()),
        'train': 'images/train',
        'val':   'images/val',
        'nc':    1,
        'names': ['knife'],
    }
    with open(out / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"[knife_dataset] Merged {len(train_pairs)} train / "
          f"{len(val_pairs)} val images → {output_dir}")
    return str(out / 'data.yaml')
