import os
import yaml
import shutil
import random
from pathlib import Path


def _get_knife_class_id(dataset_dir):
    """Read dataset's data.yaml to find which class ID corresponds to 'knife'."""
    ds = Path(dataset_dir)
    for yaml_name in ('data.yaml', 'dataset.yaml', 'classes.yaml'):
        yp = ds / yaml_name
        if not yp.exists():
            # Search one level deeper
            for subdir in ds.iterdir():
                if subdir.is_dir():
                    yp2 = subdir / yaml_name
                    if yp2.exists():
                        yp = yp2
                        break
        if yp.exists():
            with open(yp) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get('names', [])
            if isinstance(names, list):
                for i, name in enumerate(names):
                    if 'knife' in str(name).lower():
                        return i
            elif isinstance(names, dict):
                for class_id, name in names.items():
                    if 'knife' in str(name).lower():
                        return int(class_id)
    return 0  # default: assume knife is class 0


def _find_image_label_pairs(ds_root):
    """
    Find all (image_path, label_path) pairs in a YOLO dataset regardless of layout.
    Handles: images/train/, images/valid/, images/, train/images/, etc.
    """
    ds = Path(ds_root)
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    # Collect all image files, skipping anything inside a 'labels' folder
    all_images = [
        p for p in ds.rglob('*')
        if p.suffix.lower() in img_extensions
        and 'labels' not in p.parts
    ]

    pairs = []
    for img_path in all_images:
        rel = img_path.relative_to(ds)
        parts = list(rel.parts)

        found = False
        # Try replacing an 'images' component with 'labels'
        for i, part in enumerate(parts):
            if part.lower() in ('images', 'imgs', 'image'):
                label_parts = parts[:]
                label_parts[i] = 'labels'
                label_parts[-1] = img_path.stem + '.txt'
                lbl_path = ds / Path(*label_parts)
                if lbl_path.exists():
                    pairs.append((img_path, lbl_path))
                    found = True
                    break

        if not found:
            # Same directory with .txt
            lbl_path = img_path.parent / (img_path.stem + '.txt')
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))

    return pairs


def merge_knife_datasets(dataset_dirs, output_dir, val_split=0.15, seed=42):
    """
    Merge multiple YOLO-format knife datasets into a single unified dataset.

    Auto-detects layout (images/train/, images/, train/images/, etc.)
    Auto-detects knife class ID from data.yaml.
    Remaps all knife annotations to class 0.
    Outputs a unified dataset with data.yaml for YOLOv8.
    """
    random.seed(seed)
    out = Path(output_dir)
    (out / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (out / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    all_pairs = []  # [(img_path, remapped_knife_lines)]

    for ds_dir in dataset_dirs:
        ds = Path(ds_dir)
        knife_cls = _get_knife_class_id(ds_dir)
        pairs = _find_image_label_pairs(ds)

        if not pairs:
            print(f"[knife_dataset] Skipping {ds_dir} — unexpected layout")
            continue

        print(f"[knife_dataset] {ds.name}: {len(pairs)} pairs found, knife class={knife_cls}")

        kept = 0
        for img_path, lbl_path in pairs:
            with open(lbl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            # Filter to knife annotations and remap to class 0
            knife_lines = []
            for l in lines:
                parts = l.split()
                if parts and parts[0] == str(knife_cls):
                    knife_lines.append('0 ' + ' '.join(parts[1:]))
            if not knife_lines:
                continue
            all_pairs.append((img_path, knife_lines))
            kept += 1

        print(f"  → {kept} images with knife annotations")

    random.shuffle(all_pairs)
    val_count   = max(1, int(len(all_pairs) * val_split))
    val_pairs   = all_pairs[:val_count]
    train_pairs = all_pairs[val_count:]

    def copy_pairs(pairs, split):
        for i, (img_path, knife_lines) in enumerate(pairs):
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
