import os
import pickle
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


# NTU 25-joint index → COCO 17-joint index  (used only for .skeleton files)
NTU_TO_COCO = {
    3: 0,   # head → nose
    4: 5,   # left shoulder
    8: 6,   # right shoulder
    5: 7,   # left elbow
    9: 8,   # right elbow
    6: 9,   # left wrist
    10: 10, # right wrist
    12: 11, # left hip
    16: 12, # right hip
    13: 13, # left knee
    17: 14, # right knee
    14: 15, # left ankle
    18: 16, # right ankle
}

# NTU action class ID (1-indexed) → our label
# 0=normal, 1=help_signal, 2=collapse_falling, 3=fall_down
NTU_CLASS_MAP = {
    1:  0, 2:  0, 3:  0, 4:  0, 5:  0,
    6:  0, 7:  0, 10: 0, 11: 0, 23: 0,
    43: 3,  # falling down → fall_down
    44: 3,  # lying on floor → fall_down
    53: 2,  # staggering → collapse_falling
    55: 2,  # headache/dizziness → collapse_falling
    26: 1,  # waving hand → help_signal proxy
    28: 1,  # pointing → help_signal proxy
}


class NTUDataset(Dataset):
    """
    Loads NTU RGB+D skeleton data for distress gesture classification.

    Supports two formats auto-detected from the folder contents:
      1. ntu60_hrnet.pkl  — HRNet COCO-17 pre-extracted keypoints (MMAction2 format)
      2. *.skeleton files — raw NTU joint text files

    folder: path to directory containing the data
    window_size: number of frames per clip (default 30)
    augment: apply random flip + jitter during training
    """

    def __init__(self, folder, window_size=30, augment=False):
        self.window_size = window_size
        self.augment = augment
        self.samples = []   # list of (skeleton_array_or_filepath, label)
        self._is_pkl = False

        folder_path = Path(folder)
        pkl_files = list(folder_path.glob('*.pkl'))
        if pkl_files:
            print(f'[NTU] Loading HRNet pkl: {pkl_files[0].name}')
            self._load_pkl(pkl_files[0])
            self._is_pkl = True
        else:
            print('[NTU] Looking for .skeleton files...')
            self._load_skeleton_files(str(folder))

        print(f'[NTU] Loaded {len(self.samples)} samples')

    # ── pkl loader (ntu60_hrnet.pkl / MMAction2 format) ──────────────────────

    def _load_pkl(self, pkl_path):
        with open(pkl_path, 'rb') as f:
            raw = pickle.load(f)

        # Two common layouts:
        #   list of annotation dicts
        #   dict with 'annotations' key
        if isinstance(raw, dict) and 'annotations' in raw:
            annotations = raw['annotations']
        elif isinstance(raw, list):
            annotations = raw
        else:
            print('[NTU] Unknown pkl structure — skipping')
            return

        for ann in annotations:
            label_raw = ann.get('label', -1)
            action_id = int(label_raw) + 1  # pkl is 0-indexed → our map is 1-indexed
            if action_id not in NTU_CLASS_MAP:
                continue

            keypoint = ann.get('keypoint')       # (N, T, 17, 2) or (T, 17, 2)
            keypoint_score = ann.get('keypoint_score')  # (N, T, 17) or (T, 17)
            img_shape = ann.get('img_shape', (1080, 1920))  # (H, W)

            if keypoint is None:
                continue

            # Use first detected person only
            if keypoint.ndim == 4:
                kp = keypoint[0].astype(np.float32)    # (T, 17, 2)
                ks = keypoint_score[0].astype(np.float32) if keypoint_score is not None \
                     else np.ones(kp.shape[:2], dtype=np.float32)
            else:
                kp = keypoint.astype(np.float32)        # (T, 17, 2)
                ks = keypoint_score.astype(np.float32) if keypoint_score is not None \
                     else np.ones(kp.shape[:2], dtype=np.float32)

            # Normalize pixel coords → [0, 1]
            H, W = float(img_shape[0]), float(img_shape[1])
            if W > 1.0:  # pixel space
                kp[:, :, 0] /= W
                kp[:, :, 1] /= H

            T = kp.shape[0]
            skeleton = np.zeros((T, 17, 3), dtype=np.float32)
            skeleton[:, :, :2] = kp
            skeleton[:, :, 2] = ks

            skeleton = _normalize_skeleton(skeleton)
            skeleton = _sample_frames(skeleton, self.window_size)

            self.samples.append((skeleton, NTU_CLASS_MAP[action_id]))

    # ── .skeleton file loader ─────────────────────────────────────────────────

    def _load_skeleton_files(self, folder):
        for root, _dirs, files in os.walk(folder):
            for fname in files:
                if not fname.endswith('.skeleton'):
                    continue
                try:
                    action_id = int(fname[fname.index('A') + 1: fname.index('A') + 4])
                except (ValueError, IndexError):
                    continue
                if action_id not in NTU_CLASS_MAP:
                    continue
                self.samples.append((
                    os.path.join(root, fname),
                    NTU_CLASS_MAP[action_id]
                ))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]

        if isinstance(data, np.ndarray):
            skeleton = data
        else:
            skeleton_ntu = _parse_skeleton_file(data)
            skeleton = _ntu_to_coco(skeleton_ntu)
            skeleton = _normalize_skeleton(skeleton)
            skeleton = _sample_frames(skeleton, self.window_size)

        if self.augment:
            skeleton = self._augment(skeleton)

        x = skeleton.transpose(2, 0, 1).astype(np.float32)  # (C, T, V)
        return x, label

    @staticmethod
    def _augment(skeleton):
        if np.random.rand() < 0.5:
            skeleton = skeleton.copy()
            skeleton[:, :, 0] *= -1
        skeleton = skeleton + np.random.normal(0, 0.01, skeleton.shape)
        return skeleton.astype(np.float32)


# ── Skeleton file parsing helpers ─────────────────────────────────────────────

def _parse_skeleton_file(path):
    """Parse NTU .skeleton file → (T, 25, 3)."""
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')

    idx = 0
    num_frames = int(lines[idx]); idx += 1
    frames = []

    for _ in range(num_frames):
        num_bodies = int(lines[idx]); idx += 1
        bodies = []
        for _b in range(num_bodies):
            idx += 1
            num_joints = int(lines[idx]); idx += 1
            joints = []
            for _j in range(num_joints):
                vals = lines[idx].split(); idx += 1
                joints.append([float(vals[0]), float(vals[1]), float(vals[2])])
            bodies.append(np.array(joints))
        frames.append(bodies[0] if bodies else np.zeros((25, 3)))

    return np.array(frames)


def _ntu_to_coco(skeleton_ntu):
    """Convert (T, 25, 3) NTU → (T, 17, 3) COCO."""
    T = skeleton_ntu.shape[0]
    coco = np.zeros((T, 17, 3), dtype=np.float32)
    coco[:, :, 2] = 1.0
    for ntu_idx, coco_idx in NTU_TO_COCO.items():
        coco[:, coco_idx, :2] = skeleton_ntu[:, ntu_idx, :2]
    return coco


def _normalize_skeleton(skeleton):
    """Center on hip midpoint, scale by torso height. skeleton: (T, 17, 3)"""
    left_hip  = skeleton[:, 11, :2]
    right_hip = skeleton[:, 12, :2]
    center    = (left_hip + right_hip) / 2.0

    left_shoulder  = skeleton[:, 5, :2]
    right_shoulder = skeleton[:, 6, :2]
    shoulder_mid   = (left_shoulder + right_shoulder) / 2.0
    torso_height   = np.linalg.norm(shoulder_mid - center, axis=1).mean()
    scale = torso_height if torso_height > 1e-6 else 1.0

    out = skeleton.copy()
    out[:, :, :2] = (skeleton[:, :, :2] - center[:, np.newaxis, :]) / scale
    return out


def _sample_frames(skeleton, window_size):
    """Resample to exactly window_size frames."""
    T = skeleton.shape[0]
    if T == window_size:
        return skeleton
    indices = np.linspace(0, T - 1, window_size).astype(int)
    return skeleton[indices]
