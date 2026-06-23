import os
import numpy as np
from torch.utils.data import Dataset


# NTU 25-joint index → COCO 17-joint index
# Joints without a match are left as zero
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

# NTU action class ID → our label index
# 0=normal, 1=help_signal, 2=collapse_falling, 3=fall_down
NTU_CLASS_MAP = {
    1:  0, 2:  0, 3:  0, 4:  0, 5:  0,
    6:  0, 7:  0, 10: 0, 11: 0, 23: 0,  # normal daily activities
    43: 3,                                # falling down → fall_down
    44: 3,                                # lying on floor → fall_down
    53: 2,                                # staggering → collapse_falling
    55: 2,                                # headache/dizziness → collapse_falling
    26: 1,                                # waving hand → help_signal (proxy)
    28: 1,                                # pointing → help_signal (proxy)
}


def _parse_skeleton_file(path):
    """
    Parse a single NTU .skeleton file.
    Returns array of shape (T, 25, 3) — frames × joints × (x, y, z).
    Only keeps the first detected body.
    """
    with open(path, 'r') as f:
        lines = f.read().strip().split('\n')

    idx = 0
    num_frames = int(lines[idx]); idx += 1
    frames = []

    for _ in range(num_frames):
        num_bodies = int(lines[idx]); idx += 1
        bodies = []
        for b in range(num_bodies):
            idx += 1  # body info line
            num_joints = int(lines[idx]); idx += 1
            joints = []
            for j in range(num_joints):
                vals = lines[idx].split(); idx += 1
                x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                joints.append([x, y, z])
            bodies.append(np.array(joints))
        frames.append(bodies[0] if bodies else np.zeros((25, 3)))

    return np.array(frames)  # (T, 25, 3)


def _ntu_to_coco(skeleton_ntu):
    """
    Convert (T, 25, 3) NTU skeleton to (T, 17, 3) COCO skeleton.
    Unmapped joints are zero-filled.
    """
    T = skeleton_ntu.shape[0]
    coco = np.zeros((T, 17, 3), dtype=np.float32)
    coco[:, :, 2] = 1.0  # confidence = 1 for NTU (ground truth skeleton)
    for ntu_idx, coco_idx in NTU_TO_COCO.items():
        coco[:, coco_idx, :2] = skeleton_ntu[:, ntu_idx, :2]
    return coco


def _normalize_skeleton(skeleton):
    """
    Center skeleton on hip midpoint, scale by torso height.
    skeleton: (T, 17, 3)
    """
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
    """
    Fixed-length clip sampling via linear interpolation of frame indices.
    skeleton: (T, 17, 3) → (window_size, 17, 3)
    """
    T = skeleton.shape[0]
    if T == window_size:
        return skeleton
    indices = np.linspace(0, T - 1, window_size).astype(int)
    return skeleton[indices]


class NTUDataset(Dataset):
    """
    Loads NTU RGB+D 120 skeleton data for our 4 distress classes.

    folder: path to directory containing .skeleton files
    window_size: number of frames per clip
    augment: apply random flips and jitter during training
    """

    def __init__(self, folder, window_size=30, augment=False):
        self.window_size = window_size
        self.augment = augment
        self.samples = []  # list of (file_path, label)

        for fname in os.listdir(folder):
            if not fname.endswith('.skeleton'):
                continue
            action_id = int(fname[fname.index('A') + 1: fname.index('A') + 4])
            if action_id not in NTU_CLASS_MAP:
                continue
            self.samples.append((
                os.path.join(folder, fname),
                NTU_CLASS_MAP[action_id]
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        skeleton_ntu = _parse_skeleton_file(path)
        skeleton     = _ntu_to_coco(skeleton_ntu)
        skeleton     = _normalize_skeleton(skeleton)
        skeleton     = _sample_frames(skeleton, self.window_size)

        if self.augment:
            skeleton = self._augment(skeleton)

        # (T, V, C) → (C, T, V) for ST-GCN
        x = skeleton.transpose(2, 0, 1).astype(np.float32)
        return x, label

    @staticmethod
    def _augment(skeleton):
        # random horizontal flip
        if np.random.rand() < 0.5:
            skeleton = skeleton.copy()
            skeleton[:, :, 0] *= -1
        # Gaussian joint jitter
        skeleton = skeleton + np.random.normal(0, 0.01, skeleton.shape)
        return skeleton.astype(np.float32)
