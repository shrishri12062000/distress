import os
import tempfile
import numpy as np
import cv2
from torch.utils.data import Dataset

try:
    from datasets import load_dataset as hf_load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# MediaPipe BlazePose index → COCO 17-joint index
MP_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
    11: 5, 12: 6, 13: 7, 14: 8,
    15: 9, 16: 10, 23: 11, 24: 12,
    25: 13, 26: 14, 27: 15, 28: 16,
}

# OmniFall label taxonomy → our 4 classes
OMNIFALL_LABEL_MAP = {
    'fall':      2,
    'lie_down':  2,
    'fallen':    3,
    'lying':     3,
    'walk':      0,
    'run':       0,
    'stand':     0,
    'standing':  0,
    'sit_down':  0,
    'sitting':   0,
    'stand_up':  0,
    'move':      0,
    'crouching': 0,
    'unclear':   0,
}

LABEL_NORMAL           = 0
LABEL_COLLAPSE_FALLING = 2
LABEL_FALL_DOWN        = 3

_POSE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
)


def _get_pose_model_path():
    if os.path.exists('/kaggle/working'):
        return '/kaggle/working/pose_landmarker_full.task'
    from pathlib import Path
    return str(Path.home() / '.cache' / 'mediapipe' / 'pose_landmarker_full.task')


def _ensure_pose_model():
    path = _get_pose_model_path()
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        print(f'[MediaPipe] Downloading pose model (~8 MB) → {path}')
        urllib.request.urlretrieve(_POSE_MODEL_URL, path)
        print('[MediaPipe] Download complete')
    return path


def _make_pose_detector():
    model_path = _ensure_pose_model()
    opts = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(opts)


def _landmarks_to_coco(landmarks):
    sk = np.zeros((17, 3), dtype=np.float32)
    if landmarks:
        lm = landmarks[0]
        for mp_idx, coco_idx in MP_TO_COCO.items():
            if mp_idx < len(lm):
                sk[coco_idx] = [lm[mp_idx].x, lm[mp_idx].y, lm[mp_idx].visibility]
    return sk


class OmniFallDataset(Dataset):
    """
    OmniFall 2025 — unified fall detection benchmark.
    Source: https://huggingface.co/datasets/simplexsigil2/omnifall

    Loads the 'train' split (the only split available on HuggingFace).
    Extracts COCO-17 skeletons from video clips using MediaPipe Tasks API.
    """

    # HuggingFace exposes a single 'train' split
    SUPPORTED_SPLITS = ['train']

    def __init__(self, skeleton_cache_dir, splits=None, window_size=30,
                 augment=False):
        assert _HF_AVAILABLE, 'Run: pip install datasets'
        assert _MP_AVAILABLE, 'Run: pip install mediapipe'

        self.window_size = window_size
        self.augment     = augment
        self.cache_dir   = skeleton_cache_dir
        os.makedirs(skeleton_cache_dir, exist_ok=True)

        self.samples = []
        self._data   = {}

        for split in self.SUPPORTED_SPLITS:
            print(f'[OmniFall] Loading split: {split}')
            try:
                ds = hf_load_dataset('simplexsigil2/omnifall', split=split)
            except Exception as e:
                print(f'[OmniFall] Could not load split {split}: {e}')
                continue

            for i, row in enumerate(ds):
                raw_label = row.get('action', row.get('label', 'unclear'))
                if isinstance(raw_label, int):
                    feat = ds.features.get('label') or ds.features.get('action')
                    raw_label = feat.int2str(raw_label) if feat else 'unclear'
                raw_label = str(raw_label).lower().replace(' ', '_')
                label = OMNIFALL_LABEL_MAP.get(raw_label, LABEL_NORMAL)

                vid_id = f'{split}_{i:06d}'
                self._data[vid_id] = row
                self.samples.append((vid_id, label))

        print(f'[OmniFall] Total samples: {len(self.samples)}')
        self._print_distribution()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id, label = self.samples[idx]
        skeleton = self._get_skeleton(vid_id)
        if self.augment:
            skeleton = _augment(skeleton)
        return skeleton.transpose(2, 0, 1).astype(np.float32), label

    def _get_skeleton(self, vid_id):
        cache_path = os.path.join(self.cache_dir, f'{vid_id}.npy')
        if os.path.exists(cache_path):
            return np.load(cache_path)
        skeleton = self._extract_skeleton(vid_id)
        np.save(cache_path, skeleton)
        return skeleton

    def _extract_skeleton(self, vid_id):
        row = self._data[vid_id]
        video_bytes = row.get('video', None)
        if video_bytes is None:
            return np.zeros((self.window_size, 17, 3), dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            if isinstance(video_bytes, bytes):
                f.write(video_bytes)
            elif hasattr(video_bytes, 'read'):
                f.write(video_bytes.read())
            else:
                return np.zeros((self.window_size, 17, 3), dtype=np.float32)
            tmp_path = f.name

        try:
            skeleton = _extract_skeleton_from_video(tmp_path, self.window_size)
        finally:
            os.unlink(tmp_path)

        return skeleton

    def _print_distribution(self):
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        names  = {0: 'normal', 1: 'help_signal',
                  2: 'collapse_falling', 3: 'fall_down'}
        for k, v in sorted(counts.items()):
            print(f'  {names[k]}: {v}')


def _extract_skeleton_from_video(video_path, window_size):
    """Extract COCO-17 skeleton sequence from a video using MediaPipe Tasks API."""
    cap    = cv2.VideoCapture(video_path)
    frames = []

    with _make_pose_detector() as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)
            frames.append(_landmarks_to_coco(result.pose_landmarks))

    cap.release()

    if not frames:
        return np.zeros((window_size, 17, 3), dtype=np.float32)

    arr     = np.array(frames)
    indices = np.linspace(0, len(arr) - 1, window_size).astype(int)
    return arr[indices]


def _augment(skeleton):
    skeleton = skeleton.copy()
    if np.random.rand() < 0.5:
        skeleton[:, :, 0] *= -1
    skeleton += np.random.normal(0, 0.01, skeleton.shape)
    return skeleton.astype(np.float32)
