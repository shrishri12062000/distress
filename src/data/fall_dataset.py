import os
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# MediaPipe BlazePose landmark index → COCO 17-joint index
# (indices are identical between old solutions API and new Tasks API)
MP_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
    11: 5, 12: 6, 13: 7, 14: 8,
    15: 9, 16: 10, 23: 11, 24: 12,
    25: 13, 26: 14, 27: 15, 28: 16,
}

LABEL_NORMAL           = 0
LABEL_HELP_SIGNAL      = 1
LABEL_COLLAPSE_FALLING = 2
LABEL_FALL_DOWN        = 3

_POSE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
)


def _get_pose_model_path():
    if os.path.exists('/kaggle/working'):
        return '/kaggle/working/pose_landmarker_full.task'
    path = Path.home() / '.cache' / 'mediapipe' / 'pose_landmarker_full.task'
    return str(path)


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
    """Create a MediaPipe Tasks PoseLandmarker in IMAGE mode."""
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
    """Convert PoseLandmarker result landmarks to (17, 3) COCO array."""
    sk = np.zeros((17, 3), dtype=np.float32)
    if landmarks:
        lm = landmarks[0]  # first person
        for mp_idx, coco_idx in MP_TO_COCO.items():
            if mp_idx < len(lm):
                sk[coco_idx] = [lm[mp_idx].x, lm[mp_idx].y, lm[mp_idx].visibility]
    return sk


def extract_skeleton_from_video(video_path, window_size=30):
    """
    Extract COCO-17 skeleton sequence from a video using MediaPipe Tasks API.
    Returns (window_size, 17, 3) array.
    """
    assert _MP_AVAILABLE, 'mediapipe not installed'

    cap = cv2.VideoCapture(video_path)
    frames = []

    with _make_pose_detector() as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)
            frames.append(_landmarks_to_coco(result.pose_landmarks))

    cap.release()

    if not frames:
        return np.zeros((window_size, 17, 3), dtype=np.float32)

    arr = np.array(frames)
    indices = np.linspace(0, len(arr) - 1, window_size).astype(int)
    return arr[indices]


class URFDDataset(Dataset):
    """
    University of Rzeszow Fall Detection Dataset.

    Layout (Kaggle): UR_fall_detection_cam0_rgb/
                         fall-NN-cam0-rgb/  → fall_down (3)
                         adl-NN-cam0-rgb/   → normal (0)
    Each subfolder contains sequential PNG frames.
    """

    def __init__(self, root, window_size=30, augment=False,
                 skeleton_cache_dir=None):
        self.window_size = window_size
        self.augment = augment
        self.cache_dir = skeleton_cache_dir
        self.samples = []

        found = self._scan_dir(root)
        if not found:
            for sub in sorted(os.listdir(root)):
                sub_path = os.path.join(root, sub)
                if os.path.isdir(sub_path):
                    self._scan_dir(sub_path)

    def _scan_dir(self, path):
        found = False
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if not os.path.isdir(full):
                continue
            nl = name.lower()
            if nl.startswith('fall'):
                self.samples.append((full, LABEL_FALL_DOWN))
                found = True
            elif any(nl.startswith(p) for p in ('adl', 'normal', 'notfall',
                                                  'no_fall', 'daily')):
                self.samples.append((full, LABEL_NORMAL))
                found = True
        return found

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, label = self.samples[idx]
        skeleton = self._load_skeleton(seq_dir)
        if self.augment:
            skeleton = _augment(skeleton)
        return skeleton.transpose(2, 0, 1).astype(np.float32), label

    def _load_skeleton(self, seq_dir):
        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir,
                                      os.path.basename(seq_dir) + '.npy')
            if os.path.exists(cache_path):
                return np.load(cache_path)

        skeleton = self._extract_from_frames(seq_dir)
        if cache_path:
            np.save(cache_path, skeleton)
        return skeleton

    def _extract_from_frames(self, seq_dir):
        assert _MP_AVAILABLE, 'mediapipe not installed'

        frames_list = sorted([
            f for f in os.listdir(seq_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not frames_list:
            return np.zeros((self.window_size, 17, 3), dtype=np.float32)

        skeletons = []
        with _make_pose_detector() as detector:
            for fname in frames_list:
                img = cv2.imread(os.path.join(seq_dir, fname))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)
                skeletons.append(_landmarks_to_coco(result.pose_landmarks))

        if not skeletons:
            return np.zeros((self.window_size, 17, 3), dtype=np.float32)

        arr = np.array(skeletons)
        indices = np.linspace(0, len(arr) - 1, self.window_size).astype(int)
        return arr[indices]


class Le2iDataset(Dataset):
    """
    Le2i / ImViA Fall Detection Dataset.

    Kaggle layout (tuyenldvn/falldataset-imvia):
        Coffee_room_01/Coffee_room_01/Videos/*.avi  +  Annotation_files/
        Lecture_room/Lecture room/*.avi
        Office/Office/*.avi
    """

    def __init__(self, root, window_size=30, augment=False,
                 skeleton_cache_dir=None):
        self.window_size = window_size
        self.augment = augment
        self.cache_dir = skeleton_cache_dir
        self.samples = []

        for dirpath, dirnames, files in os.walk(root):
            folder = os.path.basename(dirpath).lower().replace(' ', '_')
            videos = [f for f in sorted(files)
                      if f.lower().endswith(('.avi', '.mp4', '.mkv', '.mov'))]
            if not videos:
                continue

            if folder in ('videos', 'video'):
                parent = os.path.dirname(dirpath)
                ann_dir = next(
                    (os.path.join(parent, d) for d in os.listdir(parent)
                     if 'annotation' in d.lower()
                     and os.path.isdir(os.path.join(parent, d))),
                    None,
                )
                for f in videos:
                    label = _le2i_label(f, ann_dir)
                    self.samples.append((os.path.join(dirpath, f), label))
            elif any(k in folder for k in ('lecture', 'office', 'home',
                                            'coffee', 'kitchen')):
                for f in videos:
                    self.samples.append((os.path.join(dirpath, f),
                                         LABEL_FALL_DOWN))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir,
                                      os.path.basename(video_path) + '.npy')

        if cache_path and os.path.exists(cache_path):
            skeleton = np.load(cache_path)
        else:
            skeleton = extract_skeleton_from_video(video_path, self.window_size)
            if cache_path:
                np.save(cache_path, skeleton)

        if self.augment:
            skeleton = _augment(skeleton)
        return skeleton.transpose(2, 0, 1).astype(np.float32), label


class SignalForHelpDataset(Dataset):
    """
    Signal for Help gesture dataset.
    Layout: signal/ → help_signal (1)   no_signal/ → normal (0)
    """

    def __init__(self, root, window_size=30, augment=False,
                 skeleton_cache_dir=None):
        self.window_size = window_size
        self.augment = augment
        self.cache_dir = skeleton_cache_dir
        self.samples = []

        for category in ['signal', 'no_signal']:
            label = LABEL_HELP_SIGNAL if category == 'signal' else LABEL_NORMAL
            cat_dir = os.path.join(root, category)
            if not os.path.isdir(cat_dir):
                continue
            for f in sorted(os.listdir(cat_dir)):
                if f.lower().endswith(('.avi', '.mp4', '.mkv', '.mov')):
                    self.samples.append((os.path.join(cat_dir, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        cache_path = None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir,
                                      os.path.basename(video_path) + '.npy')

        if cache_path and os.path.exists(cache_path):
            skeleton = np.load(cache_path)
        else:
            skeleton = extract_skeleton_from_video(video_path, self.window_size)
            if cache_path:
                np.save(cache_path, skeleton)

        if self.augment:
            skeleton = _augment(skeleton)
        return skeleton.transpose(2, 0, 1).astype(np.float32), label


def _le2i_label(video_name, ann_dir):
    """Determine label from Le2i annotation file. Returns fall_down by default."""
    if not ann_dir or not os.path.isdir(ann_dir):
        return LABEL_FALL_DOWN
    stem = os.path.splitext(video_name)[0]
    for ann_file in os.listdir(ann_dir):
        if stem.lower() in ann_file.lower():
            try:
                with open(os.path.join(ann_dir, ann_file)) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[-1] in ('1', '2'):
                            return LABEL_FALL_DOWN
                return LABEL_NORMAL
            except OSError:
                pass
    return LABEL_FALL_DOWN


def _augment(skeleton):
    skeleton = skeleton.copy()
    if np.random.rand() < 0.5:
        skeleton[:, :, 0] *= -1
    skeleton += np.random.normal(0, 0.01, skeleton.shape)
    return skeleton.astype(np.float32)
