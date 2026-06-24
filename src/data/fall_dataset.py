import os
import cv2
import numpy as np
from torch.utils.data import Dataset

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# MediaPipe landmark index → COCO 17-joint index
MP_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
    11: 5, 12: 6, 13: 7, 14: 8,
    15: 9, 16: 10, 23: 11, 24: 12,
    25: 13, 26: 14, 27: 15, 28: 16,
}

# Label assignments
LABEL_NORMAL           = 0
LABEL_HELP_SIGNAL      = 1
LABEL_COLLAPSE_FALLING = 2
LABEL_FALL_DOWN        = 3


def extract_skeleton_from_video(video_path, window_size=30):
    """
    Extract COCO-17 skeleton sequence from a video using MediaPipe Pose.
    Returns (window_size, 17, 3) array.
    """
    assert _MP_AVAILABLE, "mediapipe not installed — run: pip install mediapipe"

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    frames = []

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            skeleton = np.zeros((17, 3), dtype=np.float32)
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                for mp_idx, coco_idx in MP_TO_COCO.items():
                    skeleton[coco_idx] = [lm[mp_idx].x, lm[mp_idx].y,
                                          lm[mp_idx].visibility]
            frames.append(skeleton)

    cap.release()

    if not frames:
        return np.zeros((window_size, 17, 3), dtype=np.float32)

    arr = np.array(frames)
    T = arr.shape[0]
    indices = np.linspace(0, T - 1, window_size).astype(int)
    return arr[indices]


class URFDDataset(Dataset):
    """
    University of Rzeszow Fall Detection Dataset.

    Expected folder layout:
        urfd/
            fall-*/     → label: fall_down (3)
            adl-*/      → label: normal (0)

    Each subfolder contains sequential PNG frames.
    """

    def __init__(self, root, window_size=30, augment=False,
                 skeleton_cache_dir=None):
        self.window_size = window_size
        self.augment = augment
        self.cache_dir = skeleton_cache_dir
        self.samples = []  # (sequence_dir, label)

        # Walk up to 2 levels to find fall-* / adl-* sequence folders
        found_any = self._scan_dir(root, depth=0)
        if not found_any:
            # Fallback: search one level deeper (e.g. root/urfd/fall-*)
            for sub in sorted(os.listdir(root)):
                sub_path = os.path.join(root, sub)
                if os.path.isdir(sub_path):
                    self._scan_dir(sub_path, depth=1)

    def _scan_dir(self, path, depth):
        found = False
        for folder in sorted(os.listdir(path)):
            full = os.path.join(path, folder)
            if not os.path.isdir(full):
                continue
            fl = folder.lower()
            if any(fl.startswith(p) for p in ('fall', 'falls')):
                self.samples.append((full, LABEL_FALL_DOWN))
                found = True
            elif any(fl.startswith(p) for p in ('adl', 'normal', 'notfall', 'no_fall', 'daily')):
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

        x = skeleton.transpose(2, 0, 1).astype(np.float32)  # (C, T, V)
        return x, label

    def _load_skeleton(self, seq_dir):
        cache_path = None
        if self.cache_dir:
            name = os.path.basename(seq_dir) + '.npy'
            cache_path = os.path.join(self.cache_dir, name)
            if os.path.exists(cache_path):
                return np.load(cache_path)

        skeleton = self._extract_from_frames(seq_dir)

        if cache_path:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(cache_path, skeleton)

        return skeleton

    def _extract_from_frames(self, seq_dir, max_frames=60):
        assert _MP_AVAILABLE, "mediapipe not installed"

        all_frames = sorted([
            f for f in os.listdir(seq_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Sample evenly to keep extraction fast (avoid hours of compute per dataset)
        if len(all_frames) > max_frames:
            step = len(all_frames) // max_frames
            all_frames = all_frames[::step][:max_frames]

        skeletons = []
        try:
            pose_solution = mp.solutions.pose
            with pose_solution.Pose(static_image_mode=True,
                                    min_detection_confidence=0.3) as pose:
                for fname in all_frames:
                    img = cv2.imread(os.path.join(seq_dir, fname))
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    result = pose.process(rgb)

                    sk = np.zeros((17, 3), dtype=np.float32)
                    if result.pose_landmarks:
                        lm = result.pose_landmarks.landmark
                        for mp_idx, coco_idx in MP_TO_COCO.items():
                            sk[coco_idx] = [lm[mp_idx].x, lm[mp_idx].y,
                                            lm[mp_idx].visibility]
                    skeletons.append(sk)
        except Exception as e:
            print(f'  [WARN] MediaPipe extraction failed for {seq_dir}: {e}')
            return np.zeros((self.window_size, 17, 3), dtype=np.float32)

        if not skeletons:
            return np.zeros((self.window_size, 17, 3), dtype=np.float32)

        arr = np.array(skeletons)
        T = arr.shape[0]
        indices = np.linspace(0, T - 1, self.window_size).astype(int)
        return arr[indices]


class Le2iDataset(Dataset):
    """
    Le2i / ImViA Fall Detection Dataset.

    Expected folder layout:
        le2i/
            Fall/         → label: fall_down (3)
            NotFall/      → label: normal (0)

    Each subfolder contains .avi or .mp4 video files.
    """

    def __init__(self, root, window_size=30, augment=False,
                 skeleton_cache_dir=None):
        self.window_size = window_size
        self.augment = augment
        self.cache_dir = skeleton_cache_dir
        self.samples = []

        # Walk looking for Videos/ subdirectories (Coffee_room/Home layout)
        # or video files directly in scene folders (Lecture_room/Office layout)
        for dirpath, dirnames, files in os.walk(root):
            folder_name = os.path.basename(dirpath).lower().replace(' ', '_')

            videos = [f for f in sorted(files)
                      if f.lower().endswith(('.avi', '.mp4', '.mkv', '.mov'))]
            if not videos:
                continue

            if folder_name in ('videos', 'video'):
                # Inside a Videos/ subdir — check sibling Annotation_files/
                parent = os.path.dirname(dirpath)
                ann_dir = next(
                    (os.path.join(parent, d) for d in os.listdir(parent)
                     if 'annotation' in d.lower() and os.path.isdir(os.path.join(parent, d))),
                    None
                )
                for f in videos:
                    label = _le2i_label(f, ann_dir)
                    self.samples.append((os.path.join(dirpath, f), label))
            else:
                # Videos directly in scene folder (Lecture room, Office, etc.)
                for f in videos:
                    self.samples.append((os.path.join(dirpath, f), LABEL_FALL_DOWN))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        cache_path = None
        if self.cache_dir:
            name = os.path.basename(video_path) + '.npy'
            cache_path = os.path.join(self.cache_dir, name)

        if cache_path and os.path.exists(cache_path):
            skeleton = np.load(cache_path)
        else:
            skeleton = extract_skeleton_from_video(video_path, self.window_size)
            if cache_path:
                os.makedirs(self.cache_dir, exist_ok=True)
                np.save(cache_path, skeleton)

        if self.augment:
            skeleton = _augment(skeleton)

        x = skeleton.transpose(2, 0, 1).astype(np.float32)
        return x, label


class SignalForHelpDataset(Dataset):
    """
    Signal for Help gesture dataset (Politecnico di Torino).

    Expected folder layout:
        signal_for_help/
            signal/       → label: help_signal (1)
            no_signal/    → label: normal (0)

    Each subfolder contains video files.
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
            name = os.path.basename(video_path) + '.npy'
            cache_path = os.path.join(self.cache_dir, name)

        if cache_path and os.path.exists(cache_path):
            skeleton = np.load(cache_path)
        else:
            skeleton = extract_skeleton_from_video(video_path, self.window_size)
            if cache_path:
                os.makedirs(self.cache_dir, exist_ok=True)
                np.save(cache_path, skeleton)

        if self.augment:
            skeleton = _augment(skeleton)

        x = skeleton.transpose(2, 0, 1).astype(np.float32)
        return x, label


def _le2i_label(video_name, ann_dir):
    """
    Determine fall/normal label for a Le2i video from its annotation file.
    Annotation format: lines of 'frame_number label' where label 1 = fall.
    Falls back to LABEL_FALL_DOWN if annotation not found (it's a fall dataset).
    """
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
    """Shared augmentation: horizontal flip + Gaussian jitter."""
    skeleton = skeleton.copy()
    if np.random.rand() < 0.5:
        skeleton[:, :, 0] *= -1
    skeleton += np.random.normal(0, 0.01, skeleton.shape)
    return skeleton.astype(np.float32)
