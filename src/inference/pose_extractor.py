import os
import numpy as np
import cv2
from pathlib import Path

import mediapipe as mp

# MediaPipe BlazePose index → COCO 17-joint index
MP_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
    11: 5, 12: 6, 13: 7, 14: 8,
    15: 9, 16: 10, 23: 11, 24: 12,
    25: 13, 26: 14, 27: 15, 28: 16,
}

_POSE_MODEL_URL = (
    'https://storage.googleapis.com/mediapipe-models/'
    'pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
)


def _get_pose_model_path():
    # Prefer project-local models/ dir so it ships with the repo
    project_root = Path(__file__).resolve().parent.parent.parent
    local = project_root / 'models' / 'pose_landmarker_full.task'
    if local.exists():
        return str(local)
    # Fall back to user cache
    cache = Path.home() / '.cache' / 'mediapipe' / 'pose_landmarker_full.task'
    return str(cache)


def _ensure_pose_model():
    path = _get_pose_model_path()
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        print(f'[PoseExtractor] Downloading pose model (~8 MB) → {path}')
        urllib.request.urlretrieve(_POSE_MODEL_URL, path)
        print('[PoseExtractor] Download complete')
    return path


class PoseExtractor:
    """
    Extracts COCO-17 skeleton from a single BGR frame using MediaPipe Tasks API.
    Designed for real-time CPU inference.

    Uses mediapipe.tasks.vision.PoseLandmarker (works with mediapipe >= 0.10).
    """

    def __init__(self, min_detection_confidence=0.5, min_presence_confidence=0.5):
        model_path = _ensure_pose_model()
        opts = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_presence_confidence,
        )
        self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(opts)

    def extract(self, bgr_frame):
        """
        Extract skeleton from a single BGR frame.

        Returns:
            keypoints (17, 3): x_norm, y_norm, visibility — zeros if no person
            detected (bool): whether a person was found
        """
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)

        keypoints = np.zeros((17, 3), dtype=np.float32)
        if not result.pose_landmarks:
            return keypoints, False

        lm = result.pose_landmarks[0]  # first person
        for mp_idx, coco_idx in MP_TO_COCO.items():
            if mp_idx < len(lm):
                keypoints[coco_idx] = [lm[mp_idx].x, lm[mp_idx].y,
                                       lm[mp_idx].visibility]
        return keypoints, True

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SkeletonBuffer:
    """
    Sliding window of skeleton frames for ST-GCN input.
    Fills with zeros until enough frames are collected.
    """

    def __init__(self, window_size=30):
        self.window_size = window_size
        self._buffer = np.zeros((window_size, 17, 3), dtype=np.float32)
        self._count  = 0

    def push(self, keypoints):
        """Add one frame (17, 3). Returns True when buffer is full."""
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = keypoints
        self._count = min(self._count + 1, self.window_size)
        return self._count >= self.window_size

    def get_input(self):
        """Returns (1, 3, T, 17) float32 array ready for ONNX ST-GCN."""
        return self._buffer.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

    def get_wrists(self):
        """Returns (left_wrist_xy, right_wrist_xy) from the latest frame."""
        last = self._buffer[-1]
        return last[9, :2], last[10, :2]  # COCO: 9=left wrist, 10=right wrist

    def reset(self):
        self._buffer[:] = 0
        self._count = 0
