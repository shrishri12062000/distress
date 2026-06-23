import numpy as np
import cv2
import mediapipe as mp

# MediaPipe landmark index → COCO 17-joint index
MP_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4,
    11: 5, 12: 6, 13: 7, 14: 8,
    15: 9, 16: 10, 23: 11, 24: 12,
    25: 13, 26: 14, 27: 15, 28: 16,
}


class PoseExtractor:
    """
    Extracts COCO-17 skeleton from a single BGR frame using MediaPipe Pose.
    Designed for CPU inference.
    """

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, bgr_frame):
        """
        Extract skeleton from a single frame.

        Returns:
            keypoints (17, 3): x_norm, y_norm, visibility  — or zeros if no person
            detected (bool): whether a person was found
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)

        keypoints = np.zeros((17, 3), dtype=np.float32)
        if not result.pose_landmarks:
            return keypoints, False

        lm = result.pose_landmarks.landmark
        for mp_idx, coco_idx in MP_TO_COCO.items():
            keypoints[coco_idx] = [lm[mp_idx].x, lm[mp_idx].y,
                                   lm[mp_idx].visibility]
        return keypoints, True

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SkeletonBuffer:
    """
    Maintains a sliding window of skeleton frames for ST-GCN input.
    Fills the buffer with zeros until enough frames are collected.
    """

    def __init__(self, window_size=30):
        self.window_size = window_size
        self._buffer = np.zeros((window_size, 17, 3), dtype=np.float32)
        self._count = 0

    def push(self, keypoints):
        """Add one frame (17, 3). Returns True when buffer is full."""
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = keypoints
        self._count = min(self._count + 1, self.window_size)
        return self._count >= self.window_size

    def get_input(self):
        """
        Returns (1, 3, T, 17) tensor ready for ONNX ST-GCN inference.
        """
        # (T, V, C) → (C, T, V) → (1, C, T, V)
        x = self._buffer.transpose(2, 0, 1)[np.newaxis]
        return x.astype(np.float32)

    def get_wrists(self):
        """
        Returns (left_wrist_xy, right_wrist_xy) from the latest frame.
        Normalized 0-1 coordinates.
        """
        last = self._buffer[-1]
        return last[9, :2], last[10, :2]  # COCO: 9=left wrist, 10=right wrist

    def reset(self):
        self._buffer[:] = 0
        self._count = 0
