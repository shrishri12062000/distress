"""
Live webcam distress detection.
Run after downloading trained ONNX models to models/

Usage:
    python scripts/run_camera.py
    python scripts/run_camera.py --camera 1
"""

import argparse
import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pose_extractor import PoseExtractor, SkeletonBuffer
from src.inference.action_detector import ActionDetector
from src.inference.knife_detector import KnifeDetector
from src.inference.fusion import DistressDetector
from src.utils.visualization import draw_skeleton, draw_detection
from src.utils.logger import get_logger

STGCN_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'models', 'stgcn.onnx')
YOLO_MODEL  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            'models', 'yolo_knife.onnx')

log = get_logger('run_camera')


def main(camera_id=0, show_skeleton=True):
    # ── Load models ──────────────────────────────────────────────────────────
    if not os.path.exists(STGCN_MODEL):
        log.error(f'ST-GCN model not found: {STGCN_MODEL}')
        log.error('Download models from Google Drive and place in models/')
        sys.exit(1)
    if not os.path.exists(YOLO_MODEL):
        log.error(f'YOLO model not found: {YOLO_MODEL}')
        sys.exit(1)

    log.info('Loading models...')
    pose_extractor  = PoseExtractor()
    skeleton_buffer = SkeletonBuffer(window_size=30)
    action_detector = ActionDetector(STGCN_MODEL, confidence_threshold=0.60)
    knife_detector  = KnifeDetector(YOLO_MODEL,   conf_threshold=0.50)
    distress        = DistressDetector()
    log.info('Models loaded. Press Q to quit.')

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    result = {
        'action_label': 'Initializing...',
        'action_conf': 0.0,
        'knife_detected': False,
        'knife_boxes': [],
        'final_alert': False,
        'final_class': 'normal',
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error('Camera read failed')
            break

        # ── Pose extraction ──────────────────────────────────────────────────
        keypoints, person_found = pose_extractor.extract(frame)
        ready = skeleton_buffer.push(keypoints)

        if show_skeleton and person_found:
            frame = draw_skeleton(frame, keypoints)

        # ── Inference (once buffer is full) ──────────────────────────────────
        if ready:
            skeleton_input = skeleton_buffer.get_input()
            action_result  = action_detector.predict(skeleton_input)
            knife_result   = knife_detector.detect(frame)
            wrists         = skeleton_buffer.get_wrists()

            result = distress.fuse(
                frame_shape=frame.shape[:2],
                action_result=action_result,
                knife_result=knife_result,
                wrist_positions=wrists,
            )

            if result['final_alert']:
                log.info(f"ALERT: {result['final_class']} "
                         f"({result['action_conf']:.0%})")

        # ── Display ──────────────────────────────────────────────────────────
        frame = draw_detection(frame, result)
        cv2.imshow('Distress Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            skeleton_buffer.reset()
            log.info('Buffer reset')

    cap.release()
    cv2.destroyAllWindows()
    pose_extractor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera',        type=int,  default=0)
    parser.add_argument('--no-skeleton',   action='store_true')
    args = parser.parse_args()
    main(camera_id=args.camera, show_skeleton=not args.no_skeleton)
