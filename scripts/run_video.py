"""
Run distress detection on a recorded video file.

Usage:
    python scripts/run_video.py --input path/to/video.mp4
    python scripts/run_video.py --input video.mp4 --output annotated.mp4
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

log = get_logger('run_video')


def main(input_path, output_path=None, show_skeleton=True):
    if not os.path.exists(input_path):
        log.error(f'Input video not found: {input_path}')
        sys.exit(1)

    log.info('Loading models...')
    pose_extractor  = PoseExtractor()
    skeleton_buffer = SkeletonBuffer(window_size=30)
    action_detector = ActionDetector(STGCN_MODEL, confidence_threshold=0.60)
    knife_detector  = KnifeDetector(YOLO_MODEL,   conf_threshold=0.50)
    distress        = DistressDetector()

    cap = cv2.VideoCapture(input_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        log.info(f'Writing output to: {output_path}')

    alerts = []
    frame_idx = 0

    result = {
        'action_label': 'Initializing...',
        'action_conf': 0.0,
        'knife_detected': False,
        'knife_boxes': [],
        'final_alert': False,
        'final_class': 'normal',
    }

    log.info(f'Processing {total_frames} frames...')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints, person_found = pose_extractor.extract(frame)
        ready = skeleton_buffer.push(keypoints)

        if show_skeleton and person_found:
            frame = draw_skeleton(frame, keypoints)

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
                timestamp = frame_idx / fps
                alerts.append({
                    'frame': frame_idx,
                    'time':  f'{int(timestamp//60):02d}:{timestamp%60:05.2f}',
                    'class': result['final_class'],
                    'conf':  result['action_conf'],
                })

        frame = draw_detection(frame, result)

        if writer:
            writer.write(frame)

        cv2.imshow('Distress Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    pose_extractor.close()

    log.info(f'\n── Detection Summary ────────────────────────────')
    log.info(f'  Frames processed: {frame_idx}')
    log.info(f'  Alerts detected:  {len(alerts)}')
    for a in alerts:
        log.info(f'  [{a["time"]}] {a["class"]} ({a["conf"]:.0%})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',       required=True)
    parser.add_argument('--output',      default=None)
    parser.add_argument('--no-skeleton', action='store_true')
    args = parser.parse_args()
    main(args.input, args.output, show_skeleton=not args.no_skeleton)
