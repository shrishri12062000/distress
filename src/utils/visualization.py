import cv2
import numpy as np

CLASS_COLORS = {
    'normal':           (0, 200, 0),
    'help_signal':      (0, 165, 255),
    'collapse_falling': (0, 0, 255),
    'fall_down':        (0, 0, 200),
    'holding_knife':    (0, 0, 255),
}

CLASS_LABELS = {
    0: 'Normal',
    1: 'HELP SIGNAL',
    2: 'COLLAPSING',
    3: 'FALLEN DOWN',
    4: 'HOLDING KNIFE',
}

COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def draw_skeleton(frame, keypoints, color=(0, 255, 0), threshold=0.3):
    """
    Draw COCO-17 skeleton on frame.
    keypoints: (17, 3) array — (x_norm, y_norm, confidence)
    """
    h, w = frame.shape[:2]
    pts = []
    for i in range(17):
        x = int(keypoints[i, 0] * w)
        y = int(keypoints[i, 1] * h)
        conf = keypoints[i, 2]
        pts.append((x, y, conf))

    for i, j in COCO_SKELETON_PAIRS:
        if pts[i][2] > threshold and pts[j][2] > threshold:
            cv2.line(frame, pts[i][:2], pts[j][:2], color, 2, cv2.LINE_AA)

    for x, y, conf in pts:
        if conf > threshold:
            cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)

    return frame


def draw_detection(frame, result):
    """
    Draw final distress detection result on frame.

    result dict keys:
        action_class (int), action_label (str), action_conf (float),
        knife_detected (bool), knife_boxes (list of [x1,y1,x2,y2,conf]),
        final_alert (bool), final_class (str)
    """
    h, w = frame.shape[:2]

    # Draw knife bounding boxes
    if result.get('knife_detected'):
        for box in result.get('knife_boxes', []):
            x1, y1, x2, y2, conf = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'Knife {conf:.2f}',
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Draw alert banner
    final_class = result.get('final_class', 'normal')
    color = CLASS_COLORS.get(final_class, (0, 200, 0))
    label = result.get('action_label', 'Normal')
    conf  = result.get('action_conf', 0.0)

    if result.get('final_alert'):
        cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
        cv2.putText(frame, f'ALERT: {label}  ({conf:.0%})',
                    (10, 34), cv2.FONT_HERSHEY_DUPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, f'{label}  ({conf:.0%})',
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (200, 200, 200), 1, cv2.LINE_AA)

    return frame
