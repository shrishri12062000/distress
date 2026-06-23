import numpy as np

# Classes that trigger an alert
ALERT_CLASSES = {'help_signal', 'collapse_falling', 'fall_down', 'holding_knife'}

# Action class index → name
ACTION_CLASS_NAMES = {
    0: 'normal',
    1: 'help_signal',
    2: 'collapse_falling',
    3: 'fall_down',
}


class DistressDetector:
    """
    Fuses ST-GCN action classification and YOLOv8 knife detection
    into a single distress detection result.

    Holding knife logic:
        A knife is detected AND its bounding box center is within
        wrist_proximity_threshold (normalized) of either wrist keypoint.
    """

    def __init__(self,
                 action_confidence_threshold=0.60,
                 knife_confidence_threshold=0.50,
                 wrist_proximity_threshold=0.15):
        self.action_thresh    = action_confidence_threshold
        self.knife_thresh     = knife_confidence_threshold
        self.wrist_prox       = wrist_proximity_threshold

    def fuse(self, frame_shape, action_result, knife_result, wrist_positions):
        """
        Combine action and knife results into a single detection output.

        Args:
            frame_shape:     (h, w) of the original frame
            action_result:   (class_id, label, confidence, all_probs)
            knife_result:    (boxes, detected)  — boxes in pixel coords
            wrist_positions: (left_wrist_xy, right_wrist_xy) — normalized 0-1

        Returns:
            dict with keys:
                action_class, action_label, action_conf,
                knife_detected, knife_boxes,
                holding_knife, final_alert, final_class
        """
        class_id, label, conf, all_probs = action_result
        knife_boxes, knife_detected      = knife_result
        left_wrist, right_wrist          = wrist_positions

        # ── Holding knife check ──────────────────────────────────────────────
        holding_knife = False
        if knife_detected:
            h, w = frame_shape
            for box in knife_boxes:
                if box[4] < self.knife_thresh:
                    continue
                # Normalize knife center to 0-1
                knife_cx = ((box[0] + box[2]) / 2) / w
                knife_cy = ((box[1] + box[3]) / 2) / h
                knife_pt = np.array([knife_cx, knife_cy])

                for wrist in [left_wrist, right_wrist]:
                    if wrist[0] == 0 and wrist[1] == 0:
                        continue
                    dist = np.linalg.norm(knife_pt - wrist)
                    if dist < self.wrist_prox:
                        holding_knife = True
                        break
                if holding_knife:
                    break

        # ── Final classification ─────────────────────────────────────────────
        if holding_knife:
            final_class = 'holding_knife'
            final_alert = True
        elif conf >= self.action_thresh and label in ALERT_CLASSES:
            final_class = label
            final_alert = True
        else:
            final_class = 'normal'
            final_alert = False

        return {
            'action_class':   class_id,
            'action_label':   label,
            'action_conf':    conf,
            'action_probs':   all_probs.tolist(),
            'knife_detected': knife_detected,
            'knife_boxes':    knife_boxes,
            'holding_knife':  holding_knife,
            'final_alert':    final_alert,
            'final_class':    final_class,
        }
