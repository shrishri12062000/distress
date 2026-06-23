import numpy as np
import cv2
import onnxruntime as ort


class KnifeDetector:
    """
    Runs YOLOv8n ONNX model to detect knives in a frame.
    Designed for CPU inference via ONNX Runtime.
    """

    def __init__(self, model_path, conf_threshold=0.50, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.input_size     = 640

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, bgr_frame):
        """
        Detect knives in a BGR frame.

        Returns:
            boxes (list): each item [x1, y1, x2, y2, confidence] in pixel coords
            detected (bool): whether any knife was found
        """
        h, w = bgr_frame.shape[:2]
        blob, ratio, pad = self._preprocess(bgr_frame)

        outputs = self.session.run(None, {self.input_name: blob})[0]
        boxes   = self._postprocess(outputs, ratio, pad, w, h)

        return boxes, len(boxes) > 0

    def _preprocess(self, frame):
        """Letterbox resize + normalize to (1, 3, 640, 640)."""
        img, ratio, pad = _letterbox(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return img[np.newaxis], ratio, pad

    def _postprocess(self, outputs, ratio, pad, orig_w, orig_h):
        """Parse YOLO output → filtered bounding boxes in original pixel space."""
        # outputs shape: (1, 5, num_anchors) for single-class YOLOv8
        preds = outputs[0].T  # (num_anchors, 5) → x, y, w, h, conf
        mask  = preds[:, 4] > self.conf_threshold
        preds = preds[mask]

        if len(preds) == 0:
            return []

        # xywh → xyxy
        boxes_xywh = preds[:, :4]
        scores     = preds[:, 4]

        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # Remove letterbox padding and rescale to original image
        x1 = (x1 - pad[0]) / ratio
        y1 = (y1 - pad[1]) / ratio
        x2 = (x2 - pad[0]) / ratio
        y2 = (y2 - pad[1]) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes = np.stack([x1, y1, x2, y2, scores], axis=1)

        # NMS
        keep = _nms(boxes, self.iou_threshold)
        return boxes[keep].tolist()


def _letterbox(img, target_size):
    h, w = img.shape[:2]
    ratio = min(target_size / h, target_size / w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    img = cv2.copyMakeBorder(img, pad_h, target_size - new_h - pad_h,
                             pad_w, target_size - new_w - pad_w,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, ratio, (pad_w, pad_h)


def _nms(boxes, iou_threshold):
    x1, y1, x2, y2, scores = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep
