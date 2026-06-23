import numpy as np
import onnxruntime as ort

CLASS_NAMES = {
    0: 'normal',
    1: 'help_signal',
    2: 'collapse_falling',
    3: 'fall_down',
}


class ActionDetector:
    """
    Runs ST-GCN ONNX model to classify skeleton sequences into 4 distress classes.
    Designed for CPU inference via ONNX Runtime.
    """

    def __init__(self, model_path, confidence_threshold=0.6):
        self.threshold = confidence_threshold

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider'],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, skeleton_window):
        """
        Run inference on a skeleton window.

        Args:
            skeleton_window: (1, 3, T, 17) float32 numpy array

        Returns:
            class_id (int): predicted class index
            label (str): class name
            confidence (float): softmax probability of predicted class
            all_probs (array): softmax probabilities for all 4 classes
        """
        logits = self.session.run(
            [self.output_name],
            {self.input_name: skeleton_window}
        )[0]  # (1, 4)

        probs = _softmax(logits[0])
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        label = CLASS_NAMES[class_id]

        return class_id, label, confidence, probs


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()
