"""
YOLO-based object/defect detector, wrapped in a small class so the rest of
the app doesn't need to know anything about the Ultralytics API directly.
This is the "adapter" pattern -- if we ever swap YOLO for another model,
only this file changes.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from ultralytics import YOLO

from src.utils.config import DEFAULT_MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """A single detected object, in a framework-agnostic shape."""
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class YoloDetector:
    """
    Thin wrapper around an Ultralytics YOLO model.

    Note on model choice: we default to the small pretrained 'yolov8n.pt'
    (COCO classes) so the pipeline is runnable immediately without a custom
    dataset. For actual defect detection (Phase 3), replace model_path with
    a model you've fine-tuned on your own crack/scratch/dent dataset.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 conf_threshold: float = CONFIDENCE_THRESHOLD,
                 iou_threshold: float = IOU_THRESHOLD):
        logger.info(f"Loading YOLO model from {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single frame (BGR NumPy array) and return a
        clean list of Detection objects.

        conf: confidence threshold - detections below this are discarded.
              Too low -> many false positives. Too high -> missed real defects.
        iou:  IoU threshold used internally by Ultralytics for Non-Maximum
              Suppression (removing duplicate overlapping boxes for the
              same object).
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = []
        result = results[0]
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append(Detection(
                class_id=cls_id,
                class_name=names.get(cls_id, str(cls_id)),
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes + labels on a copy of the frame."""
        import cv2
        output = frame.copy()

        for det in detections:
            color = (0, 255, 0)
            cv2.rectangle(output, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(output, label, (det.x1, max(det.y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output
