from pathlib import Path
from collections import Counter

from ultralytics import YOLO


class WasteDetector:
    """YOLO-based waste detection engine."""

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image, confidence=0.25):
        """Run waste detection on an image."""
        results = self.model.predict(
            image,
            conf=confidence,
            verbose=False
        )

        return results[0]

    def get_counts(self, result):
        """Count detected objects by material type."""
        counts = Counter()

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            counts[class_name] += 1

        return counts

    def get_detections(self, result):
        """Return individual detections."""
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            detections.append({
                "class": class_name,
                "confidence": confidence
            })

        return detections

    def get_annotated_image(self, result):
        """Return image with YOLO annotations."""
        return result.plot()