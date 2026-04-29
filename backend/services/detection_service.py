"""
backend/services/detection_service.py
--------------------------------------
Business logic that wraps the detection package for use by the API layer.
"""

from pathlib import Path


def run_detection(image_path: str | Path, conf: float = 0.25) -> list[dict]:
    """Run YOLOv8 detection on an image and return structured results.

    Args:
        image_path: Path to the image file.
        conf:       Confidence threshold.

    Returns:
        List of dicts with keys: label, confidence, bbox (x1,y1,x2,y2).
    """
    # TODO: import and call detection.predict.predict()
    raise NotImplementedError("Detection service not yet implemented.")
