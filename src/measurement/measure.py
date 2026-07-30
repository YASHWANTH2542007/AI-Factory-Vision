"""
Object measurement using classical OpenCV geometry.

Deep learning (YOLO) tells us WHAT an object is and roughly WHERE it is
(a bounding box). It does NOT give precise real-world measurements.
For that, we fall back to classical CV: find the object's contour, fit a
minimum bounding rectangle, and convert pixel distances to millimeters
using a calibration factor.
"""

import cv2
import numpy as np
from src.utils.config import PIXELS_PER_MM
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_bounding_box_dimensions(contour) -> dict:
    """
    Fit a rotated minimum-area rectangle around a contour and return its
    dimensions. We use the MINIMUM AREA rectangle (not just a straight
    axis-aligned box) because products on a conveyor are rarely perfectly
    aligned to the camera axis — a straight box would overestimate size
    for a rotated object.
    """
    rect = cv2.minAreaRect(contour)  # ((cx, cy), (w, h), angle)
    (cx, cy), (w, h), angle = rect

    box_points = cv2.boxPoints(rect)  # 4 corner points of the rotated rect
    box_points = np.intp(box_points)

    width_px = max(w, h)
    height_px = min(w, h)

    return {
        "center": (cx, cy),
        "width_px": width_px,
        "height_px": height_px,
        "angle_deg": angle,
        "box_points": box_points,
    }


def pixels_to_mm(pixel_value: float, pixels_per_mm: float = PIXELS_PER_MM) -> float:
    """
    Convert a pixel measurement to millimeters using a calibration factor.

    HOW TO CALIBRATE: place an object of known size (e.g. a 50mm coin) in
    frame, measure its width in pixels, then:
        pixels_per_mm = measured_width_px / 50
    Update PIXELS_PER_MM in config.py with that value. This must be redone
    any time the camera height/zoom/angle changes.
    """
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive")
    return pixel_value / pixels_per_mm


def measure_object(contour, pixels_per_mm: float = PIXELS_PER_MM) -> dict:
    """
    Full measurement pipeline for a single contour: pixel dimensions,
    real-world dimensions (mm), and area.
    """
    dims = get_bounding_box_dimensions(contour)
    area_px = cv2.contourArea(contour)

    width_mm = pixels_to_mm(dims["width_px"], pixels_per_mm)
    height_mm = pixels_to_mm(dims["height_px"], pixels_per_mm)

    return {
        **dims,
        "area_px": area_px,
        "width_mm": round(width_mm, 2),
        "height_mm": round(height_mm, 2),
    }


def draw_measurement(img: np.ndarray, measurement: dict) -> np.ndarray:
    """Overlay the rotated bounding box and dimension text on the image."""
    output = img.copy()
    cv2.drawContours(output, [measurement["box_points"]], 0, (0, 255, 255), 2)

    cx, cy = int(measurement["center"][0]), int(measurement["center"][1])
    label = f'{measurement["width_mm"]}mm x {measurement["height_mm"]}mm'
    cv2.putText(output, label, (cx - 60, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return output
