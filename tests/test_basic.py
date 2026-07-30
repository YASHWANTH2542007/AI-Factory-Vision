"""
Basic sanity tests. Run with:  pytest tests/
These are intentionally simple -- expand them as you build each module,
especially the tracker and measurement logic, which are the easiest to
silently break.
"""

import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.image_utils import to_grayscale, threshold_image, find_contours
from src.measurement.measure import pixels_to_mm, measure_object
from src.tracking.tracker import CentroidTracker


def make_blank_image_with_square():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    return img


def test_to_grayscale_shape():
    img = make_blank_image_with_square()
    gray = to_grayscale(img)
    assert gray.shape == (200, 200)  # no channel dimension


def test_find_contours_detects_square():
    img = make_blank_image_with_square()
    gray = to_grayscale(img)
    binary = threshold_image(gray, use_otsu=True)
    contours = find_contours(binary)
    assert len(contours) == 1


def test_pixels_to_mm_conversion():
    assert pixels_to_mm(100, pixels_per_mm=10) == 10.0


def test_pixels_to_mm_invalid_calibration():
    try:
        pixels_to_mm(100, pixels_per_mm=0)
        assert False, "Expected ValueError for zero calibration factor"
    except ValueError:
        pass


def test_measure_object_square():
    img = make_blank_image_with_square()
    gray = to_grayscale(img)
    binary = threshold_image(gray, use_otsu=True)
    contours = find_contours(binary)
    result = measure_object(contours[0], pixels_per_mm=1.0)
    # square is 100x100 px
    assert 95 <= result["width_px"] <= 105
    assert 95 <= result["height_px"] <= 105


def test_tracker_registers_new_objects():
    tracker = CentroidTracker()
    objects = tracker.update([(10, 10), (50, 50)])
    assert len(objects) == 2


def test_tracker_maintains_id_across_frames():
    tracker = CentroidTracker()
    tracker.update([(10, 10)])
    objects = tracker.update([(12, 11)])  # small movement, same object
    assert len(objects) == 1
