"""
Preprocessing utilities.

These are the classical (non-deep-learning) OpenCV building blocks that
support the rest of the pipeline: loading, resizing, denoising, thresholding,
edge detection, and contour extraction. YOLO handles "what and where," but
these functions handle "clean the image up" and "measure precisely."
"""

import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as a BGR NumPy array.

    Raises:
        FileNotFoundError: if OpenCV cannot read the file (bad path/corrupt file).
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    logger.info(f"Loaded image {path} with shape {img.shape}")
    return img


def resize_image(img: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """
    Resize while preserving aspect ratio if only one dimension is given.
    Preserving aspect ratio avoids distorting the product's real shape,
    which matters a lot once we start measuring dimensions.
    """
    (h, w) = img.shape[:2]

    if width is None and height is None:
        return img

    if width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to single-channel grayscale.
    Most classical CV operations (thresholding, edge detection) work on
    intensity, not color, so this is almost always the first preprocessing step.
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to reduce sensor noise before edge detection.
    Edge detectors are sensitive to noise (a single noisy pixel can look
    like a fake edge), so blurring slightly first gives cleaner results.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def threshold_image(gray_img: np.ndarray, thresh_val: int = 127, use_otsu: bool = False) -> np.ndarray:
    """
    Binarize a grayscale image (pixels become either 0 or 255).

    Otsu's method automatically finds the optimal threshold value based on
    the image's histogram, which is useful when lighting conditions vary
    between inspection runs (a very common real factory problem).
    """
    if use_otsu:
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
    return binary


def detect_edges(gray_img: np.ndarray, low_thresh: int = 50, high_thresh: int = 150) -> np.ndarray:
    """
    Canny edge detection.

    low_thresh/high_thresh define hysteresis: pixels above high_thresh are
    "sure edges," pixels below low_thresh are discarded, and pixels in
    between are kept only if connected to a sure edge. This two-threshold
    approach makes Canny far more robust than simple gradient thresholding.
    """
    return cv2.Canny(gray_img, low_thresh, high_thresh)


def apply_morphology(binary_img: np.ndarray, operation: str = "close", kernel_size: int = 5) -> np.ndarray:
    """
    Morphological operations clean up binary masks:
      - 'dilate': grows white regions (fills small holes, connects broken parts)
      - 'erode':  shrinks white regions (removes small noise specks)
      - 'open':   erode then dilate (removes small noise, keeps overall shape)
      - 'close':  dilate then erode (fills small holes inside objects)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    ops = {
        "dilate": lambda x: cv2.dilate(x, kernel, iterations=1),
        "erode": lambda x: cv2.erode(x, kernel, iterations=1),
        "open": lambda x: cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel),
        "close": lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel),
    }

    if operation not in ops:
        raise ValueError(f"Unknown morphology operation: {operation}. Choose from {list(ops.keys())}")

    return ops[operation](binary_img)


def find_contours(binary_img: np.ndarray):
    """
    Find contours (continuous outlines of white regions) in a binary image.
    RETR_EXTERNAL only grabs outer boundaries (ignores holes/nested contours),
    which is what we want for detecting a whole product's outline.
    CHAIN_APPROX_SIMPLE compresses redundant points on straight lines,
    saving memory without losing shape accuracy.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours(img: np.ndarray, contours, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Draw contours on a copy of the image (never mutate the original)."""
    output = img.copy()
    cv2.drawContours(output, contours, -1, color, thickness)
    return output
