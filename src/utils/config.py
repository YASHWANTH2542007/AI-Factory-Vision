"""
Central configuration for AI Factory Vision.
Keeping constants in one place avoids "magic numbers" scattered across the codebase
and makes it trivial to retune the system without hunting through every file.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")

MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

DB_PATH = os.path.join(BASE_DIR, "inspection.db")

# ---------------------------------------------------------------------------
# YOLO / Detection settings
# ---------------------------------------------------------------------------
# Path to weights. Defaults to the generic pretrained COCO model so the app
# runs out-of-the-box; swap this for your custom-trained defect model
# (e.g. models/defect_best.pt) once you've trained one in Phase 3.
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n.pt")

CONFIDENCE_THRESHOLD = 0.5   # minimum confidence to accept a detection
IOU_THRESHOLD = 0.45         # IoU threshold used for Non-Maximum Suppression

# Class names for OUR custom defect-detection model (Phase 3 target).
# Replace this list with your actual trained classes.
DEFECT_CLASS_NAMES = ["good_product", "crack", "scratch", "dent", "missing_part"]

# ---------------------------------------------------------------------------
# Measurement settings (pixel-to-real-world conversion)
# ---------------------------------------------------------------------------
# A "calibration factor" — how many millimeters does one pixel represent?
# You calibrate this once using a reference object of known size.
PIXELS_PER_MM = 5.0

# ---------------------------------------------------------------------------
# Tracking settings
# ---------------------------------------------------------------------------
MAX_TRACKING_DISTANCE = 50     # max pixel distance to associate detections between frames
MAX_DISAPPEARED_FRAMES = 30    # frames an object can be missing before we drop its track

# ---------------------------------------------------------------------------
# Streamlit dashboard settings
# ---------------------------------------------------------------------------
APP_TITLE = "AI Factory Vision - Smart Quality Inspection"
