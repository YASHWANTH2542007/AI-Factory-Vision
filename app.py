"""
AI Factory Vision - Main Streamlit Application

This is the orchestrator: it wires together preprocessing, detection,
measurement, tracking, and the database into one interactive dashboard.
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import tempfile
import os

from src.utils.config import APP_TITLE, CONFIDENCE_THRESHOLD
from src.utils.logger import get_logger
from src.preprocessing.image_utils import (
    load_image, to_grayscale, denoise, threshold_image,
    apply_morphology, find_contours,
)
from src.measurement.measure import measure_object, draw_measurement
from src.detection.yolo_detector import YoloDetector
from src.tracking.tracker import CentroidTracker
from src.db.database import (
    init_db, insert_inspection, get_all_inspections,
    get_summary_stats, clear_all_inspections,
)
from src.dashboard.reports import export_csv, export_pdf

logger = get_logger(__name__)

st.set_page_config(page_title=APP_TITLE, layout="wide")
init_db()


@st.cache_resource
def load_detector(conf_threshold):
    """
    Cache the YOLO model across reruns -- Streamlit reruns the whole script
    on every UI interaction, so without caching we'd reload the model
    (slow, expensive) on every single click.
    """
    return YoloDetector(conf_threshold=conf_threshold)


def process_image_classical(img: np.ndarray, min_contour_area: int = 500):
    """
    Classical CV pipeline: grayscale -> denoise -> threshold -> morphology
    -> contours -> measurement. Used for dimension measurement, which YOLO
    alone does not provide.
    """
    gray = to_grayscale(img)
    blurred = denoise(gray)
    binary = threshold_image(blurred, use_otsu=True)
    cleaned = apply_morphology(binary, operation="close")
    contours = find_contours(cleaned)

    # Filter out tiny noise contours
    contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

    output = img.copy()
    measurements = []
    for c in contours:
        m = measure_object(c)
        measurements.append(m)
        output = draw_measurement(output, m)

    return output, measurements


def run_detection_and_log(img: np.ndarray, detector: YoloDetector, source: str):
    """Run YOLO detection, draw results, and log each detection to the DB."""
    detections = detector.detect(img)
    output = detector.draw_detections(img, detections)

    for det in detections:
        # NOTE: with the default COCO-pretrained model, classes won't be
        # "crack"/"scratch" etc. Swap in your fine-tuned defect model
        # (see models/ and config.DEFAULT_MODEL_PATH) for real defect logic.
        is_defective = det.class_name.lower() not in ("good_product",)
        insert_inspection(
            class_name=det.class_name,
            is_defective=is_defective,
            confidence=det.confidence,
            source=source,
        )

    return output, detections


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏭 AI Factory Vision")
page = st.sidebar.radio("Navigate", [
    "Image Inspection", "Video Inspection", "Dashboard & Reports",
])
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 0.9, CONFIDENCE_THRESHOLD, 0.05)

# ---------------------------------------------------------------------------
# Page: Image Inspection
# ---------------------------------------------------------------------------
if page == "Image Inspection":
    st.title("🔍 Single Image Inspection")
    st.write("Upload a product image to run detection and dimension measurement.")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        tab1, tab2 = st.tabs(["YOLO Detection", "Classical Measurement"])

        with tab1:
            detector = load_detector(conf_threshold)
            output, detections = run_detection_and_log(img, detector, source=uploaded.name)
            st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            st.write(f"Found {len(detections)} object(s).")
            if detections:
                st.dataframe(pd.DataFrame([{
                    "class": d.class_name, "confidence": round(d.confidence, 3), "bbox": d.bbox
                } for d in detections]))

        with tab2:
            output_measure, measurements = process_image_classical(img)
            st.image(cv2.cvtColor(output_measure, cv2.COLOR_BGR2RGB))
            st.write(f"Found {len(measurements)} contour(s).")
            if measurements:
                st.dataframe(pd.DataFrame([{
                    "width_mm": m["width_mm"], "height_mm": m["height_mm"],
                    "area_px": round(m["area_px"], 1),
                } for m in measurements]))

# ---------------------------------------------------------------------------
# Page: Video Inspection
# ---------------------------------------------------------------------------
elif page == "Video Inspection":
    st.title("🎥 Video / Conveyor Belt Inspection")
    st.write("Upload a short video to run detection + tracking across frames.")

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    frame_skip = st.slider("Process every Nth frame (higher = faster)", 1, 10, 2)

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())

        detector = load_detector(conf_threshold)
        tracker = CentroidTracker()

        cap = cv2.VideoCapture(tfile.name)
        frame_placeholder = st.empty()
        stats_placeholder = st.empty()

        frame_idx = 0
        unique_ids_seen = set()

        if st.button("Start Processing"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                detections = detector.detect(frame)
                centroids = [d.center for d in detections]
                tracked_objects = tracker.update(centroids)
                unique_ids_seen.update(tracked_objects.keys())

                output = detector.draw_detections(frame, detections)
                for object_id, centroid in tracked_objects.items():
                    cv2.putText(output, f"ID {object_id}", (int(centroid[0]), int(centroid[1]) - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.circle(output, (int(centroid[0]), int(centroid[1])), 4, (255, 0, 0), -1)

                frame_placeholder.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
                stats_placeholder.write(f"Frame {frame_idx} | Unique products tracked so far: {len(unique_ids_seen)}")

            cap.release()
            os.unlink(tfile.name)
            st.success(f"Done. Total unique products counted: {len(unique_ids_seen)}")

# ---------------------------------------------------------------------------
# Page: Dashboard & Reports
# ---------------------------------------------------------------------------
elif page == "Dashboard & Reports":
    st.title("📊 Inspection Dashboard")

    stats = get_summary_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Inspected", stats["total"])
    col2.metric("Good Products", stats["good"])
    col3.metric("Defective Products", stats["defective"])
    col4.metric("Defect Rate", f"{stats['defect_rate']}%")

    inspections = get_all_inspections()

    if inspections:
        df = pd.DataFrame(inspections)
        st.subheader("Recent Inspections")
        st.dataframe(df, use_container_width=True)

        st.subheader("Defects by Class")
        defect_counts = df[df["is_defective"] == 1]["class_name"].value_counts()
        if not defect_counts.empty:
            st.bar_chart(defect_counts)
        else:
            st.info("No defects logged yet.")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("Export CSV"):
                path = export_csv(inspections)
                with open(path, "rb") as f:
                    st.download_button("Download CSV", f, file_name=os.path.basename(path))
        with col_b:
            if st.button("Export PDF Summary"):
                path = export_pdf(stats)
                with open(path, "rb") as f:
                    st.download_button("Download PDF", f, file_name=os.path.basename(path))
        with col_c:
            if st.button("Clear All Records", type="secondary"):
                clear_all_inspections()
                st.rerun()
    else:
        st.info("No inspections logged yet. Go run an inspection on the Image or Video pages.")
