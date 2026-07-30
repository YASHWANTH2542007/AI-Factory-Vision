# AI Factory Vision — Smart Quality Inspection & Defect Detection System

An AI-powered industrial quality inspection pipeline combining classical
OpenCV techniques with YOLO object detection, object tracking, dimension
measurement, and a Streamlit dashboard with SQLite-backed reporting.

## Features

- Product/defect detection via YOLO (Ultralytics)
- Classical CV dimension measurement (contours + calibrated pixel-to-mm)
- Multi-object tracking across video frames (centroid tracker)
- Good vs. defective classification and counting
- SQLite-backed inspection logging
- Streamlit dashboard with live stats, CSV and PDF report export

## Project Structure

```
AI-Factory-Vision/
├── data/
│   ├── images/       # sample/test images
│   ├── videos/        # sample/test videos
│   └── datasets/       # training data for custom YOLO models
├── models/            # YOLO weights (.pt files)
├── src/
│   ├── detection/      # YOLO wrapper
│   ├── tracking/       # centroid tracker
│   ├── measurement/     # OpenCV contour-based measurement
│   ├── preprocessing/   # classical CV utilities
│   ├── dashboard/       # report generation (CSV/PDF)
│   ├── db/             # SQLite persistence layer
│   └── utils/          # config + logging
├── reports/            # generated CSV/PDF reports
├── tests/              # pytest unit tests
├── app.py              # Streamlit entry point
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The default detector uses the pretrained `yolov8n.pt` (COCO classes) so the
app runs immediately. Ultralytics will auto-download it on first run.
For real defect detection, train a custom model on your own
crack/scratch/dent/missing-part dataset (see `data/datasets/`) and point
`DEFAULT_MODEL_PATH` in `src/utils/config.py` at your trained weights.

## Running

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Running Tests

```bash
pytest tests/ -v
```

## Calibrating Measurements

Dimension measurement converts pixels to millimeters using a
`PIXELS_PER_MM` constant in `src/utils/config.py`. To calibrate:

1. Place an object of known size in the camera frame.
2. Measure its width in pixels (visible in the app's measurement output).
3. Compute `pixels_per_mm = measured_width_px / known_width_mm`.
4. Update `PIXELS_PER_MM` in `config.py`.

Recalibrate any time the camera height, zoom, or angle changes.

## Roadmap / Extension Ideas

- Swap the centroid tracker for ByteTrack/DeepSORT for occlusion-heavy scenes
- Train a custom YOLO model on real defect imagery (Phase 3)
- Add a live camera feed input option in the dashboard
- Add per-shift / per-batch report grouping in the database schema
