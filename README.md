# Varroa Mite Detector

A computer vision web application built with **Streamlit**, **OpenCV**, **SAHI**, and **YOLOv8** for detecting and quantifying *Varroa destructor* mites on beehive sticky boards. Developed within the BeeStorm project environment to support transparent, human-in-the-loop pest monitoring.

## Overview & Processing Pipeline

Because individual Varroa mites occupy only a tiny fraction of a full sticky board photograph, direct downscaling causes severe loss of visual detail. This application implements a multi-stage processing pipeline to preserve resolution and standardize physical scale:

1. **Image Standardization:** Handles EXIF orientation metadata, supports mobile formats (`.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`), and standardizes images to landscape orientation.
2. **Board Detection & Perspective Rectification (OpenCV):** Segments the bright sticky board in HSV color space, detects the four corners, and applies perspective warping to a standardized resolution of **6040 × 5040 px** (corresponding to physical board dimensions of 30.2 cm × 25.2 cm). If four corners cannot be isolated, a scaled rectangular fallback crop is applied.
3. **Tiled Object Detection (SAHI + YOLOv8s):** Slices the rectified high-resolution board into overlapping **640 × 640 px** tiles (default 20% overlap) and merges tile-level predictions back into full-board coordinates.
4. **Size-Based Post-Processing:** Filters out detections outside realistic mite pixel dimensions (default 10–50 px) to suppress oversized false positives such as wax fragments, pollen, hive debris, or bees.
5. **Interactive Inspection:** Provides a full-board detection overview (with full-resolution annotated download), a zoomed detection-by-detection inspector, and an HSV segmentation mask view to verify board alignment.

## Model Summary

- **Architecture:** YOLOv8s (`models/best.pt`)
- **Training Dataset:** 300 annotated field sticky boards (200 train / 40 validation / 60 test) with empty-tile subsampling and debris-rich negative examples.
- **Validation Performance:** Precision `0.851`, Recall `0.872`, F1-score `0.862`, mAP50 `0.913`, mAP50-95 `0.810`.

> **Note:** This tool is intended as a decision-support aid for beekeepers and researchers rather than a replacement for expert apicultural judgment.

## Repository Structure

```text
├── models/
│   └── best.pt                  # Trained YOLOv8s weights
├── packages.txt                 # System-level dependencies for Linux cloud deployment
├── requirements.txt             # Python dependencies (CPU-optimized PyTorch)
├── varroa_pipeline.py           # Image preprocessing, rectification, SAHI inference & post-processing
├── varroa_streamlit_app.py      # Streamlit web user interface
└── README.md
