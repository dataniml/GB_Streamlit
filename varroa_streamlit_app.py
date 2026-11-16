from pathlib import Path
import time

import cv2
import numpy as np
import streamlit as st

from varroa_pipeline import (
    PipelineSettings,
    load_detection_model,
    make_detection_crop,
    run_pipeline,
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"


def resize_for_preview(image: np.ndarray, max_width: int = 1600) -> np.ndarray:
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    new_size = (max_width, int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


st.set_page_config(
    page_title="Varroa Mite Detector",
    page_icon="🐝",
    layout="wide",
)

st.title("Varroa Mite Detection")
st.caption(
    "Upload a photograph of a beehive sticky board. The application rectifies the board perspective, "
    "detects individual Varroa mites, and lets you inspect each detection."
)

with st.sidebar:
    st.header("Settings")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.30,
        step=0.05,
    )
    st.caption("Lower values detect more candidates but may increase false positives.")

    with st.expander("Advanced settings"):
        tile_size = st.selectbox("SAHI tile size (px)", [512, 640, 768], index=1)
        overlap = st.slider("Tile overlap ratio", 0.0, 0.5, 0.20, 0.05)
        min_box = st.number_input("Min bounding box size (px)", 0, 100, 10, 5)
        max_box = st.number_input("Max bounding box size (px)", 10, 200, 50, 5)
        hsv_v_min = st.slider(
            "Board brightness threshold (HSV V)",
            min_value=80,
            max_value=220,
            value=150,
            step=10,
            help="Lower this value if the sticky board was photographed in dim lighting or shadow.",
        )

uploaded = st.file_uploader(
    "Upload sticky board image",
    type=["jpg", "jpeg", "png", "heic", "heif"],
)

if "result" not in st.session_state:
    st.session_state.result = None
if "detection_index" not in st.session_state:
    st.session_state.detection_index = 0
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

if uploaded is not None:
    if st.session_state.uploaded_file_id != uploaded.file_id:
        st.session_state.uploaded_file_id = uploaded.file_id
        st.session_state.result = None
        st.session_state.detection_index = 0

    st.image(uploaded.getvalue(), caption="Uploaded image", use_container_width=True)

    status_slot = st.empty()
    if st.button("Analyze image", type="primary", use_container_width=True):
        if not MODEL_PATH.exists():
            st.error(f"Model file not found at: {MODEL_PATH}")
            st.stop()

        if min_box >= max_box:
            st.error("Minimum bounding box size must be smaller than maximum bounding box size.")
            st.stop()

        settings = PipelineSettings(
            tile_size=tile_size,
            overlap_ratio=float(overlap),
            confidence_threshold=float(confidence),
            min_box_width=float(min_box),
            min_box_height=float(min_box),
            max_box_width=float(max_box),
            max_box_height=float(max_box),
            hsv_v_min=int(hsv_v_min),
            device="cpu",
        )

        try:
            with status_slot.status("Processing image...", expanded=True) as status:
                st.write("1/3 Loading detection model")
                model = load_detection_model(str(MODEL_PATH), device="cpu")

                st.write("2/3 Standardizing image and rectifying sticky board")
                started = time.perf_counter()
                raw_result = run_pipeline(uploaded.getvalue(), uploaded.name, model, settings)
                elapsed = time.perf_counter() - started

                st.write("3/3 Preparing detection views")
                ok, encoded = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(raw_result["annotated_rgb"], cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                annotated_jpeg = encoded.tobytes() if ok else None

                st.session_state.result = {
                    "filename": raw_result["filename"],
                    "board_mode": raw_result["board_mode"],
                    "detections": raw_result["detections"],
                    "processing_seconds": elapsed,
                    "rectified_rgb": raw_result["rectified_rgb"],
                    "annotated_preview": resize_for_preview(raw_result["annotated_rgb"]),
                    "rectified_preview": resize_for_preview(raw_result["rectified_rgb"]),
                    "mask_preview": resize_for_preview(raw_result["board_mask"]),
                    "annotated_jpeg": annotated_jpeg,
                }
                st.session_state.detection_index = 0
                status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

result = st.session_state.result
if result:
    detections = result["detections"]
    c1, c2 = st.columns(2)
    c1.metric("Detected mites", len(detections))
    c2.metric("Processing time", f'{result["processing_seconds"]:.1f} s')

    if result["board_mode"] == "perspective":
        st.success("Sticky board detected via 4 corners and perspective-rectified.")
    else:
        st.warning(
            "Four corners could not be detected. A rectangular fallback crop was used; "
            "please inspect the detections carefully."
        )

    tab1, tab2, tab3 = st.tabs(["Detection Overview", "Individual Detections", "Board Rectification"])

    with tab1:
        st.image(
            result["annotated_preview"],
            caption="Accepted detections (preview)",
            use_container_width=True,
        )
        if result["annotated_jpeg"] is not None:
            output_stem = Path(result["filename"]).stem
            st.download_button(
                label="Download full-resolution annotated image",
                data=result["annotated_jpeg"],
                file_name=f"{output_stem}_varroa_detections.jpg",
                mime="image/jpeg",
            )

    with tab2:
        if not detections:
            st.info("No detections passed the current confidence and size thresholds.")
        else:
            st.session_state.detection_index %= len(detections)
            b1, b2, b3 = st.columns([1, 2, 1])
            if b1.button("← Previous", use_container_width=True):
                st.session_state.detection_index = (st.session_state.detection_index - 1) % len(detections)
            if b3.button("Next →", use_container_width=True):
                st.session_state.detection_index = (st.session_state.detection_index + 1) % len(detections)

            idx = st.session_state.detection_index
            detection = detections[idx]
            b2.markdown(
                f"<h4 style='text-align:center'>Detection {idx + 1} / {len(detections)}</h4>",
                unsafe_allow_html=True,
            )
            crop = make_detection_crop(result["rectified_rgb"], detection, margin=140, output_size=480)
            left, right = st.columns([2, 1])
            left.image(crop, caption=f"Detection {idx + 1} (zoomed)", use_container_width=True)
            right.metric("Confidence", f'{detection["confidence"]:.1%}')
            right.write(f'Bounding box: {detection["width"]:.0f} × {detection["height"]:.0f} px')
            right.write(f'Center coordinates: ({detection["center_x"]:.0f}, {detection["center_y"]:.0f})')

    with tab3:
        st.image(
            result["rectified_preview"],
            caption="Rectified or cropped sticky board",
            use_container_width=True,
        )
        with st.expander("Board segmentation mask"):
            st.image(
                result["mask_preview"],
                caption="HSV brightness segmentation mask",
                use_container_width=True,
                clamp=True,
            )
