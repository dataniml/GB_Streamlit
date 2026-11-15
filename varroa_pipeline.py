from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

register_heif_opener()


@dataclass(frozen=True)
class PipelineSettings:
    tile_size: int = 640
    overlap_ratio: float = 0.20
    confidence_threshold: float = 0.30
    min_box_width: float = 0.0
    min_box_height: float = 0.0
    max_box_width: float = 50.0
    max_box_height: float = 50.0
    hsv_v_min: int = 150
    device: str = "cpu"
    board_width: int = 6040
    board_height: int = 5040


@lru_cache(maxsize=2)
def load_detection_model(model_path: str, device: str = "cpu"):
    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.30,
        device=device,
    )


def standardize_image(file_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(file_bytes))
    image = ImageOps.exif_transpose(image)
    if image.height > image.width:
        image = image.rotate(90, expand=True)
    image = image.convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def detect_and_rectify_board(
    image_bgr: np.ndarray,
    width: int = 6040,
    height: int = 5040,
    hsv_v_min: int = 150,
):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, hsv_v_min]), np.array([180, 80, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(
            "No sticky board was detected in the image. "
            "Please try an image where the board edges are more clearly visible."
        )

    largest = max(contours, key=cv2.contourArea)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    if cv2.contourArea(largest) < image_area * 0.05:
        raise ValueError("The detected bright region is too small to be a sticky board.")

    perimeter = cv2.arcLength(largest, True)
    approx = None
    for multiplier in np.arange(0.01, 0.101, 0.01):
        candidate = cv2.approxPolyDP(largest, float(multiplier) * perimeter, True)
        if len(candidate) == 4:
            approx = candidate
            break

    if approx is not None:
        rect = order_points(approx.reshape(4, 2).astype("float32"))
        avg_width = (np.linalg.norm(rect[1] - rect[0]) + np.linalg.norm(rect[2] - rect[3])) / 2.0
        avg_height = (np.linalg.norm(rect[3] - rect[0]) + np.linalg.norm(rect[2] - rect[1])) / 2.0
        if width > height and avg_height > avg_width:
            rect = np.roll(rect, 1, axis=0)

        destination = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, destination)
        result = cv2.warpPerspective(image_bgr, matrix, (width, height), flags=cv2.INTER_LANCZOS4)
        return result, closed, "perspective"

    x, y, w, h = cv2.boundingRect(largest)
    cropped = image_bgr[y:y + h, x:x + w]
    result = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LANCZOS4)
    return result, closed, "bounding_box"


def run_detection(image_rgb: np.ndarray, model, settings: PipelineSettings):
    model.confidence_threshold = settings.confidence_threshold
    result = get_sliced_prediction(
        image_rgb,
        model,
        slice_height=settings.tile_size,
        slice_width=settings.tile_size,
        overlap_height_ratio=settings.overlap_ratio,
        overlap_width_ratio=settings.overlap_ratio,
        verbose=0,
    )

    original = result.object_prediction_list
    detections = []
    removed = 0
    for prediction in original:
        x1 = float(prediction.bbox.minx)
        y1 = float(prediction.bbox.miny)
        x2 = float(prediction.bbox.maxx)
        y2 = float(prediction.bbox.maxy)
        width = x2 - x1
        height = y2 - y1
        if (
            width < settings.min_box_width
            or height < settings.min_box_height
            or width > settings.max_box_width
            or height > settings.max_box_height
        ):
            removed += 1
            continue
        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "center_x": (x1 + x2) / 2.0,
            "center_y": (y1 + y2) / 2.0,
            "confidence": float(prediction.score.value),
        })

    detections.sort(key=lambda item: (item["center_y"], item["center_x"]))
    return detections, len(original), removed


def draw_detections(image_rgb: np.ndarray, detections: list[dict]) -> np.ndarray:
    output = image_rgb.copy()
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = [int(round(detection[k])) for k in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'{index}: {detection["confidence"]:.0%}'
        cv2.putText(output, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    return output


def make_detection_crop(image_rgb: np.ndarray, detection: dict, margin: int = 140, output_size: int = 480):
    h, w = image_rgb.shape[:2]
    cx, cy = int(detection["center_x"]), int(detection["center_y"])
    x1, y1 = max(0, cx - margin), max(0, cy - margin)
    x2, y2 = min(w, cx + margin), min(h, cy + margin)
    crop = image_rgb[y1:y2, x1:x2].copy()
    local_x1 = int(detection["x1"]) - x1
    local_y1 = int(detection["y1"]) - y1
    local_x2 = int(detection["x2"]) - x1
    local_y2 = int(detection["y2"]) - y1
    cv2.rectangle(crop, (local_x1, local_y1), (local_x2, local_y2), (255, 0, 0), 2)
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_NEAREST)


def run_pipeline(file_bytes: bytes, filename: str, model, settings: PipelineSettings):
    standardized = standardize_image(file_bytes)
    rectified_bgr, mask, board_mode = detect_and_rectify_board(
        standardized,
        width=settings.board_width,
        height=settings.board_height,
        hsv_v_min=settings.hsv_v_min,
    )
    rectified_rgb = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
    detections, before_filtering, removed = run_detection(rectified_rgb, model, settings)
    annotated_rgb = draw_detections(rectified_rgb, detections)
    return {
        "filename": filename,
        "board_mode": board_mode,
        "detections": detections,
        "detections_before_filtering": before_filtering,
        "removed_too_large": removed,
        "rectified_rgb": rectified_rgb,
        "annotated_rgb": annotated_rgb,
        "board_mask": mask,
    }
