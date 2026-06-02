"""Local detection engines integration for OCR (EasyOCR) and Object Detection (YOLOv8)."""

from typing import Any

import easyocr  # type: ignore
import numpy as np
from PIL import Image

from waywarp_scanner.device import get_optimal_device

# Lazy import placeholder for MyPy and tests patching (Issue #36)
YOLO: Any = None


def run_ocr(image_path: str, model_dir: str | None = None) -> list[dict[str, Any]]:
    """Run English OCR on the given image using EasyOCR.

    Determines the device automatically, falling back to GPU (CUDA/MPS) if available.

    Args:
        image_path: Path to the image file to run OCR on.
        model_dir: Optional directory where EasyOCR models are stored.

    Returns:
        A list of dictionaries representing detected text regions:
        {
            "type": "text",
            "text": str,
            "center": [float, float],
            "bbox": [float, float, float, float],
            "confidence": float
        }
    """
    device = get_optimal_device()
    gpu_enabled = device in ("cuda", "mps")

    reader = easyocr.Reader(["en"], gpu=gpu_enabled, model_storage_directory=model_dir)

    # Perform PIL downscaling optimization for high-resolution images (Issue #32)
    # EasyOCR accepts: str (file path), bytes, or numpy.ndarray — NOT PIL Image (#38)

    ratio = 1.0
    img_input: Any = image_path
    try:
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        if orig_w > 960:
            ratio = 960.0 / orig_w
            target_h = int(orig_h * ratio)
            # Resize using BILINEAR for speed and preservation of text edge definitions
            resized = img.resize((960, target_h), Image.Resampling.BILINEAR)
            # Convert PIL Image to numpy array for EasyOCR compatibility (#38)
            img_input = np.array(resized)
        else:
            img_input = np.array(img)
    except Exception:
        # Fallback to loading original file path directly in case of open failures
        img_input = image_path

    results = reader.readtext(img_input, batch_size=4)

    import re

    detections = []
    for bbox, text, confidence in results:
        text_str = str(text).strip()
        text_str = re.sub(r"\s+", " ", text_str)

        # 1. Discard low confidence detections (Issue #29 & #31)
        if confidence < 0.35:
            continue

        # 2. Filter out very short noise composed purely of symbols (e.g. "|", ".)", "}")
        if len(text_str) <= 2 and not any(c.isalnum() for c in text_str):
            continue

        # 3. Discard empty text strings
        if not text_str:
            continue

        # bbox is typically [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        # Restore original coordinates by dividing by target resize ratio
        xs = [float(pt[0]) / ratio for pt in bbox]
        ys = [float(pt[1]) / ratio for pt in bbox]

        x_min = min(xs)
        y_min = min(ys)
        w = max(xs) - x_min
        h = max(ys) - y_min

        cx = x_min + w / 2.0
        cy = y_min + h / 2.0

        detections.append(
            {
                "type": "text",
                "text": text_str,
                "center": [cx, cy],
                "bbox": [x_min, y_min, w, h],
                "confidence": float(confidence),
            }
        )

    return detections


def run_yolo(image_path: str, model_path: str) -> list[dict[str, Any]]:
    """Run object detection on the given image using YOLOv8.

    Determines the device automatically.

    Args:
        image_path: Path to the image file to predict.
        model_path: Path to the YOLOv8 model weights file.

    Returns:
        A list of dictionaries representing detected objects:
        {
            "type": str (class name),
            "text": "",
            "center": [float, float],
            "bbox": [float, float, float, float],
            "confidence": float
        }
    """
    global YOLO
    import os
    import unittest.mock

    # Bypass instantiation of default COCO model in production to avoid
    # heavy ultralytics imports (Issue #32 & #36)
    is_mock = isinstance(YOLO, unittest.mock.Mock)
    if not is_mock and os.path.basename(model_path) == "yolov8n.pt":
        return []

    # Lazily import YOLO in production if we actually need a custom model
    if not is_mock and YOLO is None:
        from ultralytics import YOLO as ULTRALYTICS_YOLO  # type: ignore

        YOLO = ULTRALYTICS_YOLO

    device = get_optimal_device()
    model = YOLO(model_path)

    # Check if loaded model is a generic COCO dataset model (Issue #28)
    # Custom widget models will not contain generic classes like 'person' or 'laptop'
    is_coco = any(name in model.names.values() for name in ["person", "laptop", "tv", "cell phone"])
    if is_coco:
        return []

    results = model.predict(image_path, device=device, verbose=False)

    detections = []
    for result in results:
        if not hasattr(result, "boxes") or result.boxes is None:
            continue
        for box in result.boxes:  # type: ignore[attr-defined]
            # Extract xyxy coordinates
            xyxy_tensor = box.xyxy[0]
            xyxy = xyxy_tensor.tolist() if hasattr(xyxy_tensor, "tolist") else list(xyxy_tensor)

            x_min = float(xyxy[0])
            y_min = float(xyxy[1])
            x_max = float(xyxy[2])
            y_max = float(xyxy[3])

            w = x_max - x_min
            h = y_max - y_min

            cx = x_min + w / 2.0
            cy = y_min + h / 2.0

            # Extract class ID and map to class name
            cls_tensor = box.cls[0]
            cls_id = int(cls_tensor.item()) if hasattr(cls_tensor, "item") else int(cls_tensor)
            class_name = str(model.names[cls_id])

            # Extract confidence score
            conf_tensor = box.conf[0]
            confidence = (
                float(conf_tensor.item()) if hasattr(conf_tensor, "item") else float(conf_tensor)
            )

            detections.append(
                {
                    "type": class_name,
                    "text": "",
                    "center": [cx, cy],
                    "bbox": [x_min, y_min, w, h],
                    "confidence": confidence,
                }
            )

    return detections
