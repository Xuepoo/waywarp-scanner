"""Local detection engines integration for OCR (EasyOCR) and Object Detection (YOLOv8)."""

from typing import Any

import easyocr  # type: ignore
from ultralytics import YOLO  # type: ignore

from waywarp_scanner.device import get_optimal_device


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
    results = reader.readtext(image_path)

    detections = []
    for bbox, text, confidence in results:
        # bbox is typically [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        xs = [float(pt[0]) for pt in bbox]
        ys = [float(pt[1]) for pt in bbox]

        x_min = min(xs)
        y_min = min(ys)
        w = max(xs) - x_min
        h = max(ys) - y_min

        cx = x_min + w / 2.0
        cy = y_min + h / 2.0

        detections.append(
            {
                "type": "text",
                "text": str(text),
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
    device = get_optimal_device()
    model = YOLO(model_path)
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
