"""Bounding box merging and coordinate transformation logic for waywarp-scanner."""

from typing import Any

from waywarp_scanner.capture import to_logical_coords


def merge_elements(
    yolo_elements: list[dict[str, Any]],
    ocr_elements: list[dict[str, Any]],
    monitor_scales: dict[str, float] | None = None,
    monitor_name: str | None = None,
    monitor_index: int = 0,
    physical_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Merge YOLO-detected widgets with OCR-detected text elements.

    Transforms physical pixel coordinates into logical Wayland units based on monitor scaling,
    associates text elements inside widgets (left-to-right), leaves non-associated text
    as standalone elements, and assigns sequential IDs.

    Args:
        yolo_elements: List of YOLO detections with keys 'type', 'bbox', 'center', etc.
        ocr_elements: List of OCR detections with keys 'text', 'bbox', 'center', etc.
        monitor_scales: Optional dictionary mapping monitor names to their scale factors.
        monitor_name: Optional name of the monitor the screenshot was captured on.
        monitor_index: Optional integer index of the monitor (defaults to 0).
        physical_size: Optional tuple (width, height) representing the physical screen size.

    Returns:
        A list of merged elements ready for JSON serialization, where each element has keys:
        - id: Sequential integer starting at 0
        - type: Widget type or "text"
        - text: Joined text for widgets or original text for standalone text elements
        - center: Logical [cx, cy] coordinates
        - bbox: Logical [x, y, w, h] coordinates
        - monitor_index: Index of the monitor the element resides on
    """
    scale_factor = 1.0
    if monitor_scales is not None:
        if monitor_name is not None:
            scale_factor = monitor_scales.get(monitor_name, 1.0)
        elif monitor_scales:
            scale_factor = next(iter(monitor_scales.values()), 1.0)

    # Calculate logical screen dimensions for clamping bounds
    if physical_size is not None:
        logical_width = physical_size[0] / scale_factor
        logical_height = physical_size[1] / scale_factor
    else:
        # Fallback to standard 1920x1080 logical sizes if physical size isn't provided
        logical_width = 1920.0
        logical_height = 1080.0

    # 1. Convert YOLO elements to logical coordinates
    logical_yolo = []
    for elem in yolo_elements:
        cx, cy = elem["center"]
        x, y, w, h = elem["bbox"]
        cx_l, cy_l = to_logical_coords(cx, cy, scale_factor)
        x_l, y_l = to_logical_coords(x, y, scale_factor)
        w_l, h_l = to_logical_coords(w, h, scale_factor)

        # Clamp logical coordinates to screen boundaries (Issue #30)
        x_l = max(0.0, min(x_l, logical_width))
        y_l = max(0.0, min(y_l, logical_height))
        w_l = max(0.0, min(w_l, logical_width - x_l))
        h_l = max(0.0, min(h_l, logical_height - y_l))
        cx_l = x_l + w_l / 2.0
        cy_l = y_l + h_l / 2.0

        new_elem = elem.copy()
        new_elem["center"] = [cx_l, cy_l]
        new_elem["bbox"] = [x_l, y_l, w_l, h_l]
        logical_yolo.append(new_elem)

    # 2. Convert OCR elements to logical coordinates
    logical_ocr = []
    for elem in ocr_elements:
        cx, cy = elem["center"]
        x, y, w, h = elem["bbox"]
        cx_l, cy_l = to_logical_coords(cx, cy, scale_factor)
        x_l, y_l = to_logical_coords(x, y, scale_factor)
        w_l, h_l = to_logical_coords(w, h, scale_factor)

        # Clamp logical coordinates to screen boundaries (Issue #30)
        x_l = max(0.0, min(x_l, logical_width))
        y_l = max(0.0, min(y_l, logical_height))
        w_l = max(0.0, min(w_l, logical_width - x_l))
        h_l = max(0.0, min(h_l, logical_height - y_l))
        cx_l = x_l + w_l / 2.0
        cy_l = y_l + h_l / 2.0

        new_elem = elem.copy()
        new_elem["center"] = [cx_l, cy_l]
        new_elem["bbox"] = [x_l, y_l, w_l, h_l]
        logical_ocr.append(new_elem)

    # 3. Center enclosure checking helper
    def is_inside(cx: float, cy: float, bbox: list[float]) -> bool:
        bx, by, bw, bh = bbox
        return (bx <= cx <= bx + bw) and (by <= cy <= by + bh)

    merged_ocr_indices: set[int] = set()
    merged_elements: list[dict[str, Any]] = []

    # 4. Process each widget (YOLO detection)
    for yolo_elem in logical_yolo:
        inside_ocr: list[tuple[int, dict[str, Any]]] = []
        for idx, ocr_elem in enumerate(logical_ocr):
            cx, cy = ocr_elem["center"]
            if is_inside(cx, cy, yolo_elem["bbox"]):
                inside_ocr.append((idx, ocr_elem))

        # Sort text elements horizontally by center_x (ascending)
        inside_ocr.sort(key=lambda item: item[1]["center"][0])

        # Join text labels with a space
        text_labels = [item[1]["text"] for item in inside_ocr]
        joined_text = " ".join(text_labels)

        # Mark these OCR text elements as merged
        for idx, _ in inside_ocr:
            merged_ocr_indices.add(idx)

        # Build widget entry
        widget_elem = {
            "type": yolo_elem["type"],
            "text": joined_text,
            "center": yolo_elem["center"],
            "bbox": yolo_elem["bbox"],
            "monitor_index": monitor_index,
        }
        if "confidence" in yolo_elem:
            widget_elem["confidence"] = yolo_elem["confidence"]

        merged_elements.append(widget_elem)

    # 5. Process remaining standalone text elements
    for idx, ocr_elem in enumerate(logical_ocr):
        if idx not in merged_ocr_indices:
            text_elem = {
                "type": "text",
                "text": ocr_elem["text"],
                "center": ocr_elem["center"],
                "bbox": ocr_elem["bbox"],
                "monitor_index": monitor_index,
            }
            if "confidence" in ocr_elem:
                text_elem["confidence"] = ocr_elem["confidence"]

            merged_elements.append(text_elem)

    # 6. Assign sequential IDs starting at 0
    final_elements: list[dict[str, Any]] = []
    for idx, elem in enumerate(merged_elements):
        # Double clamp center coordinates to screen logical bounds for absolute safety (Issue #35)
        cx, cy = elem["center"]
        elem["center"] = [
            max(0.0, min(cx, logical_width)),
            max(0.0, min(cy, logical_height)),
        ]

        new_elem = {"id": idx}
        new_elem.update(elem)
        final_elements.append(new_elem)

    return final_elements
