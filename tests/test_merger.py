"""Unit tests for the bounding box merger and JSON serialization logic."""

from typing import Any

from waywarp_scanner.merger import merge_elements


def test_text_enclosure_and_sorting() -> None:
    """Test text enclosure and horizontal sorting in merger."""
    # Define a widget at logical [100, 100, 200, 100], center [200, 150]
    # No scaling: scale_factor = 1.0
    yolo_elements = [
        {
            "type": "button",
            "bbox": [100.0, 100.0, 200.0, 100.0],
            "center": [200.0, 150.0],
            "confidence": 0.9,
        }
    ]

    # Define OCR elements:
    # 1. Inside, on the right: center [250, 150] -> text: "World"
    # 2. Inside, on the left: center [150, 150] -> text: "Hello"
    # 3. Outside: center [50, 150] -> text: "Outside"
    ocr_elements = [
        {
            "type": "text",
            "text": "World",
            "bbox": [220.0, 140.0, 60.0, 20.0],
            "center": [250.0, 150.0],
            "confidence": 0.95,
        },
        {
            "type": "text",
            "text": "Hello",
            "bbox": [120.0, 140.0, 60.0, 20.0],
            "center": [150.0, 150.0],
            "confidence": 0.98,
        },
        {
            "type": "text",
            "text": "Outside",
            "bbox": [20.0, 140.0, 60.0, 20.0],
            "center": [50.0, 150.0],
            "confidence": 0.85,
        },
    ]

    result = merge_elements(yolo_elements, ocr_elements)

    # We expect 2 final elements:
    # - ID 0: The button widget with merged text "Hello World"
    # - ID 1: The standalone text element "Outside"
    assert len(result) == 2

    # Check the merged button element
    button_elem = result[0]
    assert button_elem["id"] == 0
    assert button_elem["type"] == "button"
    assert button_elem["text"] == "Hello World"
    assert button_elem["center"] == [200.0, 150.0]
    assert button_elem["bbox"] == [100.0, 100.0, 200.0, 100.0]
    assert button_elem["confidence"] == 0.9

    # Check the standalone text element
    text_elem = result[1]
    assert text_elem["id"] == 1
    assert text_elem["type"] == "text"
    assert text_elem["text"] == "Outside"
    assert text_elem["center"] == [50.0, 150.0]
    assert text_elem["bbox"] == [20.0, 140.0, 60.0, 20.0]
    assert text_elem["confidence"] == 0.85


def test_monitor_scaling() -> None:
    """Test physical coordinates conversion to logical units via scale factor."""
    yolo_elements = [
        {
            "type": "input",
            "bbox": [200.0, 400.0, 300.0, 100.0],
            "center": [350.0, 450.0],
        }
    ]

    ocr_elements = [
        {
            "type": "text",
            "text": "Username",
            "bbox": [220.0, 420.0, 100.0, 40.0],
            "center": [270.0, 440.0],
        }
    ]

    # scale_factor = 2.0
    monitor_scales = {"HDMI-A-1": 2.0, "eDP-1": 1.5}

    result = merge_elements(
        yolo_elements,
        ocr_elements,
        monitor_scales=monitor_scales,
        monitor_name="HDMI-A-1",
        monitor_index=1,
    )

    assert len(result) == 1
    elem = result[0]
    assert elem["id"] == 0
    assert elem["type"] == "input"
    assert elem["text"] == "Username"
    assert elem["monitor_index"] == 1
    # Check scaled center: [350.0, 450.0] / 2.0 = [175.0, 225.0]
    assert elem["center"] == [175.0, 225.0]
    # Check scaled bbox: [200.0, 400.0, 300.0, 100.0] / 2.0 = [100.0, 200.0, 150.0, 50.0]
    assert elem["bbox"] == [100.0, 200.0, 150.0, 50.0]


def test_default_scaling_missing_args() -> None:
    """Test that scale factor defaults to 1.0 if monitor arguments are missing or not matching."""
    yolo_elements = [
        {
            "type": "checkbox",
            "bbox": [10.0, 20.0, 30.0, 40.0],
            "center": [25.0, 40.0],
        }
    ]
    ocr_elements: list[dict[str, Any]] = []

    # Case 1: monitor_scales is None
    result = merge_elements(yolo_elements, ocr_elements, monitor_scales=None, monitor_name="eDP-1")
    assert result[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]

    # Case 2: monitor_name is None
    result = merge_elements(
        yolo_elements,
        ocr_elements,
        monitor_scales={"eDP-1": 2.0},
        monitor_name=None,
    )
    assert result[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]

    # Case 3: monitor_name is not in monitor_scales
    result = merge_elements(
        yolo_elements,
        ocr_elements,
        monitor_scales={"eDP-1": 2.0},
        monitor_name="HDMI-A-1",
    )
    assert result[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]


def test_schema_compatibility() -> None:
    """Verify schema compatibility of output elements."""
    yolo_elements = [
        {
            "type": "button",
            "bbox": [10.0, 10.0, 20.0, 20.0],
            "center": [20.0, 20.0],
            "confidence": 0.88,
        }
    ]
    ocr_elements = [
        {
            "type": "text",
            "text": "Click",
            "bbox": [12.0, 12.0, 5.0, 5.0],
            "center": [14.5, 14.5],
            "confidence": 0.99,
        }
    ]

    result = merge_elements(yolo_elements, ocr_elements, monitor_index=2)

    assert len(result) == 1
    elem = result[0]

    # Required keys and their expected types
    assert isinstance(elem["id"], int)
    assert isinstance(elem["type"], str)
    assert isinstance(elem["text"], str)
    assert isinstance(elem["center"], list)
    assert len(elem["center"]) == 2
    assert all(isinstance(val, float) for val in elem["center"])
    assert isinstance(elem["bbox"], list)
    assert len(elem["bbox"]) == 4
    assert all(isinstance(val, float) for val in elem["bbox"])
    assert isinstance(elem["monitor_index"], int)
    assert isinstance(elem["confidence"], float)

    assert elem["id"] == 0
    assert elem["type"] == "button"
    assert elem["text"] == "Click"
    assert elem["monitor_index"] == 2
    assert elem["confidence"] == 0.88


def test_bounds_clamping() -> None:
    """Verify that logical bounding boxes are clamped correctly inside screen resolution."""
    yolo_elements = [
        {
            "type": "button",
            "bbox": [1000.0, 800.0, 1500.0, 1000.0],
            "center": [1750.0, 1300.0],
        }
    ]

    ocr_elements: list[dict[str, Any]] = []

    # physical_size = (1920, 1080), scale = 1.0 -> logical screen width=1920, height=1080
    result = merge_elements(
        yolo_elements,
        ocr_elements,
        physical_size=(1920, 1080),
    )

    assert len(result) == 1
    elem = result[0]

    # Expected clamped bbox:
    # x = 1000 (valid)
    # y = 800 (valid)
    # w = min(1500, 1920 - 1000) = 920
    # h = min(1000, 1080 - 800) = 280
    assert elem["bbox"] == [1000.0, 800.0, 920.0, 280.0]

    # Expected clamped center:
    # cx = 1000.0 + 920.0 / 2 = 1460.0
    # cy = 800.0 + 280.0 / 2 = 940.0
    assert elem["center"] == [1460.0, 940.0]
