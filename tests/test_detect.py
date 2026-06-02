"""Unit tests for local detection engines (EasyOCR and YOLOv8)."""

from unittest.mock import MagicMock, patch

from waywarp_scanner.detect import run_ocr, run_yolo


@patch("waywarp_scanner.detect.get_optimal_device")
@patch("waywarp_scanner.detect.easyocr.Reader")
def test_run_ocr_cuda(mock_reader_cls: MagicMock, mock_get_device: MagicMock) -> None:
    """Test run_ocr with CUDA hardware accelerator."""
    mock_get_device.return_value = "cuda"
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[10, 20], [110, 20], [110, 40], [10, 40]], "Submit", 0.95)
    ]
    mock_reader_cls.return_value = mock_reader

    results = run_ocr("dummy_path.png", model_dir="/dummy/models")

    # Assert Reader initialized with correct gpu flag and model dir
    mock_reader_cls.assert_called_once_with(
        ["en"], gpu=True, model_storage_directory="/dummy/models"
    )
    mock_reader.readtext.assert_called_once_with("dummy_path.png")

    assert len(results) == 1
    det = results[0]
    assert det["type"] == "text"
    assert det["text"] == "Submit"
    assert det["bbox"] == [10.0, 20.0, 100.0, 20.0]
    assert det["center"] == [60.0, 30.0]
    assert det["confidence"] == 0.95


@patch("waywarp_scanner.detect.get_optimal_device")
@patch("waywarp_scanner.detect.easyocr.Reader")
def test_run_ocr_cpu(mock_reader_cls: MagicMock, mock_get_device: MagicMock) -> None:
    """Test run_ocr with CPU fallback (gpu=False)."""
    mock_get_device.return_value = "cpu"
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [([[50, 60], [70, 60], [70, 80], [50, 80]], "OK", 0.88)]
    mock_reader_cls.return_value = mock_reader

    results = run_ocr("dummy_path.png")

    mock_reader_cls.assert_called_once_with(["en"], gpu=False, model_storage_directory=None)
    assert len(results) == 1
    assert results[0]["text"] == "OK"
    assert results[0]["bbox"] == [50.0, 60.0, 20.0, 20.0]
    assert results[0]["center"] == [60.0, 70.0]


@patch("waywarp_scanner.detect.get_optimal_device")
@patch("waywarp_scanner.detect.YOLO")
def test_run_yolo_cuda(mock_yolo_cls: MagicMock, mock_get_device: MagicMock) -> None:
    """Test run_yolo with CUDA hardware accelerator and correct coordinate mapping."""
    mock_get_device.return_value = "cuda"

    mock_model = MagicMock()
    mock_model.names = {0: "button", 1: "input"}

    # Mock prediction result
    mock_box = MagicMock()
    mock_box.xyxy = [[100.0, 200.0, 300.0, 250.0]]
    mock_box.cls = [0]
    mock_box.conf = [0.92]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model.predict.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    results = run_yolo("dummy_path.png", "yolov8n.pt")

    mock_yolo_cls.assert_called_once_with("yolov8n.pt")
    mock_model.predict.assert_called_once_with("dummy_path.png", device="cuda", verbose=False)

    assert len(results) == 1
    det = results[0]
    assert det["type"] == "button"
    assert det["text"] == ""
    assert det["bbox"] == [100.0, 200.0, 200.0, 50.0]
    assert det["center"] == [200.0, 225.0]
    assert det["confidence"] == 0.92


@patch("waywarp_scanner.detect.get_optimal_device")
@patch("waywarp_scanner.detect.YOLO")
def test_run_yolo_no_boxes(mock_yolo_cls: MagicMock, mock_get_device: MagicMock) -> None:
    """Test run_yolo handles empty results or results without boxes gracefully."""
    mock_get_device.return_value = "cpu"
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = None  # No boxes
    mock_model.predict.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    results = run_yolo("dummy_path.png", "yolov8n.pt")

    assert len(results) == 0


@patch("waywarp_scanner.detect.get_optimal_device")
@patch("waywarp_scanner.detect.easyocr.Reader")
def test_run_ocr_filters_noise(mock_reader_cls: MagicMock, mock_get_device: MagicMock) -> None:
    """Verify run_ocr filters out low confidence detections and symbol-only short noise."""
    mock_get_device.return_value = "cpu"
    mock_reader = MagicMock()
    mock_reader.readtext.return_value = [
        ([[10, 20], [110, 20], [110, 40], [10, 40]], "Submit", 0.95),  # OK
        ([[50, 60], [70, 60], [70, 80], [50, 80]], "  Cancel  ", 0.88),  # OK, will be stripped
        ([[20, 20], [30, 20], [30, 30], [20, 30]], "|", 0.99),  # Symbol-only, discard
        ([[40, 40], [50, 40], [50, 50], [40, 50]], "a", 0.15),  # Low confidence, discard
        ([[80, 80], [90, 80], [90, 90], [80, 90]], ".)", 0.80),  # Symbol-only, discard
    ]
    mock_reader_cls.return_value = mock_reader

    results = run_ocr("dummy_path.png")

    assert len(results) == 2
    assert results[0]["text"] == "Submit"
    assert results[1]["text"] == "Cancel"


@patch("waywarp_scanner.detect.get_optimal_device")
@patch("waywarp_scanner.detect.YOLO")
def test_run_yolo_coco_fallback(mock_yolo_cls: MagicMock, mock_get_device: MagicMock) -> None:
    """Verify run_yolo discards predictions if the model is a generic COCO model."""
    mock_get_device.return_value = "cpu"
    mock_model = MagicMock()
    # Mocking standard COCO names dictionary
    mock_model.names = {0: "person", 63: "laptop"}

    mock_yolo_cls.return_value = mock_model

    results = run_yolo("dummy_path.png", "yolov8n.pt")

    # Must immediately return empty list and skip predicting to fallback gracefully
    assert results == []
    assert mock_model.predict.call_count == 0


@patch("waywarp_scanner.detect.get_optimal_device")
@patch("waywarp_scanner.detect.easyocr.Reader")
@patch("PIL.Image.open")
def test_run_ocr_downscale_restore(
    mock_image_open: MagicMock,
    mock_reader_cls: MagicMock,
    mock_get_device: MagicMock,
) -> None:
    """Verify that run_ocr downscales high-res screenshots and correctly
    restores original coords."""
    mock_get_device.return_value = "cpu"

    # Mock PIL Image
    mock_img = MagicMock()
    mock_img.size = (2560, 1600)  # Width > 1280, triggers ratio = 1280.0 / 2560.0 = 0.5

    mock_resized_img = MagicMock()
    mock_img.resize.return_value = mock_resized_img
    mock_image_open.return_value = mock_img

    mock_reader = MagicMock()
    # EasyOCR results are on the resized image (W=1280, H=800)
    # Give it a word at [50, 100, 100, 50] on resized image (logical center = [100.0, 125.0])
    mock_reader.readtext.return_value = [
        ([[50, 100], [150, 100], [150, 150], [50, 150]], "Accelerate", 0.96)
    ]
    mock_reader_cls.return_value = mock_reader

    results = run_ocr("dummy_highres.png")

    # Assert PIL.Image.open and resize were called
    mock_image_open.assert_called_once_with("dummy_highres.png")
    # Bilinear resampling is BILINEAR
    from PIL import Image

    mock_img.resize.assert_called_once_with((1280, 800), Image.Resampling.BILINEAR)

    # EasyOCR reader should have been fed with the resized image
    mock_reader.readtext.assert_called_once_with(mock_resized_img)

    # Coordinates must be successfully restored back to 2560x1600 space
    # by multiplying by 2.0 (1.0 / 0.5)
    assert len(results) == 1
    det = results[0]
    assert det["text"] == "Accelerate"
    # Resized bbox is [50.0, 100.0, 100.0, 50.0]
    # Restored bbox should be [100.0, 200.0, 200.0, 100.0]
    assert det["bbox"] == [100.0, 200.0, 200.0, 100.0]
    # Restored center should be [200.0, 250.0]
    assert det["center"] == [200.0, 250.0]
