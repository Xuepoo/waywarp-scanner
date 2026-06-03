"""Unit tests for noise filtering logic (#41)."""

from waywarp_scanner.filters import filter_noise


def test_filter_keeps_clean_gui_text() -> None:
    """Clean GUI labels should pass through all filters."""
    detections = [
        {"text": "Submit", "confidence": 0.95, "center": [100, 200], "bbox": [80, 190, 40, 20]},
        {"text": "Cancel", "confidence": 0.88, "center": [200, 200], "bbox": [180, 190, 40, 20]},
        {"text": "File", "confidence": 0.92, "center": [50, 10], "bbox": [40, 5, 20, 10]},
    ]
    result = filter_noise(detections)
    assert len(result) == 3
    assert [d["text"] for d in result] == ["Submit", "Cancel", "File"]


def test_filter_removes_code_patterns() -> None:
    """Code-like text should be filtered out."""
    detections = [
        {"text": "import os", "confidence": 0.90, "center": [100, 100], "bbox": [50, 90, 100, 20]},
        {
            "text": "def main():",
            "confidence": 0.85,
            "center": [100, 120],
            "bbox": [50, 110, 100, 20],
        },
        {
            "text": "fn get_device()",
            "confidence": 0.80,
            "center": [100, 140],
            "bbox": [50, 130, 100, 20],
        },
        {
            "text": "return self.value",
            "confidence": 0.88,
            "center": [100, 160],
            "bbox": [50, 150, 100, 20],
        },
    ]
    result = filter_noise(detections)
    assert len(result) == 0


def test_filter_removes_shell_prompts() -> None:
    """Shell prompts should be filtered out."""
    detections = [
        {
            "text": "$ git status",
            "confidence": 0.90,
            "center": [100, 100],
            "bbox": [50, 90, 100, 20],
        },
        {"text": "# comment", "confidence": 0.85, "center": [100, 120], "bbox": [50, 110, 100, 20]},
    ]
    result = filter_noise(detections)
    assert len(result) == 0


def test_filter_removes_file_paths() -> None:
    """File path strings should be filtered out."""
    detections = [
        {
            "text": "/home/user/.config/app",
            "confidence": 0.90,
            "center": [100, 100],
            "bbox": [50, 90, 200, 20],
        },
        {
            "text": "~/Documents/code.py",
            "confidence": 0.85,
            "center": [100, 120],
            "bbox": [50, 110, 200, 20],
        },
    ]
    result = filter_noise(detections)
    assert len(result) == 0


def test_filter_removes_bracket_heavy() -> None:
    """Bracket-heavy text (>30% brackets) should be filtered."""
    detections = [
        {"text": "({[]})", "confidence": 0.90, "center": [100, 100], "bbox": [50, 90, 60, 20]},
    ]
    result = filter_noise(detections)
    assert len(result) == 0


def test_filter_removes_low_confidence_single_word() -> None:
    """Single-word text below 0.5 confidence should be filtered."""
    detections = [
        {"text": "garbled", "confidence": 0.42, "center": [100, 100], "bbox": [50, 90, 70, 20]},
        {"text": "Clear", "confidence": 0.55, "center": [200, 100], "bbox": [180, 90, 40, 20]},
    ]
    result = filter_noise(detections)
    assert len(result) == 1
    assert result[0]["text"] == "Clear"


def test_filter_removes_single_char_low_confidence() -> None:
    """Single alphabetic characters with low confidence should be filtered."""
    detections = [
        {"text": "a", "confidence": 0.55, "center": [100, 100], "bbox": [95, 95, 10, 10]},
        {"text": "X", "confidence": 0.80, "center": [200, 100], "bbox": [195, 95, 10, 10]},
    ]
    result = filter_noise(detections)
    assert len(result) == 1
    assert result[0]["text"] == "X"


def test_filter_removes_very_long_text() -> None:
    """Very long text (>60 chars) likely log lines should be filtered."""
    detections = [
        {"text": "a" * 61, "confidence": 0.90, "center": [100, 100], "bbox": [10, 90, 500, 20]},
        {
            "text": "Short label",
            "confidence": 0.85,
            "center": [200, 200],
            "bbox": [180, 190, 40, 20],
        },
    ]
    result = filter_noise(detections)
    assert len(result) == 1
    assert result[0]["text"] == "Short label"


def test_filter_removes_special_char_heavy() -> None:
    """Text with <50% alphanumeric chars should be filtered."""
    detections = [
        {"text": "<<=>>=::", "confidence": 0.90, "center": [100, 100], "bbox": [50, 90, 80, 20]},
        {
            "text": "Hello World",
            "confidence": 0.85,
            "center": [200, 200],
            "bbox": [180, 190, 50, 20],
        },
    ]
    result = filter_noise(detections)
    assert len(result) == 1
    assert result[0]["text"] == "Hello World"


def test_filter_operators_in_code() -> None:
    """Arrow operators and double-colon should be filtered."""
    detections = [
        {
            "text": "value -> result",
            "confidence": 0.88,
            "center": [100, 100],
            "bbox": [50, 90, 120, 20],
        },
        {
            "text": "std::string",
            "confidence": 0.85,
            "center": [100, 120],
            "bbox": [50, 110, 100, 20],
        },
    ]
    result = filter_noise(detections)
    assert len(result) == 0


def test_filter_new_noise_patterns() -> None:
    """Verify that file path slashes, unmatched brackets, colons in middle of word,
    and underscores are successfully filtered out."""
    detections = [
        {
            "text": "src/dto",
            "confidence": 0.90,
            "center": [100, 100],
            "bbox": [50, 90, 80, 20],
        },
        {
            "text": "Serialize) ]",
            "confidence": 0.88,
            "center": [100, 120],
            "bbox": [50, 110, 80, 20],
        },
        {
            "text": "path: String",
            "confidence": 0.85,
            "center": [100, 140],
            "bbox": [50, 130, 80, 20],
        },
        {
            "text": "install_command",
            "confidence": 0.92,
            "center": [100, 160],
            "bbox": [50, 150, 80, 20],
        },
        {
            "text": "https://github.com",
            "confidence": 0.95,
            "center": [100, 180],
            "bbox": [50, 170, 100, 20],
        },
        {
            "text": "12:34",
            "confidence": 0.90,
            "center": [100, 200],
            "bbox": [50, 190, 40, 20],
        },
        {
            "text": "Username:",
            "confidence": 0.90,
            "center": [100, 220],
            "bbox": [50, 210, 60, 20],
        },
    ]
    result = filter_noise(detections)
    assert len(result) == 3
    texts = [d["text"] for d in result]
    assert "https://github.com" in texts
    assert "12:34" in texts
    assert "Username:" in texts


def test_filter_duplicate_suppression() -> None:
    """Verify duplicate overlapping boxes are suppressed keeping the highest confidence."""
    detections = [
        {
            "text": "OK",
            "confidence": 0.85,
            "center": [100, 100],
            "bbox": [80, 90, 40, 20],
        },
        {
            "text": "OK",
            "confidence": 0.95,
            "center": [101, 100],
            "bbox": [80, 90, 40, 20],
        },
        {
            "text": "Cancel",
            "confidence": 0.90,
            "center": [200, 100],
            "bbox": [180, 90, 40, 20],
        },
    ]
    result = filter_noise(detections)
    assert len(result) == 2
    assert any(d["text"] == "OK" and d["confidence"] == 0.95 for d in result)
    assert not any(d["text"] == "OK" and d["confidence"] == 0.85 for d in result)
