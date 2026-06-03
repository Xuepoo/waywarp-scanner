"""Unit tests for heuristic element type classification (#40)."""

from waywarp_scanner.classify import classify_element


def test_classify_button_by_label() -> None:
    """Common button labels should be classified as 'button'."""
    for label in ["Submit", "Cancel", "OK", "Save", "Delete", "Close", "Apply", "Next", "Sign In"]:
        elem = {
            "type": "text",
            "text": label,
            "center": [100.0, 500.0],
            "bbox": [80.0, 490.0, 40.0, 20.0],
        }
        result = classify_element(elem, screen_width=1600, screen_height=1000)
        assert (
            result["type"] == "button"
        ), f"Expected 'button' for '{label}', got '{result['type']}'"


def test_classify_menu_item() -> None:
    """Menu bar labels at top of screen should be classified as 'menu_item'."""
    for label in ["File", "Edit", "View", "Help", "Tools", "Window"]:
        elem = {
            "type": "text",
            "text": label,
            "center": [100.0, 25.0],  # top 2.5% of 1000px screen
            "bbox": [80.0, 15.0, 40.0, 20.0],
        }
        result = classify_element(elem, screen_width=1600, screen_height=1000)
        assert (
            result["type"] == "menu_item"
        ), f"Expected 'menu_item' for '{label}', got '{result['type']}'"


def test_classify_menu_not_at_top() -> None:
    """Menu labels NOT at top of screen should remain as button or text."""
    elem = {
        "type": "text",
        "text": "File",
        "center": [100.0, 500.0],  # middle of screen
        "bbox": [80.0, 490.0, 40.0, 20.0],
    }
    result = classify_element(elem, screen_width=1600, screen_height=1000)
    # "File" is not in button labels, so it stays as text
    assert result["type"] == "text"


def test_classify_tab() -> None:
    """File-like labels at top of screen should be classified as 'tab'."""
    elem = {
        "type": "text",
        "text": "main.py",
        "center": [200.0, 50.0],  # top 5% of 1000px screen
        "bbox": [170.0, 40.0, 60.0, 20.0],
    }
    result = classify_element(elem, screen_width=1600, screen_height=1000)
    assert result["type"] == "tab"


def test_classify_link() -> None:
    """URL-containing text should be classified as 'link'."""
    elem = {
        "type": "text",
        "text": "https://github.com/example",
        "center": [400.0, 300.0],
        "bbox": [200.0, 290.0, 400.0, 20.0],
    }
    result = classify_element(elem, screen_width=1600, screen_height=1000)
    assert result["type"] == "link"


def test_classify_status_item() -> None:
    """Short text at bottom of screen should be classified as 'status_item'."""
    elem = {
        "type": "text",
        "text": "Ln 42",
        "center": [800.0, 990.0],  # bottom 1% of 1000px screen
        "bbox": [780.0, 980.0, 40.0, 20.0],
    }
    result = classify_element(elem, screen_width=1600, screen_height=1000)
    assert result["type"] == "status_item"


def test_classify_keeps_regular_text() -> None:
    """Regular text in the middle of screen should remain as 'text'."""
    elem = {
        "type": "text",
        "text": "Welcome to the application dashboard",
        "center": [800.0, 500.0],
        "bbox": [500.0, 490.0, 600.0, 20.0],
    }
    result = classify_element(elem, screen_width=1600, screen_height=1000)
    assert result["type"] == "text"


def test_classify_preserves_other_fields() -> None:
    """Classification should preserve all existing element fields."""
    elem = {
        "type": "text",
        "text": "Submit",
        "center": [100.0, 500.0],
        "bbox": [80.0, 490.0, 40.0, 20.0],
        "confidence": 0.95,
        "monitor_index": 0,
    }
    result = classify_element(elem, screen_width=1600, screen_height=1000)
    assert result["type"] == "button"
    assert result["confidence"] == 0.95
    assert result["monitor_index"] == 0
    assert result["center"] == [100.0, 500.0]


def test_classify_empty_text() -> None:
    """Empty text should remain as 'text'."""
    elem = {
        "type": "text",
        "text": "",
        "center": [100.0, 500.0],
        "bbox": [80.0, 490.0, 40.0, 20.0],
    }
    result = classify_element(elem, screen_width=1600, screen_height=1000)
    assert result["type"] == "text"


def test_classify_case_insensitive_button() -> None:
    """Button classification should be case-insensitive."""
    for label in ["submit", "CANCEL", "Ok", "SAVE"]:
        elem = {
            "type": "text",
            "text": label,
            "center": [100.0, 500.0],
            "bbox": [80.0, 490.0, 40.0, 20.0],
        }
        result = classify_element(elem, screen_width=1600, screen_height=1000)
        assert result["type"] == "button", f"Expected 'button' for '{label}'"
