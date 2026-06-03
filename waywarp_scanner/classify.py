"""Heuristic element type classification for OCR-only detections (#40)."""

import re
from typing import Any

# Common button label patterns (case-insensitive)
_BUTTON_LABELS = re.compile(
    r"^(?:"
    r"submit|cancel|ok|okay|save|delete|remove|close|apply|next|back|done"
    r"|sign\s*in|sign\s*up|log\s*in|log\s*out|register|continue|confirm"
    r"|accept|decline|reject|send|reset|clear|update|create|add|edit"
    r"|yes|no|retry|skip|finish|start|stop|pause|play|search"
    r"|download|upload|install|uninstall|enable|disable|connect|disconnect"
    r"|copy|paste|cut|undo|redo|refresh|reload|share|print|export|import"
    r"|browse|open|new|run|build|test|deploy|publish|subscribe|buy|checkout"
    r")$",
    re.IGNORECASE,
)

# Common menu bar labels (case-insensitive)
_MENU_LABELS = re.compile(
    r"^(?:"
    r"file|edit|view|help|tools|window|options|settings|preferences"
    r"|format|insert|selection|terminal|go|run|debug|extensions"
    r"|bookmarks|history|tab|navigate"
    r")$",
    re.IGNORECASE,
)

# Common tab labels (case-insensitive)
_TAB_LABELS = re.compile(
    r"(?:\.(?:py|rs|js|ts|tsx|jsx|html|css|json|yaml|yml|toml|md|txt|sh|go|c|cpp|h)$)",
    re.IGNORECASE,
)


def classify_element(
    element: dict[str, Any],
    screen_width: float = 1920.0,
    screen_height: float = 1080.0,
) -> dict[str, Any]:
    """Classify an OCR text element into a GUI element type using heuristics.

    Classification priority:
    1. Menu item — top 5% of screen, matches menu labels
    2. Tab — top 10% of screen, looks like a file tab
    3. Button — matches common button labels, short text, small bbox
    4. Link — contains URL-like patterns
    5. Default: text

    Args:
        element: Detection dict with 'text', 'center', 'bbox', etc.
        screen_width: Logical screen width for position-based classification.
        screen_height: Logical screen height for position-based classification.

    Returns:
        A copy of the element dict with updated 'type' field.
    """
    result = element.copy()
    text = element.get("text", "").strip()
    _cx, cy = element.get("center", [0.0, 0.0])
    bbox = element.get("bbox", [0.0, 0.0, 0.0, 0.0])
    _bx, _by, _bw, _bh = bbox

    if not text:
        return result

    # Normalized vertical position (0.0 = top, 1.0 = bottom)
    y_ratio = cy / screen_height if screen_height > 0 else 0.5
    word_count = len(text.split())

    # 1. Menu item: top 5% of screen + matches menu labels
    if y_ratio < 0.05 and _MENU_LABELS.match(text):
        result["type"] = "menu_item"
        return result

    # 2. Tab: top 10% of screen + looks like filename tab
    if y_ratio < 0.10 and _TAB_LABELS.search(text):
        result["type"] = "tab"
        return result

    # 3. Button: matches common button labels, short text
    if _BUTTON_LABELS.match(text):
        result["type"] = "button"
        return result

    # 4. Link: contains URL patterns
    if re.search(r"https?://|www\.", text, re.IGNORECASE):
        result["type"] = "link"
        return result

    # 6. Status bar items: bottom 5% of screen
    if y_ratio > 0.95 and word_count <= 3:
        result["type"] = "status_item"
        return result

    # Default: keep as text
    return result
