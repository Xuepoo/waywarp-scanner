"""Noise filtering for OCR detections to remove terminal/code content (#41, #44)."""

import re
from typing import Any

# Compiled regex patterns for code/terminal noise detection
_CODE_PATTERNS = re.compile(
    r"(?:"
    r"^[$#>]\s"  # Shell prompts
    # Code keywords (removed common English words)
    r"|(?:^|\s)(?:import|from|def|class|return|elif|except|raise)\s"
    r"|(?:^|\s)(?:fn|let|mut|pub|impl|struct|enum|async|await)\s"  # Rust keywords
    r"|(?:^|\s)(?:const|function|export|require|module)\s"  # JS/Node keywords
    r"|(?:=>|->|::|&&|\|\||<<|>>|==|!=|\+=|-=)"  # Operators
    r"|[{}]{2,}"  # Multiple braces
    r"|;"  # Semicolons
    r"|_"  # Underscores (code/terminal variable names)
    r")",
    re.IGNORECASE,
)

# Standalone code/terminal terms that are never valid GUI labels
_CODE_TERMS = re.compile(
    r"^(?:"
    r"stderr|stdout|stdin|argv|argc|args|extract"
    r"|nil|null|void|bool|char|int|str|dict|list|tuple|vec"
    r"|fd|pid|uid|gid|eof|nul"
    r")$",
    re.IGNORECASE,
)

_PATH_PATTERN = re.compile(
    r"(?:"
    r"(?:^|[\s\"'])(?:/home/|~/|/usr/|/etc/|/tmp/|/var/|/opt/|\./|\.\./)[\w/.-]+"  # Unix paths
    r"|(?:^|[\s\"'])[A-Z]:\\\\[\w\\\\.-]+"  # Windows paths
    # Code file extensions
    r"|\b[\w-]+\.(?:py|rs|js|ts|h|c|cpp|toml|json|yaml|yml|sh|lock|cfg|md)\b"
    r")",
    re.IGNORECASE,
)

# Erratic mixed capitalization (e.g. starts with lowercase,
# has uppercase inside, like fIsh, scAnner)
_ERRATIC_CAPS = re.compile(r"\b[a-z]+[A-Z][a-z]+\b")

# Version string typos (e.g. VO.1.7, vo.1)
_VERSION_TYPO = re.compile(r"\b[a-zA-Z]*[oO]\.\d", re.IGNORECASE)

# CLI command prefixes (e.g. grep, git, echo, but not standalone Clear/Find)
_CLI_COMMANDS = re.compile(
    r"^(?:"
    r"(?:git|cd|ls|grep|echo|cat|cargo|python3?|pip|npm|npx|node|yarn|bun|uv|docker|podman|kubectl|systemctl|sudo|apt|pacman|yay|curl|wget|ssh|tar|zip|unzip)\b"
    r"|(?:find|clear)(?:\s+|$)"
    r")"
)


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Coordinates of intersection
    ix1 = max(x1, x2)
    it1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    it2 = min(y1 + h1, y2 + h2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, it2 - it1)

    intersection = iw * ih
    union = (w1 * h1) + (w2 * h2) - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def filter_noise(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out noisy OCR detections that are likely terminal/code content.

    Applies multiple heuristic filters:
    1. Check for URL/link and allow them directly with baseline confidence
    2. Check for unmatched brackets/parentheses (code noise indicator)
    3. Check for colons in the middle of words (code signature)
    4. Check for slashes in the word (path indicator, non-URL)
    5. Check for code keywords, operators, and underscores
    6. Check for file path patterns and code file extensions
    7. Apply confidence threshold (0.45 for general single-word, 0.35 for common UI labels)
    8. Duplicate suppression via IoU overlap check

    Args:
        detections: List of OCR detection dicts with 'text', 'confidence', etc.

    Returns:
        Filtered list of detections with noise removed.
    """
    from waywarp_scanner.classify import _BUTTON_LABELS, _MENU_LABELS

    filtered = []
    for det in detections:
        text = det.get("text", "").strip()
        confidence = det.get("confidence", 0.0)

        if not text:
            continue

        # 1. URL / Link check (always keep if confidence is OK)
        is_link = bool(re.search(r"https?://|www\.", text, re.IGNORECASE))
        if is_link:
            if confidence >= 0.35:
                filtered.append(det)
            continue

        # 2. Unmatched brackets/parentheses (code noise indicator like 'Serialize) ]')
        open_p = text.count("(")
        close_p = text.count(")")
        open_b = text.count("[")
        close_b = text.count("]")
        open_c = text.count("{")
        close_c = text.count("}")
        open_a = text.count("<")
        close_a = text.count(">")
        if open_p != close_p or open_b != close_b or open_c != close_c or open_a != close_a:
            continue

        # 3. Colon in the middle of words (e.g. 'path:String' or 'path: String')
        if (
            ":" in text
            and not re.search(r"\b\d{1,2}:\d{2}\b", text)
            and re.search(r"\b[a-z]+:\s*[a-zA-Z]", text)
        ):
            continue

        # 4. Slashes in text (path indicator, since links are already handled above)
        if "/" in text:
            continue

        # 5. Code keywords, operators, underscores
        if _CODE_PATTERNS.search(text):
            continue

        # 5b. Standalone code/terminal terms (stderr, stdout, etc.)
        if _CODE_TERMS.match(text):
            continue

        # 5c. Hash-prefixed short fragments (#t, #[derive], etc.)
        if text.startswith("#") and len(text) <= 10:
            continue

        # 5d. Erratic capitalization (fIsh, scAnner, etc.)
        if _ERRATIC_CAPS.search(text):
            continue

        # 5e. Version typos (VO.1.7, etc.)
        if _VERSION_TYPO.search(text):
            continue

        # 5f. Common CLI command-line inputs (grep waywarp, echo, etc.)
        if _CLI_COMMANDS.search(text):
            continue

        # 6. File paths and code file extensions
        if _PATH_PATTERN.search(text):
            continue

        # 7. Single-char alphabetic with low confidence
        if len(text) == 1 and text.isalpha() and confidence < 0.7:
            continue

        # 8. Confidence threshold based on word count and common labels
        word_count = len(text.split())
        if word_count <= 1:
            is_common_ui = bool(_BUTTON_LABELS.match(text) or _MENU_LABELS.match(text))
            threshold = 0.35 if is_common_ui else 0.45
            if confidence < threshold:
                continue
        elif word_count == 2:
            # Multi-word gibberish like "5 1", "2 n" — apply moderate threshold
            if confidence < 0.40:
                continue
        else:
            # 3+ words — slightly lower threshold since multi-word is less likely noise
            if confidence < 0.35:
                continue

        # 8b. Short alphanumeric noise: 2-3 char text with moderate-low confidence
        # Catches garbled OCR fragments like "agy", "f52", "Axo"
        if len(text) <= 3 and text.isalpha() and confidence < 0.75:
            # Allow known UI labels (e.g., "OK", "No", "Yes", "Go")
            is_common_ui = bool(_BUTTON_LABELS.match(text) or _MENU_LABELS.match(text))
            if not is_common_ui:
                continue

        # 9. Very long text (>60 chars) is likely log/code noise
        if len(text) > 60:
            continue

        # 10. Text with too many special characters (non-alphanumeric ratio < 0.5)
        if len(text) > 3:
            alnum_ratio = sum(1 for c in text if c.isalnum() or c.isspace()) / len(text)
            if alnum_ratio < 0.5:
                continue

        filtered.append(det)

    # 11. Duplicate suppression (IoU overlap filter)
    # Sort by confidence to select the best box, but preserve original layout order
    sorted_dets = sorted(filtered, key=lambda x: x.get("confidence", 0.0), reverse=True)
    kept_ids = set()
    kept_boxes: list[list[float]] = []
    for det in sorted_dets:
        bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
        overlap = False
        for kept_box in kept_boxes:
            if compute_iou(bbox, kept_box) > 0.8:
                overlap = True
                break
        if not overlap:
            kept_ids.add(id(det))
            kept_boxes.append(bbox)

    return [det for det in filtered if id(det) in kept_ids]
