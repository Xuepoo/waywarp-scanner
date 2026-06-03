"""Noise filtering for OCR detections to remove terminal/code content (#41)."""

import re
from typing import Any

# Compiled regex patterns for code/terminal noise detection
_CODE_PATTERNS = re.compile(
    r"(?:"
    r"^[$#>]\s"  # Shell prompts
    r"|(?:^|\s)(?:import|from|def|class|return|if|else|elif"
    r"|for|while|try|except|raise)\s"  # Python keywords
    r"|(?:^|\s)(?:fn|let|mut|pub|use|mod|impl|struct|enum|match|async|await)\s"  # Rust keywords
    r"|(?:^|\s)(?:const|var|function|export|require|module)\s"  # JS/Node keywords
    r"|(?:=>|->|::|&&|\|\||<<|>>)"  # Operators
    r"|[{}]{2,}"  # Multiple braces
    r")",
    re.IGNORECASE,
)

_PATH_PATTERN = re.compile(
    r"(?:"
    r"(?:^|[\s\"'])(?:/home/|~/|/usr/|/etc/|/tmp/|/var/|/opt/|\./|\.\./)[\w/.-]+"  # Unix paths
    r"|(?:^|[\s\"'])[A-Z]:\\\\[\w\\\\.-]+"  # Windows paths
    r")"
)

_BRACKET_CHARS = set("(){}[]<>")


def filter_noise(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out noisy OCR detections that are likely terminal/code content.

    Applies multiple heuristic filters:
    1. Raise confidence threshold for short (single-word) detections
    2. Reject text matching code/shell patterns
    3. Reject text matching file path patterns
    4. Reject bracket-heavy text (>30% bracket characters)
    5. Reject single-char detections with low confidence

    Args:
        detections: List of OCR detection dicts with 'text', 'confidence', etc.

    Returns:
        Filtered list of detections with noise removed.
    """
    filtered = []
    for det in detections:
        text = det.get("text", "")
        confidence = det.get("confidence", 0.0)

        # 1. Single-word short text needs higher confidence
        word_count = len(text.split())
        if word_count <= 1 and confidence < 0.5:
            continue

        # 2. Single-char alphabetic with low confidence
        if len(text) == 1 and text.isalpha() and confidence < 0.7:
            continue

        # 3. Code/terminal pattern detection
        if _CODE_PATTERNS.search(text):
            continue

        # 4. File path detection
        if _PATH_PATTERN.search(text):
            continue

        # 5. Bracket-heavy text (>30% brackets)
        if len(text) > 2:
            bracket_count = sum(1 for c in text if c in _BRACKET_CHARS)
            if bracket_count / len(text) > 0.3:
                continue

        # 6. Very long text (>60 chars) is likely a log line or code
        if len(text) > 60:
            continue

        # 7. Text with too many special characters (code-like)
        if len(text) > 3:
            alnum_ratio = sum(1 for c in text if c.isalnum() or c.isspace()) / len(text)
            if alnum_ratio < 0.5:
                continue

        filtered.append(det)

    return filtered
