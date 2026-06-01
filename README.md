# waywarp-scanner

High-performance, decoupled Wayland GUI layout scanner designed for AI Agents. It runs neural object detection and OCR locally to extract user interface controls (buttons, inputs, labels, text blocks) and exports logical coordinate JSON grids to downstream AI orchestrators (e.g. Gemini, Claude Code, Hermes).

By using `waywarp-scanner`, AI agents can bypass sending large raw screenshots to VLMs (Visual Language Models), **saving over 98% of prompt tokens** while securing **100% exact coordinates** for mouse interaction.

---

## Features

- 🧠 **Dual Neural Inference**: Combines YOLOv8 (widget detection) and Character Region Awareness for Text Detection CRAFT/EasyOCR (text localization and character recognition).
- 🧩 **Geometry-Based Widget Merger**: Automatically merges OCR text blocks into enclosed widget bounding boxes with ascending left-to-right (horizontal) sorting.
- 📐 **Wayland Scale Factor Auto-detection**: Automatically queries monitor scaling factors from the active Wayland compositor (`hyprctl`, `swaymsg`, or `wlr-randr`) and converts raw pixel coordinates into logical units.
- ⚡ **Heterogeneous Hardware Acceleration**: Auto-detects and binds to optimal compute backends, supporting Nvidia CUDA, Apple Silicon MPS (MacBook), and CPU fallbacks.
- 📦 **Standardized CLI Contract**: Outputs layout grids to `stdout` as GFM JSON and outputs all runtime errors strictly to `stderr` with JSON schema envelopes.

---

## Installation

### 1. Install CLI Tool via PyPI
Install the layout scanner package locally using `uv` (recommended) or `pip`:
```bash
uv pip install waywarp-scanner
# or
pip install waywarp-scanner
```

### 2. Install AI Agent Skill (Gemini / Claude Code / Hermes CLI)
To equip your developer agent CLI with this skill, install it instantly using the `skills` utility:
```bash
npx skills add https://github.com/Xuepoo/waywarp-skill
```

---

## Quick Start

### 1. Download Visual Models
Download CRAFT and YOLOv8 weights to your XDG-compliant cache folder:
```bash
waywarp-scanner download-models
```
*Weights are cached at `$XDG_DATA_HOME/waywarp/models/` (falls back to `~/.local/share/waywarp/models/`).*

### 2. Scan Screen Layout
Run layout scanner to analyze the active screen:
```bash
waywarp-scanner scan
```

### Output JSON Format Example:
```json
{
  "screen_width": 1920,
  "screen_height": 1080,
  "elements": [
    {
      "id": 0,
      "type": "button",
      "text": "Login",
      "center": [100.0, 50.0],
      "bbox": [80.0, 40.0, 40.0, 20.0],
      "monitor_index": 0,
      "confidence": 0.95
    }
  ]
}
```

Use these center coordinates to execute pointer warps directly via `waywarp`:
```bash
waywarp --move-to 100.0 50.0 --click left
```

---

## Development & Test

We use `uv` for local virtual environment and package management:

```bash
# Run test suite
uv run pytest

# Check code formatting & styles
uv run ruff check
uv run ruff format --check

# Check type safety
uv run mypy .
```

---

## License

MIT License.
