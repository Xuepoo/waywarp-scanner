"""CLI entry point and command definition for waywarp-scanner."""

import importlib.metadata
import json
import os
import tempfile
import urllib.request
import zipfile

import click

from waywarp_scanner.capture import capture_screen, check_prerequisites, get_monitor_scales
from waywarp_scanner.detect import run_ocr, run_yolo
from waywarp_scanner.merger import merge_elements


def get_model_dir() -> str:
    """Determine XDG compliant base cache path for models."""
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return os.path.join(xdg_data, "waywarp", "models")
    return os.path.expanduser("~/.local/share/waywarp/models")


try:
    __version__ = importlib.metadata.version("waywarp-scanner")
except Exception:
    try:
        __version__ = importlib.metadata.version("waywarp_scanner")
    except Exception:
        __version__ = "0.1.4"


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Waywarp Agent Visual Scanner CLI."""
    pass


@cli.command()
@click.option("--dest", type=click.Path(), help="Custom destination directory for models.")
def download_models(dest: str | None) -> None:
    """Download required visual models (EasyOCR + YOLO) to cache directory."""
    model_dir = dest if dest else get_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    click.echo(f"Downloading models to {model_dir}...")

    # Download EasyOCR Craft Detection model
    craft_zip = os.path.join(model_dir, "craft_mlt_25k.zip")
    craft_pth = os.path.join(model_dir, "craft_mlt_25k.pth")
    if not os.path.exists(craft_pth):
        click.echo("Downloading EasyOCR Craft Detection model...")
        urllib.request.urlretrieve(
            "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
            craft_zip,
        )
        click.echo("Extracting Craft model...")
        with zipfile.ZipFile(craft_zip, "r") as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(craft_zip)

    # Download EasyOCR English Recognition model
    eng_zip = os.path.join(model_dir, "english_g2.zip")
    eng_pth = os.path.join(model_dir, "english_g2.pth")
    if not os.path.exists(eng_pth):
        click.echo("Downloading EasyOCR English Recognition model...")
        urllib.request.urlretrieve(
            "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip",
            eng_zip,
        )
        click.echo("Extracting English model...")
        with zipfile.ZipFile(eng_zip, "r") as zip_ref:
            zip_ref.extractall(model_dir)
        os.remove(eng_zip)

    # Download YOLOv8-Nano
    yolo_pt = os.path.join(model_dir, "yolov8n.pt")
    if not os.path.exists(yolo_pt):
        click.echo("Downloading YOLOv8 widget detection model...")
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
            yolo_pt,
        )

    click.echo("All models successfully cached!")


@cli.command()
@click.option("--monitor", type=str, help="Bind capture to specific monitor name.")
@click.option("--monitor-index", type=int, default=0, help="Monitor index value.")
@click.option("--models-dir", type=click.Path(), help="Custom models directory path.")
def scan(monitor: str | None, monitor_index: int, models_dir: str | None) -> None:
    """Capture Wayland screen, run detection, and output structured layout JSON."""
    try:
        check_prerequisites()
    except RuntimeError as e:
        click.echo(json.dumps({"error": str(e)}), err=True)
        click.get_current_context().exit(1)

    m_dir = models_dir if models_dir else get_model_dir()
    os.makedirs(m_dir, exist_ok=True)

    # Standard temp image path
    image_path = os.path.join(tempfile.gettempdir(), "waywarp_scan.png")

    try:
        capture_screen(image_path, monitor)
    except Exception as e:
        click.echo(json.dumps({"error": f"Failed to capture screen: {e}"}), err=True)
        click.get_current_context().exit(1)

    # Check model files
    yolo_pt = os.path.join(m_dir, "yolov8n.pt")
    if not os.path.exists(yolo_pt):
        err_msg = "YOLOv8 model missing. Please run `waywarp-scanner download-models` first."
        click.echo(json.dumps({"error": err_msg}), err=True)
        click.get_current_context().exit(1)

    try:
        # Run detection
        ocr_res = run_ocr(image_path, m_dir)
        yolo_res = run_yolo(image_path, yolo_pt)

        # Get actual screen physical dimensions (Issue #27)
        phys_width, phys_height = 1920, 1080
        from PIL import Image

        try:
            with Image.open(image_path) as img:
                phys_width, phys_height = img.size
        except (FileNotFoundError, OSError, ValueError):
            phys_width, phys_height = 1920, 1080

        # Resolve scale factor
        scales = get_monitor_scales()
        scale_factor = 1.0
        if monitor is not None:
            scale_factor = scales.get(monitor, 1.0)
        elif scales:
            # Fallback to the scale of the first monitor found
            scale_factor = next(iter(scales.values()), 1.0)

        logical_width = int(phys_width / scale_factor)
        logical_height = int(phys_height / scale_factor)

        # Merge
        merged = merge_elements(
            yolo_res,
            ocr_res,
            scales,
            monitor,
            monitor_index=monitor_index,
            physical_size=(phys_width, phys_height),
        )

        # Clean up screenshot safely
        if os.path.exists(image_path):
            os.remove(image_path)

        # Output JSON
        click.echo(
            json.dumps(
                {
                    "screen_width": logical_width,
                    "screen_height": logical_height,
                    "elements": merged,
                },
                indent=2,
            )
        )
    except Exception as e:
        click.echo(json.dumps({"error": f"Failed during scan: {e}"}), err=True)
        click.get_current_context().exit(1)
