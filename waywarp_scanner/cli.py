"""CLI entry point and command definition for waywarp-scanner."""

import importlib.metadata
import json
import os
import tempfile
import time
import urllib.request
import zipfile
from typing import Any

import click

from waywarp_scanner.capture import capture_screen, check_prerequisites, get_monitor_scales


def get_model_dir() -> str:
    """Determine XDG compliant base cache path for models."""
    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return os.path.join(xdg_data, "waywarp", "models")
    return os.path.expanduser("~/.local/share/waywarp/models")


def _ocr_worker(image_path: str, m_dir: str | None, queue: Any) -> None:
    try:
        from waywarp_scanner.detect import run_ocr

        res = run_ocr(image_path, m_dir)
        queue.put(("ocr", res))
    except Exception as e:
        queue.put(("error", f"OCR failed: {e}"))


def _yolo_worker(image_path: str, yolo_pt: str, queue: Any) -> None:
    try:
        from waywarp_scanner.detect import run_yolo

        res = run_yolo(image_path, yolo_pt)
        queue.put(("yolo", res))
    except Exception as e:
        queue.put(("error", f"YOLO failed: {e}"))


def _ocr_thread(image_path: str, m_dir: str | None, result: list[Any]) -> None:
    """Run OCR in a thread (shares cached Reader with main process)."""
    from waywarp_scanner.detect import run_ocr

    result[0] = run_ocr(image_path, m_dir)


def _yolo_thread(image_path: str, yolo_pt: str, result: list[Any]) -> None:
    """Run YOLO in a thread (shares model cache with main process)."""
    from waywarp_scanner.detect import run_yolo

    result[0] = run_yolo(image_path, yolo_pt)


try:
    __version__ = importlib.metadata.version("waywarp-scanner")
except Exception:
    try:
        __version__ = importlib.metadata.version("waywarp_scanner")
    except Exception:
        __version__ = "0.2.0"


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
        urllib.request.urlretrieve(  # nosec B310
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
        urllib.request.urlretrieve(  # nosec B310
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
        urllib.request.urlretrieve(  # nosec B310
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
            yolo_pt,
        )

    click.echo("All models successfully cached!")


def get_socket_path(custom_path: str | None = None) -> str:
    """Resolve XDG compliant path for the Unix domain socket."""
    if custom_path:
        return custom_path
    xdg_runtime = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime:
        return os.path.join(xdg_runtime, "waywarp-scanner.sock")
    return os.path.join(tempfile.gettempdir(), "waywarp-scanner.sock")


def _send_to_server(sock_path: str, request: dict[str, Any]) -> dict[str, Any] | None:
    """Attempt to send a scan request to a running warm daemon server."""
    import socket

    if not os.path.exists(sock_path):
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(10.0)  # 10s timeout for scan
            client.connect(sock_path)
            client.sendall(json.dumps(request).encode("utf-8"))

            data = []
            while True:
                chunk = client.recv(4096)
                if not chunk:
                    break
                data.append(chunk)
            res: dict[str, Any] = json.loads(b"".join(data).decode("utf-8"))
            return res
    except Exception:
        return None


@cli.command()
@click.option("--socket-path", type=str, help="Custom path for the Unix domain socket.")
@click.option("--models-dir", type=click.Path(), help="Custom models directory path.")
def serve(socket_path: str | None, models_dir: str | None) -> None:
    """Start background daemon server to keep models warm in memory."""
    import socket

    from PIL import Image

    from waywarp_scanner.detect import _get_reader, run_ocr, run_yolo
    from waywarp_scanner.device import get_optimal_device
    from waywarp_scanner.merger import merge_elements

    m_dir = models_dir if models_dir else get_model_dir()
    os.makedirs(m_dir, exist_ok=True)

    sock_path = get_socket_path(socket_path)

    if os.path.exists(sock_path):
        try:
            os.remove(sock_path)
        except OSError as e:
            click.echo(
                json.dumps({"error": f"Failed to remove existing socket {sock_path}: {e}"}),
                err=True,
            )
            click.get_current_context().exit(1)

    click.echo("Pre-loading models...")
    device = get_optimal_device()
    gpu_enabled = device in ("cuda", "mps")

    # Warm up reader cache
    _ = _get_reader(m_dir, gpu_enabled)
    yolo_pt = os.path.join(m_dir, "yolov8n.pt")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(sock_path)
        os.chmod(sock_path, 0o600)
    except Exception as e:
        click.echo(json.dumps({"error": f"Failed to bind socket {sock_path}: {e}"}), err=True)
        click.get_current_context().exit(1)

    server.listen(5)
    click.echo(f"Server is warm and listening on {sock_path}...")

    try:
        while True:
            conn, _ = server.accept()
            try:
                req_data = conn.recv(4096)
                if not req_data:
                    conn.close()
                    continue
                req = json.loads(req_data.decode("utf-8"))

                action = req.get("action")
                if action == "scan":
                    image_path = req.get("image_path")
                    monitor = req.get("monitor")
                    monitor_index = req.get("monitor_index", 0)

                    if not image_path or not os.path.exists(image_path):
                        conn.sendall(
                            json.dumps({"error": f"Image path {image_path} does not exist"}).encode(
                                "utf-8"
                            )
                        )
                        continue

                    import time

                    t0 = time.time()
                    # Run OCR
                    ocr_res = run_ocr(image_path, m_dir)
                    t_ocr = time.time() - t0

                    # Run YOLO
                    t1 = time.time()
                    yolo_res = run_yolo(image_path, yolo_pt)
                    t_yolo = time.time() - t1

                    phys_width, phys_height = 1920, 1080
                    import contextlib

                    with contextlib.suppress(Exception), Image.open(image_path) as img:
                        phys_width, phys_height = img.size

                    scales = get_monitor_scales()
                    scale_factor = 1.0
                    if monitor is not None:
                        scale_factor = scales.get(monitor, 1.0)
                    elif scales:
                        scale_factor = next(iter(scales.values()), 1.0)

                    logical_width = int(phys_width / scale_factor)
                    logical_height = int(phys_height / scale_factor)

                    t2 = time.time()
                    merged = merge_elements(
                        yolo_res,
                        ocr_res,
                        scales,
                        monitor,
                        monitor_index=monitor_index,
                        physical_size=(phys_width, phys_height),
                    )
                    t_merge = time.time() - t2
                    total_t = time.time() - t0

                    click.echo(
                        f"Scan complete: OCR={t_ocr:.3f}s, YOLO={t_yolo:.3f}s, "
                        f"Merge={t_merge:.3f}s, Total={total_t:.3f}s (Elements: {len(merged)})"
                    )

                    response = {
                        "screen_width": logical_width,
                        "screen_height": logical_height,
                        "elements": merged,
                    }
                    conn.sendall(json.dumps(response).encode("utf-8"))
                elif action == "ping":
                    conn.sendall(json.dumps({"status": "ok"}).encode("utf-8"))
                else:
                    conn.sendall(json.dumps({"error": f"Unknown action: {action}"}).encode("utf-8"))
            except Exception as e:
                import contextlib

                with contextlib.suppress(Exception):
                    conn.sendall(
                        json.dumps({"error": f"Internal daemon error: {e}"}).encode("utf-8")
                    )
            finally:
                conn.close()
    except KeyboardInterrupt:
        click.echo("Server shutting down.")
    finally:
        server.close()
        if os.path.exists(sock_path):
            import contextlib

            with contextlib.suppress(OSError):
                os.remove(sock_path)


@cli.command()
@click.option("--monitor", type=str, help="Bind capture to specific monitor name.")
@click.option("--monitor-index", type=int, default=0, help="Monitor index value.")
@click.option("--models-dir", type=click.Path(), help="Custom models directory path.")
@click.option("--socket-path", type=str, help="Custom path for the Unix domain socket.")
@click.option(
    "--no-serve",
    is_flag=True,
    default=False,
    help="Disable auto-start of warm daemon (always cold scan).",
)
@click.option(
    "--timing",
    is_flag=True,
    default=False,
    help="Print timing breakdown to stderr.",
)
def scan(
    monitor: str | None,
    monitor_index: int,
    models_dir: str | None,
    socket_path: str | None,
    no_serve: bool,
    timing: bool,
) -> None:
    """Capture Wayland screen, run detection, and output structured layout JSON."""
    t_total_start = time.monotonic()

    try:
        check_prerequisites()
    except RuntimeError as e:
        click.echo(json.dumps({"error": str(e)}), err=True)
        click.get_current_context().exit(1)

    m_dir = models_dir if models_dir else get_model_dir()
    os.makedirs(m_dir, exist_ok=True)

    # Standard temp image path
    image_path = os.path.join(tempfile.gettempdir(), "waywarp_scan.png")

    t0 = time.monotonic()
    try:
        capture_screen(image_path, monitor)
    except Exception as e:
        click.echo(json.dumps({"error": f"Failed to capture screen: {e}"}), err=True)
        click.get_current_context().exit(1)
    t_capture = time.monotonic() - t0

    sock_path = get_socket_path(socket_path)

    # Resolve request payload
    req = {
        "action": "scan",
        "image_path": image_path,
        "monitor": monitor,
        "monitor_index": monitor_index,
    }

    # Attempt delegation to running daemon server first
    response = _send_to_server(sock_path, req)
    if response is not None:
        if "error" in response:
            click.echo(json.dumps({"error": response["error"]}), err=True)
            click.get_current_context().exit(1)

        # Clean up screenshot safely
        if os.path.exists(image_path):
            import contextlib

            with contextlib.suppress(OSError):
                os.remove(image_path)

        if timing:
            t_total = time.monotonic() - t_total_start
            click.echo(
                f"Timing: capture={t_capture:.3f}s total={t_total:.3f}s (daemon mode)",
                err=True,
            )

        click.echo(json.dumps(response, indent=2))
        return

    # No daemon running — auto-start one in background unless --no-serve
    if not no_serve:
        _auto_start_daemon(m_dir, sock_path, socket_path)
        # Retry connecting to the newly started daemon
        time.sleep(0.5)
        response = _send_to_server(sock_path, req)
        if response is not None:
            if "error" in response:
                click.echo(json.dumps({"error": response["error"]}), err=True)
                click.get_current_context().exit(1)

            if os.path.exists(image_path):
                import contextlib

                with contextlib.suppress(OSError):
                    os.remove(image_path)

            if timing:
                t_total = time.monotonic() - t_total_start
                click.echo(
                    f"Timing: capture={t_capture:.3f}s total={t_total:.3f}s (auto-serve mode)",
                    err=True,
                )

            click.echo(json.dumps(response, indent=2))
            return

    # Fallback to local cold scan execution
    yolo_pt = os.path.join(m_dir, "yolov8n.pt")
    if not os.path.exists(yolo_pt):
        err_msg = "YOLOv8 model missing. Please run `waywarp-scanner download-models` first."
        click.echo(json.dumps({"error": err_msg}), err=True)
        click.get_current_context().exit(1)

    try:
        import threading

        from waywarp_scanner.merger import merge_elements

        # Use threads instead of processes — shares cached Reader/model (Issue #45/#46)
        ocr_result: list[Any] = [[]]
        yolo_result: list[Any] = [[]]

        t1 = time.monotonic()
        t_ocr = threading.Thread(target=_ocr_thread, args=(image_path, m_dir, ocr_result))
        t_yolo = threading.Thread(target=_yolo_thread, args=(image_path, yolo_pt, yolo_result))
        t_ocr.start()
        t_yolo.start()
        t_ocr.join()
        t_yolo.join()
        t_infer = time.monotonic() - t1

        ocr_res = ocr_result[0] if ocr_result[0] else []
        yolo_res = yolo_result[0] if yolo_result[0] else []

        if isinstance(ocr_res, Exception):
            raise ocr_res
        if isinstance(yolo_res, Exception):
            raise yolo_res

        phys_width, phys_height = 1920, 1080
        from PIL import Image

        try:
            with Image.open(image_path) as img:
                phys_width, phys_height = img.size
        except (FileNotFoundError, OSError, ValueError):
            phys_width, phys_height = 1920, 1080

        scales = get_monitor_scales()
        scale_factor = 1.0
        if monitor is not None:
            scale_factor = scales.get(monitor, 1.0)
        elif scales:
            scale_factor = next(iter(scales.values()), 1.0)

        logical_width = int(phys_width / scale_factor)
        logical_height = int(phys_height / scale_factor)

        t2 = time.monotonic()
        merged = merge_elements(
            yolo_res,
            ocr_res,
            scales,
            monitor,
            monitor_index=monitor_index,
            physical_size=(phys_width, phys_height),
        )
        t_merge = time.monotonic() - t2

        if os.path.exists(image_path):
            os.remove(image_path)

        t_total = time.monotonic() - t_total_start

        if timing:
            click.echo(
                f"Timing: capture={t_capture:.3f}s inference={t_infer:.3f}s "
                f"merge={t_merge:.3f}s total={t_total:.3f}s (cold scan)",
                err=True,
            )

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


def _auto_start_daemon(
    m_dir: str, sock_path: str, custom_socket_path: str | None
) -> None:
    """Start the warm daemon in background if not already running."""
    import subprocess  # nosec B404

    if os.path.exists(sock_path):
        # Check if the existing socket is alive
        try:
            import socket as _socket

            with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as test:
                test.settimeout(1.0)
                test.connect(sock_path)
                test.sendall(json.dumps({"action": "ping"}).encode("utf-8"))
                resp = test.recv(256)
                if b'"ok"' in resp:
                    return  # Daemon is already running
        except Exception:
            # Stale socket, remove it
            import contextlib

            with contextlib.suppress(OSError):
                os.remove(sock_path)

    # Start daemon in background
    cmd = ["waywarp-scanner", "serve", "--models-dir", m_dir]
    if custom_socket_path:
        cmd.extend(["--socket-path", custom_socket_path])

    try:
        subprocess.Popen(  # nosec B603
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        click.echo("Auto-started warm daemon in background.", err=True)
    except Exception as e:
        click.echo(f"Failed to auto-start daemon: {e}", err=True)
