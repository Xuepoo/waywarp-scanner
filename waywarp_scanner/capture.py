"""Screen capture and monitor scaling capabilities for Wayland."""

import contextlib
import json
import re
import shutil
import subprocess  # nosec B404


def check_prerequisites() -> None:
    """Verify grim is in path via shutil.which.

    Raises:
        RuntimeError: If grim is not found in the system PATH.
    """
    if shutil.which("grim") is None:
        raise RuntimeError("grim is missing. Please install grim via pacman, yay, or apt.")


def capture_screen(output_path: str, monitor_name: str | None = None) -> None:
    """Invoke grim subprocess.

    Run `grim -o {monitor_name} {output_path}` if monitor_name is specified,
    else `grim {output_path}`. Handles subprocess execution failures cleanly.

    Args:
        output_path: Path where the screenshot image should be saved.
        monitor_name: Optional name of the monitor output to capture.

    Raises:
        RuntimeError: If grim execution fails or is not found.
    """
    check_prerequisites()

    cmd = ["grim"]
    if monitor_name is not None:
        cmd.extend(["-o", monitor_name])
    cmd.append(output_path)

    try:
        # grim command is safe to execute, inputs are sanitized
        subprocess.run(  # noqa: S603  # nosec B603
            cmd, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else "Unknown error"
        raise RuntimeError(f"grim failed with exit code {e.returncode}: {error_msg}") from e
    except FileNotFoundError as e:
        raise RuntimeError("grim executable not found during capture.") from e


def get_monitor_scales() -> dict[str, float]:
    """Attempt to query monitor scaling factor dynamically from the active Wayland compositor.

    Tries to query:
    1. hyprctl (Hyprland)
    2. swaymsg (Sway)
    3. wlr-randr (Other wlroots compositors)

    Returns:
        A dictionary mapping monitor names to their float scaling factors.
    """
    # 1. Try running hyprctl monitors -j
    try:
        res = subprocess.run(
            ["hyprctl", "monitors", "-j"],  # noqa: S607  # nosec B603 B607
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        scales = {}
        for mon in data:
            if isinstance(mon, dict) and "name" in mon and "scale" in mon:
                scales[str(mon["name"])] = float(mon["scale"])
        if scales:
            return scales
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ):
        pass

    # 2. Fall back to swaymsg -t get_outputs
    try:
        res = subprocess.run(
            ["swaymsg", "-t", "get_outputs"],  # noqa: S607  # nosec B603 B607
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        scales = {}
        for mon in data:
            if isinstance(mon, dict) and "name" in mon and "scale" in mon:
                scales[str(mon["name"])] = float(mon["scale"])
        if scales:
            return scales
    except (
        subprocess.SubprocessError,
        FileNotFoundError,
        json.JSONDecodeError,
        KeyError,
        ValueError,
    ):
        pass

    # 3. Fall back to wlr-randr
    try:
        res = subprocess.run(
            ["wlr-randr"],  # noqa: S607  # nosec B603 B607
            capture_output=True,
            text=True,
            check=True,
        )
        scales = {}
        current_monitor = None
        for line in res.stdout.splitlines():
            if not line:
                continue
            # Monitor name starts at column 0 (no leading whitespace)
            monitor_match = re.match(r"^(\S+)", line)
            if monitor_match:
                current_monitor = monitor_match.group(1)
            elif current_monitor:
                scale_match = re.search(r"^\s*[Ss]cale:\s*([0-9.]+)", line)
                if scale_match:
                    with contextlib.suppress(ValueError):
                        scales[current_monitor] = float(scale_match.group(1))
        if scales:
            return scales
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return {}


def to_logical_coords(x: float, y: float, scale: float) -> tuple[float, float]:
    """Convert physical pixel coordinates to Wayland logical units by dividing by the scale factor.

    Args:
        x: Physical X coordinate.
        y: Physical Y coordinate.
        scale: Scale factor (must be greater than 0).

    Returns:
        A tuple of (logical_x, logical_y).

    Raises:
        ValueError: If the scale factor is <= 0.
    """
    if scale <= 0:
        raise ValueError("Scale factor must be greater than zero.")
    return x / scale, y / scale
