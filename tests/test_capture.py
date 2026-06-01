"""Unit tests for Wayland screen capture and monitor scaling capabilities."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from waywarp_scanner.capture import (
    capture_screen,
    check_prerequisites,
    get_monitor_scales,
    to_logical_coords,
)


def test_check_prerequisites_success() -> None:
    """Test that check_prerequisites succeeds if grim is in PATH."""
    with patch("shutil.which", return_value="/usr/bin/grim"):
        # Should not raise any error
        check_prerequisites()


def test_check_prerequisites_failure() -> None:
    """Test that check_prerequisites raises RuntimeError if grim is missing."""
    with patch("shutil.which", return_value=None):
        with pytest.raises(RuntimeError) as exc_info:
            check_prerequisites()
        assert "grim is missing" in str(exc_info.value)


@patch("shutil.which", return_value="/usr/bin/grim")
@patch("subprocess.run")
def test_capture_screen_success_no_monitor(mock_run: MagicMock, mock_which: MagicMock) -> None:
    """Test capture_screen with no monitor specified."""
    mock_run.return_value = MagicMock(returncode=0)

    capture_screen("output.png")

    mock_run.assert_called_once_with(
        ["grim", "output.png"], check=True, capture_output=True, text=True
    )


@patch("shutil.which", return_value="/usr/bin/grim")
@patch("subprocess.run")
def test_capture_screen_success_with_monitor(mock_run: MagicMock, mock_which: MagicMock) -> None:
    """Test capture_screen with a specific monitor name."""
    mock_run.return_value = MagicMock(returncode=0)

    capture_screen("output.png", "HDMI-A-1")

    mock_run.assert_called_once_with(
        ["grim", "-o", "HDMI-A-1", "output.png"], check=True, capture_output=True, text=True
    )


@patch("shutil.which", return_value="/usr/bin/grim")
@patch("subprocess.run")
def test_capture_screen_failure(mock_run: MagicMock, mock_which: MagicMock) -> None:
    """Test capture_screen handling subprocess CalledProcessError cleanly."""
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=["grim"], stderr="error during capture"
    )

    with pytest.raises(RuntimeError) as exc_info:
        capture_screen("output.png")

    assert "grim failed with exit code 1" in str(exc_info.value)
    assert "error during capture" in str(exc_info.value)


@patch("shutil.which", return_value="/usr/bin/grim")
@patch("subprocess.run")
def test_capture_screen_file_not_found(mock_run: MagicMock, mock_which: MagicMock) -> None:
    """Test capture_screen handling FileNotFoundError cleanly."""
    mock_run.side_effect = FileNotFoundError("grim not found")

    with pytest.raises(RuntimeError) as exc_info:
        capture_screen("output.png")

    assert "grim executable not found during capture" in str(exc_info.value)


def test_get_monitor_scales_hyprland() -> None:
    """Test get_monitor_scales dynamically queries and parses Hyprland monitors."""
    mock_res = MagicMock()
    mock_res.stdout = '[{"name": "eDP-1", "scale": 1.25}, {"name": "HDMI-A-1", "scale": 1.0}]'

    with patch("subprocess.run", return_value=mock_res) as mock_run:
        scales = get_monitor_scales()
        assert scales == {"eDP-1": 1.25, "HDMI-A-1": 1.0}
        mock_run.assert_called_once_with(
            ["hyprctl", "monitors", "-j"], capture_output=True, text=True, check=True
        )


def test_get_monitor_scales_sway() -> None:
    """Test get_monitor_scales falls back to swaymsg when hyprctl fails."""
    mock_sway_res = MagicMock()
    mock_sway_res.stdout = '[{"name": "eDP-1", "scale": 2.0}, {"name": "DP-2", "scale": 1.5}]'

    def side_effect(cmd, *args, **kwargs):  # type: ignore
        if cmd[0] == "hyprctl":
            raise FileNotFoundError("hyprctl not found")
        return mock_sway_res

    with patch("subprocess.run", side_effect=side_effect) as mock_run:
        scales = get_monitor_scales()
        assert scales == {"eDP-1": 2.0, "DP-2": 1.5}
        assert mock_run.call_count == 2


def test_get_monitor_scales_wlr_randr() -> None:
    """Test get_monitor_scales falls back to wlr-randr when hyprctl and swaymsg fail."""
    mock_wlr_res = MagicMock()
    mock_wlr_res.stdout = """eDP-1 "Unknown product eDP-1"
  Physical size: 300x190 mm
  Enabled: yes
  Modes:
    2880x1800 px, 60.000000 Hz
  Position: 0,0
  Transform: normal
  Scale: 1.500000
DP-2 "Unknown product DP-2"
  Physical size: 600x340 mm
  Enabled: yes
  Modes:
    3840x2160 px, 60.000000 Hz
  Position: 2880,0
  Transform: normal
  Scale: 2.0"""

    def side_effect(cmd, *args, **kwargs):  # type: ignore
        if cmd[0] in ("hyprctl", "swaymsg"):
            raise FileNotFoundError(f"{cmd[0]} not found")
        return mock_wlr_res

    with patch("subprocess.run", side_effect=side_effect) as mock_run:
        scales = get_monitor_scales()
        assert scales == {"eDP-1": 1.5, "DP-2": 2.0}
        assert mock_run.call_count == 3


def test_get_monitor_scales_all_fail() -> None:
    """Test get_monitor_scales returns empty dict if all commands fail."""
    with patch("subprocess.run", side_effect=FileNotFoundError("command not found")):
        scales = get_monitor_scales()
        assert scales == {}


def test_to_logical_coords_success() -> None:
    """Test physical pixel coordinates to Wayland logical units division."""
    logical_x, logical_y = to_logical_coords(300.0, 450.0, 1.5)
    assert logical_x == 200.0
    assert logical_y == 300.0


def test_to_logical_coords_invalid_scale() -> None:
    """Test to_logical_coords raises ValueError when scale <= 0."""
    with pytest.raises(ValueError) as exc_info:
        to_logical_coords(100.0, 100.0, 0.0)
    assert "Scale factor must be greater than zero." in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        to_logical_coords(100.0, 100.0, -1.0)
    assert "Scale factor must be greater than zero." in str(exc_info.value)
