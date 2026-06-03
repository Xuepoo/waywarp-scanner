"""Unit tests for waywarp-scanner CLI commands."""

import json
import os
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from waywarp_scanner.cli import cli, get_model_dir


def test_get_model_dir_default() -> None:
    """Verify standard XDG data home fallback logic."""
    with patch.dict(os.environ, {}, clear=True):
        path = get_model_dir()
        assert path == os.path.expanduser("~/.local/share/waywarp/models")


def test_get_model_dir_custom_xdg() -> None:
    """Verify customized XDG data home setting."""
    with patch.dict(os.environ, {"XDG_DATA_HOME": "/custom/xdg"}, clear=True):
        path = get_model_dir()
        assert path == os.path.join("/custom/xdg", "waywarp", "models")


def test_cli_version_option() -> None:
    """Verify waywarp-scanner --version output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "version" in result.output


@patch("urllib.request.urlretrieve")
@patch("zipfile.ZipFile")
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
def test_download_models_command(
    mock_makedirs: MagicMock,
    mock_exists: MagicMock,
    mock_zipfile: MagicMock,
    mock_urlretrieve: MagicMock,
) -> None:
    """Verify download-models successfully handles mocks and extracts ZIP files."""
    runner = CliRunner()
    with patch("os.remove") as mock_remove:
        result = runner.invoke(cli, ["download-models", "--dest", "/tmp/mock_models"])  # noqa: S108
        assert result.exit_code == 0
        assert "Downloading models to /tmp/mock_models" in result.output
        assert "All models successfully cached!" in result.output
        assert mock_urlretrieve.call_count == 3
        assert mock_zipfile.call_count == 2
        assert mock_remove.call_count == 2


@patch("waywarp_scanner.cli._send_to_server", return_value=None)
@patch("waywarp_scanner.cli.check_prerequisites")
@patch("waywarp_scanner.cli.capture_screen")
@patch("waywarp_scanner.detect.run_ocr")
@patch("waywarp_scanner.detect.run_yolo")
@patch("os.path.exists", return_value=True)
def test_scan_command_success(
    mock_exists: MagicMock,
    mock_yolo: MagicMock,
    mock_ocr: MagicMock,
    mock_capture: MagicMock,
    mock_prereq: MagicMock,
    mock_send: MagicMock,
) -> None:
    """Verify scan command captures screenshot, runs pipelines, and prints JSON output."""
    mock_ocr.return_value = [
        {
            "type": "text",
            "text": "Login",
            "center": [100.0, 50.0],
            "bbox": [80, 40, 40, 20],
            "confidence": 0.99,
        }
    ]
    mock_yolo.return_value = [
        {
            "type": "button",
            "text": "",
            "center": [100.0, 50.0],
            "bbox": [70, 30, 60, 40],
            "confidence": 0.95,
        }
    ]

    runner = CliRunner()
    with (
        patch("waywarp_scanner.cli.get_monitor_scales", return_value={"DP-1": 1.0}),
        patch("os.remove"),
    ):
        result = runner.invoke(
            cli,
            ["scan", "--monitor", "DP-1", "--monitor-index", "1", "--models-dir", "/tmp"],  # noqa: S108
        )
        assert result.exit_code == 0

        # Parse output JSON
        data = json.loads(result.output)
        assert "elements" in data
        assert len(data["elements"]) == 1
        element = data["elements"][0]
        assert element["id"] == 0
        assert element["type"] == "button"
        assert element["text"] == "Login"
        assert element["monitor_index"] == 1


@patch("waywarp_scanner.cli.check_prerequisites", side_effect=RuntimeError("grim missing"))
def test_scan_command_prerequisite_failure(mock_prereq: MagicMock) -> None:
    """Verify scan command gracefully prints errors when prerequisites are missing."""
    runner = CliRunner()
    result = runner.invoke(cli, ["scan"])
    assert result.exit_code != 0
    data = json.loads(result.stderr)
    assert "error" in data
    assert "grim missing" in data["error"]


@patch("waywarp_scanner.device.get_optimal_device", return_value="cpu")
@patch("waywarp_scanner.detect._get_reader")
@patch("waywarp_scanner.detect.run_ocr")
@patch("waywarp_scanner.detect.run_yolo")
@patch("waywarp_scanner.cli.get_monitor_scales", return_value={"DP-1": 1.0})
@patch("os.chmod")
def test_serve_command_success(
    mock_chmod: MagicMock,
    mock_scales: MagicMock,
    mock_yolo: MagicMock,
    mock_ocr: MagicMock,
    mock_reader: MagicMock,
    mock_device: MagicMock,
) -> None:
    """Verify serve command binds to socket, reads request, runs models, and responds."""
    mock_ocr.return_value = [
        {
            "type": "text",
            "text": "Hello",
            "center": [10, 10],
            "bbox": [5, 5, 10, 10],
            "confidence": 0.99,
        }
    ]
    mock_yolo.return_value = []

    mock_conn = MagicMock()
    mock_socket_instance = MagicMock()
    mock_socket_instance.accept.side_effect = [(mock_conn, None), KeyboardInterrupt()]

    mock_conn.recv.return_value = json.dumps(
        {
            "action": "scan",
            "image_path": "/tmp/nonexistent.png",  # noqa: S108
            "monitor": "DP-1",
            "monitor_index": 0,
        }
    ).encode("utf-8")

    sent_data = []

    def save_sent(data: bytes) -> None:
        sent_data.append(data)

    mock_conn.sendall.side_effect = save_sent

    runner = CliRunner()
    with (
        patch("socket.socket", return_value=mock_socket_instance),
        patch("os.path.exists", return_value=True),
        patch("os.remove"),
        patch("PIL.Image.open") as mock_image_open,
    ):
        mock_img = MagicMock()
        mock_img.size = (1920, 1080)
        mock_image_open.return_value.__enter__.return_value = mock_img

        result = runner.invoke(
            cli,
            [
                "serve",
                "--socket-path",
                "/tmp/mock.sock",  # noqa: S108
                "--models-dir",
                "/tmp",  # noqa: S108
            ],
        )
        assert "Server is warm and listening" in result.output
        assert len(sent_data) == 1
        res_json = json.loads(sent_data[0].decode("utf-8"))
        assert "elements" in res_json
        assert len(res_json["elements"]) == 1
        assert res_json["elements"][0]["text"] == "Hello"
