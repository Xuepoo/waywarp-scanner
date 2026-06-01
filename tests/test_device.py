"""Unit tests for multi-platform hardware and device auto-detection."""

import os
from unittest.mock import MagicMock, patch

from waywarp_scanner.device import get_optimal_device


def test_get_optimal_device_returns_valid_device() -> None:
    """Ensure get_optimal_device returns one of the expected devices on any system."""
    device = get_optimal_device()
    assert device in ["cuda", "mps", "cpu"]


@patch("torch.cuda.is_available")
def test_get_optimal_device_cuda(mock_cuda_available: MagicMock) -> None:
    """Test that CUDA is preferred when available."""
    mock_cuda_available.return_value = True

    device = get_optimal_device()
    assert device == "cuda"


def test_get_optimal_device_mps_available() -> None:
    """Test that MPS is selected when CUDA is not available but MPS is."""
    # Ensure PYTORCH_ENABLE_MPS_FALLBACK is not set initially
    os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

    mock_mps = MagicMock()
    mock_mps.is_available.return_value = True

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps", mock_mps, create=True),
        patch("waywarp_scanner.device.hasattr", return_value=True),
    ):
        device = get_optimal_device()
        assert device == "mps"
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_get_optimal_device_mps_unavailable() -> None:
    """Test that CPU is selected when both CUDA and MPS are not available."""
    mock_mps = MagicMock()
    mock_mps.is_available.return_value = False

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps", mock_mps, create=True),
        patch("waywarp_scanner.device.hasattr", return_value=True),
    ):
        device = get_optimal_device()
        assert device == "cpu"


def test_get_optimal_device_no_mps_backend() -> None:
    """Test fallback to CPU when torch.backends has no mps attribute at all."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("waywarp_scanner.device.hasattr", return_value=False),
    ):
        device = get_optimal_device()
        assert device == "cpu"
