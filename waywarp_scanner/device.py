"""Device helper to determine the optimal PyTorch hardware accelerator.

Torch is lazily imported on first call to avoid top-level import overhead (#39).
"""

import os


def get_optimal_device() -> str:
    """Determine the optimal PyTorch hardware accelerator available on the current system.

    Checks for CUDA and MPS (Apple Silicon), falling back to CPU.
    When MPS is selected, PYTORCH_ENABLE_MPS_FALLBACK is set to '1'.

    Returns:
        str: "cuda", "mps", or "cpu".
    """
    import torch  # Lazy import (#39)

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    return "cpu"
