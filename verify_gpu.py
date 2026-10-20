"""Print the active PyTorch/CUDA status before running inference."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Run: python -m pip install -r requirements-gpu.txt")
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA runtime: {torch.version.cuda or 'not available'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("GPU inference is unavailable. Check the NVIDIA driver and CUDA PyTorch install.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
