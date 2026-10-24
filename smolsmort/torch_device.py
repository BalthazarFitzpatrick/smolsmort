"""the training and inference device: cuda, then apple's mps, then cpu"""

from __future__ import annotations


def default_device(torch) -> str:
    """the fastest device this machine has for torch"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
