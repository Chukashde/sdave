import warnings
from typing import Optional

import torch


def resolve_device(requested: Optional[str] = None) -> str:
    """Resolve the compute device for training.

    DGL has no Metal/MPS backend, so a requested 'mps' is mapped to 'cpu' (the
    graph and its message-passing kernels must live on CPU or CUDA). Deeper GPU
    determinism is left to the caller's environment.

    Args:
        requested: 'cpu', 'cuda', 'mps', or None to auto-detect.

    Returns:
        'cuda' or 'cpu'.
    """
    if requested is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif requested == 'mps':
        warnings.warn(
            "DGL has no MPS backend; falling back to CPU for graph ops.",
            stacklevel=2,
        )
        device = 'cpu'
    else:
        device = requested

    if device == 'cuda':
        torch.cuda.init()
        torch.cuda.set_device(0)

    return device


__all__ = ["resolve_device"]
