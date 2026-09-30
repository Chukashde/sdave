import warnings
from typing import Optional

import torch


def resolve_device(requested: Optional[str] = None) -> str:
    """Resolve the compute device for training.

    The DGL graph itself stays on CPU (DGL has no MPS backend), but the models
    (PIGCN / PIGATv2) gather edges from it and run all tensor math on the
    model's device, so 'mps' is allowed when available.
    """
    if requested is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    elif requested == 'mps':
        if torch.backends.mps.is_available():
            device = 'mps'
        else:
            warnings.warn("MPS is not available; falling back to CPU.", stacklevel=2)
            device = 'cpu'
    else:
        device = requested

    if device == 'cuda':
        torch.cuda.init()
        torch.cuda.set_device(0)

    return device


__all__ = ["resolve_device"]
