"""
Deterministic experiment reproducibility utilities for Nebium.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """
    Seeds all random number generators across Python, NumPy, PyTorch CPU, and CUDA.

    Enforces cuDNN deterministic mode to guarantee reproducible experimental runs.

    Args:
        seed: Target integer seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
