"""Global seeding for reproducible experiments.

Judges score reproducibility explicitly, so every run must be pinned to a seed.
Call ``seed_everything(seed)`` once at the start of a run and log the seed.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) RNGs.

    Args:
        seed: The seed value to apply across all RNGs.
        deterministic: If True, force deterministic cuDNN/algorithm selection.
            Slightly slower but required for bit-reproducible runs.

    Returns:
        The seed that was applied (echoed for convenient logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
