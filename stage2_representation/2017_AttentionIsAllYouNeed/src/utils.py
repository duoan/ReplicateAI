import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Set all relevant random seeds to make experiments fully reproducible.

    This covers:
      - Python built-in `random`
      - NumPy RNG
      - PyTorch CPU & GPU RNGs
      - cuDNN deterministic mode
      - Python hash seed

    Args:
        seed (int): the base random seed to use (default: 42)

    Notes:
        * Setting cudnn.deterministic=True and cudnn.benchmark=False
          can slightly reduce performance but ensures determinism.
        * For multi-GPU or distributed training, each worker should
          call this function (optionally with a worker-specific offset).
    """
    # --- Standard library ---
    random.seed(seed)  # Python RNG
    os.environ["PYTHONHASHSEED"] = str(seed)  # Hash-based ops (dict/set order)

    # --- NumPy ---
    np.random.seed(seed)

    # --- PyTorch (CPU + all GPUs) ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- cuDNN backend settings ---
    torch.backends.cudnn.deterministic = True  # Force deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable autotuner (non-deterministic)

    print(f"✅ Random seeds set to {seed}")
