import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """
    Seed everything: python, numpy and (if present) torch

    torch is imported lazily inside the try-block
    Ideally, the package root must be importable without torch
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
