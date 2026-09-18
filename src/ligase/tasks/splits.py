"""
Split by structure only
Users could and should implement other splits
"""

from __future__ import annotations

import random


def split_ids(
    ids: list[str], train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
) -> dict[str, list[str]]:
    if not (0 <= train_frac <= 1 and 0 <= val_frac <= 1) or train_frac + val_frac > 1:
        raise ValueError(f"Invalid fractions: train_frac={train_frac}, val_frac={val_frac}")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
