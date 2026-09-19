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
        raise ValueError(f"invalid fractions: train={train_frac}, val={val_frac}")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    if n < 3:
        raise ValueError(f"need >= 3 structures for train/val/test, got {n}")
    n_train = max(1, min(int(n * train_frac), n - 2))
    n_val = max(1, min(int(n * val_frac), n - n_train - 1))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
