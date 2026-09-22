"""
Split by structure only
Users could and should implement other splits
"""

from __future__ import annotations

import json
import random
from collections.abc import Hashable, Sequence
from pathlib import Path


def split_ids(
    ids: list[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    groups: Sequence[Hashable] | None = None,
) -> dict[str, list[str]]:
    if not (0 <= train_frac <= 1 and 0 <= val_frac <= 1) or train_frac + val_frac > 1:
        raise ValueError(f"invalid fractions: train={train_frac}, val={val_frac}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in input; split ids must be unique")
    n = len(ids)
    if n < 3:
        raise ValueError(f"need >= 3 structures for train/val/test, got {n}")
    if groups is None:
        shuffled = sorted(ids)
        random.Random(seed).shuffle(shuffled)
        n_train = max(1, min(int(n * train_frac), n - 2))
        n_val = max(1, min(int(n * val_frac), n - n_train - 1))
        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train : n_train + n_val],
            "test": shuffled[n_train + n_val :],
        }
    if len(groups) != n:
        raise ValueError(f"groups length {len(groups)} != ids length {n}")
    member_groups = _canonical_member_groups(ids, groups)
    if len(member_groups) < 3:
        raise ValueError(
            "group-aware split needs >= 3 groups for a non-empty "
            f"train/val/test, got {len(member_groups)}"
        )
    budgets = (n * train_frac, n * val_frac, n * (1.0 - train_frac - val_frac))
    train, val, test = _allocate_groups(member_groups, budgets, random.Random(seed))
    return {"train": train, "val": val, "test": test}


def _canonical_member_groups(ids: list[str], labels: Sequence[Hashable]) -> list[tuple[str, ...]]:
    buckets: dict[Hashable, list[str]] = {}
    for id_, label in zip(ids, labels, strict=True):
        buckets.setdefault(label, []).append(id_)
    return sorted(
        (tuple(sorted(members)) for members in buckets.values()),
        key=lambda members: members[0],
    )


def _allocate_groups(
    member_groups: list[tuple[str, ...]],
    budgets: tuple[float, float, float],
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    order = list(member_groups)
    rng.shuffle(order)
    remaining: list[float] = list(budgets)
    slots: list[list[tuple[str, ...]]] = [[], [], []]
    for group in order:
        target = max(range(3), key=lambda s: (remaining[s], -s))
        slots[target].append(group)
        remaining[target] -= len(group)
    for empty in (0, 1, 2):
        if slots[empty]:
            continue
        donor = max(range(3), key=lambda s: (len(slots[s]), -s))
        smallest = min(slots[donor], key=lambda g: (len(g), g))
        slots[donor].remove(smallest)
        slots[empty].append(smallest)
    return (
        sorted(m for g in slots[0] for m in g),
        sorted(m for g in slots[1] for m in g),
        sorted(m for g in slots[2] for m in g),
    )


def load_groups(path: str | Path) -> dict[str, str]:
    """Read a groups JSON as written by ``scripts/build_cb513.py`` (stage ``groups``).

    Returns ``{token: group_label}`` where the label is the group's first
    (lexicographically smallest) member; readable in logs and stable for a
    committed file. The split path consumes this via ``split_ids(groups=…)``.

    Raises:
        FileNotFoundError: missing file.
        ValueError: malformed payload; missing top-level ``groups``, a group
            that is not a non-empty list of non-empty strings, or a token
            appearing in two groups.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"groups file not found: {p}")
    try:
        payload = json.loads(p.read_text())
        raw = payload["groups"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"{p}: not a groups JSON (need a top-level 'groups' list)") from exc
    lookup: dict[str, str] = {}
    for i, members in enumerate(raw):
        if (
            not isinstance(members, list)
            or not members
            or not all(isinstance(m, str) and m for m in members)
        ):
            raise ValueError(f"{p}: group {i} must be a non-empty list of non-empty strings")
        for m in members:
            if m in lookup:
                raise ValueError(f"{p}: duplicate token {m!r} across groups")
            lookup[m] = members[0]
    return lookup
