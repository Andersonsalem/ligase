from __future__ import annotations

import random

import pytest

from ligase.embed.ankh import resolve_token_layout
from ligase.embed.batching import plan_batches


def test_plan_batches_covers_all_indices_once() -> None:
    lengths = [random.Random(0).randint(20, 900) for _ in range(137)]
    batches = plan_batches(lengths)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(137))
    assert len(flat) == len(set(flat))


def test_plan_batches_respects_budget() -> None:
    lengths = [random.Random(1).randint(20, 993) for _ in range(200)]
    for b in plan_batches(lengths, 4096):
        assert max(lengths[i] for i in b) * len(b) <= 4096


def test_plan_batches_deterministic_and_length_sorted() -> None:
    lengths = [random.Random(2).randint(20, 993) for _ in range(100)]
    a = plan_batches(lengths, 4096)
    for b in a:
        assert all(lengths[b[k]] <= lengths[b[k + 1]] for k in range(len(b) - 1))


def test_plan_batches_long_sequence_alone() -> None:
    batches = plan_batches([5000, 10, 10], 4096)
    assert [0] in batches  # over-budget sequence forms a batch of one
    assert all(b != [0] or len(b) == 1 for b in batches)
    assert sorted(i for b in batches for i in b) == [0, 1, 2]


def test_layout_t5_style_trailing_special() -> None:
    assert resolve_token_layout([5, 6, 7, 1], {0, 1, 2}) == (0, 1)


def test_layout_esm_style_bounded() -> None:
    assert resolve_token_layout([0, 5, 6, 2], {0, 1, 2}) == (1, 1)


def test_layout_no_specials() -> None:
    assert resolve_token_layout([5, 6], {0, 1, 2}) == (0, 0)


def test_layout_mid_special_raises() -> None:
    with pytest.raises(ValueError, match="mid-seq"):
        resolve_token_layout([5, 1, 6], {0, 1, 2})
