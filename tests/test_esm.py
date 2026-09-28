from __future__ import annotations

import random

import numpy as np
import pytest

from ligase.embed.esm import ESM2Source, SequenceTooLong, _plan_batches, sanitize_sequence

pytestmark = pytest.mark.slow

TINY = "facebook/esm2_t6_8M_UR50D"


@pytest.fixture(scope="module")
def tiny():
    return ESM2Source(TINY)


def test_sanitize_maps_unknowns_to_x():
    assert sanitize_sequence("acde1-") == "ACDEXX"
    assert sanitize_sequence("MKT-X") == "MKTXX"


def test_real_alignment_and_dim(tiny):
    src = tiny
    assert src.dim == 320
    seqs = ["MKTAYIAK", "acDEX"]
    out = src.embed(seqs)
    assert set(out) == set(seqs)
    assert out["MKTAYIAK"].shape == (8, 320)
    assert out["acDEX"].shape == (5, 320)
    assert out["acDEX"].dtype == np.float16


def test_real_determinism(tiny):
    src = tiny
    a = src.embed(["MKTAYIAK"])["MKTAYIAK"]
    b = src.embed(["MKTAYIAK"])["MKTAYIAK"]
    assert np.array_equal(a, b)


def test_overlength_raises(tiny):
    src = tiny
    with pytest.raises(SequenceTooLong):
        src.embed(["A" * 2000])


def test_plan_batches_covers_all_indices_once() -> None:
    lengths = [random.Random(0).randint(20, 900) for _ in range(137)]
    batches = _plan_batches(lengths, 4096)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(137))
    assert len(flat) == len(set(flat))


def test_plan_batches_respects_budget() -> None:
    lengths = [random.Random(1).randint(20, 993) for _ in range(200)]
    budget = 4096
    for b in _plan_batches(lengths, budget):
        assert max(lengths[i] for i in b) * len(b) <= budget


def test_plan_batches_deterministic_and_length_sorted() -> None:
    lengths = [random.Random(2).randint(20, 993) for _ in range(100)]
    a = _plan_batches(lengths, 4096)
    assert a == _plan_batches(lengths, 4096)
    for b in a:
        assert [lengths[i] for i in b] == sorted((lengths[i] for i in b), reverse=False) or True
        assert all(
            lengths[b[k]] <= lengths[b[k + 1]] for k in range(len(b) - 1)
        )  # within-batch ascending length (stable-sort property)


def test_plan_batches_long_sequence_alone() -> None:
    batches = _plan_batches([5000, 10, 10], 4096)
    assert batches[0] == [0]  # over-budget sequence still forms a batch of one
