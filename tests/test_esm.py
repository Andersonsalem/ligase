from __future__ import annotations

import numpy as np
import pytest

from ligase.embed.esm import ESM2Source, SequenceTooLong, sanitize_sequence

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
