"""
Any embedding source must pass this test
Third party sources must run this file against their own implementation
"""

from __future__ import annotations

import numpy as np
import pytest

from ligase.utils.caching import cached

SOURCES = ["mock_source", "cached_mock"]  # raw and cache-wrapped


@pytest.mark.parametrize("fixture_name", SOURCES)
def test_alignment(fixture_name, request, seqs_edge_cases):
    source = request.getfixturevalue(fixture_name)
    out = source.embed(seqs_edge_cases)
    for seq in seqs_edge_cases:
        emb = out[seq]
        assert emb.shape == (len(seq), source.dim), seq  # L == len(seq)
        assert emb.dtype == np.float16, seq


@pytest.mark.parametrize("fixture_name", SOURCES)
def test_determinism(fixture_name, request, seqs_edge_cases):
    source = request.getfixturevalue(fixture_name)
    out1 = source.embed(seqs_edge_cases)
    out2 = source.embed(seqs_edge_cases)
    for seq in seqs_edge_cases:
        assert np.array_equal(out1[seq], out2[seq]), seq  # bitwise


def test_cache_roundtrip_zero_recompute(mock_source, seqs_edge_cases, tmp_path):
    first = cached(mock_source, tmp_path / "cache")
    r1 = first.embed(seqs_edge_cases)
    calls_after_first = mock_source.calls
    assert calls_after_first >= 1

    # same wrapper on repeated sequences
    r2 = first.embed(seqs_edge_cases)
    assert mock_source.calls == calls_after_first

    # new wrapper over same dir must hit disk and not model
    second = cached(mock_source, tmp_path / "cache")
    r3 = second.embed(seqs_edge_cases)
    assert mock_source.calls == calls_after_first

    for seq in seqs_edge_cases:
        assert np.array_equal(r1[seq], r2[seq])
        assert np.array_equal(r1[seq], r3[seq])


def test_cache_preserves_protocol(mock_source, cached_mock):
    assert cached_mock.dim == mock_source.dim
    assert cached_mock.cache_id == mock_source.cache_id


def test_cache_distinguishes_sources(mock_source, tmp_path):
    from ligase.utils.caching import _file_name

    other = type(mock_source)(dim=4)
    assert mock_source.cache_id != other.cache_id
    assert _file_name(mock_source.cache_id) != _file_name(other.cache_id)
