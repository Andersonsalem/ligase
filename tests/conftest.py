from __future__ import annotations

import hashlib

import numpy as np
import pytest


class MockSource:
    """Deterministic, dependency free EmbeddingSource for tests"""

    def __init__(self, dim: int = 8):
        self._dim = dim
        self.calls = 0

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def cache_id(self) -> str:
        return f"mock|d{self._dim}|v1"

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]:
        self.calls += 1
        out: dict[str, np.ndarray] = {}
        for seq in seqs:
            rows = []
            for i in range(len(seq)):
                seed = int.from_bytes(
                    hashlib.sha256(f"{self.cache_id}|{seq}|{i}".encode()).digest()[:8], "little"
                )
                rows.append(np.random.default_rng(seed).standard_normal(self._dim))
            out[seq] = (
                np.stack(rows).astype(np.float16) if rows else np.zeros((0, self._dim), np.float16)
            )
        return out


@pytest.fixture
def mock_source() -> MockSource:
    return MockSource()


@pytest.fixture
def cached_mock(mock_source: MockSource, tmp_path):  # noqa: ANN001
    from ligase.utils.caching import cached

    return cached(mock_source, tmp_path / "cache")


@pytest.fixture
def seqs_edge_cases() -> list[str]:
    return [
        "ACDEFGHIKLMNPQRSTVWY",
        "AXBUZ",
        "A",
        "acde",
        "MKTAYIAKQRQISFVKSHFSRQ",
    ]
