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


def _helix_coords(n: int = 8) -> np.ndarray:
    """Ideal Cα helix: 2.3 Å radius, 100°/residue, 1.5 Å rise. Consecutive
    Cα–Cα ≈ 3.8 Å — physically honest test geometry."""
    t = np.arange(n) * np.deg2rad(100.0)
    return np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(n)], axis=1)


AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "GLY",
}


def _helix_pdb(coords: np.ndarray, codes: str, b_factors: np.ndarray) -> str:
    lines = [
        f"ATOM  {i + 1:5d}  CA  {AA3[c]} A{i + 1:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C"
        for i, ((x, y, z), c, b) in enumerate(zip(coords, codes, b_factors, strict=True))
    ]
    return "\n".join(lines) + "\nEND\n"


@pytest.fixture
def toy_pdb_path(tmp_path, toy_coords, toy_codes, toy_plddt):
    p = tmp_path / "toy.pdb"
    p.write_text(_helix_pdb(toy_coords, toy_codes, toy_plddt))
    return p


@pytest.fixture
def toy_coords() -> np.ndarray:
    return _helix_coords()


@pytest.fixture
def toy_codes() -> str:
    return "ACDEFGHI"


@pytest.fixture
def toy_plddt() -> np.ndarray:
    return np.full(8, 88.0)
