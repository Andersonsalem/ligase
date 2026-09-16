from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from ligase.cli import _resolve_sequences, _resolve_structure_ids


def _cfg(**kw: object) -> object:
    base: dict[str, object] = dict(
        sequences=[],
        sequences_file=None,
        structure=None,
        structures=[],
        structures_file=None,
    )
    base.update(kw)
    return OmegaConf.create(base)


def test_structure_ids_single() -> None:
    assert _resolve_structure_ids(_cfg(structure="1UBQ")) == ["1UBQ"]  # type: ignore[arg-type]


def test_structure_ids_inline() -> None:
    assert _resolve_structure_ids(_cfg(structures=["1UBQ", "4HHB"])) == [  # type: ignore[arg-type]
        "1UBQ",
        "4HHB",
    ]


def test_structure_ids_from_file(tmp_path: Path) -> None:
    f = tmp_path / "ids.txt"
    f.write_text("1UBQ\n4HHB\n")
    ids = _resolve_structure_ids(_cfg(structures_file=str(f)))  # type: ignore[arg-type]
    assert ids == ["1UBQ", "4HHB"]


def test_structure_ids_none_is_error() -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        _resolve_structure_ids(_cfg())  # type: ignore[arg-type]


def test_structure_ids_ambiguous_is_error() -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        _resolve_structure_ids(_cfg(structure="1UBQ", structures=["4HHB"]))  # type: ignore[arg-type]


def test_sequences_ambiguous_is_error() -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        _resolve_sequences(_cfg(sequences=["AAA"], sequences_file="x.fa"))  # type: ignore[arg-type]
