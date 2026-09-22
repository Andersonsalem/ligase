from __future__ import annotations

from pathlib import Path

import pytest

from ligase.graph.tokens import parse_token, resolve_token


def test_bare_pdb_id_has_no_chain() -> None:
    assert parse_token("1UBQ") == ("1UBQ", None)


def test_chain_token_splits_on_first_dot() -> None:
    assert parse_token("1UBQ.A") == ("1UBQ", "A")


def test_multichar_chain_allowed() -> None:
    assert parse_token("6Y2G.ZZ") == ("6Y2G", "ZZ")


def test_surrounding_whitespace_stripped() -> None:
    assert parse_token("  1UBQ.A ") == ("1UBQ", "A")


def test_af_token_stays_whole() -> None:
    assert parse_token("AF_P69905F1") == ("AF_P69905F1", None)


def test_uniprot_accession_stays_whole() -> None:
    assert parse_token("P69905") == ("P69905", None)


def test_af_token_rejects_chain_part() -> None:
    with pytest.raises(ValueError, match="whole-chain"):
        parse_token("AF_P69905F1.A")


def test_chain_token_requires_pdb_shape() -> None:
    with pytest.raises(ValueError, match="chain tokens"):
        parse_token("UBIQ.A")


def test_empty_chain_part_raises() -> None:
    with pytest.raises(ValueError, match="invalid chain"):
        parse_token("1UBQ.")


def test_internal_space_raises() -> None:
    with pytest.raises(ValueError, match="token"):
        parse_token("1UBQ A")


def test_existing_path_passes_through(tmp_path: Path) -> None:
    p = tmp_path / "1ubq.pdb"
    p.write_text("x")
    assert resolve_token(str(p)) == (str(p), None)


def test_bare_chain_token_parses_without_existing_file() -> None:
    assert resolve_token("1UBQ.A") == ("1UBQ", "A")


def test_path_shaped_string_passes_whole() -> None:
    bogus = "/no/such/dir/1UBQ.A"
    assert resolve_token(bogus) == (bogus, None)
