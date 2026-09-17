from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from ligase.graph.build import build_graph, is_protein_ca, select_chain
from ligase.tasks import secondary_structure as ss


def _dssp_line(num: int, chain: str, aa: str, ss_char: str) -> str:
    """One DSSP body line, built from the column contract (0-based).

    resnum [5:10] right-justified · icode [10] · chain [11] · one-letter
    AA [13] · SS [16] --- the classic layout Bio.PDB's parser also reads.
    """
    cols = [" "] * 80
    cols[5:10] = list(f"{num:>5d}")
    cols[11] = chain
    cols[13] = aa
    cols[16] = ss_char
    return "".join(cols).rstrip()


def test_map_q8_to_q3_full_alphabet() -> None:
    # Convention: H,G,I -> H (helix) · E,B -> E (strand) · T,S,- -> C (coil)
    assert ss.Q8_TO_Q3 == {
        "H": "H",
        "G": "H",
        "I": "H",
        "E": "E",
        "B": "E",
        "T": "C",
        "S": "C",
        "-": "C",
    }
    q8 = "HBEGITS-"
    assert len(q8) == len(ss.Q8_ALPHABET) == 8
    assert ss.map_q8_to_q3(q8) == "HEEHHCCC"
    assert ss.map_q8_to_q3(list(q8)) == "HEEHHCCC"  # list input works too


def test_map_q8_to_q3_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown Q8 label"):
        ss.map_q8_to_q3("HX")


def test_encode_roundtrip() -> None:
    assert ss.encode_q8("HBEGITS-").tolist() == list(range(8))  # canonical order pinned
    # "HBEGITS-" -> Q3 "HEEHHCCC" -> indices into Q3_ALPHABET="HEC"
    assert ss.encode_q3("HBEGITS-").tolist() == [0, 1, 1, 0, 0, 2, 2, 2]
    assert ss.Q3_ALPHABET == "HEC"  # pinned: index 0/1/2 = H/E/C


def test_classification_metrics_hand_computed() -> None:
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    # Matches: indices 0,2,3,4 -> accuracy 4/5.
    m = ss.classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(0.8)
    # a: TP=1 (i0), FN=1 (i1), FP=0 -> P=1.0 R=0.5 -> F1=2/3
    assert m["f1_a"] == pytest.approx(2 / 3)
    # b: TP=2 (i2,i3), FN=0, FP=1 (i1) -> P=2/3 R=1.0 -> F1=0.8
    assert m["f1_b"] == pytest.approx(0.8)
    # c: TP=1, FP=0, FN=0 -> F1=1.0
    assert m["f1_c"] == pytest.approx(1.0)


def test_classification_metrics_zero_division_is_zero() -> None:
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 0, 0])  # class 1 never predicted -> f1_b must be 0.0, not NaN
    m = ss.classification_metrics(y_true, y_pred, ["a", "b"])
    assert m["f1_b"] == 0.0
    assert np.isfinite(list(m.values())).all()


def test_classification_metrics_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        ss.classification_metrics(np.zeros(3), np.zeros(4), ["a"])


def test_report_metrics_names() -> None:
    m = ss.report_metrics(np.array([0, 1]), np.array([0, 1]), states=3)
    assert m["Q3_accuracy"] == 1.0
    assert set(m) == {"Q3_accuracy", "f1_H", "f1_E", "f1_C"}


def test_parse_dssp_synthetic() -> None:
    lines = [
        "  #  RESIDUE AA STRUCTURE BP1 BP2  ACC",
        _dssp_line(1, "A", "M", "H"),
        _dssp_line(2, "A", "R", "G"),
        _dssp_line(10, "A", "G", "T"),
        _dssp_line(11, "A", "!", "!"),  # chain-break row: '!' in the AA column
        _dssp_line(12, "B", "K", "E"),
    ]
    assert len(lines[1]) == 17, f"helper drifted: {lines[1]!r} ({len(lines[1])} cols)"
    expected = {
        ("A", 1, " "): "H",
        ("A", 2, " "): "G",
        ("A", 10, " "): "T",
        ("B", 12, " "): "E",
    }
    d = ss._parse_dssp("\n".join(lines))
    assert d == expected, f"parsed={d!r}\nexpected={expected!r}\n" + "\n".join(
        f"  {ln!r}" for ln in lines
    )


def test_missing_binary_degrades_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="mkdssp not found"):
        ss.compute_dssp(gemmi.Structure())


def test_dssp_command_by_generation() -> None:
    """Pins the 4.0 CLI split that produced the original 'unknown option' failure."""
    assert ss._dssp_command("mkdssp", (4, 4), Path("i"), Path("o")) == ["mkdssp", "i", "o"]
    assert ss._dssp_command("dssp", (2, 2), Path("i"), Path("o")) == ["dssp", "-i", "i", "-o", "o"]


def test_dssp_version_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class _P:
        stdout, stderr = "mkdssp 4.4.5", ""

    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: _P())
    assert ss.dssp_version("x") == (4, 4)


def test_dssp_version_unparseable_assumes_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    class _P:
        stdout, stderr = "", ""

    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: _P())
    assert ss.dssp_version("x") == (2,)


def test_labels_align_with_graph_nodes(toy_pdb_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """labels[i] is node i's label --- proven by sharing the traversal's own keys."""
    st = gemmi.read_structure(str(toy_pdb_path))
    n = build_graph(st).num_nodes
    chain = select_chain(st, None)
    fake = {ss.residue_key(chain.name, res): "H" for res in chain if is_protein_ca(res)}
    monkeypatch.setattr(ss, "compute_dssp", lambda structure: fake)
    labels = ss.secondary_structure_labels(st)
    assert labels == "H" * n
    assert len(labels) == 8  # toy fixture size, pinned


@pytest.mark.slow
@pytest.mark.skipif(not ss.dssp_available(), reason="mkdssp not installed")
def test_real_ubiquitin_labels(tmp_path) -> None:
    from ligase.graph.io import load_structure

    st = load_structure("1UBQ", tmp_path)
    labels = ss.secondary_structure_labels(st)
    g = build_graph(st)
    assert len(labels) == g.num_nodes
    assert set(labels) <= set(ss.Q8_ALPHABET)
    assert labels.count("H") >= 10  # ubiquitin's α-helix (residues 23–34)
