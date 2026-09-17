from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import gemmi
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ligase.graph.build import is_protein_ca, select_chain

logger = logging.getLogger(__name__)

Q8_ALPHABET = "HBEGITS-"  # canonical order = class indices
Q3_ALPHABET = "HEC"
Q8_TO_Q3 = {"H": "H", "G": "H", "I": "H", "E": "E", "B": "E", "T": "C", "S": "C", "-": "C"}

INSTALL_HINT = (
    "mkdssp not found on PATH --- install DSSP (conda: `conda install -c conda-forge dssp`, "
    "debian: `apt install dssp`). Only label computation needs it; embedding and "
    "graph building never do."
)


def residue_key(chain_name: str, res: gemmi.Residue) -> tuple[str, int, str]:
    """DSSP lookup key: (chain, author residue number, insertion code).

    Absent insertion codes normalize to ``" "`` --- the character DSSP prints
    in the icode column when there is none.
    """
    icode = res.seqid.icode
    if icode in ("\x00", ""):
        icode = " "
    return (chain_name, res.seqid.num, icode)


def map_q8_to_q3(labels: str | list[str]) -> str:
    """Map Q8 DSSP labels to Q3 (H/G/I->H, E/B->E, T/S/-->C). Total over the alphabet."""
    out = []
    for lab in labels:
        try:
            out.append(Q8_TO_Q3[lab])
        except KeyError:
            raise ValueError(
                f"unknown Q8 label {lab!r} --- expected one of {Q8_ALPHABET!r}"
            ) from None
    return "".join(out)


def encode_q8(labels: str | list[str]) -> np.ndarray:
    """Q8 labels -> int64 class indices into ``Q8_ALPHABET``."""
    return np.array([Q8_ALPHABET.index(lab) for lab in labels], dtype=np.int64)


def encode_q3(labels: str | list[str]) -> np.ndarray:
    """Q8 labels -> int64 class indices into ``Q3_ALPHABET`` (via the named mapping)."""
    return np.array([Q3_ALPHABET.index(c) for c in map_q8_to_q3(labels)], dtype=np.int64)


def dssp_available() -> bool:
    """True when a DSSP binary (mkdssp 4.x or legacy dssp) is on PATH."""
    return shutil.which("mkdssp") is not None or shutil.which("dssp") is not None


def dssp_version(binary: str) -> tuple[int, ...]:
    try:
        proc = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
        match = re.search(r"(\d+)\.(\d+)", proc.stdout + proc.stderr)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    except (OSError, subprocess.SubprocessError):
        pass
    return (2,)


def _dssp_command(binary: str, version: tuple[int, ...], inp: Path, outp: Path) -> list[str]:
    if version >= (4,):
        return [binary, str(inp), str(outp)]
    return [binary, "-i", str(inp), "-o", str(outp)]


def compute_dssp(structure: gemmi.Structure) -> dict[tuple[str, int, str], str]:
    binary = shutil.which("mkdssp") or shutil.which("dssp")
    if binary is None:
        raise RuntimeError(INSTALL_HINT)
    version = dssp_version(binary)
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.pdb"
        outp = Path(tmp) / "out.dssp"
        inp.write_text(structure.make_pdb_string())
        proc = subprocess.run(
            _dssp_command(binary, version, inp, outp),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"DSSP failed (exit {proc.returncode}, probed version {version}): "
                f"{proc.stderr[-400:]}"
            )
        text = outp.read_text()
    return _parse_dssp(text)


def _parse_dssp(text: str) -> dict[tuple[str, int, str], str]:
    """Parse classic fixed-column DSSP: resseq 5:10, icode 10, chain 11, SS 16.

    Skips headers, blank/short lines, and chain-break markers ('!' rows).
    A space in the SS column is the loop state, normalized to '-'.
    """
    labels: dict[tuple[str, int, str], str] = {}
    in_body = False
    for line in text.splitlines():
        if line.startswith("  #  RESIDUE"):
            in_body = True
            continue
        if not in_body or len(line) < 17:
            continue
        num_field = line[5:10].strip()
        aa = line[13]
        if not num_field.isdigit() or not aa.isalpha():
            continue
        ss_char = line[16]
        if ss_char == " ":
            ss_char = "-"
        if ss_char not in Q8_ALPHABET:
            continue
        icode = line[10] if line[10].isalpha() else " "
        labels[(line[11], int(num_field), icode)] = ss_char
    return labels


def secondary_structure_labels(structure: gemmi.Structure, chain: str | None = None) -> str:
    """Q8 labels for the residues ``build_graph`` admits.

    Parameters
    ----------
    structure : gemmi.Structure
        Loaded structure (e.g. from ``graph.io.load_structure``).
    chain : str, optional
        Chain id; default follows ``select_chain`` (first non-empty + log).

    Returns
    -------
    str
        One Q8 character per admitted residue, in graph-node order.
    """
    chain_obj = select_chain(structure, chain)
    dssp = compute_dssp(structure)

    labels: list[str] = []
    missing = 0
    for res in chain_obj:
        if not is_protein_ca(res):
            continue
        lab = dssp.get(residue_key(chain_obj.name, res))
        if lab is None:
            lab = "-"  # unassigned is considered a coil
            missing += 1
        labels.append(lab)
    if not labels:
        raise ValueError("no labelable residues --- structure admits no protein CAs")
    if missing:
        logger.info("%d/%d residues without a DSSP assignment -> '-'", missing, len(labels))
    return "".join(labels)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    """Accuracy + per-class F1 (zero_division=0: unpredicted classes score 0.0)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    per_class = f1_score(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        **{f"f1_{name}": float(v) for name, v in zip(class_names, per_class, strict=True)},
    }


def report_metrics(y_true: np.ndarray, y_pred: np.ndarray, states: int) -> dict[str, float]:
    """Q3- or Q8-shaped convenience wrapper: ``Q3_accuracy``, ``f1_H``, ... ."""
    if states not in (3, 8):
        raise ValueError(f"states must be 3 or 8, got {states}")
    names = list(Q3_ALPHABET if states == 3 else Q8_ALPHABET)
    metrics = classification_metrics(y_true, y_pred, names)
    accuracy = metrics.pop("accuracy")
    return {f"Q{states}_accuracy": accuracy, **metrics}
