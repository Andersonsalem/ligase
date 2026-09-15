from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

FASTA_EXTS = {".fasta", ".fa", ".faa"}


def load_sequences(source: str | Path) -> list[str]:
    """
    Load sequences from a file, dispatching on extension.

    - .fasta/.fa/.faa : FASTA; headers parsed but not returned (v1)
    - .txt            : one sequence per line; '#' starts a comment
    - .csv            : needs a 'sequence' column (header, case-insensitive)
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = path.read_text()
    ext = path.suffix.lower()
    if ext in FASTA_EXTS:
        return _from_fasta(text)
    if ext == ".txt":
        return _from_txt(text)
    if ext == ".csv":
        return _from_csv(text)
    raise ValueError(f"Unsupported file extension: {ext} - use FASTA, TXT, or CSV")


def _from_fasta(text: str) -> list[str]:
    seqs: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if buf:
                seqs.append("".join(buf))
                buf = []
        elif line.strip():
            buf.append(line.strip())
    if buf:
        seqs.append("".join(buf))
    if not seqs:
        raise ValueError("No sequences found in FASTA file")
    return seqs


def _from_txt(text: str) -> list[str]:
    seqs = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not seqs:
        raise ValueError("No sequences found in TXT file")
    return seqs


def _from_csv(text: str) -> list[str]:
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError("No rows found in CSV file")
    key = next((k for k in rows[0] if k and k.strip().lower() == "sequence"), None)
    if key is None:
        raise ValueError("CSV must have a sequence column")
    seqs: list[str] = []
    skipped = 0
    for r in rows:
        val = (r.get(key) or "").strip()
        if val:
            seqs.append(val)
        elif any((v or "").strip() for v in r.values()):
            skipped += 1  # row has data but no sequence
    if skipped:
        logger.warning("skipped %d malformed CSV row(s) missing a sequence", skipped)
    if not seqs:
        raise ValueError("No sequences found in CSV file")
    return seqs
