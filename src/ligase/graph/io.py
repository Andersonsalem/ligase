from __future__ import annotations

import logging
import re
from pathlib import Path

import gemmi

logger = logging.getLogger(__name__)

PDB_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
UNIPROT_ACC = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$"
)
AFDB_MODEL = re.compile(r"^AF_([A-Z0-9]+)F\d+$", re.IGNORECASE)


PDB_URL = "https://files.rcsb.org/download/{id}.cif"
AFDB_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"

PLDDT_SOURCE_KEY = "_ligase_plddt_source"


def safe_name(identifier: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", identifier)


def _af_accession(identifier: str) -> str | None:
    m = AFDB_MODEL.match(identifier)
    if m:
        return m.group(1).upper()
    if UNIPROT_ACC.match(identifier):
        return identifier.upper()
    return None


def resolve_url(identifier: str) -> str:
    acc = _af_accession(identifier)
    if acc:
        import json
        import urllib.request

        req = urllib.request.Request(
            AFDB_API_URL.format(acc=acc),
            headers={"User-Agent": "ligase/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not data:
                    raise ValueError(f"No AlphaFold prediction returned for {acc}")
                return data[0]["cifUrl"]
        except Exception as e:
            raise RuntimeError(f"Failed to fetch AlphaFold URL for {acc}: {e}") from e

    if PDB_ID.match(identifier):
        return PDB_URL.format(id=identifier.upper())
    raise ValueError(
        f"'{identifier}' is not an existing path, a 4-char PDB ID, "
        f"an AF_...F1 model, or a UniProt accession"
    )


def load_structure(
    path_or_id: str, cache_dir: Path = Path("data/cache/structures")
) -> gemmi.Structure:
    """
    Local path, 4-char PDB ID, AF_{UNIPROT}F1, or bare UniProt accession.

    Downloads land in cache_dir as CIF; a cached file short-circuits the
    network. st.info is tagged so build_graph can interpret the B-factor
    column (pLDDT for AlphaFold models, ignored for experimental).
    """
    path = Path(path_or_id)
    if path.exists():
        st = gemmi.read_structure(str(path))
        st.setup_entities()
        st.info[PLDDT_SOURCE_KEY] = (
            "alphafold" if path.stem.upper().startswith("AF_") else "experimental"
        )
        return st

    identifier = path_or_id.strip()
    is_af = _af_accession(identifier) is not None
    local = cache_dir / f"{safe_name(identifier)}.cif"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        _download(resolve_url(identifier), local)

    st = gemmi.read_structure(str(local))
    st.setup_entities()
    st.info[PLDDT_SOURCE_KEY] = "alphafold" if is_af else "experimental"
    return st


def _download(url: str, dest: Path) -> None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "ligase/1.0"})
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load_structure_ids(path: str | Path) -> list[str]:
    """
    Read a manifest of structure identifiers.

    ```.txt```: one identifier per line, # comments are ignored
    ```.csv```: must have a "structure" column (case-insensitive)
    Duplicates are dropped and order is preserved.
    """
    path = Path(path)
    if not path.exists():
        raise FileExistsError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = (line.strip() for line in path.read_text().splitlines())
        ids = [ln for ln in lines if ln and not ln.startswith("#")]
    elif suffix == ".csv":
        import csv

        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"{path}: empty CSV - expected a 'structure' column")
            header = {name.strip().lower(): name for name in reader.fieldnames}
            if "structure" not in header:
                raise ValueError(f"{path}: CSV must have a 'structure' column")
            ids = [(row[header["structure"]] or "").strip() for row in reader]
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return list(dict.fromkeys(i for i in ids if i))
