from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from ligase.graph.io import PDB_ID

_AF_PREFIX = "AF_"
_CHAIN_OK = re.compile(r"^[A-Za-z0-9_-]+$")


class ParsedToken(NamedTuple):
    structure_id: str
    chain: str | None


def parse_token(token: str) -> ParsedToken:
    t = token.strip()
    if not t or any(c.isspace() for c in t):
        raise ValueError(f"Invalid token: {token!r}")
    if t.upper().startswith(_AF_PREFIX):
        if "." in t:
            raise ValueError(f"AlphaFold models are whole-chain --- no chain: {token!r}")
        return ParsedToken(t, None)
    if "." in t:
        sid, chain = t.split(".", 1)
        if not PDB_ID.match(sid):
            raise ValueError(f"{token!r}: chain tokens must be <PDB-ID>.<chain>, e.g. 1A2B.A")
        if not chain or not _CHAIN_OK.match(chain):
            raise ValueError(f"{token!r}: invalid chain part")
        return ParsedToken(sid, chain)
    return ParsedToken(t, None)


def resolve_token(structure_id: str) -> ParsedToken:
    if Path(structure_id).exists() or "/" in structure_id:
        return ParsedToken(structure_id, None)
    return parse_token(structure_id)
