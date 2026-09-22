from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import torch
from torch_geometric.data import Data

from ligase.graph.build import GraphParams, build_graph
from ligase.graph.io import load_structure, safe_name
from ligase.graph.tokens import resolve_token


def cache_key(structure_id: str, params: GraphParams) -> str:
    """
    Deterministic digest binding a cached graph to its exact inputs.

    The cache filename must change whenever anything that defines the graph
    changes (structure identity or GraphParams) or a stale graph is served
    silently. ``structure_id`` is the FULL caller-facing identity — bare id,
    chain token (``1UBQ.A``), AFDB accession, or path — so a chain graph and
    its whole-entry sibling never collide.

    Known limitation: a local structure file edited in place keeps its key
    (identity is the string passed, not the file bytes). Delete the cached
    artifact or use a distinct id.

    Parameters
    ----------
    structure_id : str
        Identifier as passed by the caller.
    params : GraphParams
        Frozen graph-construction parameters.

    Returns
    -------
    str
        First 12 hex characters of the SHA-256 digest.
    """
    fields = ",".join(f"{f.name}={getattr(params, f.name)!r}" for f in dataclasses.fields(params))
    payload = f"graph|{structure_id}|{fields}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def get_or_build_graph(
    structure_id: str,
    params: GraphParams,
    cache_dir: Path,
) -> tuple[Data, bool]:
    """
    Return the graph for ``(structure_id, params)``, building on miss.

    Chain tokens (``1UBQ.A``) are first-class: the FULL token is the cache
    identity; on miss, ``resolve_token`` splits it and only the bare entry id
    is downloaded/parsed, with ``chain=`` threaded into ``build_graph``. Local
    paths pass through whole. Because the token is parsed here (not by
    callers), the CLI ``build`` path is chain-aware for free.

    Parameters
    ----------
    structure_id : str
        Bare id, chain token, AFDB accession, or path to a local file.
    params : GraphParams
        Frozen graph-construction parameters.
    cache_dir : Path
        Cache root; graphs land in ``cache_dir/graphs``, structure downloads
        in ``cache_dir/structures``.

    Returns
    -------
    tuple[Data, bool]
        ``(data, cache_hit)``: cache_hit is True when served from disk.

    Notes
    -----
    Writes are atomic (``.part`` -> ``replace``): a killed run can never leave
    a half-written graph that a later run reads as a cache hit.
    """
    out_dir = Path(cache_dir) / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_name(structure_id)}-{cache_key(structure_id, params)}.pt"

    if path.exists():
        saved = torch.load(path, weights_only=False)
        return saved["data"], True

    token = resolve_token(structure_id)
    structure = load_structure(token.structure_id, Path(cache_dir) / "structures")
    data = build_graph(structure, params, chain=token.chain)
    tmp = path.with_name(path.name + ".part")
    torch.save({"data": data, "params": params, "structure_id": structure_id}, tmp)
    tmp.replace(path)
    return data, False
