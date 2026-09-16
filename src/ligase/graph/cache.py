from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import torch
from torch_geometric.data import Data

from ligase.graph.build import GraphParams, build_graph
from ligase.graph.io import load_structure, safe_name


def cache_key(structure_id: str, params: GraphParams) -> str:
    """
    Deterministic digest binding a cached graph to its exact inputs.

    The cache filename must change whenever anything that defines the graph
    changes (structure identity or GraphParams) or a stale graph is served
    silently.

    Known limitation: a local structure file edited in place keeps its key
    (identity is the string passed, not the file bytes). Delete the cached
    artifact or use a distinct id.

    Parameters
    ----------
    structure_id : str
        Identifier as passed by the caller: PDB ID, AFDB accession, or path.
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

    Parameters
    ----------
    structure_id : str
        PDB ID, AFDB accession, or path to a local structure file.
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

    structure = load_structure(structure_id, Path(cache_dir) / "structures")
    data = build_graph(structure, params)
    tmp = path.with_name(path.name + ".part")
    torch.save({"data": data, "params": params, "structure_id": structure_id}, tmp)
    tmp.replace(path)
    return data, False
