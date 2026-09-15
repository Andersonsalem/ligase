from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, fields

import gemmi
import numpy as np
import torch
from torch_geometric.data import Data

from ligase.graph.io import PLDDT_SOURCE_KEY

logger = logging.getLogger(__name__)

AA = "ACDEFGHIKLMNPQRSTVWYX"  # 20 + ambiguity code
OFFSET_CLAMP = 8  # beyond 8 is considered far in sequence


@dataclass(frozen=True)
class GraphParams:
    """Frozen to be hashable"""

    edge_knn: int = 10
    edge_radius: float = 10.0
    rbf_min: float = 0.0
    rbf_max: float = 20.0
    rbf_count: int = 20
    rbf_sigma: float = 2.5


DEFAULT_PARAMS = GraphParams()


def build_graph(
    structure: gemmi.Structure,
    params: GraphParams = DEFAULT_PARAMS,
    chain: str | None = None,
) -> Data:
    coords, codes, plddt, is_af = _extract(structure, chain)
    if is_af:
        logger.info("pLDDT taken from the B-factor column (AlphaFold model)")
    else:
        logger.info("experimental structure: we assume no pLDDT is available --- using 1.0")
        plddt = np.full_like(plddt, 100.0)
    return build_graph_from_coords(coords, codes, plddt, params)


def build_graph_from_coords(
    coords: np.ndarray,
    codes: np.ndarray,
    plddt: np.ndarray,
    params: GraphParams = DEFAULT_PARAMS,
) -> Data:
    """
    Every feature is a function of pairwise distances, graph structure and sequence identity.
    """
    coords = np.asarray(coords, dtype=np.float64)
    plddt = np.asarray(plddt, dtype=np.float64)
    n = len(codes)
    if len(coords) != n or len(plddt) != n:
        raise ValueError("coords, codes and plddt must agree in length")

    pairs = _edge_pairs(coords, params.edge_knn, params.edge_radius)  # (E, 2) sorted
    if len(pairs):
        d = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
        rbf = _rbf(d, params)
        offset = np.clip(pairs[:, 1] - pairs[:, 0], -OFFSET_CLAMP, OFFSET_CLAMP)
        edge_attr = np.concatenate([rbf, offset[:, None].astype(np.float64)], axis=1)
    else:
        edge_attr = np.zeros((0, params.rbf_count + 1))

    data = Data(
        x=torch.tensor(plddt / 100.0, dtype=torch.float32).unsqueeze(1),
        residue_type=torch.tensor([AA.index(c) for c in codes], dtype=torch.long),
        edge_index=torch.tensor(pairs.T, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        pos=torch.tensor(coords, dtype=torch.float32),
        residue_index=torch.arange(n, dtype=torch.long),
        num_nodes=n,
    )
    data.seq = codes
    return data


def _edge_pairs(coords: np.ndarray, knn: int, radius: float) -> np.ndarray:
    """
    Directed both-ways edges: sequence adjacency ∪ kNN ∪ radius ball.
    Sorted pairs give a canonical order; the invariance tests rely on it.
    """
    n = len(coords)
    pairs: set[tuple[int, int]] = set()
    for i in range(n - 1):
        pairs.add((i, i + 1))
        pairs.add((i + 1, i))
    if n > 1:
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        k = min(knn, n - 1)
        if k > 0:
            for i, js in enumerate(d.argsort(axis=1)[:, :k]):
                pairs.update((i, int(j)) for j in js)
        ii, jj = np.nonzero(np.triu(d <= radius, k=1))
        pairs.update((int(i), int(j)) for i, j in zip(ii, jj, strict=True))
        pairs.update((int(j), int(i)) for i, j in zip(ii, jj, strict=True))
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def _rbf(d: np.ndarray, params: GraphParams) -> np.ndarray:
    centers = np.linspace(params.rbf_min, params.rbf_max, params.rbf_count)
    return np.exp(-((d[:, None] - centers) ** 2) / (2 * params.rbf_sigma**2))


def _is_alphafold(structure: gemmi.Structure) -> bool:
    try:
        return structure.info[PLDDT_SOURCE_KEY] == "alphafold"
    except (KeyError, IndexError):
        return False


def _extract(
    structure: gemmi.Structure, chain_id: str | None = None
) -> tuple[np.ndarray, str, np.ndarray, bool]:
    model = structure[0]
    if chain_id is not None:
        chains = [model[chain_id]]
    else:
        chains = [ch for ch in model if len(ch)]
        if not chains:
            raise ValueError("Structure has no residues")
        if len(chains) > 1:
            logger.info(
                "%d chains found — using '%s' (pass chain= to override)",
                len(chains),
                chains[0].name,
            )

    coords: list[tuple[float, float, float]] = []
    codes: list[str] = []
    plddts: list[float] = []
    for ch in chains:
        for res in ch:
            info = gemmi.find_tabulated_residue(res.name)
            if info is None or not info.is_amino_acid():
                continue
            ca = res.find_atom("CA", "*")
            if ca is None:
                logger.debug("no CA in %s%s — skipped", ch.name, res.seqid.num)
                continue
            code = info.one_letter_code.upper()
            coords.append((ca.pos.x, ca.pos.y, ca.pos.z))
            codes.append(code if code in AA else "X")
            plddts.append(ca.b_iso)
    if not codes:
        raise ValueError(
            "no residues with a C-alpha found --- make sure this is a protein structure"
        )

    is_af = _is_alphafold(structure)
    return np.array(coords), "".join(codes), np.array(plddts), is_af


def cache_key(structure_id: str, params: GraphParams = DEFAULT_PARAMS) -> str:
    field = ",".join(f"{f.name}={getattr(params, f.name)!r}" for f in fields(params))
    payload = f"graph|{structure_id}|{field}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
