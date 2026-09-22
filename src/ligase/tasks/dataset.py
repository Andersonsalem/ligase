from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch_geometric.data import Data

from ligase.embed import EmbeddingSource
from ligase.graph.build import DEFAULT_PARAMS, GraphParams
from ligase.graph.cache import get_or_build_graph
from ligase.graph.io import load_structure
from ligase.graph.tokens import resolve_token
from ligase.tasks.secondary_structure import secondary_structure_labels
from ligase.utils.caching import cached

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """One structure's aligned triple. ``embeddings`` is None until attached."""

    structure_id: str
    graph: Data
    labels: str
    embeddings: np.ndarray | None = None


def build_examples(
    ids: list[str],
    params: GraphParams = DEFAULT_PARAMS,
    cache_dir: Path | str = "data/cache",
) -> tuple[list[Example], list[tuple[str, str]]]:
    """Build (graph, labels) per structure; skip and log failures.

    ``ids`` are manifest tokens: local paths, bare ids, or ``PDBID.CHAIN``
    chain tokens. Chain tokens produce single-chain graphs, and labels share
    the same chain selection — the alignment triple holds per token. The
    token is parsed here for labels and again inside
    ``get_or_build_graph``; ``parse_token`` is pure, so the duplication is
    free and keeps the cache choke point self-contained.

    Returns ``(examples, skipped)`` where ``skipped`` is
    ``[(structure_id, reason), ...]``.
    """
    cache_dir = Path(cache_dir)
    examples: list[Example] = []
    skipped: list[tuple[str, str]] = []
    for structure_id in ids:
        try:
            token = resolve_token(structure_id)
            structure = load_structure(token.structure_id, cache_dir / "structures")
            graph = get_or_build_graph(structure_id, params, cache_dir)[0]
            labels = secondary_structure_labels(structure, chain=token.chain)
        except Exception as exc:
            logger.warning("skipped %s: %s: %s", structure_id, type(exc).__name__, exc)
            skipped.append((structure_id, f"{type(exc).__name__}: {exc}"))
            continue
        if len(labels) != graph.num_nodes:
            skipped.append(
                (structure_id, f"alignment: {len(labels)} labels vs {graph.num_nodes} nodes")
            )
            continue
        examples.append(Example(structure_id=structure_id, graph=graph, labels=labels))
    if skipped:
        logger.info("%d/%d structures skipped", len(skipped), len(skipped) + len(examples))
    return examples, skipped


def attach_embeddings(
    examples: list[Example],
    source: EmbeddingSource,
    cache_dir: Path | str = "data/cache",
) -> None:
    """Attach per-residue embeddings, node-aligned, in place.

    The embedding cache's fp16 is upcast to fp32.
    Raises on L != num_nodes:
    misalignment is the disease this library exists to cure.
    """
    wrapped = cached(source, Path(cache_dir) / "embeddings")
    out = wrapped.embed([ex.graph.seq for ex in examples])  # duplicates dedupe in cache
    for ex in examples:
        emb = out[ex.graph.seq]
        if emb.shape[0] != ex.graph.num_nodes:
            raise ValueError(
                f"{ex.structure_id}: embedding L={emb.shape[0]} != {ex.graph.num_nodes} "
                "nodes --- alignment impossible; refusing to attach"
            )
        ex.embeddings = emb.astype(np.float32)
