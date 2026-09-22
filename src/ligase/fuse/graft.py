from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch


def concat(node_feats: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
    return torch.cat([embeddings, node_feats], dim=-1)


class ProjectionGraft(nn.Module):
    """Trainable linear-first graft: project the embedding block, keep struct.

    ``forward`` on ``x`` of width ``emb_dim + n_struct``: the first
    ``emb_dim`` columns (the embedding block, per the pinned order) run
    through one ``Linear(emb_dim, proj_width)``; the remaining columns
    (pLDDT) pass through bitwise. Output width ``proj_width + n_struct``.
    A third-party EmbeddingSource may return any D (Section 5.1) — this
    module absorbs it into a fixed width so ablation rows stay
    capacity-comparable.
    """

    def __init__(self, emb_dim: int, proj_width: int) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.proj = nn.Linear(emb_dim, proj_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb, struct = x[:, : self.emb_dim], x[:, self.emb_dim :]
        return torch.cat([self.proj(emb), struct], dim=-1)


class GraftedEncoder(nn.Module):
    """
    graft(batch.x), then the inner GraphEncoder.

    The graft is upstream of the encoder and trains jointly with it;
    ``build_model`` returns this wrapper when a project graft is
    configured, so state_dict keys (``graft.*``, ``encoder.*``) are
    deterministic between training and eval reconstruction. The collated
    Batch is fresh per iteration, so the in-place ``batch.x`` assignment is
    safe and avoids a per-batch clone.
    """

    def __init__(self, graft: nn.Module, encoder: nn.Module) -> None:
        super().__init__()
        self.graft = graft
        self.encoder = encoder

    def forward(self, batch: Batch) -> torch.Tensor:
        batch.x = self.graft(batch.x)
        return self.encoder(batch)
