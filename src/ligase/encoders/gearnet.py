from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from ligase.encoders.gvp import GVPFeedForward
from ligase.encoders.protocol import mean_pool
from ligase.graph.build import AA

SEQ_EDGE, SPATIAL_EDGE = 1, 0


class GearNetConv(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w_msg = nn.Linear(2 * dim, dim)
        self.type_emb = nn.Embedding(2, dim)
        self.w_edge = nn.Linear(2 * dim, dim)

    def forward(
        self, h: Tensor, e: Tensor, edge_index: Tensor, edge_type: Tensor
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        m = self.w_msg(torch.cat([h[src], e], dim=-1)) + self.type_emb(edge_type)
        e = e + self.w_edge(torch.cat([m, e], dim=-1))
        agg = scatter(m, dst, dim=0, dim_size=h.size(0), reduce="sum")
        return h + agg, e


class GearNetEncoder(nn.Module):
    """GearNet-lite over the pinned graph schema.

    - coordinate invariance: outputs bitwise unchanged under rotation or
      translation of ``pos`` (which is never read);
    - structure sensitivity: outputs change when ``edge_index``,
      ``edge_attr``, or the edge-type derivation changes.

    Parameters
    ----------
    x_dim : int
        Width of ``batch.x``. Wiring-resolved, never hardcoded: 1 for
        ``features=struct``; embedding dim + 1 for ``features=both``.
    node_dim : int
        Internal width everywhere (nodes AND edge states).
    edge_in_dim : int
        Width of ``batch.edge_attr``; must match ``GraphParams.rbf_count + 1``
        (default 21). Used once, in the raw→state projection.
    depth : int
        Number of conv + feed-forward rounds, >= 1.
    ffn_mult : int
        Hidden-width multiple of the shared gated feed-forward.
    dropout : float
        0.0 default: train == eval bitwise, as the honesty tests require.
    out_dim : int | None
        Per-node output width; defaults to ``node_dim``. The task owns any
        final head; the policy budget counts encoder + head together.

    Notes
    -----
    Parameter budget at defaults, ``x_dim=1``: ≈0.52M. At the graft width
    ``x_dim=1281`` (ESM-2 650M + pLDDT): ≈0.68M. Pinned by test.
    """

    def __init__(
        self,
        x_dim: int,
        node_dim: int = 128,
        edge_in_dim: int = 21,
        depth: int = 3,
        ffn_mult: int = 2,
        dropout: float = 0.0,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        if x_dim < 1 or node_dim < 1 or edge_in_dim < 1:
            raise ValueError("x_dim, node_dim and edge_in_dim must be >= 1")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        out_dim = node_dim if out_dim is None else out_dim

        self.type_emb = nn.Embedding(len(AA), node_dim)
        self.x_proj = nn.Linear(x_dim, node_dim)
        self.input_norm = nn.LayerNorm(node_dim)

        self.edge_proj = nn.Linear(edge_in_dim, node_dim)

        self.convs = nn.ModuleList(GearNetConv(node_dim) for _ in range(depth))
        self.ffns = nn.ModuleList(GVPFeedForward(node_dim, ffn_mult, dropout) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(node_dim, out_dim)

    def forward(self, batch: Data) -> Tensor:
        """Per-node features: (num_nodes, out_dim)"""
        x = batch.x
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"expected fp32 or fp64 node features, got {x.dtype} — the "
                "pipeline upcasts fp16 cache hits before grafting; nothing "
                "should reach an encoder as fp16"
            )

        edge_type = (batch.edge_attr[:, -1].abs() == 1.0).long()
        h = self.input_norm(self.x_proj(x) + self.type_emb(batch.residue_type))
        e = self.edge_proj(batch.edge_attr)  # raw → state, once; states persist
        for conv, ffn in zip(self.convs, self.ffns, strict=True):
            h, e = conv(h, e, batch.edge_index, edge_type)
            h = self.dropout(h)
            h = ffn(h)
        return self.readout(h)

    def pool(self, out: Tensor, batch: Data) -> Tensor:
        """Mean-pool per-node features: (num_graphs, out_dim)"""
        return mean_pool(out, batch)
