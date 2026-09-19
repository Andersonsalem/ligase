from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from ligase.encoders.protocol import mean_pool
from ligase.graph.build import AA


class GVPBlock(nn.Module):
    """
    The scalar half of a geometric vector perceptron.

        s' = LayerNorm(W s) + sigmoid(W_gate s) ⊙ (W_value s)

    The gate modulates how much of the value branch passes per channel;
    LayerNorm on the residual branch keeps activation scale stable across
    proteins of wildly different sizes. The full GVP also updates a vector
    stream jointly which is deliberately absent here (module docstring).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w = nn.Linear(dim, dim)
        self.w_gate = nn.Linear(dim, dim)
        self.w_value = nn.Linear(dim, dim)

    def forward(self, s: Tensor) -> Tensor:
        return self.norm(self.w(s)) + torch.sigmoid(self.w_gate(s)) * self.w_value(s)


class GVPFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.w_in = nn.Linear(dim, mult * dim)
        self.w_gate = nn.Linear(dim, mult * dim)
        self.w_out = nn.Linear(mult * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, s: Tensor) -> Tensor:
        return s + self.drop(self.w_out(torch.sigmoid(self.w_gate(s)) * self.w_in(s)))


class GVPConv(MessagePassing):
    """
    One invariant message-passing round.

        m_ij = W_m [h_i || h_j || e_ij]
        h_i' = GVPBlock(h_i + Σ_j m_ij)

    e_ij is the pinned edge feature row: 20 RBF columns of the Cα distance
    plus the raw sequence offset clipped to ±8. The columns live on very
    different scales; the first message Linear is expected to calibrate
    them the schema stays unnormalized by design. Sum aggregation is
    edge-order-agnostic (the equivariance test relies on this); isolated
    nodes receive a zero message and pass through the block.
    """

    def __init__(self, dim: int, edge_dim: int) -> None:
        super().__init__()
        self.message_proj = nn.Linear(2 * dim + edge_dim, dim)
        self.block = GVPBlock(dim)

    def forward(self, h: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.block(h + self.propagate(edge_index, h=h, edge_attr=edge_attr))

    def message(self, h_i: Tensor, h_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.message_proj(torch.cat([h_i, h_j, edge_attr], dim=-1))


class GVPEncoder(nn.Module):
    """
    GVP-GNN in invariant mode over the pinned graph schema.

    Honesty contract, both sides tested:

    - coordinate invariance: outputs bitwise unchanged under rotation or
        translation of ``pos`` (which is never read);
    - structure sensitivity: outputs change when ``edge_index`` or
        ``edge_attr`` change (structure enters only through edges).

    Parameters
    ----------
    x_dim : int
        Width of ``batch.x``. Resolved by the wiring, never hardcoded:
        1 for ``features=struct`` (pLDDT/100); embedding dim + 1 for
        ``features=both`` (graft; the fp16→fp32 upcast happens upstream).
    node_dim : int
        Internal width everywhere.
    edge_dim : int
        Width of ``batch.edge_attr``; must match ``GraphParams.rbf_count + 1``
        (default 21).
    depth : int
        Number of GVPConv + feed-forward rounds, >= 1.
    ffn_mult : int
        Hidden-width multiple of the gated feed-forward.
    dropout : float
        0.0 default: train == eval bitwise, which the honesty tests require.
    out_dim : int | None
        Per-node output width; defaults to ``node_dim``. Set it to the task's
        class count and the task head becomes a bare readout, the training
        policy budget counts encoder + head together.

    Notes
    -----
    Parameter budget at defaults, ``x_dim=1``: ≈0.57M. At the graft width
    ``x_dim=1281`` (ESM-2 650M + pLDDT): ≈0.74M. Pinned by test.
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

        self.convs = nn.ModuleList(GVPConv(node_dim, edge_in_dim) for _ in range(depth))
        self.ffns = nn.ModuleList(GVPFeedForward(node_dim, ffn_mult, dropout) for _ in range(depth))
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(node_dim, out_dim)

    def forward(self, batch: Data) -> Tensor:
        """Per-node features (num_nodes, out_dim)"""
        x = batch.x
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"expected fp32 node features, got {x.dtype}: the pipeline "
                "upcasts fp16 cache hits before grafting; nothing should "
                "reach an encoder as fp16"
            )
        h = self.input_norm(self.x_proj(x) + self.type_emb(batch.residue_type))
        for conv, ffn in zip(self.convs, self.ffns, strict=True):
            h = self.dropout(conv(h, batch.edge_index, batch.edge_attr))
            h = ffn(h)
        return self.readout(h)

    def pool(self, out: Tensor, batch: Data) -> Tensor:
        """Mean-pooling over node features (num_graphs, out_dim)"""
        return mean_pool(out, batch)
