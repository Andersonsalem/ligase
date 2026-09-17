from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from ligase.encoders.protocol import mean_pool


class MLPEncoder(nn.Module):
    """
    Per residue MLP over node features; structure blind by construction
    Parameters
    ----------
    in_dim : int
        Node-feature width. Supplied by the training wiring, never
        hardcoded: ``features=seq`` -> embedding dim; ``struct`` -> 1
        (pLDDT/100); ``both`` -> embedding dim + 1. Third-party
        ``model=my_gnn`` yamls follow the same rule.
    hidden_dim : int
        Width of every hidden layer.
    out_dim : int
        Per-node output width. Encoders stop here; the task owns the
        final linear head to class logits (D2: task-agnostic encoders).
    depth : int
        Number of hidden layers, >= 1.
    dropout : float
        Default 0.0. Tiny heads on frozen features rarely need it, and
        0.0 keeps train == eval so the honesty tests are bitwise.

    Notes
    -----
    Param budget: the training policy (README Section 4) caps trained
    stacks at ~1M parameters. Defaults with ``in_dim=1280`` (the shipped
    ESM-2 650M dim) come to ~0.92M; ``tests/test_mlp.py`` pins that
    boundary so a widening default fails loudly instead of silently
    drifting past the policy.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        out_dim: int = 3,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        blocks: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            blocks += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        blocks.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, batch: Data) -> Tensor:
        """
        Per-node features from node features alone.

        ``Batch`` subclasses ``Data``, so batched and singleton graphs
        take the same path. Raises ``TypeError`` on non-fp32 input: the
        embedding cache stores float16, and the wiring upcasts when
        grafting. If you see this error, the graft skipped the upcast.
        """
        x = batch.x
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"Expected fp32 input, got {x.dtype}")
        return self.net(x)

    def pool(self, out: Tensor, batch: Data) -> Tensor:
        """
        Mean-Pool per-node features into graph features.
        """
        return mean_pool(out, batch)
