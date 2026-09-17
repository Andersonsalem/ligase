from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import scatter


@runtime_checkable
class GraphEncoder(Protocol):
    def forward(self, data: Data) -> Tensor: ...

    def pool(self, out: Tensor, batch: Data) -> Tensor: ...


def mean_pool(out: Tensor, batch: Data) -> Tensor:
    index = getattr(batch, "batch", None)
    if index is None:
        return out.mean(dim=0, keepdim=True)
    return scatter(out, index, dim=0, reduce="mean")
