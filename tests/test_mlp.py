from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data

from ligase.encoders.mlp import MLPEncoder

N_EDGES = 16


def _toy_graph(n: int = 8, in_dim: int = 4, seed: int = 0) -> Data:
    """Deterministic graph with every field the schema pins. MLP may read x only."""
    g = torch.Generator().manual_seed(seed)
    return Data(
        x=torch.randn(n, in_dim, generator=g),
        edge_index=torch.randint(0, n, (2, N_EDGES), generator=g),
        edge_attr=torch.randn(N_EDGES, 21, generator=g),  # rbf_count(20) + offset
        pos=torch.randn(n, 3, generator=g),
    )


def test_forward_and_pool_shapes() -> None:
    m = MLPEncoder(in_dim=4, out_dim=3)
    g = _toy_graph(n=8)
    assert m(g).shape == (8, 3)
    batch = Batch.from_data_list([g, _toy_graph(n=5, seed=1)])
    assert m.pool(m(batch), batch).shape == (2, 3)


def test_structure_blindness() -> None:
    """Edges, edge features and coordinates change; outputs must not. Bitwise."""
    m = MLPEncoder(in_dim=4)
    m.eval()
    g1 = _toy_graph()
    g2 = g1.clone()
    gen = torch.Generator().manual_seed(99)
    g2.edge_index = torch.randint(0, 8, (2, N_EDGES), generator=gen)
    g2.edge_attr = torch.randn(N_EDGES, 21, generator=gen)
    g2.pos = torch.randn(8, 3, generator=gen) * 10.0
    assert torch.equal(m(g1), m(g2))


def test_permutation_equivariance() -> None:
    m = MLPEncoder(in_dim=4)
    m.eval()
    g = _toy_graph()
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(7))
    g_perm = g.clone()
    g_perm.x = g.x[perm]
    with torch.no_grad():
        assert torch.allclose(m(g_perm), m(g)[perm], atol=1e-6)


def test_batching_equivalence() -> None:
    m = MLPEncoder(in_dim=4)
    m.eval()
    g1, g2 = _toy_graph(n=8, seed=0), _toy_graph(n=5, seed=1)
    batched = Batch.from_data_list([g1, g2])
    with torch.no_grad():
        out_b = m(batched)
        assert torch.allclose(out_b[:8], m(g1), atol=1e-6)
        assert torch.allclose(out_b[8:], m(g2), atol=1e-6)
        pooled = m.pool(out_b, batched)
        solo = m.pool(m(g1), Batch.from_data_list([g1]))
        assert torch.allclose(pooled[0], solo, atol=1e-6)


def test_parameter_budget() -> None:
    m = MLPEncoder(in_dim=1280)  # esm2_t33_650M, out_dim=3, hidden 512, depth 2
    total = sum(p.numel() for p in m.parameters())
    assert 100_000 < total <= 1_000_000


def test_rejects_bad_construction() -> None:
    with pytest.raises(ValueError, match="depth"):
        MLPEncoder(in_dim=4, depth=0)
    with pytest.raises(ValueError, match="in_dim"):
        MLPEncoder(in_dim=0)


def test_rejects_fp16_input() -> None:
    m = MLPEncoder(in_dim=4)
    g = _toy_graph()
    g.x = g.x.half()
    with pytest.raises(TypeError, match="Expected"):
        m(g)
