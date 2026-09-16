from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from ligase.encoders.gvp import GVPEncoder
from ligase.graph.build import build_graph_from_coords

CODES = "ACDEFGHIKLMNPQRSTVWYX"  # build.AA order; helpers use prefixes


def _coords(n: int = 8) -> np.ndarray:
    """Same ideal helix as conftest._helix_coords (local copy: tests stay standalone)."""
    t = np.arange(n) * np.deg2rad(100.0)
    return np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(n)], axis=1)


def _graph(n: int = 8, plddt: float = 88.0) -> Data:
    """Real pinned-schema graph from the real builder — n <= 21."""
    return build_graph_from_coords(_coords(n), CODES[:n], np.full(n, plddt))


def _f64(g: Data) -> Data:
    h = g.clone()
    for field in ("x", "edge_attr", "pos"):
        h[field] = h[field].double()
    return h


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_forward_and_pool_shapes() -> None:
    enc = GVPEncoder(x_dim=1, node_dim=16, out_dim=8)
    g = _graph(8)
    assert enc(g).shape == (8, 8)
    batch = Batch.from_data_list([g, _graph(5)])
    assert enc.pool(enc(batch), batch).shape == (2, 8)


def test_coordinate_invariance() -> None:
    """pos is never read: rotate + translate it, outputs bitwise identical."""
    enc = GVPEncoder(x_dim=1)
    enc.eval()
    g1 = _graph(8)
    rot = torch.as_tensor(_random_rotation(np.random.default_rng(3)), dtype=torch.float32)
    g2 = g1.clone()
    g2.pos = g1.pos @ rot.T + 17.0
    assert not torch.equal(g1.pos, g2.pos)
    with torch.no_grad():
        assert torch.equal(enc(g1), enc(g2))


def test_edge_features_matter() -> None:
    """Corrupting edge features must change the output"""
    enc = GVPEncoder(x_dim=1)
    enc.eval()
    g2 = _graph(8)
    g2.edge_attr = g2.edge_attr.roll(shifts=1, dims=1)
    with torch.no_grad():
        assert not torch.equal(enc(_graph(8)), enc(g2))


def test_edge_index_matters() -> None:
    enc = GVPEncoder(x_dim=1)
    enc.eval()
    g1 = _graph(8)
    g2 = g1.clone()
    g2.edge_index = g1.edge_index[:, ::2]  # keep one direction of each pair
    g2.edge_attr = g1.edge_attr[::2]
    assert g2.edge_index.shape[1] == g1.edge_index.shape[1] // 2
    with torch.no_grad():
        assert not torch.equal(enc(g1), enc(g2))


def test_permutation_equivariance() -> None:
    enc = GVPEncoder(x_dim=1).double()
    enc.eval()
    g = _f64(_graph(8))
    perm = torch.as_tensor([5, 0, 7, 2, 1, 6, 3, 4])
    inv = torch.argsort(perm)  # inv[old] = new position of old node
    g_perm = g.clone()
    for field in ("x", "pos", "residue_type", "residue_index"):
        g_perm[field] = g[field][perm]
    g_perm.seq = "".join(g.seq[i] for i in perm.tolist())
    g_perm.edge_index = inv[g.edge_index]  # old (a->b) becomes new (inv[a]->inv[b])
    with torch.no_grad():
        out, out_perm = enc(g), enc(g_perm)
    assert torch.allclose(out_perm, out[perm], atol=1e-9)


def test_batching_equivalence() -> None:
    """
    Solo encoding == batched encoding, per graph and for pooling
    """
    enc = GVPEncoder(x_dim=1).double()
    enc.eval()
    g1, g2 = _f64(_graph(8)), _f64(_graph(5))
    batch = Batch.from_data_list([g1, g2])
    with torch.no_grad():
        out_b = enc(batch)
        assert torch.allclose(out_b[:8], enc(g1), atol=1e-9)
        assert torch.allclose(out_b[8:], enc(g2), atol=1e-9)
        pooled = enc.pool(out_b, batch)
        solo = enc.pool(enc(g1), Batch.from_data_list([g1]))
        assert torch.allclose(pooled[0], solo, atol=1e-9)


def test_single_residue_empty_edges() -> None:
    """n=1 graph: empty edges must flow through propagate as zero messages."""
    enc = GVPEncoder(x_dim=1, node_dim=16)
    g = build_graph_from_coords(np.zeros((1, 3)), "A", np.array([50.0]))
    assert g.edge_index.shape == (2, 0)
    out = enc(g)
    assert out.shape == (1, 16)
    assert torch.isfinite(out).all()
    assert enc.pool(out, g).shape == (1, 16)  # unbatched pool path


def test_determinism() -> None:
    enc = GVPEncoder(x_dim=1)
    enc.eval()
    g = _graph(8)
    with torch.no_grad():
        assert torch.equal(enc(g), enc(g))


def test_parameter_budget() -> None:
    """Must stay within the 1million parameters self imposed rule"""
    enc = GVPEncoder(x_dim=1)
    total = sum(p.numel() for p in enc.parameters())
    assert 100_000 < total <= 1_000_000


def test_rejects_fp16_input() -> None:
    enc = GVPEncoder(x_dim=4)
    g = _graph(8)
    g.x = g.x.half()
    with pytest.raises(TypeError, match="fp32"):
        enc(g)


def test_rejects_bad_construction() -> None:
    with pytest.raises(ValueError, match="depth"):
        GVPEncoder(x_dim=1, depth=0)
    with pytest.raises(ValueError, match="x_dim"):
        GVPEncoder(x_dim=0)
