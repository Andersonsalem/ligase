from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch.nn import Module
from torch_geometric.data import Batch, Data

from ligase.encoders.gearnet import GearNetEncoder
from ligase.encoders.gvp import GVPEncoder
from ligase.encoders.mlp import MLPEncoder
from ligase.graph.build import build_graph_from_coords

FACTORIES: dict[str, Callable[[], Module]] = {
    "mlp": lambda: MLPEncoder(in_dim=1, hidden_dim=16, out_dim=8),
    "gvp": lambda: GVPEncoder(x_dim=1, node_dim=16, edge_dim=21, out_dim=8),
    "gearnet": lambda: GearNetEncoder(x_dim=1, node_dim=16, edge_in_dim=21, out_dim=8),
}

GRAPH_CONSUMING = ("gvp", "gearnet")

ENCODERS = list(FACTORIES)


def _coords(n: int) -> np.ndarray:
    t = np.arange(n) * np.deg2rad(100.0)
    return np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 1.5 * np.arange(n)], axis=1)


def _graph(n: int = 8, plddt: float = 88.0) -> Data:
    codes = "ACDEFGHIKLMNPQRSTVWYX"[:n]
    return build_graph_from_coords(_coords(n), codes, np.full(n, plddt))


def _f64(g: Data) -> Data:
    h = g.clone()
    for field in ("x", "edge_attr", "pos"):
        h[field] = h[field].double()
    return h


def _rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


@pytest.mark.parametrize("name", ENCODERS)
def test_forward_and_pool_shapes(name: str) -> None:
    enc = FACTORIES[name]()
    g = _graph(8)
    assert enc(g).shape == (8, 8)
    batch = Batch.from_data_list([g, _graph(5)])
    assert enc.pool(enc(batch), batch).shape == (2, 8)


@pytest.mark.parametrize("name", ENCODERS)
def test_determinism(name: str) -> None:
    enc = FACTORIES[name]()
    enc.eval()
    g = _graph(8)
    with torch.no_grad():
        assert torch.equal(enc(g), enc(g))


@pytest.mark.parametrize("name", ENCODERS)
def test_coordinate_invariance(name: str) -> None:
    enc = FACTORIES[name]()
    enc.eval()
    g1 = _graph(8)
    rot = torch.as_tensor(_rotation(np.random.default_rng(3)), dtype=torch.float32)
    g2 = g1.clone()
    g2.pos = g1.pos @ rot.T + 17.0
    assert not torch.equal(g1.pos, g2.pos)
    with torch.no_grad():
        assert torch.equal(enc(g1), enc(g2))


@pytest.mark.parametrize("name", ENCODERS)
def test_permutation_equivariance(name: str) -> None:
    enc = FACTORIES[name]().double()
    enc.eval()
    g = _f64(_graph(8))
    perm = torch.as_tensor([5, 0, 7, 2, 1, 6, 3, 4])
    inv = torch.argsort(perm)
    g_perm = g.clone()
    for field in ("x", "pos", "residue_type", "residue_index"):
        g_perm[field] = g[field][perm]
    g_perm.seq = "".join(g.seq[i] for i in perm.tolist())
    g_perm.edge_index = inv[g.edge_index]
    with torch.no_grad():
        out, out_perm = enc(g), enc(g_perm)
    assert torch.allclose(out_perm, out[perm], atol=1e-9)


@pytest.mark.parametrize("name", ENCODERS)
def test_batching_equivalence(name: str) -> None:
    enc = FACTORIES[name]().double()
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


@pytest.mark.parametrize("name", ENCODERS)
def test_rejects_fp16(name: str) -> None:
    enc = FACTORIES[name]()
    g = _graph(8)
    g.x = g.x.half()
    with pytest.raises(TypeError, match="fp"):
        enc(g)


@pytest.mark.parametrize("name", ENCODERS)
def test_parameter_budget(name: str) -> None:
    total = sum(p.numel() for p in FACTORIES[name]().parameters())
    assert 0 < total <= 1_000_000


@pytest.mark.parametrize("name", GRAPH_CONSUMING)
def test_edge_features_matter(name: str) -> None:
    enc = FACTORIES[name]()
    enc.eval()
    g2 = _graph(8)
    g2.edge_attr = g2.edge_attr.roll(shifts=1, dims=1)
    with torch.no_grad():
        assert not torch.equal(enc(_graph(8)), enc(g2))


@pytest.mark.parametrize("name", GRAPH_CONSUMING)
def test_edge_index_matters(name: str) -> None:
    enc = FACTORIES[name]()
    enc.eval()
    g1 = _graph(8)
    g2 = g1.clone()
    g2.edge_index = g1.edge_index[:, ::2]  # keep one direction of each pair
    g2.edge_attr = g1.edge_attr[::2]
    assert g2.edge_index.shape[1] == g1.edge_index.shape[1] // 2
    with torch.no_grad():
        assert not torch.equal(enc(g1), enc(g2))
