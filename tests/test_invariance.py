import numpy as np
import pytest
import torch

from ligase.graph.build import (
    OFFSET_CLAMP,
    GraphParams,
    _rbf,
    build_graph_from_coords,
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Random proper rotation (det = +1) via QR with a column-flip fix."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_rotation_and_translation_invariance(toy_coords, toy_codes, toy_plddt):
    rng = np.random.default_rng(0)
    base = build_graph_from_coords(toy_coords, toy_codes, toy_plddt)
    for _ in range(5):
        R = _random_rotation(rng)
        t = rng.uniform(-50, 50, size=3)
        g = build_graph_from_coords(toy_coords @ R.T + t, toy_codes, toy_plddt)
        assert torch.equal(base.x, g.x)  # pLDDT: no coords involved
        assert torch.equal(base.residue_type, g.residue_type)
        assert torch.equal(base.edge_index, g.edge_index)  # kNN/radius: distance-based
        assert torch.allclose(base.edge_attr, g.edge_attr, atol=1e-5)
        assert not torch.allclose(base.pos, g.pos)  # pos is DATA — it must move


def test_determinism(toy_coords, toy_codes, toy_plddt):
    a = build_graph_from_coords(toy_coords, toy_codes, toy_plddt)
    b = build_graph_from_coords(toy_coords, toy_codes, toy_plddt)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.edge_index, b.edge_index)
    assert torch.equal(a.edge_attr, b.edge_attr)
    assert a.seq == b.seq


def test_edge_attr_corresponds_to_edge_index(toy_coords, toy_codes, toy_plddt):
    """Row k of edge_attr must describe column k of edge_index:
    edge_attr[k] = [RBF(d_ij) (20) | clip(j - i)]."""
    params = GraphParams()
    g = build_graph_from_coords(toy_coords, toy_codes, toy_plddt, params)
    ei = g.edge_index.numpy()
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        d = np.linalg.norm(toy_coords[i] - toy_coords[j])
        assert np.allclose(g.edge_attr[k, :-1].numpy(), _rbf(np.array([d]), params)[0], atol=1e-5)
        assert g.edge_attr[k, -1].item() == float(np.clip(j - i, -OFFSET_CLAMP, OFFSET_CLAMP))


def test_sequence_adjacency_symmetric(toy_coords, toy_codes, toy_plddt):
    g = build_graph_from_coords(toy_coords, toy_codes, toy_plddt)
    pairs = set(map(tuple, g.edge_index.t().tolist()))
    for i in range(len(toy_coords) - 1):
        assert (i, i + 1) in pairs and (i + 1, i) in pairs


def test_radius_edges_complete(toy_coords, toy_codes, toy_plddt):
    """A huge radius makes every ordered pair an edge (kNN ⊆ full set): n(n-1)."""
    n = len(toy_coords)
    g = build_graph_from_coords(toy_coords, toy_codes, toy_plddt, GraphParams(edge_radius=100.0))
    assert g.edge_index.shape[1] == n * (n - 1)


@pytest.mark.skip(
    reason="Milestone 3 encoder conformance: batching + permutation are model properties"
)
def test_batching_equivalence(): ...
