from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from ligase.encoders.mlp import MLPEncoder
from ligase.fuse import GraftedEncoder, ProjectionGraft, concat
from ligase.tasks.train import build_model, x_width


def _batch(x: torch.Tensor) -> Batch:
    d = Data(x=x, edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=x.shape[0])
    return Batch.from_data_list([d])


def test_concat_order_is_emb_then_struct() -> None:
    emb = torch.arange(12.0).reshape(4, 3)
    nf = torch.arange(4.0).reshape(4, 1)
    out = concat(node_feats=nf, embeddings=emb)
    assert out.shape == (4, 4)
    assert torch.equal(out[:, :3], emb)
    assert torch.equal(out[:, 3:], nf)


def test_concat_bitwise_equals_inline_wiring() -> None:
    emb = torch.randn(7, 5)
    nf = torch.randn(7, 1)
    assert torch.equal(concat(node_feats=nf, embeddings=emb), torch.cat([emb, nf], dim=1))


def test_x_width_project_mode() -> None:
    assert x_width("both", 320) == 321
    assert x_width("both", 320, "project", 128) == 129
    assert x_width("seq", 320) == 320
    assert x_width("struct", 0) == 1
    with pytest.raises(ValueError, match="concat|project"):
        x_width("both", 320, "gelu")
    with pytest.raises(ValueError, match="requires features=both"):
        x_width("seq", 320, "project")


def test_projection_graft_passthrough_and_budget() -> None:
    torch.manual_seed(0)
    d, w, n = 6, 3, 5
    graft = ProjectionGraft(d, w)
    x = torch.randn(n, d + 1)
    out = graft(x)
    assert out.shape == (n, w + 1)
    assert torch.equal(out[:, -1], x[:, -1])  # struct column passes through bitwise
    assert torch.equal(out[:, :-1], graft.proj(x[:, :-1]))
    assert sum(p.numel() for p in graft.parameters()) == d * w + w  # F3 budget


def test_graft_trains_jointly_with_encoder() -> None:
    torch.manual_seed(0)
    model = GraftedEncoder(
        ProjectionGraft(6, 3), MLPEncoder(in_dim=4, hidden_dim=8, depth=1, out_dim=3)
    )
    loss = torch.nn.functional.cross_entropy(
        model(_batch(torch.randn(9, 7))), torch.randint(0, 3, (9,))
    )
    loss.backward()
    assert model.graft.proj.weight.grad is not None  # F3: joint training


def test_grafted_encoder_batching_equivalence() -> None:
    torch.manual_seed(0)
    d, w = 4, 3
    model = GraftedEncoder(
        ProjectionGraft(d, w), MLPEncoder(in_dim=w + 1, hidden_dim=8, depth=1, out_dim=3)
    ).double()  # tolerance doctrine: batching equivalence is cross-computation-graph
    model.eval()
    da = Data(
        x=torch.randn(5, d + 1, dtype=torch.float64),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        num_nodes=5,
    )
    db = Data(
        x=torch.randn(7, d + 1, dtype=torch.float64),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        num_nodes=7,
    )
    with torch.no_grad():
        sa = model(Batch.from_data_list([da]))
        sb = model(Batch.from_data_list([db]))
        both = model(Batch.from_data_list([da, db]))
    assert torch.allclose(both[:5], sa, atol=1e-9)
    assert torch.allclose(both[5:], sb, atol=1e-9)


def test_build_model_graft_roundtrip() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"_target_": "ligase.encoders.mlp.MLPEncoder", "hidden_dim": 8, "depth": 1},
            "task": {"labels": "q3"},
        }
    )
    model = build_model(cfg, width=4, edge_dim=21, graft=ProjectionGraft(6, 3))
    assert isinstance(model, GraftedEncoder)
    model2 = build_model(cfg, width=4, edge_dim=21, graft=ProjectionGraft(6, 3))
    model2.load_state_dict(model.state_dict())  # eval replay: exact reconstruction
    for (k1, p1), (k2, p2) in zip(model.named_parameters(), model2.named_parameters(), strict=True):
        assert k1 == k2 and torch.equal(p1, p2)
