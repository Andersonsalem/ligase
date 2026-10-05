"""End-to-end training tests — the done-when, in miniature, offline.

CI never downloads weights (MockSource supplies embeddings) and never
needs mkdssp (faked here; validated for real in test_secondary_structure).
Determinism is pinned: same seed -> identical test metrics.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from ligase.encoders.mlp import MLPEncoder
from ligase.graph.build import is_protein_ca
from ligase.tasks import dataset as ds
from ligase.tasks import secondary_structure as ss
from ligase.tasks.splits import split_ids
from ligase.tasks.train import (
    build_model,
    encode_examples,
    evaluate_saved,
    evaluate_saved_per_structure,
    run_training,
    train_model,
    x_width,
)


@pytest.fixture
def toy_set(toy_pdb_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pattern = "HET-" * 2

    def _fake_dssp(structure) -> dict:
        chain = next(ch for ch in structure[0] if len(ch))
        keys = [ss.residue_key(chain.name, r) for r in chain if is_protein_ca(r)]
        return {k: pattern[i % len(pattern)] for i, k in enumerate(keys)}

    monkeypatch.setattr(ss, "compute_dssp", _fake_dssp)
    ids = []
    for i in range(8):
        p = tmp_path / f"toy{i}.pdb"
        shutil.copy(toy_pdb_path, p)
        ids.append(str(p))
    examples, skipped = ds.build_examples(ids)
    assert not skipped and len(examples) == 8
    return examples


def _cfg(tmp_path: Path, features: str = "struct") -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": 42,
            "cache_dir": str(tmp_path / "cache"),
            "features": {"name": features},
            "embed": None,
            "graph": {
                "edge_knn": 10,
                "edge_radius": 10.0,
                "rbf_min": 0.0,
                "rbf_max": 20.0,
                "rbf_count": 20,
                "rbf_sigma": 2.5,
            },
            "model": {
                "_target_": "ligase.encoders.mlp.MLPEncoder",
                "hidden_dim": 16,
                "depth": 1,
                "dropout": 0.0,
            },
            "task": {
                "name": "secondary_structure",
                "labels": "q3",
                "num_classes": 3,
                "train_frac": 0.5,
                "val_frac": 0.25,
            },
            "profile": {"name": "cpu", "device": "cpu", "batch_size": 2, "epochs": 3, "lr": 0.003},
        }
    )


def test_x_width() -> None:
    assert x_width("seq", 320) == 320
    assert x_width("struct", 320) == 1
    assert x_width("both", 320) == 321
    with pytest.raises(ValueError, match="seq|struct|both"):
        x_width("nope", 4)


def test_encode_examples_attaches_y(toy_set) -> None:
    data = encode_examples(toy_set, "struct", states=3)
    assert data[0].y.shape == (data[0].num_nodes,)
    assert data[0].y.dtype == np.int64 or data[0].y.dtype == torch.long
    assert set(data[0].y.tolist()) <= {0, 1, 2}


def test_end_to_end_struct(toy_set, tmp_path: Path) -> None:
    metrics = run_training(
        _cfg(tmp_path), [ex.structure_id for ex in toy_set], run_dir=tmp_path / "run"
    )
    assert 0.0 <= metrics["Q3_accuracy"] <= 1.0
    assert set(metrics) == {"Q3_accuracy", "f1_H", "f1_E", "f1_C"}
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "test_metrics.json").exists()
    saved = json.loads((tmp_path / "run" / "test_metrics.json").read_text())
    assert saved == metrics  # file matches return value
    splits = json.loads((tmp_path / "run" / "splits.json").read_text())
    assert set(splits) == {"train", "val", "test"}
    assert sorted(i for v in splits.values() for i in v) == sorted(
        ex.structure_id for ex in toy_set
    )  # file matches return value


def test_training_is_seed_deterministic(toy_set, tmp_path: Path) -> None:
    ids = [ex.structure_id for ex in toy_set]
    a = run_training(_cfg(tmp_path / "a"), ids, run_dir=tmp_path / "a" / "run")
    b = run_training(_cfg(tmp_path / "b"), ids, run_dir=tmp_path / "b" / "run")
    assert a == b  # same seed -> bitwise-same metrics on CPU


def test_end_to_end_seq_with_mock(toy_set, cached_mock, tmp_path: Path) -> None:
    from torch_geometric.loader import DataLoader

    ds.attach_embeddings(toy_set, cached_mock, tmp_path / "cache")
    by_id = {ex.structure_id: ex for ex in toy_set}
    splits = split_ids([ex.structure_id for ex in toy_set], 0.5, 0.25, 42)
    data = {k: encode_examples([by_id[i] for i in v], "seq", 3) for k, v in splits.items()}
    assert data["train"][0].x.shape[1] == cached_mock.dim  # x really is embeddings
    model = MLPEncoder(in_dim=cached_mock.dim, hidden_dim=16, depth=1, out_dim=3)
    loader = DataLoader(
        data["train"], batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42)
    )
    m = train_model(
        model,
        loader,
        data["val"],
        epochs=3,
        lr=0.003,
        device="cpu",
        states=3,
        metrics_path=tmp_path / "m.jsonl",
    )
    assert 0.0 <= m <= 1.0


def _fake_dssp(structure) -> dict:
    pattern = "HEC" * 3
    chain = next(ch for ch in structure[0] if len(ch))
    keys = [ss.residue_key(chain.name, r) for r in chain if is_protein_ca(r)]
    return {k: pattern[i % len(pattern)] for i, k in enumerate(keys)}


def test_labels_solely_determine_num_classes(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.task.labels = "q8"
    model = build_model(cfg, width=1, edge_dim=21)
    assert model.net[-1].out_features == 8


def test_build_model_wiring_all_gnns() -> None:
    """Wiring passes edge_in_dim to all GNNs."""
    from ligase.encoders.gearnet import GearNetEncoder
    from ligase.encoders.gvp import GVPEncoder

    cfg = OmegaConf.create(
        {
            "model": {"_target_": "ligase.encoders.gvp.GVPEncoder", "node_dim": 16, "depth": 1},
            "task": {"labels": "q3"},
        }
    )
    assert isinstance(build_model(cfg, width=1, edge_dim=21), GVPEncoder)
    cfg.model._target_ = "ligase.encoders.gearnet.GearNetEncoder"
    assert isinstance(build_model(cfg, width=1, edge_dim=21), GearNetEncoder)


def test_evaluate_saved_replays_saved_split(toy_set, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    cfg = _cfg(tmp_path)  # trains with train_frac 0.5, val_frac 0.25
    metrics = run_training(cfg, [ex.structure_id for ex in toy_set], run_dir=run_dir)
    eval_cfg = _cfg(tmp_path / "eval")
    eval_cfg.task.train_frac = 0.34
    eval_cfg.task.val_frac = 0.33
    again = evaluate_saved(eval_cfg, [ex.structure_id for ex in toy_set], run_dir)
    assert again == metrics  # same weights + same test set => bitwise-same metrics


def test_per_structure_reconstructs_aggregate(toy_set, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    metrics = run_training(_cfg(tmp_path), [ex.structure_id for ex in toy_set], run_dir=run_dir)
    rows = evaluate_saved_per_structure(
        _cfg(tmp_path / "e"), [ex.structure_id for ex in toy_set], run_dir
    )
    assert len(rows) == len(json.loads((run_dir / "splits.json").read_text())["test"])
    total_res = sum(r["n_residues"] for r in rows)
    total_ok = sum(r["n_correct"] for r in rows)
    assert total_res > 0
    assert abs((total_ok / total_res) - metrics["Q3_accuracy"]) < 1e-9  # weighted mean == aggregate
