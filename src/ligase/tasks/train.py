from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from ligase.embed import EmbeddingSource
from ligase.fuse import GraftedEncoder, ProjectionGraft, concat
from ligase.graph.build import GraphParams
from ligase.tasks.dataset import Example, attach_embeddings, build_examples
from ligase.tasks.secondary_structure import encode_q3, encode_q8, report_metrics
from ligase.tasks.splits import load_groups, split_ids
from ligase.utils.logging import append_json
from ligase.utils.seeding import seed_everything

logger = logging.getLogger(__name__)

STATES = {"q3": 3, "q8": 8}


def x_width(features: str, emb_dim: int, mode: str = "concat", proj_width: int = 128) -> int:
    try:
        base = {"seq": emb_dim, "struct": 1, "both": emb_dim + 1}[features]
    except KeyError:
        raise ValueError(f"features must be seq|struct|both, got {features!r}") from None
    if mode not in ("concat", "project"):
        raise ValueError(f"features.mode must be concat|project, got {mode!r}")
    if mode == "project":
        if features != "both":
            raise ValueError(f"features.mode='project' requires features=both, got {features!r}")
        return proj_width + 1
    return base


def encode_examples(examples: list[Example], features: str, states: int) -> list[Data]:
    encode = encode_q3 if states == 3 else encode_q8
    out: list[Data] = []
    for ex in examples:
        g = ex.graph.clone()
        if features in ("seq", "both"):
            if ex.embeddings is None:
                raise ValueError(
                    f"{ex.structure_id}: features={features!r} needs embeddings --- "
                    "call attach_embeddings first"
                )
            emb = torch.from_numpy(ex.embeddings)
            g.x = emb if features == "seq" else concat(node_feats=g.x, embeddings=emb)
        g.y = torch.from_numpy(encode(ex.labels))
        out.append(g)
    return out


def run_epoch(model, loader: DataLoader, opt: torch.optim.Optimizer, device: str) -> float:
    """Single training pass ... returns loss averaged per graph"""
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(batch), batch.y)  # TODO implement what you need
        loss.backward()
        opt.step()
        total += float(loss) * batch.num_graphs
        count += batch.num_graphs
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, data_list: list[Data], states: int, device: str) -> dict[str, float]:
    model.eval()
    ys, ps = [], []
    for data in data_list:
        batch = Batch.from_data_list([data]).to(device)
        ps.append(model(batch).argmax(-1).cpu().numpy())
        ys.append(data.y.cpu().numpy())
    if not ys:
        raise ValueError("evaluate called with an empty split --- check split fractions")
    return report_metrics(np.concatenate(ys), np.concatenate(ps), states)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_list: list[Data],
    *,
    epochs: int,
    lr: float,
    device: str,
    states: int,
    metrics_path: Path,
) -> float:
    """Full training, restores and returns the best-val state"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    key = f"Q{states}_accuracy"
    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, opt, device)
        val_m = evaluate(model, val_list, states, device)
        append_json(metrics_path, {"epoch": epoch, "split": "train", "loss": tr_loss})
        append_json(metrics_path, {"epoch": epoch, "split": "val", **val_m})
        logger.info("epoch %d: train loss %.4f --- val %s=%.4f", epoch, tr_loss, key, val_m[key])
        if val_m[key] > best_acc:
            best_acc = val_m[key]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return best_acc


def build_model(
    cfg: DictConfig, width: int, edge_dim: int, graft: torch.nn.Module | None = None
) -> torch.nn.Module:
    raw = OmegaConf.to_container(cfg.model, resolve=True) or {}
    raw.pop("name", None)
    target = str(raw.get("_target_", ""))
    out_dim = STATES[str(cfg.task.labels)]
    if target.endswith("MLPEncoder"):
        inner: torch.nn.Module = instantiate(raw, in_dim=width, out_dim=out_dim)
    else:
        inner = instantiate(raw, x_dim=width, edge_in_dim=edge_dim, out_dim=out_dim)
    return GraftedEncoder(graft, inner) if graft is not None else inner


def _device(cfg: DictConfig) -> str:
    want = cfg.profile.get("device", "cpu")
    return "cuda" if want == "cuda" and torch.cuda.is_available() else "cpu"


def _group_labels(ids: list[str], groups_file: str | Path | None) -> list[str] | None:
    if not groups_file:
        return None
    lookup = load_groups(groups_file)
    missing = [i for i in ids if i not in lookup]
    if missing:
        raise SystemExit(
            f"{len(missing)} ids absent from groups file (first: {missing[0]!r})"
            "--- regenerate groups from the same manifest"
        )
    return [lookup[i] for i in ids]


def run_training(cfg: DictConfig, ids: list[str], run_dir: Path | None = None) -> dict[str, float]:
    """Config -> trained model -> test metrics. Returns the final test metrics.

    ``run_dir`` defaults to Hydra's runtime output dir; tests pass tmp paths.
    Saves: ``best.pt`` (weights + config snapshot), ``metrics.jsonl``,
    ``test_metrics.json``.
    """
    if run_dir is None:
        from hydra.core.hydra_config import HydraConfig

        run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir = Path(run_dir)
    seed_everything(cfg.seed)
    device = _device(cfg)

    try:
        states = STATES[cfg.task.labels]
    except KeyError:
        raise SystemExit(f"task.labels must be 'q3' or 'q8', got {cfg.task.labels!r}") from None

    examples, skipped = build_examples(
        ids,
        GraphParams(**{k: cfg.graph[k] for k in GraphParams.__dataclass_fields__}),
        cfg.cache_dir,
    )
    for sid, reason in skipped:
        logger.warning("skipped %s: %s", sid, reason)
    if len(examples) < 3:
        raise SystemExit(f"need >= 3 structures for a split, got {len(examples)}")

    emb_dim = 0
    mode = str(cfg.features.get("mode", "concat"))
    proj_width = int(cfg.features.get("width", 128))
    if cfg.features.name in ("seq", "both"):
        source: EmbeddingSource = instantiate(cfg.embed)
        attach_embeddings(examples, source, cfg.cache_dir)
        emb_dim = source.dim

    by_id = {ex.structure_id: ex for ex in examples}
    surviving = [ex.structure_id for ex in examples]
    splits = split_ids(
        surviving,
        cfg.task.train_frac,
        cfg.task.val_frac,
        cfg.seed,
        groups=_group_labels(surviving, cfg.task.get("groups_file")),
    )
    data = {
        name: encode_examples([by_id[i] for i in ids_], cfg.features.name, states)
        for name, ids_ in splits.items()
    }
    width = x_width(cfg.features.name, emb_dim, mode, proj_width)
    graft = ProjectionGraft(emb_dim, proj_width) if mode == "project" else None
    model = build_model(cfg, width, cfg.graph.rbf_count + 1, graft=graft).to(device)

    gen = torch.Generator().manual_seed(cfg.seed)  # shuffle determinism
    loader = DataLoader(
        data["train"], batch_size=cfg.profile.batch_size, shuffle=True, generator=gen
    )
    best_val = train_model(
        model,
        loader,
        data["val"],
        epochs=cfg.profile.epochs,
        lr=cfg.profile.lr,
        device=device,
        states=states,
        metrics_path=run_dir / "metrics.jsonl",
    )
    test_m = evaluate(model, data["test"], states, device)
    append_json(run_dir / "metrics.jsonl", {"epoch": -1, "split": "test", **test_m})

    torch.save(
        {"state_dict": model.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)},
        run_dir / "best.pt",
    )
    (run_dir / "test_metrics.json").write_text(json.dumps(test_m, indent=2))
    print(
        f"[{cfg.features.name}/{Path(str(cfg.model._target_)).stem}] "
        f"best Q{states}_acc={best_val:.4f}| test Q{states}_acc={test_m[f'Q{states}_accuracy']:.4f}"
    )
    print(f"run dir: {run_dir}")
    return test_m


def evaluate_saved(cfg: DictConfig, ids: list[str], run_dir: Path) -> dict[str, float]:
    ckpt = torch.load(Path(run_dir) / "best.pt", weights_only=False)  # we wrote it
    saved = OmegaConf.create(ckpt["config"])
    try:
        states = STATES[str(saved.task.labels)]
    except KeyError:
        raise SystemExit(
            f"saved task.labels must be 'q3' or 'q8', got {saved.task.labels!r}"
        ) from None
    examples, _ = build_examples(
        ids,
        GraphParams(**{k: saved.graph[k] for k in GraphParams.__dataclass_fields__}),
        cfg.cache_dir,
    )
    emb_dim = 0
    mode = "concat"
    proj_width = 128
    if saved.features.name in ("seq", "both"):
        source: EmbeddingSource = instantiate(saved.embed)
        attach_embeddings(examples, source, cfg.cache_dir)
        emb_dim = source.dim
        mode = str(saved.features.get("mode", "concat"))
        proj_width = int(saved.features.get("width", 128))
    by_id = {ex.structure_id: ex for ex in examples}
    surviving = [ex.structure_id for ex in examples]
    splits = split_ids(
        surviving,
        saved.task.train_frac,
        saved.task.val_frac,
        saved.seed,
        groups=_group_labels(surviving, saved.task.get("groups_file")),
    )
    test_data = encode_examples([by_id[i] for i in splits["test"]], saved.features.name, states)
    width = x_width(saved.features.name, emb_dim, mode, proj_width)
    graft = ProjectionGraft(emb_dim, proj_width) if mode == "project" else None
    model = build_model(saved, width, saved.graph.rbf_count + 1, graft=graft)
    model.load_state_dict(ckpt["state_dict"])
    return evaluate(model, test_data, states, _device(cfg))
