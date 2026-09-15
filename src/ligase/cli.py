from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

CONFIG_PKG = "pkg://ligase.configs"


def main() -> None:
    commands = {
        "embed": embed_app,
        "build": build_app,
        "train": _not_yet,
        "eval": _not_yet,
        "help": help_app,
    }
    if len(sys.argv) < 2:
        help_app()
        raise SystemExit(2)
    cmd = sys.argv.pop(1)
    if cmd not in commands:
        print(f"unknown command: {cmd}\n")
        help_app()
        raise SystemExit(2)
    commands[cmd]()


def help_app() -> None:
    print(
        "ligase <command> [hydra overrides]\n"
        "  embed                  extract per-residue embeddings into the cache\n"
        "  build | train | eval   (Milestones 2 / 3 / 4 - see README Section 2)\n\n"
        "example:\n"
        "  uv run ligase embed 'sequences=[ACDEFGHIKLMNPQRSTVWY, MKTAYIAKQRQISFVKSHFSRQ]'\n"
        "  uv run ligase embed embed.model_name=facebook/esm2_t6_8M_UR50D \\\n"
        "      'sequences=[ACDEFGHIKLMNPQRSTVWY]'   # 8M model: quick smoke\n"
    )


def _not_yet() -> None:
    raise SystemExit("not yet implemented --- see README Section 2")


def _resolve_sequences(cfg: DictConfig) -> list[str]:
    from ligase.utils.seqio import load_sequences

    has_list = bool(cfg.sequences)
    has_file = cfg.get("sequences_file") is not None
    if has_list and has_file:
        raise SystemExit(
            "set exactly one of 'sequences' or 'sequences_file' — "
            "ambiguity is how irreproducible runs happen"
        )
    if has_file:
        return load_sequences(str(cfg.sequences_file))
    return list(cfg.sequences)


@hydra.main(config_path=CONFIG_PKG, config_name="config", version_base=None)
def embed_app(cfg: DictConfig) -> None:
    from hydra.utils import instantiate

    from ligase.utils.caching import cached

    sequences = _resolve_sequences(cfg)
    if not sequences:
        raise SystemExit("no sequences provided")

    source = instantiate(cfg.embed)
    wrapped = cached(source, Path(cfg.cache_dir) / "embeddings")
    out = wrapped.embed(sequences)
    for seq, arr in out.items():
        print(f"{seq[:24]:<26} -> {arr.shape}")
    print(f"cache: {Path(cfg.cache_dir) / 'embeddings'}")


@hydra.main(config_path=CONFIG_PKG, config_name="config", version_base=None)
def build_app(cfg: DictConfig) -> None:
    import torch
    from omegaconf import OmegaConf

    from ligase.graph.build import GraphParams, build_graph, cache_key
    from ligase.graph.io import load_structure, safe_name

    if not cfg.structure:
        raise SystemExit("no structure given — try: structure=1UBQ or structure=AF_P69905F1")
    fields = GraphParams.__dataclass_fields__
    raw = OmegaConf.to_container(cfg.graph) or {}
    unknown = sorted(set(raw) - set(fields) - {"name"})
    if unknown:
        raise SystemExit(
            f"unknown graph options: {', '.join(unknown)} — "
            "typo'd configs must fail loudly, not silently build the default"
        )
    params = GraphParams(**{k: v for k, v in raw.items() if k in fields})
    structure_id = str(cfg.structure)

    out_dir = Path(cfg.cache_dir) / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{safe_name(structure_id)}-{cache_key(structure_id, params)}.pt"

    if out.exists():
        # weights_only=False: we wrote this file; it holds a PyG Data, not raw tensors
        saved = torch.load(out, weights_only=False)
        data = saved["data"]
        note = f"cache hit: {out}"
    else:
        structure = load_structure(structure_id, Path(cfg.cache_dir) / "structures")
        data = build_graph(structure, params)
        torch.save({"data": data, "params": params, "structure_id": structure_id}, out)
        note = f"saved: {out}"

    print(f"{structure_id}: {data.num_nodes} residues, {data.edge_index.shape[1]} directed edges")
    print(f"seq[:60]: {data.seq[:60]}")
    print(note)
