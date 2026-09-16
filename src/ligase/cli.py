from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from ligase.graph.build import GraphParams

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
        "  build                  structures -> PyG graphs (single, list, or manifest file)\n"
        "  train | eval           (Milestones 3 / 4 - see README Section 2)\n\n"
        "example:\n"
        "  uv run ligase embed 'sequences=[ACDEFGHIKLMNPQRSTVWY, MKTAYIAKQRQISFVKSHFSRQ]'\n"
        "  uv run ligase build structure=1UBQ\n"
        "  uv run ligase build structures_file=ids.txt   # one PDB/AFDB id per line\n"
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


def _resolve_structure_ids(cfg: DictConfig) -> list[str]:
    """Exactly one of structure / structures / structures_file; ambiguity exits."""
    from ligase.graph.io import load_structure_ids

    given = [
        name
        for name, has in (
            ("structure", bool(cfg.structure)),
            ("structures", bool(list(cfg.structures))),
            ("structures_file", cfg.get("structures_file") is not None),
        )
        if has
    ]
    if len(given) != 1:
        raise SystemExit(
            "set exactly one of 'structure', 'structures', or 'structures_file' — "
            f"got {', '.join(given) if given else 'none'}"
        )
    if given[0] == "structures_file":
        return load_structure_ids(str(cfg.structures_file))
    if given[0] == "structures":
        return [str(s) for s in cfg.structures]
    return [str(cfg.structure)]


def _graph_params(cfg: DictConfig) -> GraphParams:
    from omegaconf import OmegaConf

    from ligase.graph.build import GraphParams

    fields = GraphParams.__dataclass_fields__
    raw = OmegaConf.to_container(cfg.graph) or {}
    unknown = sorted(set(raw) - set(fields) - {"name"})
    if unknown:
        raise SystemExit(
            f"unknown graph options: {', '.join(unknown)} — "
            "typo'd configs must fail loudly, not silently build the default"
        )
    return GraphParams(**{k: v for k, v in raw.items() if k in fields})


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
    from ligase.graph.cache import get_or_build_graph

    ids = _resolve_structure_ids(cfg)
    if not ids:
        raise SystemExit("no structures given — try: structure=1UBQ or structures_file=ids.txt")
    params = _graph_params(cfg)

    hits = 0
    for i, structure_id in enumerate(ids, start=1):
        data, hit = get_or_build_graph(structure_id, params, Path(cfg.cache_dir))
        hits += int(hit)
        state = "cache hit" if hit else "built"
        print(
            f"[{i}/{len(ids)}] {structure_id}: {data.num_nodes} residues, "
            f"{data.edge_index.shape[1]} directed edges ({state})"
        )
    print(f"{len(ids) - hits} built, {hits} cached -> {Path(cfg.cache_dir) / 'graphs'}")
