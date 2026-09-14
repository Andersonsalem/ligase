from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

CONFIG_PKG = "pkg://ligase.configs"


def main() -> None:
    commands = {
        "embed": embed_app,
        "build": _not_yet,
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
        "  build | train | eval   (M2 / M3 / M4 — see README Section 2)\n\n"
        "example:\n"
        "  uv run ligase embed 'sequences=[ACDEFGHIKLMNPQRSTVWY, MKTAYIAKQRQISFVKSHFSRQ]'\n"
        "  uv run ligase embed embed.model_name=facebook/esm2_t6_8M_UR50D \\\n"
        "      'sequences=[ACDEFGHIKLMNPQRSTVWY]'   # 8M model: quick smoke\n"
    )


def _not_yet() -> None:
    raise SystemExit("not yet implemented --- see README Section 2")


@hydra.main(config_path=CONFIG_PKG, config_name="config", version_base=None)
def embed_app(cfg: DictConfig) -> None:
    from hydra.utils import instantiate

    from ligase.utils.caching import cached

    sequences = list(cfg.sequences)
    if not sequences:
        raise SystemExit("no sequences provided")

    source = instantiate(cfg.embed)
    wrapped = cached(source, Path(cfg.cache_dir) / "embeddings")
    out = wrapped.embed(sequences)
    for seq, arr in out.items():
        print(f"{seq[:24]:<26} -> {arr.shape}")
    print(f"cache: {Path(cfg.cache_dir) / 'embeddings'}")
