"""
Extract per-structure test accuracy from saved ablation runs.

    uv run python scripts/per_protein.py --runs "outputs/ablation/*" \
        --ids benchmarks/cb513_chains.txt --out benchmarks/per_protein.csv

Only for the interested

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ligase.graph.io import load_structure_ids
from ligase.tasks.train import evaluate_saved_per_structure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="outputs/ablation/*")
    parser.add_argument("--ids", default="benchmarks/cb513_chains.txt")
    parser.add_argument("--out", default="benchmarks/per_protein.csv")
    args = parser.parse_args()

    ids = load_structure_ids(args.ids)
    rows: list[dict] = []
    run_dirs = sorted(p for p in Path(".").glob(args.runs) if (p / "best.pt").exists())
    if not run_dirs:
        raise SystemExit(f"no runs with best.pt under {args.runs!r}")
    for run_dir in tqdm(run_dirs, desc="runs"):
        ckpt = torch.load(run_dir / "best.pt", weights_only=False)
        saved = OmegaConf.create(ckpt["config"])
        embed = str(getattr(saved.embed, "model_name", "none"))
        for r in evaluate_saved_per_structure(OmegaConf.create({}), ids, run_dir):
            rows.append({"run": run_dir.name, "embed": embed, **r})
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} structures across {len(run_dirs)} runs")


if __name__ == "__main__":
    main()
