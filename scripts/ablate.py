from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from hydra import compose, initialize_config_module

from ligase.graph.io import load_structure_ids
from ligase.tasks.train import run_training

STATES = {"q3": 3, "q8": 8}
GRID = [(f, m, "concat") for f in ("seq", "struct", "both") for m in ("mlp", "gvp", "gearnet")]
EXTRA = [("both", "gearnet", "project")]


def _overrides(features: str, model: str, mode: str, args: argparse.Namespace) -> list[str]:
    ov = [
        f"features={features}",
        f"model={model}",
        f"embed={args.embed}",
        f"profile={args.profile}",
        f"task.labels={args.labels}",
    ]
    if mode == "project":
        ov.append("features.mode=project")
    if args.groups != "none":
        ov.append(f"task.groups_file={args.groups}")
    if args.epochs is not None:
        ov.append(f"profile.epochs={args.epochs}")
    return ov


def _command(features: str, model: str, mode: str, args: argparse.Namespace) -> str:
    cmd = (
        f"uv run ligase train task=secondary_structure features={features} "
        f"model={model} embed={args.embed} profile={args.profile} task.labels={args.labels}"
    )
    if mode == "project":
        cmd += " features.mode=project"
    if args.groups != "none":
        cmd += f" task.groups_file={args.groups}"
    if args.epochs is not None:
        cmd += f" profile.epochs={args.epochs}"
    return cmd


def _run_row(
    features: str, model: str, mode: str, args: argparse.Namespace, ids: list[str]
) -> dict:
    overrides = _overrides(features, model, mode, args)
    if not isinstance(overrides, list) or not all(
        isinstance(o, str) and "=" in o for o in overrides
    ):
        raise SystemExit(f"_overrides must return a list of key=value strings, got: {overrides!r}")
    cfg = compose(config_name="config", overrides=overrides)
    cfg = compose(config_name="config", overrides=_overrides(features, model, mode, args))
    if args.groups != "none":
        cfg.task.groups_file = args.groups  # yaml key is optional
    run_dir = Path("outputs/ablation") / f"{features}_{model}_{mode}"
    t0 = time.perf_counter()
    run_training(cfg, ids, run_dir=run_dir)
    minutes = round((time.perf_counter() - t0) / 60, 1)
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    splits = json.loads((run_dir / "splits.json").read_text())
    acc_key = f"Q{STATES[args.labels]}_accuracy"
    f1s = {k: v for k, v in metrics.items() if k.startswith("f1_")}
    return {
        "features": features,
        "model": model,
        "mode": mode,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        acc_key: metrics[acc_key],
        "f1_macro": round(sum(f1s.values()) / max(1, len(f1s)), 4),
        **f1s,
        "minutes": minutes,
        "command": _command(features, model, mode, args),
        "_splits": splits,
    }


def _caveat(args: argparse.Namespace) -> str:
    lines = [
        "",
        "Honesty notes:",
        "  - structure-level splits; identical splits/seed/metrics across rows "
        "(verified via splits.json)",
        "  - homology: pool pre-culled at <=25% identity (PISCES)",
    ]
    gp = Path("benchmarks/cb513_groups.json")
    if gp.exists():
        d = json.loads(gp.read_text())
        v = d.get("verification_at_cull_threshold", {})
        lines.append(
            f"  - independent identity re-verification: {v.get('n_pairs_over_threshold')} "
            f"of 131,328 pairs >25% under our metric; {v.get('n_groups')} components "
            "at the cull threshold"
        )
    if args.groups != "none":
        lines.append(f"  - split protection: group-aware splits from {args.groups}")
    else:
        lines.append("  - split protection: NOT used (legacy random split)")
    lines.append("  - the baseline is the conscience: report either way.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ablation grid (see module docstring).")
    parser.add_argument("--ids", default="benchmarks/cb513_chains.txt")
    parser.add_argument("--embed", default="esm2_t33_650M")
    parser.add_argument("--labels", default="q3", choices=sorted(STATES))
    parser.add_argument("--profile", default="cpu", choices=("cpu", "gpu"))
    parser.add_argument("--groups", default="none", help="groups JSON; 'none' = legacy split")
    parser.add_argument("--epochs", type=int, default=None, help="override profile epochs")
    parser.add_argument("--out", default="benchmarks/ablation.csv")
    args = parser.parse_args()

    ids = load_structure_ids(args.ids)
    print(
        f"ablation grid: {len(GRID) + len(EXTRA)} rows over {len(ids)} ids "
        f"(embed={args.embed}, profile={args.profile}, labels={args.labels})"
    )
    rows: list[dict] = []
    base_splits: dict | None = None
    with initialize_config_module(version_base=None, config_module="ligase.configs"):
        for features, model, mode in GRID + EXTRA:
            print(f"\n=== row: features={features} model={model} mode={mode} ===")
            row = _run_row(features, model, mode, args, ids)
            as_sets = {k: frozenset(v) for k, v in row["_splits"].items()}
            if base_splits is None:
                base_splits = as_sets
            elif as_sets != base_splits:
                raise SystemExit(
                    f"split drift at features={features} model={model} mode={mode}; "
                    "rows are not comparable; investigate (transient skip?) before "
                    "trusting the table"
                )
            rows.append({k: v for k, v in row.items() if k != "_splits"})

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    acc_key = f"Q{STATES[args.labels]}_accuracy"
    compact = df[["features", "model", "mode", acc_key, "f1_macro", "n_test", "minutes"]]
    print("\n=== ablation table ===")
    print(compact.to_string(index=False))
    print("\nreproducing commands:")
    for cmd in df["command"]:
        print(f"  {cmd}")
    print(_caveat(args))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
