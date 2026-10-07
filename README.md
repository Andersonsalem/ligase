# Ligase

Initially this project was meant for me to learn Github actions, the basics of CI/CD, how to reproduce and slightly modify research papers (the GNNs were just models I reproduced from older notes I had in my notebook) and how to use pytest (because unit testing is important)... And that's what I am currently satisfied with. I estimate LLMs to have written at most a tenth of the code but they were super useful for debugging (thanks Z.ai for GLM 5.3 flash).
Anyways here go...

Ligase is a benchmark for combining protein language model embeddings with structure-based residue graphs. It lets you test whether adding structure to a sequence model actually improves predictions.

You embed sequences with any protein language model, build residue graphs from structures, combine the two, and train. Splits, seed, and metrics stay the same across runs, and switching between sequence-only, structure-only, and combined inputs takes one flag.

> **Status: pre-alpha.** The API will change urgently if people want me to add new tests, but realistically, I will only work on it during my free time if I find an interesting research question.

## Results

Secondary structure Q3 accuracy on a CB513-class set: 513 PISCES-culled chains at ≤25% identity, with group-aware splits. PLMs are frozen, heads are ≤1M parameters, and everything runs on CPU.

| Features | Encoder      | ESM-2 650M | Ankh base |
|----------|--------------|-----------:|----------:|
| seq      | MLP          | 0.838      | 0.842     |
| both     | MLP          | 0.839      | 0.841     |
| struct   | GearNet      | 0.688      | 0.688     |
| both     | GearNet      | 0.759      | n/a       |
| struct   | MLP (pLDDT)  | 0.357      | n/a       |

Structure did not beat the language model here. A structure-only model gets real signal from geometry (0.688, versus 0.357 from pLDDT features alone), but it never reaches the sequence baseline, whether the splits are random or homology-protected. The full analysis is in [`notebooks/ablation_analysis.ipynb`](notebooks/ablation_analysis.ipynb), and every number above can be reproduced with the commands below.

## Install

```bash
git clone https://github.com/<you>/ligase && cd ligase
uv sync --all-extras          # Python 3.10-3.12, CPU torch by default
uv run pytest -m "not gpu and not slow"   # ~2 min, no downloads, no GPU
uv run ligase --help
```

Structure tasks need `mkdssp` on your PATH:

```bash
apt install dssp
# or
conda install -c bioconda dssp
```

## Quickstart

```bash
# 1. Embed sequences (cached on disk, keyed by model)
uv run ligase embed embed=esm2_t6_8M 'sequences=[ACDEFGHIKLMNPQRSTVWY]'

# 2. Build residue graphs from structures
#    (PDB ID, AF-...F1, UniProt accession, or local file)
uv run ligase build structure=1UBQ

# 3. Train
uv run ligase train features=seq model=mlp embed=esm2_t6_8M   # PLM only
uv run ligase train features=struct model=gearnet             # geometry only
uv run ligase train features=both model=gearnet               # both combined

# 4. Score a finished run
uv run ligase eval run_dir=outputs/secondary_structure/seq/mlp/<timestamp>
```

Each training run prints its best and test Q3 accuracy, and saves `best.pt`, `metrics.jsonl`, `test_metrics.json`, and `splits.json` to its run directory.

To produce the full results table, use the ablation runner. This is a ~2 minute smoke test:

```bash
uv run python scripts/ablate.py --ids examples/structures_example.txt \
    --embed esm2_t6_8M --epochs 3 --out benchmarks/ablation_smoke.csv
```

## Bring your own model

Any class that implements one small protocol gets the rest of the harness: caching, splits, baselines, and conformance checks.

And for the future - I would like to have a more extensive conformance script but that's for me :))
```bash
# TODO: placeholder until the conformance CLI ships
uv run ligase conformance embed --target mypackage.MySource
```
