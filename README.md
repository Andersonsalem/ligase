# Ligase

> Join what protein ML keeps apart: sequence language models and structure graphs.

Ligase is a small, readable library for protein representation learning. Embed a sequence with a protein language model, build a residue graph from structure, graft the two together, predict, and, above all, find out whether structure was worth it.

**Status: PRE-ALPHA.** The API will break. The direction will not.

---

## 0. READ THIS FIRST — protocol for contributors (human or AI)

This README is the single source of truth. It is a living document: **every completed task must update Section 2 (Status dashboard) and Section 16 (Changelog) in the same commit that completes the task.**

Working rules:

1. Read Section 2, pick the lowest-numbered unchecked task. Do not skip ahead.
2. Implement to the *contract* written in the relevant folder's Section 7 entry. Contracts may evolve, but only with a Changelog entry explaining why.
3. Respect the hard rules (Section 11). They are non-negotiable.
4. Before declaring a task done, run the full quality gate (Section 12) — all green.
5. Update: status checkboxes, changelog row, and any contract/open-questiontext that the work settled. Commit with a Conventional Commit message.
6. Do not add dependencies without a Changelog entry and a one-line rationale.
7. Files stay under \~300 lines. If a file grows past that, split it and note it.
8. Never import lab data, lab code, or anything non-public. Public data only (Section 9).

---

## 1. Why this exists

Protein representation learning is fragmented. ESM gives you embeddings,Graphein builds graphs, ProteinWorkshop runs benchmarks — and every paper repoglues them together differently, once, invisibly. Meanwhile the field's mostcommon overclaim, "structure helped!", is almost never tested against a decentstructure-free baseline on identical splits.

Ligase makes that test a one-flag operation and treats the result as the product.

## 2. Status dashboard ← UPDATE THIS SECTION AS TASKS COMPLETE

### Milestones

| **#Milestone DeliverableDone whenStatus** |                   |                                                |                                                                               |   |
| ---------------------------------------- | ----------------- | ---------------------------------------------- | ----------------------------------------------------------------------------- | - |
| M0                                       | Skeleton          | Repo layout, uv env, CI, this README           | CI green from first commit; `uv run pytest` passes locally                    | ☐ |
| M1                                       | `embed`           | ESM-2 extraction → disk cache                  | round-trip + zero-recompute + determinism tests pass                          | ☐ |
| M2                                       | `graph`           | Structures → PyG residue graphs                | rotation-invariance & permutation-equivariance tests pass (unskipped)         | ☐ |
| M3                                       | `encoders` + task | MLP / GVP / GearNet-lite + secondary structure | trains end-to-end on a small CPU-able config; Q3 > seq-only baseline reported | ☐ |
| M4                                       | Fusion + honesty  | Ablation runs, benchmark table, docs site      | the one-flag promise works verbatim from README commands                      | ☐ |
| M5                                       | Launch            | HF Space + launch post                         | a stranger runs the demo unaided                                              | ☐ |

### Task ledger

| **Task Milestone StatusEvidence (commit/PR)Notes** |       |   |   |                               |
| ------------------------------------------------ | ----- | - | - | ----------------------------- |
| Skeleton + README + uv + CI                      | M0    | ☐ | — | this commit                   |
| Hydra config skeleton (M1-relevant groups)       | M0/M1 | ☐ | — | Section 8                     |
| `embed/esm.py` extractor                         | M1    | ☐ | — |                               |
| Embedding cache (h5, hash-addressed)             | M1    | ☐ | — |                               |
| Cache tests (round-trip, zero-recompute)         | M1    | ☐ | — |                               |
| `graph/io.py` structure loader                   | M2    | ☐ | — |                               |
| `graph/build.py` graph builder                   | M2    | ☐ | — |                               |
| Invariance tests unskipped & green               | M2    | ☐ | — | the moment of truth           |
| `encoders/mlp.py`                                | M3    | ☐ | — | the conscience of the library |
| `encoders/gvp.py`                                | M3    | ☐ | — |                               |
| `encoders/gearnet.py`                            | M3    | ☐ | — |                               |
| `tasks/secondary_structure.py`                   | M3    | ☐ | — |                               |
| Ablation runner + benchmark table                | M4    | ☐ | — |                               |
| Docs site (mkdocs) live                          | M4    | ☐ | — |                               |
| HF Space                                         | M5    | ☐ | — |                               |
| Launch post                                      | M5    | ☐ | — |                               |

---

## 3. The one-flag promise

The entire reason this library exists, expressed as three commands:

```bash
uv run ligase train task=secondary_structure features=seq     # pooled ESM-2 → MLP
uv run ligase train task=secondary_structure features=struct  # geometry → GNN
uv run ligase train task=secondary_structure features=both    # the graft
```

Same splits, same seed, same metrics. The ablation table falls out.Every performance claim in the docs must be reproducible from one of these runs.If a comparison can't be run this way, it doesn't go in the README.

## 4. Training policy (read before touching training code)

The maintainer does not want to train models. Honor that:

- **Foundation models: never trained, never fine-tuned.** ESM-2 is used frozen, inference-only.
- **Task heads are the only training that happens**: ≤ \~1M parameters, minutes on CPU,default `profile=cpu`. GPU is an optional convenience, never a requirement (Section 8, profiles).
- No large-scale experiments, no hyperparameter sweeps by default. If an experiment needs more than a coffee break on CPU, it needs a Changelog justification.

## 5. Scope

**In scope (v1)**

- ESM-2 (and optionally Ankh) embedding extraction: batched, bf16 on GPU / fp32 on CPU,disk-cached, hash-addressed by sequence
- PDB / mmCIF → PyTorch Geometric residue graphs: sequence edges, kNN + radius edgeson Cα distance, RBF distance features, pLDDT as a node feature
- Three encoders, each in one readable file: pooled-embedding MLP (the honestbaseline), GVP-GNN (invariant mode), GearNet-lite
- Tasks: 3-/8-state secondary structure first; binding-site prediction next
- Tests that verify the *math*: permutation equivariance, rotation invariance,batching equivalence, cache round-trips

**Deliberately out of scope**

- SE(3)-equivariance / vector neurons — invariant features (distances, angles) coverthe industrial majority; this is a design decision, not an omission
- Generative modeling (diffusion, flow matching, inverse folding) — explicitly excluded
- Training or fine-tuning foundation models (see Section 4)
- MSA-based models, protein–ligand complexes, docking, antibody-specific tools (v2 maybe)

## 6. Tech stack (chosen deliberately — this repo doubles as a job portfolio)

| **ConcernToolWhy it's here** |                                                           |                                                                                           |
| ---------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Env & package management     | **uv** (`uv.lock` committed)                              | Reproducible envs in seconds; the modern standard. `uv sync` must be all a newcomer needs |
| Python                       | 3.11 (CI matrix: 3.10–3.12)                               | Boring, fast, broadly compatible                                                          |
| Configs & run management     | **Hydra** (config groups + overrides)                     | The one-flag promise *is* Hydra composition; also standard in industrial ML               |
| Tensors / models             | torch, torch-geometric                                    | Industry default                                                                          |
| Structure parsing            | gemmi                                                     | One dependency does PDB + mmCIF correctly                                                 |
| Caching                      | h5py (hash-addressed)                                     | Zero-recompute guarantee, testable                                                        |
| Metrics                      | scikit-learn                                              | Q3/Q8, F1 — standard, auditable                                                           |
| Config/Metric tables         | pandas (or polars, decide in M4)                          | Benchmark tables                                                                          |
| Tests                        | pytest + pytest-cov                                       | The math tests are the differentiator                                                     |
| Lint + format                | ruff (one tool, both jobs)                                | Zero-config credibility                                                                   |
| Types                        | mypy (gradually strict)                                   | Types on public contracts; strict per-module as it stabilizes                             |
| Hooks                        | pre-commit                                                | Quality gates run before commit, not after review                                         |
| CI                           | GitHub Actions (uv-native action)                         | Lint + types + tests on every push; matrix 3.10–3.12                                      |
| Docs                         | mkdocs-material (+ mkdocstrings)                          | Deployed to GH Pages in M4; API docs generated from docstrings                            |
| CLI                          | `[project.scripts]` entry point via Hydra's `@hydra.main` | `uv run ligase ...` works from day one                                                    |
| Experiment tracking          | CSV/JSON by default; wandb as optional extra              | No accounts required to reproduce results                                                 |

**Versioning:** SemVer. Pre-1.0 means anything may break; after 1.0, breaking changes require a major bump.

## 7. Repository layout — what each piece is and why

 
Legend: **plain** = exists now. `(M#)` = not yet scaffolded; arrives with that milestone.
 
```
ligase/
├── README.md                  ← this document; the contract + status dashboard
├── LICENSE                    ← MIT
├── pyproject.toml             ← single source for deps, tools, CLI entry point
├── uv.lock                    ← (M0) committed once `uv sync` is first run; exact reproducible env
├── Makefile                   ← thin aliases over uv commands (make test/lint/format)
├── .gitignore                 ← includes data/, *.h5, outputs/, .dssp/
├── .pre-commit-config.yaml    ← (M0) ruff lint+format, ruff-check hook; mypy on changed files
├── .github/
│   └── workflows/ci.yml       ← uv setup → sync → ruff → mypy → pytest (no GPU)
│
├── configs/                   ← (M1) HYDRA. Objective: every run reproducible from configs alone.
│   ├── config.yaml            ←   root defaults list; see Section 8
│   ├── features/               ←   seq.yaml | struct.yaml | both.yaml   (THE flag)
│   ├── embed/                  ←   one yaml per embedding model (esm2_t33_650M.yaml, …)
│   ├── graph/                  ←   graph construction params (knn k, radius, RBF centers)
│   ├── model/                  ←   mlp.yaml | gvp.yaml | gearnet.yaml
│   ├── task/                   ←   secondary_structure.yaml | binding_site.yaml | none.yaml
│   └── profile/                ←   cpu.yaml (default) | gpu.yaml
│
├── src/ligase/
│   ├── __init__.py             ← version only. HARD RULE: imports nothing heavy (Section 11)
│   ├── cli.py                  ← (M1) @hydra.main entry points: `embed`, `build`, `train`, `eval`
│   │
│   ├── embed/                  ← OBJECTIVE: sequence → per-residue embeddings, once, cached forever.
│   │   ├── __init__.py
│   │   └── esm.py              ← (M1)
│   │
│   ├── graph/                  ← OBJECTIVE: structure → one opinionated PyG residue graph.
│   │   ├── __init__.py
│   │   ├── io.py               ← (M2) loading (paths or PDB/AFDB IDs), download+cache
│   │   └── build.py            ← (M2) node/edge features; THE invariance guarantee lives here
│   │
│   ├── fuse/                   ← OBJECTIVE: the graft. pLM embeddings onto graph nodes.
│   │   └── __init__.py         ← (M4) dimension handling, frozen-vs-tuned switch, projections
│   │
│   ├── encoders/                ← OBJECTIVE: three encoders, three files, one idea each.
│   │   ├── __init__.py
│   │   ├── mlp.py               ← (M3) pooled-embedding baseline. Never delete: it's the conscience.
│   │   ├── gvp.py               ← (M3) GVP-GNN in invariant mode
│   │   └── gearnet.py           ← (M3) relational GNN, RBF edge features, edge-type embeddings
│   │
│   ├── tasks/                   ← OBJECTIVE: labels in, metrics out, per-residue heads.
│   │   ├── __init__.py
│   │   └── secondary_structure.py  ← (M3)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py           ← seed_everything; determinism policy — the only real code today, and it sets the tone
│       ├── caching.py           ← (M1) hash-addressed cache helpers
│       └── logging.py           ← (M3) tiny CSV/JSON metric logger — no accounts, ever
│
├── tests/
│   ├── conftest.py              ← fixtures: tiny toy structure, 5 fake sequences, tmp caches
│   ├── test_smoke.py            ← import policy, version, determinism — green at M0
│   └── test_invariance.py       ← M2 acceptance math (Section 10); skipped, but the math is written down
│
├── examples/                    ← notebooks land M3+: 01_embed_and_cache, 02_build_graphs,
│                                 ← 03_secondary_structure_ablation
├── docs/                        ← (M4) mkdocs; API reference generated from docstrings
└── data/                        ← gitignored. Download + cache root. Never commit data.
```
 
**Where things stand right now (pre-M0 complete):** the package skeleton, empty module files, and `utils/seeding.py` exist; `configs/`, `uv.lock`, `.pre-commit-config.yaml`, `cli.py`, and `docs/` do not yet. Closing that gap — plus a green CI run — is exactly what M0 is (Section 2). Every `(M#)` tag above should flip to plain text, and its checkbox in the Task ledger should flip to [V], in the same commit.

### 7.1 Module contracts (implement exactly these; signatures are the API)

**`embed/esm.py`** — M1

```python
embed_sequences(    seqs: list[str],    model_name: str = "esm2_t33_650M",   # keys of configs/embed/*.yaml    layer: int | None = None,            # default from the embed config    cache_dir: Path = ...,               # default: data/cache/embeddings) -> dict[str, np.ndarray]               # {seq: (L, D) float16}; L = residue count
```

Open questions (decide while building M1, then move answers into this README):

- Backend: HF `transformers` (recommended: tokenizer + models + AutoModel in one) vs fair-esm
- Cache layout: one h5 per model vs per sequence (benchmark on 1k seqs, pick, document)
- Non-canonical residues (X, U, B, Z, gaps): map-and-log vs pass-through — pick one policy, test it
- Which layers: last hidden default; multi-layer export only if a task proves need

**`graph/io.py`** — M2

```python
load_structure(path_or_id: str) -> gemmi.Structure
# Accepts local .pdb/.cif paths AND 4-char PDB IDs / AFDB UniProt accessions.# Downloads to data/cache/structures. AlphaFold pLDDT must survive into build().
```

Open questions: asymmetric unit vs assembly (lean: asymmetric unit); missing residues= mask node + log (lean yes); one graph per chain (v1: yes).

**`graph/build.py`** — M2

```python
build_graph(    structure, *,    edge_knn: int = 10,    edge_radius: float = 10.0,           # Ångströms    rbf_centers: ... ,                    # from configs/graph) -> torch_geometric.data.Data
```

Node features: pLM embedding (if seq) · residue code · pLDDT.Edges: union of sequence-adjacency (i, i±1) and kNN/radius on Cα distance.Edge features: RBF(Cα distance) · sequence offset.**INVARIANCE GUARANTEE:** every feature is a function of distances/angles only.Rotating or translating coordinates must not change any feature. Enforced by tests.

**`fuse/`** — M4

```python
graft(node_feats: Tensor, embeddings: Tensor, mode: str) -> Tensor# mode: "concat" | "project". Dimension mismatches resolved HERE, not in encoders.
```

Open questions: linear vs 2-layer projection (start linear); frozen-only in v1.

**`encoders/*.py`** — M3. Each: `forward(batch) -> per-residue logits` plus a`pool() -> per-protein embedding`. Three files, \~250 lines each, math commented.

**`tasks/secondary_structure.py`** — M3Labels from DSSP (system dep `mkdssp`; document install, degrade gracefully).
Metrics: Q3/Q8 accuracy + per-class F1 via scikit-learn. 8-state → 3-state mapping is a named, tested function, not an inline lambda.

## 8. Hydra configuration system

Root `configs/config.yaml` (defaults list — this *is* the architecture of a run):

```yaml
defaults:  - profile: cpu  - features: seq            # THE flag: seq | struct | both  - embed: esm2_t33_650M  - graph: default  - model: mlp  - task: secondary_structure  - _seed: 42hydra:  run:    dir: outputs/${task.name}/${features.name}/${now:%Y-%m-%d_%H-%M-%S}
```

Group files must be self-contained and heavily commented; each sets `name:` sooutput dirs and benchmark tables are human-readable.

Commands that must work verbatim by M4 (test them in CI as a smoke check):

```bash
uv run ligase embed  embed=esm2_t33_650M                       # fill the cache
uv run ligase build graph=default                               # graphs → data/cache/graphs
uv run ligase train task=secondary_structure features=seq       # CPU, minutes
uv run ligase train task=secondary_structure features=both profile=gpu
uv run ligase eval   task=secondary_structure features=both     # metrics from existing outputs
```

`profile=cpu` is the default: fp32, small batches, tiny splits — everything runson a laptop. `profile=gpu` switches dtype/batch/split sizes only. Hardware isnever load-bearing for correctness.

## 9. Data policy

- **Public data only**: PDB, AlphaFold DB, standard benchmark sets (e.g. CB513 andfriends for secondary structure; ProteinGym subsets later). No lab data. No exceptions.
- All downloads go through `graph/io.py` / task loaders into `data/cache/`, never committed.
- Record provenance: every dataset used in a benchmark table gets a line in`docs/data_sources.md` (source URL, version/date downloaded, license). Reviewers andinterviewers both ask this.
- Cache keys are content hashes. Same input = cache hit = zero recompute (tested in M1).

## 10. Testing strategy — verify the math, not just the plumbing

These acceptance tests are written **now**, in `tests/test_invariance.py`, skippeduntil M2. They may not be quietly deleted or weakened:

1. **Rotation invariance** — for random rotation R:`build_graph(R·coords)` produces node/edge features identical (atol 1e-5) to`build_graph(coords)`. Holds because every feature is a distance or angle.
2. **Permutation equivariance** — permuting residue order permutes per-residueoutputs identically; pooled output is unchanged.
3. **Batching equivalence** — encoding two graphs separately equals encoding thembatched (guards PyG `Batch` correctness: masks, offsets, pooling).
4. **Cache round-trip** — `embed_sequences` twice yields identical arrays; the secondcall performs zero model forward passes.
5. **Determinism** — same seed ⇒ bitwise-identical metrics on the CPU profile.

The test suite is the library's argument. An interviewer who reads these testsunderstands the design without reading a line of model code.

## 11. Hard rules

1. `import ligase` must never import torch. Heavy imports are lazy, inside functions/submodules.
2. No SE(3)-equivariant layers, no generative modeling (see Section 5).
3. No foundation-model training or fine-tuning (Section 4).
4. Every feature is invariant: distances and angles only.
5. Public data only; nothing from any employer or lab (Section 9).
6. Files < \~300 lines; every public function has a typed signature and a docstring.
7. No result in the README without a reproducing command from Section 3/8.

## 12. Quality gate (run before every "done")

```bash
uv sync --all-extras          # env matches uv.lock
uv run ruff check . && uv run ruff format --check .
uv run mypy src               # gradually strict; no NEW suppressions without comment
uv run pytest -m "not gpu" --cov=ligase --cov-report=term
```

CI runs exactly this on push/PR, Python 3.10–3.12. A task is not done until CI is green.

## 13. Conventions

- **Commits:** Conventional Commits (`feat(embed): hash-addressed h5 cache`).Changelog section + commit = one unit of work.
- **Branches:** `m1-embed-cache` style, short-lived, squash-merged.
- **Style:** ruff IS the style guide. Line length 100.
- **Docstrings:** NumPy-style, written for mkdocstrings. The math goes in the docstring.
- **Naming:** functions `snake_case`, configs kebab-agnostic but file names `snake_case.yaml`.

## 14. Getting started (the newcomer test)

```bash
git clone https://github.com/<you>/ligase && cd ligase
uv sync --all-extras
uv run pytest -m "not gpu"
uv run ligase --help
```

If any of those four lines fails for a newcomer, that is a P0 bug. This block isalso the README's install section from v0.1 onward.

## 15. Landscape (keep current; update if a competitor ships)

| **ProjectWhat it isLigase's relation** |                                 |                                                     |
| -------------------------------------- | ------------------------------- | --------------------------------------------------- |
| fair-esm / ESM                         | The models                      | We use them; we don't reimplement                   |
| Graphein                               | Deep, broad graph construction  | We ship one opinionated graph, not a menu           |
| ProteinWorkshop                        | Heavyweight benchmark framework | We ship one readable path, not a framework          |
| BioPython                              | Parsing swiss-army knife        | gemmi covers our structural needs with less surface |

## 16. Changelog ← APPEND A ROW FOR EVERY COMPLETED TASK

| **DateTask (from ledger)What changedDecisions made / open questions settledCommit** |   |   |   |   |
| ----------------------------------------------------------------------------------- | - | - | - | - |
| —                                                                                   | — | — | — | — |

## 17. Open questions ledger

| **#QuestionCurrent leanSettled in** |                                             |                           |    |
| ----------------------------------- | ------------------------------------------- | ------------------------- | -- |
| Q1                                  | ESM backend: HF transformers vs fair-esm    | HF transformers           | M1 |
| Q2                                  | Cache layout (per-model vs per-sequence h5) | benchmark 1k seqs, decide | M1 |
| Q3                                  | Non-canonical residue policy                | map-to-X + log            | M1 |
| Q4                                  | Missing residues: mask node vs skip         | mask + log                | M2 |
| Q5                                  | Graft projection: linear vs MLP             | linear                    | M4 |
| Q6                                  | pandas vs polars for benchmark tables       | pandas (boring wins)      | M4 |
