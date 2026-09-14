# Ligase

> Join what protein ML keeps apart: sequence language models and structure graphs.

Ligase is a small, readable library for protein representation learning. Embed a sequence with a protein language model, build a residue graph from structure, graft the two together, predict, and, above all, find out whether structure was worth it.

It is model-agnostic by design: any protein language model or embedding function that satisfies one small protocol (Section 5.1) gets the entire pipeline — caching, conformance tests, tasks, and the honest baseline comparison. **Ligase is a benchmark harness for protein representations that happens to ship good defaults.**

**Status: PRE-ALPHA.** The API will break. The direction will not.

---

## 0. READ THIS FIRST — protocol for contributors (human or AI)

This README is the single source of truth. It is a living document: **every completed task must update Section 2 (Status dashboard) and Section 16 (Changelog) in the same commit that completes the task.**

Working rules:

1. Read Section 2, pick the lowest-numbered unchecked task. Do not skip ahead.
2. Implement to the *contract* written in the relevant folder's Section 7 entry. Contracts may evolve, but only with a Changelog entry explaining why.
3. Respect the hard rules (Section 11). They are non-negotiable.
4. Before declaring a task done, run the full quality gate (Section 12) — all green.
5. Update: status checkboxes, changelog row, and any contract/open-question text that the work settled. Commit with a Conventional Commit message.
6. Do not add dependencies without a Changelog entry and a one-line rationale.
7. Files stay under ~300 lines. If a file grows past that, split it and note it.
8. Never import lab data, lab code, or anything non-public. Public data only (Section 9).

---

## 1. Why this exists

Protein representation learning is fragmented. ESM gives you embeddings, Graphein builds graphs, ProteinWorkshop runs benchmarks — and every paper repo glues them together differently, once, invisibly. Meanwhile the field's most common overclaim, "structure helped!", is almost never tested against a decent structure-free baseline on identical splits.

Ligase makes that test a one-flag operation and treats the result as the product.

## 2. Status dashboard ← UPDATE THIS SECTION AS TASKS COMPLETE

### Milestones

| # | Milestone | Deliverable | Done when | Status |
|---|---|---|---|---|
| M0 | Skeleton | Repo layout, uv env, CI, this README | CI green from first commit; `uv run pytest` passes locally | ☑ |
| M1 | `embed` | ESM-2 extraction → disk cache + protocol extraction | round-trip + zero-recompute + determinism + conformance tests pass | ☐ |
| M2 | `graph` | Structures → PyG residue graphs | rotation-invariance & permutation-equivariance tests pass (unskipped) | ☐ |
| M3 | `encoders` + task | MLP / GVP / GearNet-lite + secondary structure | trains end-to-end on a small CPU-able config; Q3 > seq-only baseline reported | ☐ |
| M4 | Fusion + honesty | Ablation runs, benchmark table, docs site, plugin docs + conformance CLI | the one-flag promise works verbatim from README commands | ☐ |
| M5 | Launch | HF Space + launch post | a stranger runs the demo unaided | ☐ |

### Task ledger

| Task | Milestone | Status | Evidence (commit/PR) | Notes |
|---|---|---|---|---|
| Skeleton + README + uv + CI | M0 | ☑ | a29d036  | this commit |
| Hydra config skeleton (M1-relevant groups) | M0/M1 | ☐ | — | Section 8 |
| `embed/esm.py` extractor (implements `EmbeddingSource`) | M1 | ☐ | — | Section 5.1, 7.1 |
| `embed/protocol.py` — protocol extracted from working code | M1 | ☐ | — | rule of two: extract, don't invent |
| Cache decorator wraps any `EmbeddingSource` | M1 | ☐ | — | `utils/caching.py` |
| Cache tests (round-trip, zero-recompute) | M1 | ☐ | — | |
| Conformance suite, parametrized over sources | M1 | ☐ | — | `tests/test_conformance.py` |
| `graph/io.py` structure loader | M2 | ☐ | — | |
| `graph/build.py` graph builder | M2 | ☐ | — | |
| Invariance tests unskipped & green | M2 | ☐ | — | the moment of truth |
| `encoders/mlp.py` | M3 | ☐ | — | the conscience of the library |
| `encoders/gvp.py` | M3 | ☐ | — | |
| `encoders/gearnet.py` | M3 | ☐ | — | |
| `tasks/secondary_structure.py` | M3 | ☐ | — | |
| Ablation runner + benchmark table | M4 | ☐ | — | |
| Plugin docs + `ligase conformance` CLI (rule of two) | M4 | ☐ | — | ships with second implementation (Ankh) |
| Docs site (mkdocs) live | M4 | ☐ | — | |
| HF Space | M5 | ☐ | — | |
| Launch post | M5 | ☐ | — | |

---

## 3. The one-flag promise

The entire reason this library exists, expressed as three commands:

```bash
uv run ligase train task=secondary_structure features=seq     # pooled ESM-2 → MLP
uv run ligase train task=secondary_structure features=struct  # geometry → GNN
uv run ligase train task=secondary_structure features=both    # the graft
```

And it works unchanged on your model (Section 5.1):

```bash
uv run ligase train task=secondary_structure embed=my_plm features=seq   # your PLM, our harness
```

Same splits, same seed, same metrics. The ablation table falls out. Every performance claim in the docs must be reproducible from one of these runs. If a comparison can't be run this way, it doesn't go in the README.

## 4. Training policy (read before touching training code)

The maintainer does not want to train models. Honor that:

- **Foundation models: never trained, never fine-tuned.** ESM-2 is used frozen, inference-only.
- **Task heads are the only training that happens**: ≤ ~1M parameters, minutes on CPU, default `profile=cpu`. GPU is an optional convenience, never a requirement (Section 8, profiles).
- No large-scale experiments, no hyperparameter sweeps by default. If an experiment needs more than a coffee break on CPU, it needs a Changelog justification.

This policy binds Ligase's *maintainers*, not its users. You may bring your own fine-tuned PLM as an `EmbeddingSource` (Section 5.1) and evaluate it here — the harness is model-agnostic; the "we don't train" rule governs what we ship and run by default.

## 5. Scope

**In scope (v1)**

- Model-agnostic core: the `EmbeddingSource` protocol, the cache decorator, and the parametrized conformance suite (Section 5.1)
- ESM-2 (and optionally Ankh) embedding extraction: batched, bf16 on GPU / fp32 on CPU, disk-cached, hash-addressed by sequence
- PDB / mmCIF → PyTorch Geometric residue graphs: sequence edges, kNN + radius edges on Cα distance, RBF distance features, pLDDT as a node feature
- Three encoders, each in one readable file: pooled-embedding MLP (the honest baseline), GVP-GNN (invariant mode), GearNet-lite
- Tasks: 3-/8-state secondary structure first; binding-site prediction next
- Tests that verify the *math*: permutation equivariance, rotation invariance, batching equivalence, cache round-trips

**Deliberately out of scope**

- SE(3)-equivariance / vector neurons — invariant features (distances, angles) cover the industrial majority; this is a design decision, not an omission
- Generative modeling (diffusion, flow matching, inverse folding) — explicitly excluded
- Training or fine-tuning foundation models (Section 4 — binds us, not users bringing their own weights)
- MSA-based models, protein–ligand complexes, docking, antibody-specific tools (v2 maybe)

### 5.1 The model-agnostic core (why Ligase is a harness, not a wrapper)

Ligase's machinery — splits, graphs, tasks, metrics, honest baselines — is useful to anyone holding a protein representation, not just to users of ESM-2. So the core is model-agnostic:

**One small protocol is the entire plugin surface.** Anything that can do this can be benchmarked:

```python
from typing import Protocol
import numpy as np


class EmbeddingSource(Protocol):
    """Anything that turns sequences into per-residue embeddings."""

    @property
    def dim(self) -> int: ...

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]:
        """Returns {seq: (L, D) float16}.

        ALIGNMENT CLAUSE: L == len(seq), always — BOS/EOS and other
        special tokens stripped, exactly one vector per residue,
        aligned 1:1 with the input string. X/B/U/Z each count as one
        residue. This clause is where most DIY evaluation scripts
        silently misalign; the conformance suite (Section 10, item 6)
        enforces it.
        """
```

**The cache is a decorator around the protocol, not a feature of ESM-2.** `cached(source)` wraps *any* `EmbeddingSource` and gives it hash-addressed caching, zero-recompute, and the round-trip tests for free. The cache knows the protocol, not the model.

**Hydra makes plugging in a zero-code-change operation.** Embedding configs are objects, not names — each `configs/embed/*.yaml` instantiates a source via `_target_`:

```yaml
# configs/embed/my_plm.yaml — lives in the USER's repo, zero Ligase changes
_target_: mypackage.MyProteinLM   # any class satisfying EmbeddingSource
checkpoint: weights/my_finetuned.ckpt
layer: 24
```

Then the entire one-flag promise runs on their model:

```bash
uv run ligase train task=secondary_structure embed=my_plm features=seq
uv run ligase train task=secondary_structure embed=my_plm features=both
```

Same splits, same seed, same baseline — comparing a custom PLM against ESM-2 is one command. That is the product.

**Conformance is a test suite, not an honor system.** Any `EmbeddingSource` — built-in or third-party — must pass: alignment (incl. X/B/U/Z and length-1), determinism (bitwise), cache round-trip with zero recompute. Structure feature sources get a second protocol (`StructureSource`) with a rotation-invariance conformance check — on the same rule of two: only when a second real implementation exists.

The boundary:

| Ours (the moat) | Theirs (pluggable) |
|---|---|
| Graph construction + invariance guarantee | The PLM / embedding function |
| Tasks, splits, metrics, DSSP plumbing | Their fine-tuned weights (fine — Section 4 binds us, not them) |
| Honest baselines (MLP / GVP / GearNet-lite) | Optionally, their structure feature source (second protocol, later) |
| Caching, determinism, configs | Optionally, their own task (v2 — don't rush it) |

**The rule of two:** the public extensibility surface — plugin docs section and the `ligase conformance` CLI — ships only when a second concrete implementation exists (Ankh counts; likely M4). The protocol itself is extracted in M1 from the working ESM-2 code, because an interface designed before its second implementation is almost always wrong.

### 5.2 The encoder contract (bring your own GNN)

Symmetric with 5.1: the encoder consuming our graphs is pluggable. Input schema is pinned by `build_graph` (Section 7.1): `data.x`, `data.edge_index`, `data.edge_attr` (RBF + sequence offset),`data.pos`  (Cα coordinates — may be ignored by graph-free encoders like the MLP, never assumed absent).

Instantiation via Hydra `_target_` in `configs/model/*.yaml` — zero library changes, same as 5.1
Encoder conformance (Section 10): permutation equivariance, batching equivalence, and rotationinvariance of outputs for encoders that consume coordinates
Rule of two: satisfied at birth by the built-in trio; protocol still extracted from workingcode in M3, plugin docs ship in M4 with the conformance CLI
Full-harness composition: user's PLM (5.1) + user's encoder (5.2) on our graphs, splits, metrics
**Task admission rule.** A task joins the suite when it (1) has public, license-clear labels,(2) trains a ≤1M-param head on CPU in minutes, and (3) probes a facet of representation qualityexisting tasks don't. Growth axis: local geometry → functional sites → global fitness(see Changelog for the candidate table).

## 6. Tech stack (chosen deliberately — this repo doubles as a job portfolio)

| Concern | Tool | Why it's here |
|---|---|---|
| Env & package management | **uv** (`uv.lock` committed) | Reproducible envs in seconds; the modern standard. `uv sync` must be all a newcomer needs |
| Python | 3.11 (CI matrix: 3.10–3.12) | Boring, fast, broadly compatible |
| Configs & run management | **Hydra** (config groups + overrides) | The one-flag promise *is* Hydra composition; also standard in industrial ML |
| Tensors / models | torch, torch-geometric | Industry default |
| Structure parsing | gemmi | One dependency does PDB + mmCIF correctly |
| Caching | h5py (hash-addressed) | Zero-recompute guarantee, testable |
| Metrics | scikit-learn | Q3/Q8, F1 — standard, auditable |
| Config/Metric tables | pandas (or polars, decide in M4) | Benchmark tables |
| Tests | pytest + pytest-cov | The math tests are the differentiator |
| Lint + format | ruff (one tool, both jobs) | Zero-config credibility |
| Types | mypy (gradually strict) | Types on public contracts; strict per-module as it stabilizes |
| Hooks | pre-commit | Quality gates run before commit, not after review |
| CI | GitHub Actions (uv-native action) | Lint + types + tests on every push; matrix 3.10–3.12 |
| Docs | mkdocs-material (+ mkdocstrings) | Deployed to GH Pages in M4; API docs generated from docstrings |
| CLI | `[project.scripts]` entry point via Hydra's `@hydra.main` | `uv run ligase ...` works from day one |
| Experiment tracking | CSV/JSON by default; wandb as optional extra | No accounts required to reproduce results |

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
│   ├── embed/                  ←   one yaml per embedding SOURCE — each instantiates an
│   │                            ←   EmbeddingSource via _target_ (esm2_t33_650M.yaml, my_plm.yaml, …)
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
│   │   ├── __init__.py         ←   exports EmbeddingSource + cached()
│   │   ├── protocol.py         ← (M1) the one-interface plugin surface (~50 lines; Section 5.1)
│   │   └── esm.py              ← (M1) ESM-2 implementation of the protocol; cache via decorator
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
│   │   ├── protocol.py             ← (M3) GraphEncoder contract, extracted from the working trio
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
│       ├── caching.py           ← (M1) cached(): decorator over ANY EmbeddingSource
│       └── logging.py           ← (M3) tiny CSV/JSON metric logger — no accounts, ever
│
├── tests/
│   ├── conftest.py              ← fixtures: tiny toy structure, 5 fake sequences, tmp caches,
│   │                             ←   a MOCK EmbeddingSource (tiny dim) for CI conformance runs
│   ├── test_smoke.py            ← import policy, version, determinism — green at M0
│   ├── test_conformance.py      ← (M1) parametrized suite any EmbeddingSource must pass
│   └── test_invariance.py       ← M2 acceptance math (Section 10); skipped, but the math is written down
│
├── examples/                    ← notebooks land M3+: 01_embed_and_cache, 02_build_graphs,
│                                 ← 03_secondary_structure_ablation
├── docs/                        ← (M4) mkdocs; API reference generated from docstrings
└── data/                        ← gitignored. Download + cache root. Never commit data.
```

**Where things stand right now (pre-M0 complete):** the package skeleton, empty module files, and `utils/seeding.py` exist; `configs/`, `uv.lock`, `.pre-commit-config.yaml`, `cli.py`, and `docs/` do not yet. Closing that gap — plus a green CI run — is exactly what M0 is (Section 2). Every `(M#)` tag above should flip to plain text, and its checkbox in the Task ledger should flip to ☑, in the same commit.

### 7.1 Module contracts (implement exactly these; signatures are the API)

**`embed/protocol.py`** — M1 (extracted, not invented)

```python
class EmbeddingSource(Protocol): ...


# The FULL contract text — including the alignment clause — lives in Section 5.1.
# Extraction rule: write esm.py first, then lift the interface out of the working
# code. This file should be ~50 lines and import nothing heavier than numpy.
```

**`embed/esm.py`** — M1 (implements the protocol; see Section 5.1)

```python
class ESM2Source:                    # satisfies EmbeddingSource
    def __init__(self, model_name: str = "esm2_t33_650M", layer: int | None = None): ...

    @property
    def dim(self) -> int: ...

    def embed(self, seqs: list[str]) -> dict[str, np.ndarray]: ...


def embed_sequences(seqs, model_name=..., layer=None, cache_dir=...) -> dict[str, np.ndarray]
# functional sugar over the class: cached(ESM2Source(...)).embed(seqs)
# {seq: (L, D) float16}; L = residue count
```

Open questions (decide while building M1, then move answers into this README):

- Backend: HF `transformers` (recommended: tokenizer + models + AutoModel in one) vs fair-esm
- Cache layout: one h5 per model vs per sequence (benchmark on 1k seqs, pick, document)
- Non-canonical residues (X, U, B, Z, gaps): map-and-log vs pass-through — pick one policy, test it
- Which layers: last hidden default; multi-layer export only if a task proves need

**`utils/caching.py`** — M1

```python
def cached(source: EmbeddingSource, cache_dir: Path = ...) -> EmbeddingSource
# Wraps ANY EmbeddingSource with the hash-addressed h5 cache: zero-recompute on
# repeats, round-trip tested. This decorator is what makes the library
# model-agnostic rather than "ESM with extras".
```

**`tests/test_conformance.py`** — M1 (parametrized; the trust engine)

```python
# @pytest.mark.parametrize over sources: [mock_source (fixture, tiny dim), esm2_tiny]
# CI conformance runs against the MOCK source (and optionally, gpu-marked, the
# smallest real ESM-2) — CI must NEVER download 650M-parameter weights.


def test_alignment(source): ...  # L == len(seq); X/B/U/Z; length-1; specials stripped
def test_determinism(source): ...  # bitwise-identical on repeat
def test_cache_roundtrip(source): ...  # identical arrays; second call = zero forwards


# Third-party sources run this same file against their implementation — that is
# what makes a green conformance checkmark mean something.
```

**`graph/io.py`** — M2

```python
load_structure(path_or_id: str) -> gemmi.Structure
# Accepts local .pdb/.cif paths AND 4-char PDB IDs / AFDB UniProt accessions.
# Downloads to data/cache/structures. AlphaFold pLDDT must survive into build().
```

Open questions: asymmetric unit vs assembly (lean: asymmetric unit); missing residues = mask node + log (lean yes); one graph per chain (v1: yes).

**`graph/build.py`** — M2

```python
build_graph(
    structure, *,
    edge_knn: int = 10,
    edge_radius: float = 10.0,           # Ångströms
    rbf_centers: ... ,                    # from configs/graph
) -> torch_geometric.data.Data
```

Node features: pLM embedding (if seq) · residue code · pLDDT. Edges: union of sequence-adjacency (i, i±1) and kNN/radius on Cα distance. Edge features: RBF(Cα distance) · sequence offset. **INVARIANCE GUARANTEE:** every feature is a function of distances/angles only. Rotating or translating coordinates must not change any feature. Enforced by tests.

**`fuse/`** — M4

```python
graft(node_feats: Tensor, embeddings: Tensor, mode: str) -> Tensor
# mode: "concat" | "project". Dimension mismatches resolved HERE, not in encoders.
# Hidden dims are arbitrary by design — a third-party EmbeddingSource may return
# any D; fuse absorbs it (Section 5.1).
```

Open questions: linear vs 2-layer projection (start linear); frozen-only in v1.

**`encoders/*.py`** — M3. Each: `forward(batch) -> per-residue logits` plus a `pool() -> per-protein embedding`. Three files, ~250 lines each, math commented.

**`tasks/secondary_structure.py`** — M3. Labels from DSSP (system dep `mkdssp`; document install, degrade gracefully). Metrics: Q3/Q8 accuracy + per-class F1 via scikit-learn. 8-state → 3-state mapping is a named, tested function, not an inline lambda.

## 8. Hydra configuration system

Root `configs/config.yaml` (defaults list — this *is* the architecture of a run):

```yaml
defaults:
  - profile: cpu
  - features: seq            # THE flag: seq | struct | both
  - embed: esm2_t33_650M
  - graph: default
  - model: mlp
  - task: secondary_structure
  - _self_

seed: 42

hydra:
  run:
    dir: outputs/${task.name}/${features.name}/${now:%Y-%m-%d_%H-%M-%S}
```

Group files must be self-contained and heavily commented; each sets `name:` so output dirs and benchmark tables are human-readable.

Embedding configs are objects, not names: each `configs/embed/*.yaml` instantiates an `EmbeddingSource` via Hydra's `_target_` (Section 5.1). A third-party model therefore needs zero library changes — drop a yaml into the group (or point at one in the user's repo with `embed=my_plm`) and every command below works on it unchanged.

Commands that must work verbatim by M4 (test them in CI as a smoke check):

```bash
uv run ligase embed  embed=esm2_t33_650M                       # fill the cache
uv run ligase build graph=default                               # graphs → data/cache/graphs
uv run ligase train task=secondary_structure features=seq       # CPU, minutes
uv run ligase train task=secondary_structure features=both profile=gpu
uv run ligase eval   task=secondary_structure features=both     # metrics from existing outputs
```

`profile=cpu` is the default: fp32, small batches, tiny splits — everything runs on a laptop. `profile=gpu` switches dtype/batch/split sizes only. Hardware is never load-bearing for correctness.

## 9. Data policy

- **Public data only**: PDB, AlphaFold DB, standard benchmark sets (e.g. CB513 and friends for secondary structure; ProteinGym subsets later). No lab data. No exceptions.
- All downloads go through `graph/io.py` / task loaders into `data/cache/`, never committed.
- Third-party embedding configs may point at local checkpoints (`my_plm.yaml`) — those paths are the user's business; never commit weights, ours or theirs.
- Record provenance: every dataset used in a benchmark table gets a line in `docs/data_sources.md` (source URL, version/date downloaded, license). Reviewers and interviewers both ask this.
- Cache keys are content hashes. Same input = cache hit = zero recompute (tested in M1).

## 10. Testing strategy — verify the math, not just the plumbing

These acceptance tests are written **now**, in `tests/test_invariance.py`, skipped until M2. They may not be quietly deleted or weakened:

1. **Rotation invariance** — for random rotation R: `build_graph(R·coords)` produces node/edge features identical (atol 1e-5) to `build_graph(coords)`. Holds because every feature is a distance or angle.
2. **Permutation equivariance** — permuting residue order permutes per-residue outputs identically; pooled output is unchanged.
3. **Batching equivalence** — encoding two graphs separately equals encoding them batched (guards PyG `Batch` correctness: masks, offsets, pooling).
4. **Cache round-trip** — `embed_sequences` twice yields identical arrays; the second call performs zero model forward passes.
5. **Determinism** — same seed ⇒ bitwise-identical metrics on the CPU profile.
6. **Conformance (parametrized)** — any `EmbeddingSource` passes `tests/test_conformance.py`: alignment (`L == len(seq)`, including X/B/U/Z and length-1), determinism (bitwise), and cache round-trip with zero recompute. Built-in sources run it in CI (against a mock/tiny fixture — never 650M weights); third-party sources run the same file; `ligase conformance` (M4, rule of two) exposes it as a CLI.
7. **Encoder conformance (parametrized)** — built-in and third-party encoders pass the sameinvariance suite: permutation equivariance, batching equivalence; rotation invariance whenthe encoder consumes coordinates.

The test suite is the library's argument. An interviewer who reads these tests understands the design without reading a line of model code.

## 11. Hard rules

1. `import ligase` must never import torch. Heavy imports are lazy, inside functions/submodules.
2. No SE(3)-equivariant layers, no generative modeling (see Section 5).
3. No foundation-model training or fine-tuning (Section 4).
4. Every feature is invariant: distances and angles only.
5. Public data only; nothing from any employer or lab (Section 9).
6. Files < ~300 lines; every public function has a typed signature and a docstring.
7. No result in the README without a reproducing command from Section 3/8.
8. **Rule of two.** No public plugin API (plugin docs, `ligase conformance` CLI) until a second concrete implementation exists. Interfaces are extracted from working code, never designed up front (Section 5.1).

## 12. Quality gate (run before every "done")

```bash
uv sync --all-extras          # env matches uv.lock
uv run ruff check . && uv run ruff format --check .
uv run mypy src               # gradually strict; no NEW suppressions without comment
uv run pytest -m "not gpu" --cov=ligase --cov-report=term
```

CI runs exactly this on push/PR, Python 3.10–3.12. A task is not done until CI is green.

## 13. Conventions

- **Commits:** Conventional Commits (`feat(embed): hash-addressed h5 cache`). Changelog section + commit = one unit of work.
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

If any of those four lines fails for a newcomer, that is a P0 bug. This block is also the README's install section from v0.1 onward.

## 15. Landscape (keep current; update if a competitor ships)

| Project | What it is | Ligase's relation |
|---|---|---|
| fair-esm / ESM | The models | We use them; we don't reimplement |
| Graphein | Deep, broad graph construction | We ship one opinionated graph, not a menu |
| ProteinWorkshop | Heavyweight benchmark framework | We ship one readable path, not a framework |
| BioPython | Parsing swiss-army knife | gemmi covers our structural needs with less surface |
| MTEB | Benchmark harness for text embeddings | The playbook: the harness, not the models, is the moat |
| TAPE / PEER | Protein benchmark suites | Precedent and comparison point; we optimize for readability + the ablation flag they lack |

## 16. Changelog ← APPEND A ROW FOR EVERY COMPLETED TASK

| Date | Task (from ledger) | What changed | Decisions made / open questions settled | Commit |
|---|---|---|---|---|
| September 13 2026 | README patch: model-agnostic core | Sections 0/2/3/4/5(+5.1)/7/8/9/10/11/15/17 updated; ledger +4 tasks; conformance suite + mock fixture added to tree | Harness framing adopted; rule of two becomes Hard Rule 8; Q7/Q8 opened; Hydra defaults-list syntax fixed (`- _seed: 42` → plain `seed: 42`) | — |
| September 13 2026 | Skeleton + README + uv + CI | — | created the main skeleton | (see task ledger) |
| September 13 2026 | README patch: encoder contract + task admission rule | Sections 5.2/7/10/15/17; tree +1 file | GraphEncoder protocol adopted; task growth axis recorded (Q9, Q10) | |
| — | — | — | — | — |

## 17. Open questions ledger

| # | Question | Current lean | Settled in |
|---|---|---|---|
| Q1 | ESM backend: HF transformers vs fair-esm | HF transformers | M1 |
| Q2 | Cache layout (per-model vs per-sequence h5) | benchmark 1k seqs, decide | M1 |
| Q3 | Non-canonical residue policy | map-to-X + log | M1 |
| Q4 | Missing residues: mask node vs skip | mask + log | M2 |
| Q5 | Graft projection: linear vs MLP | linear | M4 |
| Q6 | pandas vs polars for benchmark tables | pandas (boring wins) | M4 |
| Q7 | When does the public plugin API ship (plugin docs + `ligase conformance` CLI)? | With the second implementation (Ankh) — likely M4 | M4 |
| Q8 | Second protocol `StructureSource` for external structure feature extractors? | On demand only; rule of two applies | — |
| Q9 | Encoder plugin docs timing | protocol extracted M3 (trio satisfies rule of two); docs + CLI in M4 | M3 || Q10 | Task-suite growth order after SS | binding sites (v1.5) → ProteinGym DMS (v2) | v1.5 |

---

*This README is the project. Code is its implementation.*

Handoff prompt for the next instance stays the same: paste this and say "Read Section 0, pick the first unchecked task in Section 2, follow the Definition of Done in Section 12." The one thing worth flagging verbally to it: M1's protocol extraction happens *after* `esm.py` works — if the next instance starts by writing `protocol.py` first, gently point it at Hard Rule 8. That ordering discipline is what keeps the abstraction honest.
