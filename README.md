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
| M1 | `embed` | ESM-2 extraction → disk cache + protocol extraction | round-trip + zero-recompute + determinism + conformance tests pass | ☑ |
| M2 | `graph` | Structures → PyG residue graphs | rotation-invariance & permutation-equivariance tests pass (unskipped) | ☑ |
| M3 | `encoders` + task | MLP / GVP / GearNet-lite + secondary structure | trains end-to-end on a small CPU-able config; Q3 vs seq-only baseline reported | ☑ |
| M4 | Fusion + honesty | Ablation runs, benchmark table, docs site, plugin docs + conformance CLI | the one-flag promise works verbatim from README commands | ☐ |
| M5 | Launch | HF Space + launch post | a stranger runs the demo unaided | ☐ |

### Task ledger

| Task | Milestone | Status | Evidence | Notes |
|---|---|---|---|---|
| Skeleton + README + uv + CI | M0 | ☑ | a29d036 | |
| Hydra config skeleton | M0/M1 | ☑ | <hash> | configs inside package (pkg:// resolution) |
| `embed/esm.py` (implements protocol) | M1 | ☑ | <hash> | Q1: HF transformers; sanitize+overlength policies |
| `embed/protocol.py` — extracted from working code | M1 | ☑ | <hash> | Hard Rule 8 order kept; `cache_id` added under pressure |
| Cache decorator over any source | M1 | ☑ | <hash> | one h5 per cache_id (Q2); torn-write = miss |
| Cache tests (round-trip, zero-recompute) | M1 | ☑ | <hash> | |
| Conformance suite, parametrized | M1 | ☑ | <hash> | mock + cached-mock; real ESM behind @slow |
| `utils/seqio.py` loaders (FASTA/TXT/CSV) | M1 | ☑ | <hash> | raw strings out; sanitization lives in esm.py |
| CLI `embed` + dispatch | M1 | ☑ | <hash> | |
| `graph/io.py` (PDB/AFDB, CIF-everywhere) | M2 | ☑ | <hash> | AFDB v4 pinned; atomic downloads; pLDDT policy |
| `graph/build.py` + GraphParams | M2 | ☑ | <hash> | Q4: missing-CA skip+log; frozen hashable params |
| Invariance tests unskipped & green | M2 | ☑ | <hash> | the guarantee demonstrated |
| `graph/cache.py` — keyed + atomic graph cache | M2 | ☑ | <hash> | cache_id lesson one layer up; extracted at 2nd consumer |
| Structure manifests + batch build | M2 | ☑ | <hash> | txt/csv; params per-run, not per-row |
| `configs/graph/default.yaml` | M2 | ☑ | <hash> | knn 10 · radius 10 Å · RBF 20 @ σ 2.5 |
| CLI `build` | M2 | ☑ | <hash> | |
| `encoders/mlp.py` | M3 | ☑ | <hash> | structure-blind by construction, proven bitwise |
| `encoders/gvp.py` | M3 | ☑ | <hash> | invariant mode: vector stream amputated; pos never read |
| `encoders/gearnet.py` | M3 | ☑ | <hash> | edge states + relational bias; type from offset column |
| `encoders/protocol.py` — GraphEncoder extracted | M3 | ☑ | <hash> | trio first, lift after; mean_pool dedup ×3 |
| Encoder conformance suite (parametrized) | M3 | ☑ | <hash> | M2 batching skip absorbed; admission gate for M4 CLI |
| `tasks/secondary_structure.py` | M3 | ☑ | <hash> | alignment contract via shared traversal; mkdssp lazy |
| `tasks/dataset.py` — assembly + alignment triple | M3 | ☑ | <hash> | fp16→fp32 upcast lives in attach_embeddings |
| `tasks/splits.py` — seeded structure splits | M3 | ☑ | <hash> | homology leakage documented as v1 limitation |
| Training loop + features wiring | M3 | ☑ | <hash> | encoder out_dim = class head (D2 amended) |
| `train`/`eval` CLI + config groups | M3 | ☑ | <hash> | seq: Q3=<acc>; struct/gvp: Q3=<acc> — 5-structure demo set |
| Ablation runner + benchmark table | M4 | ☐ | — | full CB513-class set + docs/data_sources.md provenance |
| Plugin docs + `ligase conformance` CLI | M4 | ☐ | — | ships with Ankh (second source; Q7) |
| Docs site (mkdocs) live | M4 | ☐ | — | |
| HF Space | M5 | ☐ | — | |
| Launch post | M5 | ☐ | — | |

---

## 3. The one-flag promise

The entire reason this library exists, expressed as three commands:

```bash
uv run ligase train task=secondary_structure features=seq model=mlp embed=esm2_t6_8M
uv run ligase train task=secondary_structure features=struct model=gvp
uv run ligase train task=secondary_structure features=both model=gearnet
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

    @property
    def cache_id(self) -> str: ...

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
.
├── LICENSE
├── Makefile
├── README.md
├── data
│   └── cache
│       ├── embeddings
│       ├── graphs
│       └── structures
├── docs
├── examples
│   └── structures_examples.txt
├── outputs
│   ├── cli
│   ├── embed
│   └── secondary_structure
│       ├── seq
│       │   └── mlp
│       └── struct
│           └── gvp
├── pyproject.toml
├── src
│   └── ligase
│       ├── __init__.py
│       ├── cli.py
│       ├── configs
│       │   ├── __init__.py
│       │   ├── config.yaml
│       │   ├── embed
│       │   │   ├── __init__.py
│       │   │   ├── esm2_t33_650M.yaml
│       │   │   └── esm2_t6_8M.yaml
│       │   ├── features
│       │   │   ├── both.yaml
│       │   │   ├── seq.yaml
│       │   │   └── struct.yaml
│       │   ├── graph
│       │   │   ├── __init__.py
│       │   │   └── default.yaml
│       │   ├── model
│       │   │   ├── gearnet.yaml
│       │   │   ├── gvp.yaml
│       │   │   └── mlp.yaml
│       │   ├── profile
│       │   │   ├── __init__.py
│       │   │   ├── cpu.yaml
│       │   │   └── gpu.yaml
│       │   └── task
│       │       └── secondary_structure.yaml
│       ├── embed
│       │   ├── __init__.py
│       │   ├── esm.py
│       │   └── protocol.py
│       ├── encoders
│       │   ├── __init__.py
│       │   ├── gearnet.py
│       │   ├── gvp.py
│       │   ├── mlp.py
│       │   └── protocol.py
│       ├── fuse
│       │   └── __init__.py
│       ├── graph
│       │   ├── __init__.py
│       │   ├── build.py
│       │   ├── cache.py
│       │   └── io.py
│       ├── tasks
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── secondary_structure.py
│       │   ├── splits.py
│       │   └── train.py
│       └── utils
│           ├── __init__.py
│           ├── caching.py
│           ├── logging.py
│           ├── seeding.py
│           └── seqio.py
├── tests
│   ├── conftest.py
│   ├── test_cli.py
│   ├── test_conformance.py
│   ├── test_dataset.py
│   ├── test_encoder_conformance.py
│   ├── test_esm.py
│   ├── test_gearnet.py
│   ├── test_graph.py
│   ├── test_gvp.py
│   ├── test_invariance.py
│   ├── test_mlp.py
│   ├── test_secondary_structure.py
│   ├── test_seqio.py
│   ├── test_smoke.py
│   └── test_train.py
└── uv.lock
```

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
# graph/io.py — shipped: load_structure(path_or_id, cache_dir), resolve_url,
# load_structure_ids (txt/csv manifests). CIF everywhere (RCSB + AFDB mmCIF,
# one parsing path); AFDB pinned v4; atomic .part→replace downloads.
# pLDDT from B-factor ONLY for AlphaFold models (info["_ligase_plddt_source"]);
# experimental → uniform 1.0. Settled (Q4): residues without CA → skip + log.
```

Open questions: asymmetric unit vs assembly (lean: asymmetric unit); missing residues = mask node + log (lean yes); one graph per chain (v1: yes).

**`graph/build.py`** — M2

```python
# graph/build.py — shipped:
@dataclass(frozen=True)
class GraphParams:  # frozen + hashable → cache-keyable
    edge_knn: int = 10; edge_radius: float = 10.0
    rbf_min: float = 0.0; rbf_max: float = 20.0
    rbf_count: int = 20; rbf_sigma: float = 2.5

def build_graph(structure, params=DEFAULT_PARAMS, chain=None) -> Data
def build_graph_from_coords(coords, codes, plddt, params) -> Data
def select_chain(structure, chain_id=None) -> gemmi.Chain   # shared with the labeler
def is_protein_ca(res) -> bool                              # shared with the labeler

# PINNED SCHEMA (encoder conformance enforces):
#   x (n,1) fp32 pLDDT/100 · residue_type int64 into AA="ACDEFGHIKLMNPQRSTVWYX"
#   edge_index (2,E) directed both ways · edge_attr (E, rbf_count+1) = [RBF(d) | clamp(±8) seq-offset]
#   pos (n,3) Cα = DATA not feature · residue_index · seq: str
# INVARIANCE GUARANTEE: every feature is a function of distances/graph-structure/sequence only.
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

**`encoders/*.py`** — M3. Each: `forward(batch) -> per-residue logits` plus a `pool() -> per-protein embedding`.
```python
# encoders/protocol.py — shipped (extracted from the working trio; Hard Rule 8):
class GraphEncoder(Protocol)   # forward(batch)->(n,out_dim); pool(out,batch)->(num_graphs,out_dim)
def mean_pool(out, batch)      # the one true pool; unbatched Data = one graph
# Constructor contract (ecosystem-facing): MLP takes (in_dim, …, out_dim);
# non-MLP encoders take (x_dim, edge_in_dim, out_dim). Widths are
# wiring-resolved (train.py), never hardcoded in model yamls.
# Admission: tests/test_encoder_conformance.py — the same suite the built-ins pass.
# M4: `ligase conformance` CLI + plugin docs ship with Ankh (Q7/Q9).

# tasks/secondary_structure.py — shipped:
def secondary_structure_labels(structure, chain=None) -> str
    # ALIGNMENT CONTRACT: labels[i] == label of build_graph node i. Shared traversal.
def map_q8_to_q3 / encode_q8 / encode_q3   # named, total, tested; unknown chars raise
def classification_metrics / report_metrics  # accuracy + per-class F1, zero_division=0
# mkdssp = lazy system dep (4.x positional + legacy -i/-o both handled); seq paths never need it.

# tasks/dataset.py — shipped: build_examples (skip-and-log policy),
#   attach_embeddings (L==num_nodes enforced; THE fp16→fp32 upcast lives here).
# tasks/splits.py — shipped: split_ids (seeded, order-free, min-guarantee);
#   v1 limitation: structure-level split, homology leakage documented.
# tasks/train.py — shipped: run_training / evaluate_saved; features wiring
#   (seq→x=emb · struct→x=pLDDT · both→[emb|pLDDT]); seed-determinism pinned by test.```

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
uv run ligase embed  embed=esm2_t6_8M 'sequences=[ACDEFGHIKLMNPQRSTVWY]'
uv run ligase build structure=1UBQ
uv run ligase build structures_file=examples/structures_example.txt
uv run ligase train features=seq model=mlp embed=esm2_t6_8M
uv run ligase train features=struct model=gvp
uv run ligase train features=both model=gearnet profile=gpu
uv run ligase eval   run_dir=outputs/secondary_structure/seq/mlp/<timestamp>
```

`profile=cpu` is the default: fp32, small batches, tiny splits — everything runs on a laptop. `profile=gpu` switches dtype/batch/split sizes only. Hardware is never load-bearing for correctness.

## 9. Data policy

- **Public data only**: PDB, AlphaFold DB, standard benchmark sets (e.g. CB513 and friends for secondary structure; ProteinGym subsets later). No lab data. No exceptions.
- All downloads go through `graph/io.py` / task loaders into `data/cache/`, never committed.
- Third-party embedding configs may point at local checkpoints (`my_plm.yaml`) — those paths are the user's business; never commit weights, ours or theirs.
- Record provenance: every dataset used in a benchmark table gets a line in `docs/data_sources.md` (source URL, version/date downloaded, license). Reviewers and interviewers both ask this.
- Cache keys are content hashes. Same input = cache hit = zero recompute (tested in M1).

## 10. Testing strategy — verify the math, not just the plumbing

These acceptance tests have been green and unskipped since M2; the encoder suite (item 7) has been green since M3. They may not be quietly deleted or weakened:" Append to item 3: "(Enforced at the encoder layer from M3 — a graph has no forward(). Documented re-scope, since closed by test_encoder_conformance.py.)" Append to item 7: "(Shipped: tests/test_encoder_conformance.py — bitwise for same-computation comparisons, fp64 + atol 1e-9 for cross-graph properties. Tolerance doctrine lives in its docstring.)

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
| <date> | M1: embed module | esm/protocol/caching/seqio/conformance/CLI | Q1 HF transformers; Q2 one h5 per cache_id; Q3 map-to-X+log; contract amendment: cache_id added to EmbeddingSource | <hash> |
| <date> | M2: graph io + build | io.py/build.py/default.yaml/CLI build | Q4 settled: skip+log (overturns mask lean); GraphParams frozen dataclass; CIF everywhere, AFDB v4, pLDDT policy | <hash> |
| <date> | M2: invariance green | test_invariance unskipped | batching-equivalence re-homed to encoder conformance (a graph has no forward()); documented re-scope, since closed in M3 | <hash> |
| <date> | M2: graph cache + batch build | graph/cache.py, manifests, batch CLI | cache extracted at 2nd consumer; keyed by (structure, GraphParams) digest — cache_id lesson one layer up; atomic writes; per-row params rejected until 2nd need | <hash> |
| <date> | M3: MLP baseline | mlp.py + test_mlp | D1: one pipeline, honesty by test (blindness bitwise); D2: task-agnostic encoders; D3: explicit wiring-resolved widths, no lazy modules | <hash> |
| <date> | M3: GVP invariant mode | gvp.py + test_gvp | scalar pathway only; pos never read (bitwise); edges-only structure entry; raw ±8 offset calibrated by first message Linear | <hash> |
| <date> | M3: GearNet-lite | gearnet.py + test_gearnet | relational bias not weights (documented deviation); edge states persist across layers; edge type = |offset|==1 | <hash> |
| <date> | M3: dtype contract | mlp/gvp guards | encoders accept fp32+fp64, fp16 forbidden; property tests standardized: fp64, atol 1e-9 | <hash> |
| <date> | M3: encoder protocol + conformance | protocol.py, test_encoder_conformance, pool delegation, M2 skip deleted | Hard Rule 8 order kept; tolerance doctrine codified; MLP excluded from edge tests BY DESIGN (blindness is its honesty) | <hash> |
| <date> | M3: task module | secondary_structure.py + build.py traversal extraction | select_chain/is_protein_ca public at 2nd consumer — labels and graphs share ONE admission rule; missing DSSP → '-' + log; unknown Q8 raises; mkdssp 4.x + legacy both handled (found by first real-oracle run) | <hash> |
| <date> | M3: dataset + splits | dataset.py, splits.py, example manifest | small bundled manifest for the M3 done-when; full CB513-class set + provenance deferred to M4; alignment triple pinned; fp16→fp32 upcast located at attach_embeddings; homology leakage documented | <hash> |
| <date> | M3: training loop + CLI | train.py, logging.py, train/eval apps, config groups | D2 amended: encoder out_dim IS the head; JSONL metrics; features=struct skips embed instantiation; run dir task/features/model for ablation side-by-side | <hash> |
| <date> | M3: integration-run fixes | configs, split min-guarantee, num_classes derivation, edge_in_dim unification | first real CLI run caught four unit-invisible bugs (missing name keys, empty val at n=5, top-level structures_file default, GVP/GearNet constructor mismatch) — the gate's third layer: unit, conformance, integration | <hash> |
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
