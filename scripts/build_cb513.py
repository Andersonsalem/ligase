from __future__ import annotations

import argparse
import datetime
import gzip
import json
import logging
import random
from pathlib import Path

import gemmi
from tqdm import tqdm

from ligase.graph.io import load_structure
from ligase.graph.tokens import parse_token

logger = logging.getLogger(__name__)

BENCH = Path("benchmarks")
CACHE_DIR = Path("data/cache/structures")
MIN_LEN, MAX_LEN = 30, 1000
SUBSAMPLE = 513
SEED = 42
IDENTITY_THRESHOLD = 0.25  # cull's threshold ; verification reference, recorded not enforced
MERGE_THRESHOLD = 0.30  # split-protection clustering, above the cull to absorb aligner divergence
NEAR_THRESHOLD = 0.20
AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYXBUZ")


LENGTH_FILTER = 0.30


SCORING = gemmi.AlignmentScoring()
SCORING.match = 1
SCORING.mismatch = -1
SCORING.gapo = -10
SCORING.gape = -1


def read_pisces(path: Path) -> list[tuple[str, str]]:
    """Parse a PISCES cull list into (token, sequence) pairs.

    Supports both FASTA files (>XXXXC or >XXXX_C headers) and two-column
    tabular dumps. Sequence alphabet is sanity-checked and length bounds
    (MIN_LEN <= len <= MAX_LEN) are applied.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    out: list[tuple[str, str]] = []
    skipped = 0

    with opener(path, "rt") as handle:
        current_header: str | None = None
        current_seq_parts: list[str] = []

        def flush_current():
            nonlocal skipped
            if not current_header:
                return
            seq = "".join(current_seq_parts).upper()
            if set(seq) - AA_ALPHABET or not (MIN_LEN <= len(seq) <= MAX_LEN):
                skipped += 1
                return
            raw_id = current_header.lstrip(">").split()[0]
            if "_" in raw_id:
                pid, _, chain = raw_id.partition("_")
            elif len(raw_id) >= 5:
                pid, chain = raw_id[:4], raw_id[4:]
            else:
                pid, chain = raw_id, ""
            token = f"{pid.upper()}.{chain}" if chain else pid.upper()
            out.append((token, seq))

        first_char = handle.read(1)
        handle.seek(0)

        if first_char == ">":
            # FASTA format
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    flush_current()
                    current_header = line
                    current_seq_parts = []
                else:
                    current_seq_parts.append(line)
            flush_current()
        else:
            # Tabular format
            for line in handle:
                fields = line.split()
                if len(fields) < 2:
                    continue
                raw, seq = fields[0], fields[1].upper()
                if set(seq) - AA_ALPHABET or not (MIN_LEN <= len(seq) <= MAX_LEN):
                    skipped += 1
                    continue
                pid, _, chain = raw.partition("_")
                token = f"{pid.upper()}.{chain}" if chain else pid.upper()
                out.append((token, seq))

    if skipped:
        logger.info("skipped %d entries failing length or alphabet checks", skipped)
    return out


def _manifest_tokens() -> list[str]:
    manifest = BENCH / "cb513_chains.txt"
    if not manifest.exists():
        raise SystemExit(f"{manifest} not found --- run the 'list' stage first")
    return [ln for ln in manifest.read_text().splitlines() if ln and not ln.startswith("#")]


def stage_list(pisces: Path) -> None:
    """Stage 1: seeded 513-chain subsample into cb513_chains.txt"""
    tokens = sorted(dict.fromkeys(t for t, _ in read_pisces(pisces)))
    if len(tokens) < SUBSAMPLE:
        raise SystemExit(f"cull list yields {len(tokens)} chains, need >= {SUBSAMPLE}")
    draw = sorted(random.Random(SEED).sample(tokens, SUBSAMPLE))
    BENCH.mkdir(exist_ok=True)
    # criteria text MUST match the downloaded cull file ; this header is provenance
    header = [
        "# CB513-class benchmark manifest | the committed file is the dataset definition.",
        f"# drawn by scripts/build_cb513.py from: {pisces.name}",
        "# criteria: PISCES cull (<=25% id, res <= 2.5 A, R <= 0.3, X-ray only);"
        f" length {MIN_LEN}-{MAX_LEN}",
        f"# draw: {SUBSAMPLE} of {len(tokens)} chains, random.Random({SEED}).sample",
        f"# over sorted tokens generated: {datetime.date.today().isoformat()}",
    ]
    out = BENCH / "cb513_chains.txt"
    out.write_text("\n".join(header + draw) + "\n")
    print(f"wrote {out} ({SUBSAMPLE} of {len(tokens)} candidate chains, seed {SEED})")


def _compute_identity(seq1: str, seq2: str) -> float:
    """Identity in [0, 1]: calculate_identity(3)/100 under the explicit SCORING.

    No documented gemmi mode semantics are trusted here ; the oracle below
    determines the mode-3 denominator empirically at runtime and the verdict
    is recorded in the payload. What is pinned: explicit scoring (defaults
    produced ratio-inflated counts), fraction scale, and LENGTH_FILTER.
    """
    if not seq1 or not seq2:
        return 0.0
    l_min, l_max = min(len(seq1), len(seq2)), max(len(seq1), len(seq2))
    if l_min / l_max < LENGTH_FILTER:
        return 0.0
    res = gemmi.align_string_sequences(list(seq1), list(seq2), [], SCORING)
    return res.calculate_identity(3) / 100.0


def _identity_oracle() -> str:
    """Determine the mode-3 denominator empirically; pin the metric to it.

    Under the explicit SCORING (mismatch 1 vs gap 11), the aligner never
    gaps the shorter sequence and never inserts extra gaps, so all candidate
    denominators collapse to two behaviors ; distinguished by probes whose
    alignment structure is forced:

    * ('AX','A')       -> forced 1 match: reads 1.0 iff the denominator is
                          the shorter length; 0.5 iff the longer/alignment.
    * ('AB','CAB')     -> 2 matches: 1.0 (shorter) vs 2/3 (longer).
    * ('AAAA','AAQAA') -> the Q is gapped out (gap 11 < mismatch-1+gap 12):
                          1.0 (shorter) vs 0.8 (longer); 0.75 would mean the
                          Q was mismatched ; the scoring regime drifted.
    * ('AAAQ','AAAA')  -> 3 of 4 identical, no gaps: 0.75 under BOTH
                          candidates (universal scoring guard).
    * shuffled 227-mers < 0.15 ; the null baseline (the probe whose absence
                          let accessor artifacts masquerade as pool
                          properties for five rounds).

    Returns the determined family label (embedded in the payload). Anything
    unclassifiable exits loudly ; paste, don't tweak.
    """
    tol = 1e-6
    p_forced = _compute_identity("AX", "A")
    p_shift = _compute_identity("AB", "CAB")
    p_gap = _compute_identity("AAAA", "AAQAA")
    p_plain = _compute_identity("AAAQ", "AAAA")
    if abs(p_plain - 0.75) > tol:
        raise SystemExit(
            f"identity oracle failed: universal probe ('AAAQ','AAAA') = "
            f"{p_plain:.6f}, expected 0.750000 --- scoring regime drifted; "
            "paste this output"
        )
    if abs(p_forced - 1.0) <= tol:
        family = "matches / shorter sequence length"
        checks = [("AB", "CAB", 1.0), ("AAAA", "AAQAA", 1.0)]
    elif abs(p_forced - 0.5) <= tol:
        family = "matches / longer sequence length (= alignment length under this scoring)"
        checks = [("AB", "CAB", 2.0 / 3.0), ("AAAA", "AAQAA", 0.8)]
    else:
        raise SystemExit(
            "identity oracle could not classify the denominator: "
            f"('AX','A')={p_forced:.6f}, ('AB','CAB')={p_shift:.6f}, "
            f"('AAAA','AAQAA')={p_gap:.6f} --- paste this output"
        )
    for s1, s2, want in checks:
        got = _compute_identity(s1, s2)
        if abs(got - want) > tol:
            raise SystemExit(
                f"identity oracle failed: ({s1!r},{s2!r}) = {got:.6f}, expected "
                f"{want:.6f} under {family!r} --- paste this output"
            )
    rng = random.Random(0)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    a = "".join(rng.choice(alphabet) for _ in range(227))
    b = "".join(rng.choice(alphabet) for _ in range(227))
    base = _compute_identity(a, b)
    if base >= 0.15:
        raise SystemExit(
            f"identity oracle failed on the random baseline: {base:.3f} >= 0.15 "
            "--- the metric counts something other than identical residues"
        )
    return family


def stage_groups(pisces: Path) -> None:
    """Stage 2: verification record + split-protection groups in one pass.

    Unions run at MERGE_THRESHOLD (above the cull, absorbing aligner
    divergence) for the split-protection groups; the cull-threshold
    verification (pair counts, components at 0.25) is recorded, not
    enforced. Single pass over all pairs computes both.
    """
    tokens = _manifest_tokens()
    seq_of = dict(read_pisces(pisces))
    missing = [t for t in tokens if t not in seq_of]
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest tokens absent from the cull file | "
            "regenerate manifest and groups from the same file"
        )
    if not hasattr(gemmi, "align_string_sequences") or not hasattr(gemmi, "AlignmentScoring"):
        raise SystemExit(
            f"gemmi {gemmi.__version__} lacks the alignment API this stage needs | "
            "run under the locked env: uv run python scripts/build_cb513.py groups ..."
        )
    family = _identity_oracle()
    seqs = [seq_of[t] for t in tokens]

    parent_v = list(range(len(tokens)))  # verification unions at the cull threshold
    parent_g = list(range(len(tokens)))  # split-protection unions at MERGE_THRESHOLD

    def find(parent: list[int], i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    near = n_over_cull = 0
    n_pairs = len(tokens) * (len(tokens) - 1) // 2
    with tqdm(total=n_pairs, desc="identity") as bar:
        for a in range(len(tokens)):
            for b in range(a + 1, len(tokens)):
                ident = _compute_identity(seqs[a], seqs[b])
                if ident > IDENTITY_THRESHOLD:
                    parent_v[find(parent_v, a)] = find(parent_v, b)
                    n_over_cull += 1
                    if ident > MERGE_THRESHOLD:
                        parent_g[find(parent_g, a)] = find(parent_g, b)
                elif ident > NEAR_THRESHOLD:
                    near += 1
                bar.update(1)

    def components(parent: list[int]) -> list[list[str]]:
        buckets: dict[int, list[str]] = {}
        for i, token in enumerate(tokens):
            buckets.setdefault(find(parent, i), []).append(token)
        return sorted((sorted(m) for m in buckets.values()), key=lambda g: g[0])

    groups_v = components(parent_v)
    groups_g = components(parent_g)
    payload = {
        "source": pisces.name,
        "gemmi_version": gemmi.__version__,
        "scoring": {"match": 1, "mismatch": -1, "gapo": -10, "gape": -1},
        "identity_mode": 3,
        "metric": (
            f"identity = calculate_identity(3)/100, determined at runtime to be "
            f"({family}) under the explicit scoring above; LENGTH_FILTER={LENGTH_FILTER} "
            "skips pairs whose length ratio alone caps identity below every threshold; "
            "pinned by hand-computed probes including a shuffled-sequence baseline. "
            "gemmi's default scoring and identity modes were rejected empirically "
            "(ratio-inflated counts)."
        ),
        "verification_at_cull_threshold": {
            "threshold": IDENTITY_THRESHOLD,
            "n_pairs_over_threshold": n_over_cull,
            "n_pairs_in_20_25_band": near,
            "n_groups": len(groups_v),
            "note": (
                "verification result as measured by the metric above, compared "
                "against the cull's own method; divergences are recorded, not enforced"
            ),
        },
        "split_protection": {
            "merge_threshold": MERGE_THRESHOLD,
            "n_groups": len(groups_g),
            "n_multi_groups": sum(1 for g in groups_g if len(g) > 1),
            "note": (
                "groups cluster ABOVE the cull threshold to absorb aligner "
                "divergence; these groups feed split_ids(groups=...) so "
                "near-threshold relatives never straddle train/test"
            ),
        },
        "groups": groups_g,
    }
    out = BENCH / "cb513_groups.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {out}: verification@{IDENTITY_THRESHOLD} -> {n_over_cull} pairs over, "
        f"{len(groups_v)} groups | split-protection@{MERGE_THRESHOLD} -> "
        f"{len(groups_g)} groups ({payload['split_protection']['n_multi_groups']} multi-chain)"
    )
    if len(groups_g) < 3:
        print(
            "WARNING: <3 groups at the merge threshold | leave "
            "task.groups_file=null (legacy random split) and paste this output"
        )


def stage_fetch() -> None:
    """Stage 3: prime the structure cache via ligase's own loader."""
    tokens = _manifest_tokens()
    entries = list(dict.fromkeys(parse_token(t)[0] for t in tokens))
    logger.info("fetching %d entries for %d chain tokens", len(entries), len(tokens))
    dead: list[str] = []
    for entry in tqdm(entries, desc="fetch"):
        try:
            load_structure(entry, cache_dir=CACHE_DIR)
        except Exception as exc:  # dead entries are expected
            dead.append(f"{entry}\t{type(exc).__name__}: {exc}")
    if dead:
        out = BENCH / "cb513_unavailable.txt"
        out.write_text("\n".join(dead) + "\n")
    print(
        f"fetched {len(entries) - len(dead)}/{len(entries)} entries; "
        f"{len(dead)} dead" + (f" -> {BENCH / 'cb513_unavailable.txt'}" if dead else "")
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Assemble the CB513-class benchmark set (see module docstring)."
    )
    sub = parser.add_subparsers(dest="stage", required=True)
    p_list = sub.add_parser("list", help="seeded 513-chain subsample from a PISCES cull")
    p_list.add_argument("pisces", type=Path)
    p_groups = sub.add_parser("groups", help="all-vs-all identity -> groups.json (offline)")
    p_groups.add_argument("pisces", type=Path)
    sub.add_parser("fetch", help="prime data/cache/structures; dead entries reported")
    args = parser.parse_args()
    if args.stage == "list":
        stage_list(args.pisces)
    elif args.stage == "groups":
        stage_groups(args.pisces)
    else:
        stage_fetch()


if __name__ == "__main__":
    main()
