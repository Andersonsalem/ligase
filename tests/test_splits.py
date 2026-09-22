from __future__ import annotations

import random

import pytest

from ligase.tasks.splits import load_groups, split_ids


def _reference_v1_split(
    ids: list[str], train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42
) -> dict[str, list[str]]:
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, min(int(n * train_frac), n - 2))
    n_val = max(1, min(int(n * val_frac), n - n_train - 1))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def test_groups_none_matches_reference_oracle() -> None:
    ids = [f"CHAIN{i:03d}" for i in range(37)]
    assert split_ids(ids, seed=7) == _reference_v1_split(ids, seed=7)
    ids2 = ["b", "a", "d", "c", "e", "g", "f"]
    assert split_ids(ids2, train_frac=0.6, val_frac=0.2, seed=3) == (
        _reference_v1_split(ids2, train_frac=0.6, val_frac=0.2, seed=3)
    )


def test_determinism_bitwise() -> None:
    ids = [f"S{i:03d}" for i in range(50)]
    a = split_ids(ids, seed=123)
    b = split_ids(ids, seed=123)
    assert a["train"] == b["train"]
    assert a["val"] == b["val"]
    assert a["test"] == b["test"]


def test_splits_disjoint_and_exhaustive() -> None:
    ids = [f"X{i:03d}" for i in range(20)]
    out = split_ids(ids)
    assert sorted(out["train"] + out["val"] + out["test"]) == sorted(ids)
    assert not set(out["train"]) & set(out["val"])
    assert not set(out["train"]) & set(out["test"])
    assert not set(out["val"]) & set(out["test"])
    assert all(out[s] for s in ("train", "val", "test"))


def test_duplicate_ids_raise() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        split_ids(["A", "B", "A"])


def test_order_free_legacy() -> None:
    ids = [f"O{i:02d}" for i in range(15)]
    assert split_ids(ids, seed=5) == split_ids(ids[::-1], seed=5)


def test_groups_never_straddle() -> None:
    ids: list[str] = []
    labels: list[int] = []
    for g in range(12):
        for m in range(1 + (g % 4)):  # group sizes 1..4
            ids.append(f"G{g:02d}M{m}")
            labels.append(g)
    out = split_ids(ids, seed=11, groups=labels)
    where = {id_: s for s in ("train", "val", "test") for id_ in out[s]}
    for label in set(labels):
        members = [i for i, lab in zip(ids, labels, strict=True) if lab == label]
        assert len({where[m] for m in members}) == 1


def test_groups_order_free_and_deterministic() -> None:
    ids = [f"P{i:02d}" for i in range(24)]
    labels = [i % 7 for i in range(24)]
    a = split_ids(ids, seed=9, groups=labels)
    assert a == split_ids(ids, seed=9, groups=labels)
    perm = list(range(24))
    random.Random(0).shuffle(perm)
    ids_p = [ids[i] for i in perm]
    labels_p = [labels[i] for i in perm]
    assert split_ids(ids_p, seed=9, groups=labels_p) == a


def test_group_path_budget_fidelity() -> None:
    ids = [f"B{i:03d}" for i in range(513)]
    out = split_ids(ids, groups=list(range(513)))  # singletons
    n = len(ids)
    assert abs(len(out["train"]) - n * 0.7) <= 2
    assert abs(len(out["val"]) - n * 0.15) <= 2
    assert abs(len(out["test"]) - n * 0.15) <= 2


def test_lumpy_groups_min_guarantee() -> None:
    sizes = (5, 5, 1)
    ids = [f"R{g}_{m}" for g, size in enumerate(sizes) for m in range(size)]
    labels = [g for g, size in enumerate(sizes) for _ in range(size)]
    out = split_ids(ids, groups=labels)
    assert all(out[s] for s in ("train", "val", "test"))
    assert sorted(out["train"] + out["val"] + out["test"]) == sorted(ids)
    where = {id_: s for s in ("train", "val", "test") for id_ in out[s]}
    for label in set(labels):
        members = [i for i, lab in zip(ids, labels, strict=True) if lab == label]
        assert len({where[m] for m in members}) == 1


def test_too_few_groups_raises() -> None:
    with pytest.raises(ValueError, match=">= 3 groups"):
        split_ids(["A1", "A2", "B1", "B2"], groups=["A", "A", "B", "B"])


def test_groups_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="groups length"):
        split_ids(["A", "B", "C"], groups=[1, 2])


def test_invalid_fractions_raise() -> None:
    with pytest.raises(ValueError, match="invalid fractions"):
        split_ids(["A", "B", "C"], train_frac=0.8, val_frac=0.3)


def test_load_groups_roundtrip(tmp_path: object) -> None:
    p = tmp_path / "g.json"  # type: ignore[attr-defined]
    p.write_text('{"groups": [["B2", "B1"], ["A1"]]}')
    assert load_groups(p) == {"B2": "B2", "B1": "B2", "A1": "A1"}


def test_load_groups_missing_file(tmp_path: object) -> None:
    with pytest.raises(FileNotFoundError):
        load_groups(tmp_path / "nope.json")  # type: ignore[attr-defined]


def test_load_groups_malformed(tmp_path: object) -> None:
    p = tmp_path / "bad.json"  # type: ignore[attr-defined]
    p.write_text('{"nope": 1}')
    with pytest.raises(ValueError, match="groups JSON"):
        load_groups(p)
    p.write_text('{"groups": [["A"], "B"]}')
    with pytest.raises(ValueError, match="group 1"):
        load_groups(p)


def test_load_groups_duplicate_token(tmp_path: object) -> None:
    p = tmp_path / "dup.json"  # type: ignore[attr-defined]
    p.write_text('{"groups": [["A", "B"], ["B", "C"]]}')
    with pytest.raises(ValueError, match="duplicate token"):
        load_groups(p)
