from __future__ import annotations

import gemmi
import numpy as np
import pytest

from ligase.graph.build import is_protein_ca, select_chain
from ligase.tasks import dataset as ds
from ligase.tasks import secondary_structure as ss
from ligase.tasks.splits import split_ids


@pytest.fixture
def toy_example(toy_pdb_path, monkeypatch: pytest.MonkeyPatch) -> ds.Example:
    """One built example via the real pipeline, DSSP faked to all-H."""
    st = gemmi.read_structure(str(toy_pdb_path))
    chain = select_chain(st, None)
    fake = {ss.residue_key(chain.name, r): "H" for r in chain if is_protein_ca(r)}
    monkeypatch.setattr(ss, "compute_dssp", lambda structure: fake)
    examples, skipped = ds.build_examples([str(toy_pdb_path)])
    assert not skipped
    assert len(examples) == 1
    return examples[0]


def test_alignment_triple(toy_example: ds.Example) -> None:
    g = toy_example.graph
    assert len(toy_example.labels) == g.num_nodes == 8
    assert toy_example.embeddings is None  # not attached yet


def test_build_skips_dead_ids(toy_pdb_path, monkeypatch: pytest.MonkeyPatch) -> None:
    st = gemmi.read_structure(str(toy_pdb_path))
    chain = select_chain(st, None)
    fake = {ss.residue_key(chain.name, r): "H" for r in chain if is_protein_ca(r)}
    monkeypatch.setattr(ss, "compute_dssp", lambda structure: fake)
    examples, skipped = ds.build_examples([str(toy_pdb_path), "/no/such/thing.cif"])
    assert len(examples) == 1
    assert len(skipped) == 1
    assert skipped[0][0] == "/no/such/thing.cif" and "Error" in skipped[0][1]


def test_attach_embeddings_upcasts_and_aligns(
    toy_example: ds.Example, cached_mock, tmp_path
) -> None:
    ds.attach_embeddings([toy_example], cached_mock, tmp_path)
    assert toy_example.embeddings is not None
    assert toy_example.embeddings.dtype == np.float32
    assert toy_example.embeddings.shape == (toy_example.graph.num_nodes, cached_mock.dim)


def test_attach_embeddings_misalignment_raises(toy_example: ds.Example, tmp_path) -> None:
    class _WrongL:
        dim = 4
        cache_id = "wrongl|v1"

        def embed(self, seqs):
            return {s: np.zeros((len(s) + 1, 4), dtype=np.float16) for s in seqs}

    with pytest.raises(ValueError, match="alignment"):
        ds.attach_embeddings([toy_example], _WrongL(), tmp_path)


def test_split_deterministic_and_order_free() -> None:
    ids = [f"S{i:02d}" for i in range(40)]
    a = split_ids(ids, seed=7)
    b = split_ids(list(reversed(ids)), seed=7)  # input order must not matter
    assert a == b


def test_split_partitions_and_fractions() -> None:
    ids = [f"S{i:02d}" for i in range(40)]
    s = split_ids(ids)
    all_ids = s["train"] + s["val"] + s["test"]
    assert sorted(all_ids) == sorted(ids)
    assert len(set(all_ids)) == len(all_ids)
    assert len(s["train"]) == 28 and len(s["val"]) == 6  # int(40*0.7), int(40*0.15)
    assert len(s["test"]) == 6


def test_split_seed_changes_assignment() -> None:
    ids = [f"S{i:02d}" for i in range(40)]
    assert split_ids(ids, seed=1) != split_ids(ids, seed=2)


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError, match="invalid fractions"):
        split_ids(["a"], train_frac=0.8, val_frac=0.3)


def test_split_small_sets_keep_every_split_nonempty() -> None:
    assert [len(v) for v in split_ids([f"S{i}" for i in range(3)]).values()] == [1, 1, 1]
    assert [len(v) for v in split_ids([f"S{i}" for i in range(5)]).values()] == [3, 1, 1]
