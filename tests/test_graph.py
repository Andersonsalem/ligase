from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from ligase.graph.build import GraphParams, build_graph, build_graph_from_coords, cache_key
from ligase.graph.cache import get_or_build_graph
from ligase.graph.io import load_structure, load_structure_ids, resolve_url


def test_resolve_url_pdb():
    assert resolve_url("1abc") == "https://files.rcsb.org/download/1ABC.cif"


def test_resolve_url_afdb():
    assert "AF-P69905" in resolve_url("AF_P69905F1")
    assert "AF-P69905" in resolve_url("P69905")  # bare accession


def test_resolve_url_rejects_garbage():
    with pytest.raises(ValueError):
        resolve_url("not-a-thing")


def test_toy_pdb_roundtrip(toy_pdb_path):
    data = build_graph(load_structure(str(toy_pdb_path)))
    assert data.num_nodes == 8
    assert data.seq == "ACDEFGHI"
    assert data.x.shape == (8, 1)
    assert torch.allclose(data.x, torch.ones(8, 1))  # experimental -> pLDDT 1.0
    assert data.edge_attr.shape[1] == GraphParams().rbf_count + 1
    pairs = set(map(tuple, data.edge_index.t().tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs  # sequence adjacency, both ways


def test_knn_clamps_on_small_graphs(toy_coords, toy_codes, toy_plddt):
    g = build_graph_from_coords(toy_coords, toy_codes, toy_plddt, GraphParams(edge_knn=10))
    assert g.num_nodes == 8  # k clamped to n-1=7, no crash


def test_single_residue_graph():
    g = build_graph_from_coords(np.array([[0.0, 0.0, 0.0]]), "A", np.array([50.0]))
    assert g.num_nodes == 1 and g.edge_index.shape == (2, 0)


@pytest.mark.slow
def test_real_download_ubiquitin(tmp_path):
    """1UBQ is 76 residues --- should be representative enough"""
    data = build_graph(load_structure("1UBQ", tmp_path))
    assert data.num_nodes == 76
    assert float(data.x.min()) >= 0.0 and float(data.x.max()) <= 1.0


def test_cache_key_changes_with_params() -> None:
    assert cache_key("1UBQ", GraphParams()) != cache_key(
        "1UBQ", replace(GraphParams(), edge_knn=15)
    )


def test_cache_key_changes_with_structure() -> None:
    assert cache_key("1UBQ", GraphParams()) != cache_key("AF_P69905F1", GraphParams())


def test_cache_key_is_deterministic() -> None:
    assert cache_key("1UBQ", GraphParams()) == cache_key("1UBQ", GraphParams())


def test_load_ids_txt(tmp_path: Path) -> None:
    f = tmp_path / "ids.txt"
    f.write_text("# comment\n\n1UBQ\n  AF_P69905F1  \n1UBQ\n")
    assert load_structure_ids(f) == ["1UBQ", "AF_P69905F1"]  # dedupe, order kept


def test_load_ids_csv(tmp_path: Path) -> None:
    f = tmp_path / "ids.csv"
    f.write_text("Structure,note\n1UBQ,ubiquitin\nAF_P69905F1,actin model\n")
    assert load_structure_ids(f) == ["1UBQ", "AF_P69905F1"]


def test_load_ids_csv_missing_column(tmp_path: Path) -> None:
    f = tmp_path / "ids.csv"
    f.write_text("pdb_id\n1UBQ\n")
    with pytest.raises(ValueError, match="structure"):
        load_structure_ids(f)


def test_load_ids_unknown_ext(tmp_path: Path) -> None:
    f = tmp_path / "ids.json"
    f.write_text("[]")
    with pytest.raises(ValueError, match="Unsupported"):
        load_structure_ids(f)


def test_graph_cache_roundtrip(tmp_path: Path, toy_pdb_path: Path) -> None:
    params = GraphParams()
    data1, hit1 = get_or_build_graph(str(toy_pdb_path), params, tmp_path)
    data2, hit2 = get_or_build_graph(str(toy_pdb_path), params, tmp_path)
    assert not hit1 and hit2
    assert torch.equal(data1.x, data2.x)
    assert torch.equal(data1.edge_index, data2.edge_index)
    assert torch.equal(data1.edge_attr, data2.edge_attr)
    assert data1.seq == data2.seq


def test_graph_cache_distinguishes_params(tmp_path: Path, toy_pdb_path: Path) -> None:
    a = GraphParams()
    b = replace(GraphParams(), edge_knn=15)
    _, hit_a = get_or_build_graph(str(toy_pdb_path), a, tmp_path)
    _, hit_b = get_or_build_graph(str(toy_pdb_path), b, tmp_path)
    assert not hit_a and not hit_b  # different keys -> both built
    _, again = get_or_build_graph(str(toy_pdb_path), a, tmp_path)
    assert again  # original entry untouched by the second build


def test_load_ids_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_structure_ids(tmp_path / "nope.txt")


def test_cache_key_changes_with_chain() -> None:
    assert cache_key("1UBQ.A", GraphParams()) != cache_key("1UBQ", GraphParams())


def test_build_graph_chain_selection_matches_default_traversal(toy_pdb_path) -> None:
    st = load_structure(str(toy_pdb_path))
    name = next(ch.name for ch in st[0] if len(ch))
    assert build_graph(st, chain=name).seq == build_graph(st).seq


@pytest.mark.slow
def test_chain_token_end_to_end(tmp_path):
    """1UBQ.A: parses, downloads the ENTRY once, caches under the FULL token."""
    from ligase.graph.tokens import parse_token

    assert parse_token("1UBQ.A") == ("1UBQ", "A")
    data, hit = get_or_build_graph("1UBQ.A", GraphParams(), tmp_path)
    assert not hit and data.num_nodes == 76
    again, hit2 = get_or_build_graph("1UBQ.A", GraphParams(), tmp_path)
    assert hit2 and torch.equal(data.x, again.x)
