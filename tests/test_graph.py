import numpy as np
import pytest
import torch

from ligase.graph.build import GraphParams, build_graph, build_graph_from_coords
from ligase.graph.io import load_structure, resolve_url


def test_resolve_url_pdb():
    assert resolve_url("1abc") == "https://files.rcsb.org/download/1ABC.cif"


def test_resolve_url_afdb():
    assert "AF_P69905" in resolve_url("AF_P69905F1")
    assert "AF_P69905" in resolve_url("P69905")  # bare accession


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
