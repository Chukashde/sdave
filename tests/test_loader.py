"""Tests for the unified graph loader (most breakage-prone component)."""
import glob

import pytest

from pigat.data import load_graph, detect_format, read_gset, read_graph


def test_detect_gset(dataset_dir):
    assert detect_format(str(dataset_dir / "Gset" / "G14")) == "gset"


def test_detect_adjacency(dataset_dir):
    assert detect_format(str(dataset_dir / "real_world" / "USAir97.txt")) == "adj"
    assert detect_format(str(dataset_dir / "SBM" / "normal_sbm_800_k_10.txt")) == "adj"


def test_gset_parsing(dataset_dir):
    g, info = load_graph(str(dataset_dir / "Gset" / "G14"))
    assert info["n_nodes"] == 800
    assert info["n_edges"] == 4694
    # 1-indexed file edge "1 7" -> 0-indexed (0, 6)
    assert g.has_edge(0, 6)
    # node ids are 0-indexed
    assert min(g.nodes()) == 0 and max(g.nodes()) == 799


def test_adjacency_parsing(dataset_dir):
    g, info = load_graph(str(dataset_dir / "real_world" / "USAir97.txt"))
    assert info["n_nodes"] == 332
    assert g.number_of_edges() == info["n_edges"]


@pytest.mark.parametrize("path", sorted(
    glob.glob("dataset/Gset/*") +
    glob.glob("dataset/real_world/*") +
    glob.glob("dataset/SBM/*")
))
def test_load_graph_matches_legacy_dispatch(path):
    """load_graph must reproduce the legacy path-substring dispatch exactly."""
    legacy_fmt = "gset" if "Gset" in path else "adj"
    assert detect_format(path) == legacy_fmt
    g_new, _ = load_graph(path)
    g_old, _ = (read_gset(path) if legacy_fmt == "gset" else read_graph(path))
    assert g_new.number_of_nodes() == g_old.number_of_nodes()
    assert g_new.number_of_edges() == g_old.number_of_edges()
