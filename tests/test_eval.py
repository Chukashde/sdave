"""Tests for cut-value and MIS evaluation (most breakage-prone component)."""
import networkx as nx

from pigat.eval.maxcut import evaluate_maxcut_solution
from pigat.eval.mis import evaluate_mis_solution, is_valid_independent_set


def test_cut_value_triangle():
    g = nx.Graph([(0, 1), (1, 2), (0, 2)])
    # one node split off cuts exactly two edges
    res = evaluate_maxcut_solution([0, 1, 0], g)
    assert res["cut_value"] == 2
    assert res["partition_0_size"] == 2 and res["partition_1_size"] == 1


def test_cut_value_path_full_cut():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    res = evaluate_maxcut_solution([0, 1, 0, 1], g)
    assert res["cut_value"] == 3  # alternating bipartition cuts every edge


def test_cut_value_weighted():
    g = nx.Graph()
    g.add_edge(0, 1, weight=5)
    g.add_edge(1, 2, weight=3)
    res = evaluate_maxcut_solution([0, 1, 0], g)
    assert res["cut_value"] == 8  # both edges cut, summed by weight


def test_mis_valid_and_invalid():
    g = nx.Graph([(0, 1), (1, 2), (2, 3)])
    # {0, 2} is independent
    assert is_valid_independent_set([1, 0, 1, 0], g) is True
    res = evaluate_mis_solution([1, 0, 1, 0], g)
    assert res["independent_set_size"] == 2 and res["is_valid"] is True
    # {0, 1} are adjacent -> invalid
    bad = evaluate_mis_solution([1, 1, 0, 0], g)
    assert bad["is_valid"] is False and bad["penalty_count"] == 1
