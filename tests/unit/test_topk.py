import networkx as nx
import pytest

from pallas.graph import k_best_paths

pytestmark = pytest.mark.unit


def _three_route_graph():
    """Three M1->M2 routes with bottleneck edges 3.0 < 4.0 < 5.0."""
    G = nx.Graph()
    G.add_edge(1, 2, weight=5.0)               # direct
    G.add_edge(1, 3, weight=3.0)
    G.add_edge(3, 2, weight=2.0)               # via 3: bottleneck 3.0
    G.add_edge(1, 4, weight=4.0)
    G.add_edge(4, 2, weight=1.0)               # via 4: bottleneck 4.0
    return G


def test_k_best_ordered():
    paths = k_best_paths(_three_route_graph(), 1, 2, k=3)
    assert [p for p, _ in paths] == [[1, 3, 2], [1, 4, 2], [1, 2]]
    assert [b for _, b in paths] == pytest.approx([3.0, 4.0, 5.0])


def test_k_larger_than_routes():
    paths = k_best_paths(_three_route_graph(), 1, 2, k=10)
    assert len(paths) == 3  # stops when disconnected


def test_k_one_equals_minimax():
    paths = k_best_paths(_three_route_graph(), 1, 2, k=1)
    assert paths == [([1, 3, 2], pytest.approx(3.0))]
