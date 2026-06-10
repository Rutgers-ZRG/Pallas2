import networkx as nx
import pytest

from pallas import minimax_barrier, minimax_path, minimax_path_kinetic

pytestmark = pytest.mark.unit


def test_minimax_path_avoids_high_edge():
    G = nx.Graph()
    G.add_edge(1, 2, weight=5.0)   # direct, high bottleneck
    G.add_edge(1, 3, weight=3.0)
    G.add_edge(3, 2, weight=2.0)   # detour, bottleneck 3.0
    path, bottleneck = minimax_path(G, 1, 2)
    assert path == [1, 3, 2]
    assert bottleneck == pytest.approx(3.0)


def test_minimax_path_disconnected_raises():
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    with pytest.raises(nx.NetworkXNoPath):
        minimax_path(G, 1, 2)


def test_minimax_barrier_max_node_energy():
    G = nx.Graph()
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(2, 3, weight=1.0)
    nx.set_node_attributes(G, {1: 0.0, 2: 1.7, 3: 0.2}, name='energy')
    barrier, path = minimax_barrier(G, 1, 3)
    assert path == [1, 2, 3]
    assert barrier == pytest.approx(1.7)


def _trap_graph():
    """Two routes M1->M3. Route A passes a deep trap M2 (low global
    saddles but huge LOCAL escape barrier); route B has one higher
    saddle but a smaller worst local barrier."""
    G = nx.Graph()
    nodes = {
        1: ('M1', 0.0),
        10: ('S1', 1.0),
        2: ('M2', -2.0),   # deep trap: escape costs 2.5
        20: ('S2', 0.5),
        3: ('M3', 0.0),
        30: ('S3', 1.8),   # direct route saddle
    }
    for n, (xname, e) in nodes.items():
        G.add_node(n, xname=xname, e=e)
    for u, v in [(1, 10), (10, 2), (2, 20), (20, 3), (1, 30), (30, 3)]:
        # minimax_path edge weight = max endpoint enthalpy (as in PALLAS)
        w = max(G.nodes[u]['e'], G.nodes[v]['e'])
        G.add_edge(u, v, weight=w)
    return G


def test_kinetic_minimax_avoids_deep_trap():
    path, bottleneck = minimax_path_kinetic(_trap_graph(), 1, 3)
    assert path == [1, 30, 3]
    assert bottleneck == pytest.approx(1.8)


def test_plain_minimax_prefers_trap_route():
    # Sanity: the plain (global) minimax picks the trap route, which is
    # exactly why run() uses the kinetic variant.
    path, bottleneck = minimax_path(_trap_graph(), 1, 3)
    assert path == [1, 10, 2, 20, 3]
    assert bottleneck == pytest.approx(1.0)


def test_kinetic_disconnected_raises():
    G = _trap_graph()
    G.remove_edges_from(list(G.edges))
    with pytest.raises(nx.NetworkXNoPath):
        minimax_path_kinetic(G, 1, 3)
