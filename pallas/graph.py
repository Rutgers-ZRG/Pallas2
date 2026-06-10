# minimax_path.py
import heapq
from collections import deque

import networkx as nx


# ----------------------------------------------------------------------
# 1.  A tiny, fast Union-Find with path-compression + union-by-rank
# ----------------------------------------------------------------------
class UnionFind:
    __slots__ = ("p", "r")

    def __init__(self, nodes):
        self.p = {v: v for v in nodes}   # parent
        self.r = {v: 0 for v in nodes}   # rank

    def find(self, v):                   # iterative, path-compressed
        p = self.p
        while p[v] != v:
            p[v] = p[p[v]]
            v = p[v]
        return v

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


# ----------------------------------------------------------------------
# 2.  Kruskal with early stop -> O(E log E) worst-case but
#     usually stops as soon as start & goal are connected.
#     Returns both the path and the bottleneck value.
# ----------------------------------------------------------------------
def minimax_path(G: nx.Graph, start, goal, wkey: str = "weight"):
    uf   = UnionFind(G.nodes)
    adj  = {v: [] for v in G.nodes}          # adjacency in the partial MST
    edges = sorted(G.edges(data=True), key=lambda e: e[2][wkey])

    for u, v, data in edges:
        uf.union(u, v)
        adj[u].append(v)
        adj[v].append(u)

        # once the components touch, we have the minimax bottleneck
        if uf.find(start) == uf.find(goal):
            bottleneck = data[wkey]
            path = _restore_path(adj, start, goal)
            return path, bottleneck

    raise nx.NetworkXNoPath(f"{start} and {goal} are disconnected")


def _restore_path(adj, s, t):
    """BFS in the partial MST to get the actual path."""
    q, prev = deque([s]), {s: None}
    while q and t not in prev:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)

    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


# ----------------------------------------------------------------------
# 3.  Kinetic minimax: minimize the rate-limiting LOCAL barrier.
#
#     For a path M1-S1-M3-S2-M2, the local barrier of each step is
#     H(saddle) - H(preceding minimum).  A deep intermediate trap makes
#     the local barrier much larger than the overall forward barrier.
#
#     Uses modified Dijkstra on directed weights derived from node
#     enthalpies:  min->saddle = H(S)-H(M),  saddle->min = 0.
# ----------------------------------------------------------------------
def minimax_path_kinetic(G, start, goal):
    """Find path minimizing the rate-limiting (max local) barrier.

    Parameters
    ----------
    G : nx.Graph -- nodes must have 'xname' (M* or S*) and 'e' (enthalpy).
    start, goal : node IDs.

    Returns
    -------
    path : list of node IDs
    bottleneck : float -- the rate-limiting local barrier (eV).
    """
    dist = {v: float('inf') for v in G.nodes}
    dist[start] = 0.0
    prev = {start: None}
    pq = [(0.0, 0, start)]  # (bottleneck, tiebreak, node)
    counter = 1

    while pq:
        d, _, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == goal:
            break

        h_u = G.nodes[u]['e']
        u_is_saddle = G.nodes[u]['xname'].startswith('S')

        for v in G.neighbors(u):
            h_v = G.nodes[v]['e']
            v_is_saddle = G.nodes[v]['xname'].startswith('S')

            # Directed weight based on node types
            if not u_is_saddle and v_is_saddle:
                # min -> saddle: cost = local barrier (climbing)
                edge_w = max(0.0, h_v - h_u)
            elif u_is_saddle and not v_is_saddle:
                # saddle -> min: descending, free
                edge_w = 0.0
            else:
                # min->min (direct connection) or saddle->saddle
                edge_w = max(0.0, h_v - h_u)

            new_d = max(d, edge_w)
            if new_d < dist[v]:
                dist[v] = new_d
                prev[v] = u
                heapq.heappush(pq, (new_d, counter, v))
                counter += 1

    if dist[goal] == float('inf'):
        raise nx.NetworkXNoPath(f"{start} and {goal} are disconnected")

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[goal]


# ----------------------------------------------------------------------
# 4.  Legacy helper
# ----------------------------------------------------------------------
def minimax_barrier(G, start, goal, weight="weight", energy="energy"):
    path, _ = minimax_path(G, start, goal, weight)
    return max(G.nodes[n][energy] for n in path), path


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # build a toy graph
    G = nx.Graph()
    G.add_edge(0, 1, weight=2)
    G.add_edge(1, 2, weight=5)
    G.add_edge(0, 2, weight=6)
    nx.set_node_attributes(G, {0: 1.3, 1: 2.7, 2: 0.8}, name="energy")

    path, bottleneck = minimax_path(G, 0, 2)
    print("Path :", path, "  bottleneck edge weight:", bottleneck)

    barrier, path = minimax_barrier(G, 0, 2)
    print("Path :", path, "  barrier node energy  :", barrier)


# ----------------------------------------------------------------------
# 5.  k-best diverse pathways (bottleneck-edge removal)
# ----------------------------------------------------------------------
def k_best_paths(G, s, t, k=5, weight="weight"):
    """Up to ``k`` minimax paths, diversified by bottleneck edge.

    Iteratively: find the minimax path, record it, remove its bottleneck
    edge (the max-weight edge on the path), repeat. Each successive path
    must therefore avoid all previous bottlenecks; stops early once s and
    t disconnect.

    Returns
    -------
    list of (path, bottleneck) — best first.
    """
    H = G.copy()
    out = []
    for _ in range(k):
        try:
            path, bottleneck = minimax_path(H, s, t, weight)
        except (nx.NetworkXNoPath, KeyError):
            break
        out.append((path, bottleneck))
        bn_edge = max(zip(path[:-1], path[1:]),
                      key=lambda e: H.edges[e][weight])
        H.remove_edge(*bn_edge)
    return out
