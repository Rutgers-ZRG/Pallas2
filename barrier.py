# minimax_path.py
from collections import deque
import networkx as nx


# ----------------------------------------------------------------------
# 1.  A tiny, fast Union–Find with path-compression + union-by-rank
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
# 2.  Kruskal with **early stop**  ➜ O(E log E) worst-case but
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
# 3.  If you store a *node* attribute called 'energy' and the real
#     “barrier” is the max energy along the path, wrap the helper below
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
