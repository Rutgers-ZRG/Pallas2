"""Screen quoted paths: any saddle with min(fwd, rev) local barrier < eps
is a fake-saddle suspect (duplicate of a flanking minimum)."""
import glob
import pickle

import networkx as nx

from pallas.graph import minimax_path_kinetic

EPS = 0.005
for g in sorted(glob.glob('/Users/li/dev/Pallas2/runs/*/graph.pkl')):
    wd = g.split('/')[-2]
    if wd.startswith('ssneb') or wd.startswith('carbon'):
        continue
    try:
        G = pickle.load(open(g, 'rb'))
        path, bn = minimax_path_kinetic(G, 1, 2)
    except Exception as exc:
        print('%-22s (%s)' % (wd, type(exc).__name__))
        continue
    sus = []
    for i, n in enumerate(path):
        if not G.nodes[n]['xname'].startswith('S'):
            continue
        hs = G.nodes[n]['e']
        left = next(G.nodes[path[j]]['e'] for j in range(i - 1, -1, -1)
                    if G.nodes[path[j]]['xname'].startswith('M'))
        right = next(G.nodes[path[j]]['e'] for j in range(i + 1, len(path))
                     if G.nodes[path[j]]['xname'].startswith('M'))
        m = min(hs - left, hs - right)
        if m < EPS:
            sus.append('%s(min-side %.4f)' % (G.nodes[n]['xname'], m))
    names = ' '.join(G.nodes[n]['xname'] for n in path)
    print('%-22s %.4f  %s%s' % (wd, bn, names,
          ('  SUSPECT: ' + '; '.join(sus)) if sus else ''))
