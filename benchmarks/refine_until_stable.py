#!/usr/bin/env python3
"""Iteratively refine the current path's rate-limiting saddle until it is a
refined one. Each saddle is refined at most once (avoids the re-refine
double-count). Usage:

  python refine_until_stable.py WORKDIR S24,S35,S14   # already-refined names
"""
import json
import os
import pickle
import sys

import numpy as np

import ase.db

from mattersim.forcefield import MatterSimCalculator
from pallas import Pallas
from pallas.config import PallasConfig
from pallas.graph import minimax_path_kinetic
from pallas.optimize import GeometryGuard, set_calculator
from pallas.structure import PallasAtom

wd = sys.argv[1]
refined = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else set()
os.chdir(wd)
cj = json.load(open('config.json'))
cfg = PallasConfig(
    znucl=cj['znucl'], natx=cj['natx'], fpcutoff=cj['fpcutoff'],
    press=float(cj['press']), opt_steps=800, opt_fmax=cj['opt_fmax'],
    saddle_steps=800, saddle_fmax=cj['saddle_fmax'],
    fp_push_scale=cj['fp_push_scale'],
    dist_threshold=cj['dist_threshold'], ediff=cj['ediff'])
set_calculator(GeometryGuard(MatterSimCalculator(device='cpu')))

p = Pallas(cfg)
p.db = ase.db.connect('pallas.db')
p._probe_stats = {}
with open('graph.pkl', 'rb') as f:
    p.G = pickle.load(f)
p.init_minima = []
for nid in (1, 2):
    row = p.db.get(id=nid)
    pa = PallasAtom(p.db.get_atoms(nid))
    pa.znucl, pa.natx, pa.fpcutoff = cfg.znucl, cfg.natx, cfg.fpcutoff
    pa.fp = np.array(row.data['fp'])
    pa.id = nid
    p.init_minima.append(pa)

log = []
for it in range(8):
    path, bn = minimax_path_kinetic(p.G, 1, 2)
    names = [p.G.nodes[n]['xname'] for n in path]
    # rate-limiting saddle = max(H_sad - H_preceding_min)
    hmin, rl, rl_b = None, None, -1e9
    for n in path:
        nd = p.G.nodes[n]
        if nd['xname'].startswith('M'):
            hmin = nd['e']
        elif nd['e'] - hmin > rl_b:
            rl_b, rl = nd['e'] - hmin, n
    rl_name = p.G.nodes[rl]['xname'] if rl is not None else None
    print(f"iter {it}: bottleneck {bn:.6f} | {' -> '.join(names)} "
          f"| rate-limiting {rl_name} ({rl_b:.4f})")
    log.append({'iter': it, 'bn': float(bn), 'path': names,
                'rate_limiting': rl_name})
    if rl is None or rl_name in refined:
        print(f"STABLE: {bn:.6f} (rate-limiting {rl_name} already refined)")
        break
    rep = p.refine_path_saddles(path=[rl], fmax=0.01)
    refined.add(rl_name)
    if not rep or not rep[0]['accepted']:
        print(f"WARNING: refine of {rl_name} not accepted "
              f"({rep[0]['reason'] if rep else 'no report'}) — energy kept")

json.dump(log, open('refine_until_stable.json', 'w'), indent=1, default=str)
