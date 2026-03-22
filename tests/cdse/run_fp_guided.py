#!/usr/bin/env python3
"""Test FP-guided chain-growing: CdSe rocksalt → wurtzite with MatterSim."""
import os
import sys
import time

sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')  # project root

import numpy as np
from ase.build import bulk
from ase.io import write

from pallas import Pallas, PallasConfig

t0 = time.time()

# ── Build CdSe structures ────────────────────────────────────────────
rs = bulk('CdSe', 'rocksalt', a=6.05).repeat((2, 1, 1))  # 4 atoms
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01)             # 4 atoms

print("=" * 60)
print("PALLAS FP-Guided Test: CdSe RS → WZ")
print("=" * 60)
print(f"RS: {len(rs)} atoms, V = {rs.get_volume():.1f} A^3")
print(f"WZ: {len(wz)} atoms, V = {wz.get_volume():.1f} A^3")

# Clean previous run files
for f in ['pallas.db', 'graph.pkl', 'graph.gml', 'graph.gexf', 'dij.pkl']:
    if os.path.exists(f):
        os.remove(f)

write('POSCAR1', rs, format='vasp', direct=True, sort=True)
write('POSCAR2', wz, format='vasp', direct=True, sort=True)

# ── Configure ─────────────────────────────────────────────────────────
config = PallasConfig(
    znucl=sorted(set(rs.get_atomic_numbers().tolist())),
    fpcutoff=5.5,
    natx=100,
    press=0.0,
    # Chain-growing parameters
    maxstep=5,
    fp_step_scale=0.5,
    fp_push_scale=0.1,
    max_retries=2,
    # Reduced steps for speed
    opt_steps=500,
    opt_fmax=0.005,
    saddle_steps=500,
    saddle_fmax=0.05,
    bias_steps=40,
    # Convergence
    ediff=0.01,
    dist_threshold=0.1,
)

print(f"\nConfig: znucl={config.znucl}, maxstep={config.maxstep}")
print(f"  fp_step_scale={config.fp_step_scale}, fp_push_scale={config.fp_push_scale}")

# ── Run FP-guided search ─────────────────────────────────────────────
print("\n" + "=" * 60)
pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
G = pallas.run_fp_guided()

t_total = time.time() - t0

# ── Results ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

n_min = sum(1 for _, d in G.nodes(data=True) if d['xname'].startswith('M'))
n_sad = sum(1 for _, d in G.nodes(data=True) if d['xname'].startswith('S'))
print(f"Graph: {n_min} minima, {n_sad} saddles, {G.number_of_edges()} edges")
print(f"Runtime: {t_total:.0f}s ({t_total/60:.1f} min)")

print("\nAll nodes:")
for n, d in sorted(G.nodes(data=True)):
    ntype = 'MIN' if d['xname'].startswith('M') else 'SAD'
    print(f"  {ntype:3s} {n:3d}: E = {d['e']:8.4f} eV, V = {d['volume']:8.1f} A^3")

try:
    path, bottleneck = pallas.find_best_path()
    print(f"\nBest path: {' -> '.join(str(n) for n in path)}")
    print(f"Bottleneck: {bottleneck:.4f} eV")
    for node in path:
        nd = G.nodes[node]
        ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
        print(f"  {ntype} {node}: E = {nd['e']:.4f} eV")
except Exception as e:
    print(f"\nNo complete path: {e}")

print(f"\nDone! {t_total:.0f}s")
