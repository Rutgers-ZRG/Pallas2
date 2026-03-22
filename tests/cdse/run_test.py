#!/usr/bin/env python3
"""Test PALLAS: CdSe rocksalt → wurtzite pathway with MatterSim.

Quick test with minimal PSO parameters:
- 4 atoms per structure (2 Cd + 2 Se)
- popsize=1, maxstep=2 → 4 saddle searches total
- Reduced optimization steps for speed
- Expected runtime: ~15-30 min on CPU
"""
import os
import sys
import time

# Add paths
sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')  # project root

import numpy as np
from ase.build import bulk
from ase.io import write

from pallas import Pallas, PallasConfig

t0 = time.time()

# ── Build CdSe structures ────────────────────────────────────────────

# CdSe rocksalt (NaCl structure)
# Primitive cell has 2 atoms; repeat to get 4
rs = bulk('CdSe', 'rocksalt', a=6.05)
rs = rs.repeat((2, 1, 1))  # → 4 atoms

# CdSe wurtzite
# Conventional cell already has 4 atoms (2 Cd + 2 Se)
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01)

print("=" * 60)
print("PALLAS Test: CdSe Rocksalt → Wurtzite")
print("=" * 60)
print(f"RS: {len(rs)} atoms ({rs.get_chemical_formula()})")
print(f"    Cell lengths: {np.array(rs.cell.lengths()).round(3)}")
print(f"    Volume: {rs.get_volume():.1f} A^3")
print(f"WZ: {len(wz)} atoms ({wz.get_chemical_formula()})")
print(f"    Cell lengths: {np.array(wz.cell.lengths()).round(3)}")
print(f"    Volume: {wz.get_volume():.1f} A^3")

# Write POSCARs (sort by species for consistent ordering)
write('POSCAR1', rs, format='vasp', direct=True, sort=True)
write('POSCAR2', wz, format='vasp', direct=True, sort=True)
print("Wrote POSCAR1 (RS) and POSCAR2 (WZ)")

# ── Configure PALLAS ─────────────────────────────────────────────────

config = PallasConfig(
    znucl=sorted(set(rs.get_atomic_numbers().tolist())),  # [34, 48]
    fpcutoff=5.5,
    natx=100,
    press=0.0,
    # Quick test parameters
    maxstep=2,
    popsize=1,
    # Reduced optimization steps for speed
    opt_steps=500,
    opt_fmax=0.005,
    saddle_steps=500,
    saddle_fmax=0.05,
    bias_steps=40,
    # Looser convergence for test
    ediff=0.01,
    dist_threshold=0.1,
)

print(f"\nConfig:")
print(f"  znucl = {config.znucl}  (Se=type1, Cd=type2)")
print(f"  maxstep = {config.maxstep}, popsize = {config.popsize}")
print(f"  opt_steps = {config.opt_steps}, saddle_steps = {config.saddle_steps}")
print(f"  fpcutoff = {config.fpcutoff}, natx = {config.natx}")

# ── Run PALLAS ────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STARTING PSO PATHWAY SEARCH")
print("=" * 60 + "\n")

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
G = pallas.run_pso()

t_total = time.time() - t0

# ── Results ───────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

n_min = sum(1 for _, d in G.nodes(data=True) if d['xname'].startswith('M'))
n_sad = sum(1 for _, d in G.nodes(data=True) if d['xname'].startswith('S'))
print(f"Graph: {G.number_of_nodes()} nodes ({n_min} minima, {n_sad} saddles)")
print(f"       {G.number_of_edges()} edges")
print(f"Total runtime: {t_total:.0f} s ({t_total/60:.1f} min)")

# List all nodes
print("\nAll nodes:")
for n, d in sorted(G.nodes(data=True)):
    ntype = 'MIN' if d['xname'].startswith('M') else 'SAD'
    print(f"  {ntype:3s} {n:3d}: E = {d['e']:8.4f} eV, V = {d['volume']:8.1f} A^3")

# List edges
print("\nAll edges:")
for u, v, d in G.edges(data=True):
    print(f"  {u:3d} -- {v:3d}: weight = {d.get('weight', 0):.4f}, "
          f"fp_dist = {d.get('dist', 0):.5f}")

# Try to find path
try:
    path, bottleneck = pallas.find_best_path()
    print(f"\nBest path: {' → '.join(str(n) for n in path)}")
    print(f"Bottleneck energy: {bottleneck:.4f} eV")
    for node in path:
        nd = G.nodes[node]
        ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
        print(f"  {ntype} {node}: E = {nd['e']:.4f} eV")
except Exception as e:
    print(f"\nNo complete path found yet: {e}")
    print("(This is expected for a quick test with few iterations)")

print(f"\nDone! Total time: {t_total:.0f}s")
