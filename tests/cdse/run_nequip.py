#!/usr/bin/env python3
"""Test with NequIP CdSe model: search + refine + validate, 8 atoms."""
import os, sys, time
sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')  # project root

import numpy as np
from ase.build import bulk
from ase.io import write

from pallas.nequip_calc import NequIPDualCalc
from pallas.optimize import set_calculator
from pallas import Pallas, PallasConfig

t0 = time.time()

# Set NequIP CdSe calculator
calc = NequIPDualCalc(
    ef_model_path='/Users/li/dev/Pallas2/model/cdse_energy_force_model.pth',
    stress_model_path='/Users/li/dev/Pallas2/model/cdse_stress_model.pth')
set_calculator(calc)
print("Using NequIP CdSe model")

# Build structures
rs = bulk('CdSe', 'rocksalt', a=6.05, cubic=True)
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01).repeat((2, 1, 1))

print("=" * 60)
print("NequIP CdSe: RS -> WZ (8 atoms)")
print("=" * 60)

for f in ['pallas.db', 'graph.pkl', 'graph.gml', 'graph.gexf', 'dij.pkl',
          'opt.log', 'ssdimer.log', 'xcal.log', 'xcal_escape.log']:
    if os.path.exists(f):
        os.remove(f)

write('POSCAR1', rs, format='vasp', direct=True, sort=True)
write('POSCAR2', wz, format='vasp', direct=True, sort=True)

config = PallasConfig(
    znucl=sorted(set(rs.get_atomic_numbers().tolist())),
    fpcutoff=5.5, natx=100, press=0.0,
    maxstep=5,
    opt_steps=2000, opt_fmax=0.001,
    saddle_steps=2000, saddle_fmax=0.01,
    bias_steps=60,
    fp_step_scale=0.05, fp_push_scale=0.05,
    ediff=0.01, dist_threshold=0.1,
    refine_rounds=3, refine_probes=5,
)

# Phase 1: Search
print("\n>>> PHASE 1: Multi-probe search (n_probes=3)")
pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
G = pallas.run_fp_guided(n_probes=3)
t1 = time.time()
print(f"Phase 1: {t1-t0:.0f}s")

# Phase 2: Refine
print("\n>>> PHASE 2: Barrier refinement")
path, bn = pallas.refine_barrier()
t2 = time.time()
print(f"Phase 2: {t2-t1:.0f}s")

# Phase 3: Validate
print("\n>>> PHASE 3: Saddle validation")
stats = pallas.validate_graph()
t3 = time.time()
print(f"Phase 3: {t3-t2:.0f}s")

# Final
print("\n>>> FINAL RESULT")
try:
    path, bn = pallas.find_best_path()
    print(f"Best path: {' -> '.join(str(n) for n in path)}")
    print(f"Bottleneck: {bn:.4f} eV")
    for node in path:
        nd = G.nodes[node]
        ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
        print(f"  {ntype} {node}: E = {nd['e']:.4f} eV")
except Exception as e:
    print(f"No path: {e}")

print(f"\nTotal: {t3-t0:.0f}s ({(t3-t0)/60:.1f} min)")
