#!/usr/bin/env python3
"""Full pipeline with NequIP CdSe model."""
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

calc = NequIPDualCalc(
    ef_model_path='/Users/li/dev/Pallas2/tests/cdse/model/cdse_energy_force_model.pth',
    stress_model_path='/Users/li/dev/Pallas2/tests/cdse/model/cdse_stress_model.pth')
set_calculator(calc)

rs = bulk('CdSe', 'rocksalt', a=6.05, cubic=True)
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01).repeat((2, 1, 1))

print("=" * 60)
print("Full Pipeline: NequIP CdSe, FP-drag+dimer")
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
    maxstep=3,
    opt_steps=2000, opt_fmax=0.001,
    saddle_steps=2000, saddle_fmax=0.01,
    bias_steps=60,
    fp_step_scale=3.0,
    fp_push_scale=0.1,
    ediff=0.01, dist_threshold=0.1,
    refine_rounds=2, refine_probes=3,
)

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])

print("\n>>> PHASE 1: Multi-probe FP-drag+dimer")
G = pallas.run_fp_guided(n_probes=3)
t1 = time.time()
print(f"Phase 1: {t1-t0:.0f}s ({(t1-t0)/60:.1f} min)")

print("\n>>> PHASE 2: Barrier refinement")
path, bn = pallas.refine_barrier()
t2 = time.time()
print(f"Phase 2: {t2-t1:.0f}s ({(t2-t1)/60:.1f} min)")

print("\n>>> PHASE 3: Validation")
stats = pallas.validate_graph()
t3 = time.time()
print(f"Phase 3: {t3-t2:.0f}s")

print("\n>>> FINAL")
try:
    path, bn = pallas.find_best_path()
    print(f"Path: {' -> '.join(str(n) for n in path)}")
    print(f"Bottleneck: {bn:.4f} eV ({bn/4:.4f} eV/f.u.)")
    for node in path:
        nd = G.nodes[node]
        ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
        print(f"  {ntype} {node}: E = {nd['e']:.4f} eV, V = {nd['volume']:.1f}")
except Exception as e:
    print(f"No path: {e}")

print(f"\nTotal: {t3-t0:.0f}s ({(t3-t0)/60:.1f} min)")
