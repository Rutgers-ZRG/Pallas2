#!/usr/bin/env python3
"""Test search + barrier refinement: CdSe RS -> WZ, 8 atoms."""
import os, sys, time
sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')  # project root

import numpy as np
from ase.build import bulk
from ase.io import write
from pallas import Pallas, PallasConfig

t0 = time.time()

rs = bulk('CdSe', 'rocksalt', a=6.05, cubic=True)   # 8 atoms
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01).repeat((2, 1, 1))  # 8 atoms

print("=" * 60)
print("Search + Refinement: CdSe RS -> WZ (8 atoms)")
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
    saddle_steps=1000, saddle_fmax=0.02,
    bias_steps=60,
    fp_step_scale=0.5, fp_push_scale=0.1,
    ediff=0.01, dist_threshold=0.1,
    refine_rounds=3, refine_probes=5,
)

# Phase 1: Initial search with 3 probes
print("\n>>> PHASE 1: Initial multi-probe search")
pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
G = pallas.run_fp_guided(n_probes=3)

t1 = time.time()
print(f"\nPhase 1 time: {t1-t0:.0f}s")

# Phase 2: Barrier refinement
print("\n>>> PHASE 2: Barrier refinement")
best_path, best_bn = pallas.refine_barrier()

t_total = time.time() - t0
print(f"\nTotal runtime: {t_total:.0f}s ({t_total/60:.1f} min)")
