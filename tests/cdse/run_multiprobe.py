#!/usr/bin/env python3
"""Test multi-probe FP-guided search: CdSe RS → WZ with MatterSim."""
import os, sys, time
sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')  # project root

import numpy as np
from ase.build import bulk
from ase.io import write
from pallas import Pallas, PallasConfig

t0 = time.time()

rs = bulk('CdSe', 'rocksalt', a=6.05).repeat((2, 1, 1))
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01)

print("=" * 60)
print("Multi-Probe FP-Guided: CdSe RS -> WZ")
print("=" * 60)

for f in ['pallas.db', 'graph.pkl', 'graph.gml', 'graph.gexf', 'dij.pkl']:
    if os.path.exists(f):
        os.remove(f)

write('POSCAR1', rs, format='vasp', direct=True, sort=True)
write('POSCAR2', wz, format='vasp', direct=True, sort=True)

config = PallasConfig(
    znucl=sorted(set(rs.get_atomic_numbers().tolist())),
    fpcutoff=5.5, natx=100, press=0.0,
    maxstep=3,
    opt_steps=500, opt_fmax=0.005,
    saddle_steps=500, saddle_fmax=0.05,
    bias_steps=40,
    fp_step_scale=0.5, fp_push_scale=0.1,
    ediff=0.01, dist_threshold=0.1,
)

N_PROBES = 3
print(f"n_probes={N_PROBES}, maxstep={config.maxstep}")

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
G = pallas.run_fp_guided(n_probes=N_PROBES)

t_total = time.time() - t0
print(f"\nTotal runtime: {t_total:.0f}s ({t_total/60:.1f} min)")
