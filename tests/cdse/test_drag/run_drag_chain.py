#!/usr/bin/env python3
"""Test FP-drag chain: CdSe RS -> WZ, 8 atoms, MatterSim."""
import os, sys, time
sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')  # project root

import numpy as np
from ase.build import bulk
from ase.io import write
from pallas import Pallas, PallasConfig

t0 = time.time()

rs = bulk('CdSe', 'rocksalt', a=6.05, cubic=True)
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01).repeat((2, 1, 1))

print("=" * 60)
print("FP-Drag Chain: CdSe RS -> WZ (8 atoms)")
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
    opt_steps=2000, opt_fmax=0.001,
    saddle_steps=2000, saddle_fmax=0.01,
    fp_step_scale=3.0,
    fp_push_scale=0.1,
    bias_steps=60,
    ediff=0.01, dist_threshold=0.1,
)

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])

G = pallas.run_fp_drag_chain(
    drag_images=15,
    relax_steps=50,
    relax_fmax=0.03,
    max_segments=5,
)

t_total = time.time() - t0
print(f"\nTotal: {t_total:.0f}s ({t_total/60:.1f} min)")
