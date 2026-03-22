#!/usr/bin/env python3
"""Test FP-drag method: CdSe RS -> WZ, 8 atoms, MatterSim."""
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
print("FP-Drag: CdSe RS -> WZ (8 atoms, MatterSim)")
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
    fp_step_scale=3.0,  # total budget: 3.0/30 = 0.1 per image
)

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])

# Run FP-drag
result = pallas.run_fp_drag(n_images=30, relax_steps=100, relax_fmax=0.02)

t_total = time.time() - t0

# Print energy profile
print("\n>>> Energy Profile:")
print(f"{'Image':>5s} {'H (eV)':>10s} {'d_fp':>10s} {'Progress':>10s}")
for i, (h, d) in enumerate(zip(result['energies'], result['fp_dists'])):
    prog = 1.0 - d / max(result['fp_dists'][0], 1e-10)
    marker = " <-- saddle" if i == result['saddle_idx'] else ""
    print(f"{i:5d} {h:10.4f} {d:10.5f} {prog:10.1%}{marker}")

print(f"\nBarrier: {result['barrier']:.4f} eV ({result['barrier']/4:.4f} eV/f.u.)")
print(f"Total: {t_total:.0f}s ({t_total/60:.1f} min)")
