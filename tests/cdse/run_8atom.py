#!/usr/bin/env python3
"""Test multi-probe FP-guided search: CdSe RS -> WZ, 8 atoms, tight convergence."""
import os, sys, time
sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')  # project root

import numpy as np
from ase.build import bulk
from ase.io import write
from pallas import Pallas, PallasConfig

t0 = time.time()

# ── 8-atom cells ──────────────────────────────────────────────────────
# RS: conventional cubic NaCl cell (4 Cd + 4 Se)
rs = bulk('CdSe', 'rocksalt', a=6.05, cubic=True)   # 8 atoms

# WZ: conventional hexagonal (2 Cd + 2 Se) x2 supercell
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01)       # 4 atoms
wz = wz.repeat((2, 1, 1))                             # 8 atoms

print("=" * 60)
print("8-Atom Multi-Probe: CdSe RS -> WZ")
print("=" * 60)
print(f"RS: {len(rs)} atoms ({rs.get_chemical_formula()})")
print(f"    Cell: {np.round(rs.cell.lengths(), 3)}")
print(f"    Volume: {rs.get_volume():.1f} A^3")
print(f"WZ: {len(wz)} atoms ({wz.get_chemical_formula()})")
print(f"    Cell: {np.round(wz.cell.lengths(), 3)}")
print(f"    Volume: {wz.get_volume():.1f} A^3")

# Clean previous run
for f in ['pallas.db', 'graph.pkl', 'graph.gml', 'graph.gexf', 'dij.pkl',
          'opt.log', 'ssdimer.log', 'xcal.log', 'xcal_escape.log']:
    if os.path.exists(f):
        os.remove(f)

write('POSCAR1', rs, format='vasp', direct=True, sort=True)
write('POSCAR2', wz, format='vasp', direct=True, sort=True)

# ── Tight convergence config ──────────────────────────────────────────
config = PallasConfig(
    znucl=sorted(set(rs.get_atomic_numbers().tolist())),
    fpcutoff=5.5,
    natx=100,
    press=0.0,
    # Search
    maxstep=5,
    fp_step_scale=0.5,
    fp_push_scale=0.1,
    max_retries=2,
    # Tight endpoint optimization
    opt_steps=2000,
    opt_fmax=0.001,
    # Dimer
    saddle_steps=1000,
    saddle_fmax=0.02,
    bias_steps=60,
    # Convergence
    ediff=0.01,
    dist_threshold=0.1,
)

N_PROBES = 3
print(f"\nConfig: opt_fmax={config.opt_fmax}, saddle_fmax={config.saddle_fmax}")
print(f"n_probes={N_PROBES}, maxstep={config.maxstep}")
print()

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
G = pallas.run_fp_guided(n_probes=N_PROBES)

t_total = time.time() - t0
print(f"\nTotal runtime: {t_total:.0f}s ({t_total/60:.1f} min)")
