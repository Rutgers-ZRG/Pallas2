#!/usr/bin/env python3
"""Benchmark: CdSe RS->WZ with MatterSim (no-drag, seed via argv).

Same protocol as run_nequip.py / tests/golden (3 probes, max_gen 10,
dist_threshold 0.1) — the CLI cannot express fp_step_scale/dist_threshold/
ediff, hence a driver. Run from a fresh workdir; afterwards
``python benchmarks/revalidate.py WORKDIR --calc mattersim`` gives the
FINAL-comparable validated+refined number.
"""
import json
import os
import random
import sys
import time

import numpy as np

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
COMMIT = sys.argv[2] if len(sys.argv) > 2 else 'unrecorded'
random.seed(SEED)
np.random.seed(SEED)

sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')

from ase.build import bulk
from ase.io import write

from pallas import Pallas, PallasConfig

t0 = time.time()

rs = bulk('CdSe', 'rocksalt', a=6.05).repeat((2, 1, 1))
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01)

for f in ['pallas.db', 'graph.pkl', 'graph.gml', 'graph.gexf', 'dij.pkl',
          'opt.log', 'ssdimer.log']:
    if os.path.exists(f):
        os.remove(f)

write('POSCAR1', rs, format='vasp', direct=True, sort=True)
write('POSCAR2', wz, format='vasp', direct=True, sort=True)

config = PallasConfig(
    znucl=sorted(set(rs.get_atomic_numbers().tolist())),
    fpcutoff=5.5, natx=100, press=0.0,
    n_probes=3, max_gen=10, patience=3,
    opt_steps=500, opt_fmax=0.005,
    saddle_steps=500, saddle_fmax=0.05,
    fp_step_scale=0.5, fp_push_scale=0.1,
    ediff=0.01, dist_threshold=0.1,
)

with open('config.json', 'w') as f:
    json.dump({'seed': SEED, 'commit': COMMIT, 'calc': 'mattersim',
               **{k: getattr(config, k) for k in config.__dataclass_fields__}},
              f, indent=1)

print(f"CdSe RS->WZ MatterSim, seed={SEED}, commit={COMMIT}")
pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
path, barrier = pallas.run()

print(f"\nTotal runtime: {time.time() - t0:.0f}s")
if path:
    print(f"RAW barrier: {barrier:.4f} eV ({barrier / 4:.4f} eV/f.u.)")
