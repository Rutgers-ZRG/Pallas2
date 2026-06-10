#!/usr/bin/env python3
"""Benchmark: Graphite → Hexagonal Diamond at 15 GPa, NO-DRAG dimer.

Retest of the Mar-22 drag-init run (job 5699449, barrier 4.483 eV) with the
current no-drag code (tag bench/carbon-nodrag-20260610). Identical structures,
potential, pressure, and config for direct comparability.

Allegro r2SCAN+rVV10 carbon MLIP, 16 atoms, Amarel GPU (conda env nequip).
"""
import json
import os
import random
import time

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from ase.spacegroup import crystal
from ase.io import write
from pallas import Pallas, PallasConfig, set_calculator

t0 = time.time()

# --- Build structures (identical to drag run) ---
graphite = crystal(['C', 'C'], [(0, 0, 1/4), (1/3, 2/3, 1/4)],
                   spacegroup=194,
                   cellpar=[2.464, 2.464, 6.711, 90, 90, 120])
graphite = graphite.repeat((1, 1, 4))

hd = crystal('C', [(1/3, 2/3, 0.0625)],
             spacegroup=194,
             cellpar=[2.522, 2.522, 4.119, 90, 90, 120])
hd = hd.repeat((1, 1, 4))

print(f"Graphite: {len(graphite)} atoms, cell = {graphite.cell.lengths()}")
print(f"HD:       {len(hd)} atoms, cell = {hd.cell.lengths()}")

# --- Allegro r2SCAN+rVV10 calculator (Amarel scratch path) ---
from nequip.ase import NequIPCalculator
model_path = '/scratch/lz432/allegro_r2scan_finetune/allegro_r2scan_carbon.nequip.pth'
calc = NequIPCalculator.from_compiled_model(model_path, device='cuda')
set_calculator(calc)
print("Calculator: Allegro r2SCAN+rVV10 carbon (cuda)")

print("=" * 60)
print("NO-DRAG unified run(): Graphite -> HD at 15 GPa (16 atoms)")
print("=" * 60)

for f in ['pallas.db', 'graph.pkl', 'graph.gml', 'graph.gexf', 'dij.pkl']:
    if os.path.exists(f):
        os.remove(f)

write('POSCAR1', graphite, format='vasp', direct=True, sort=True)
write('POSCAR2', hd, format='vasp', direct=True, sort=True)

# --- Config: identical to Mar-22 drag run ---
press_gpa = 15.0
press_ev_a3 = press_gpa / 160.2176634  # GPa -> eV/A^3

config = PallasConfig(
    znucl=[6],
    fpcutoff=5.5, natx=200, press=press_ev_a3,
    n_probes=5,
    max_gen=20,
    patience=5,
    opt_steps=1000, opt_fmax=0.005,
    saddle_steps=500, saddle_fmax=0.05,
    bias_steps=40,
    fp_step_scale=0.5, fp_push_scale=0.1,
    ediff=0.01, dist_threshold=0.1,
)

with open('config.json', 'w') as f:
    json.dump({'seed': SEED, 'commit': 'bench/carbon-nodrag-20260610',
               'press_gpa': press_gpa,
               **{k: getattr(config, k) for k in config.__dataclass_fields__}},
              f, indent=1)

print(f"Pressure: {press_gpa} GPa ({press_ev_a3:.5f} eV/A^3)")
print(f"n_probes={config.n_probes}, max_gen={config.max_gen}, "
      f"patience={config.patience}, seed={SEED}")

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
path, barrier = pallas.run()

t_total = time.time() - t0
print(f"\nTotal runtime: {t_total:.0f}s ({t_total/60:.1f} min)")

if path:
    nat = len(graphite)
    print(f"Barrier: {barrier:.4f} eV ({barrier/nat:.4f} eV/atom)")
    print(f"Drag-run reference: 4.483 eV (0.280 eV/atom)")
