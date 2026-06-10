#!/usr/bin/env python3
"""Test unified run(): Graphite → Hexagonal Diamond at 15 GPa.

Uses Allegro r2SCAN+rVV10 carbon MLIP.
16 atoms: graphite (1×1×4) and HD (1×1×4).
"""
import os, sys, time
sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')

from ase.spacegroup import crystal
from ase.io import write
from pallas import Pallas, PallasConfig, set_calculator

t0 = time.time()

# --- Build structures ---
# Graphite AB (P6_3/mmc, 4 atoms/cell) × (1,1,4) = 16 atoms
graphite = crystal(['C', 'C'], [(0, 0, 1/4), (1/3, 2/3, 1/4)],
                   spacegroup=194,
                   cellpar=[2.464, 2.464, 6.711, 90, 90, 120])
graphite = graphite.repeat((1, 1, 4))

# Hexagonal diamond (P6_3/mmc, 4 atoms/cell) × (1,1,4) = 16 atoms
hd = crystal('C', [(1/3, 2/3, 0.0625)],
             spacegroup=194,
             cellpar=[2.522, 2.522, 4.119, 90, 90, 120])
hd = hd.repeat((1, 1, 4))

print(f"Graphite: {len(graphite)} atoms, cell = {graphite.cell.lengths()}")
print(f"HD:       {len(hd)} atoms, cell = {hd.cell.lengths()}")

# --- Set Allegro r2SCAN+rVV10 calculator ---
from nequip.ase import NequIPCalculator
model_path = '/Users/li/dev/RA/mlip-active-learn/models/allegro_r2scan_carbon.nequip.pth'
calc = NequIPCalculator.from_compiled_model(model_path, device='cpu')
set_calculator(calc)
print(f"Calculator: Allegro r2SCAN+rVV10 carbon")

# --- Clean old files ---
print("=" * 60)
print("Unified run(): Graphite → HD at 15 GPa (16 atoms)")
print("=" * 60)

for f in ['pallas.db', 'graph.pkl', 'graph.gml', 'graph.gexf', 'dij.pkl']:
    if os.path.exists(f):
        os.remove(f)

write('POSCAR1', graphite, format='vasp', direct=True, sort=True)
write('POSCAR2', hd, format='vasp', direct=True, sort=True)

# --- Configure and run ---
press_gpa = 15.0
press_ev_a3 = press_gpa / 160.2176634  # GPa → eV/Å³

config = PallasConfig(
    znucl=[6],
    fpcutoff=5.5, natx=200, press=press_ev_a3,
    # Generational search
    n_probes=5,
    max_gen=20,
    patience=5,
    # Optimization
    opt_steps=1000, opt_fmax=0.005,
    saddle_steps=500, saddle_fmax=0.05,
    bias_steps=40,
    # FP-guided
    fp_step_scale=0.5, fp_push_scale=0.1,
    # Convergence
    ediff=0.01, dist_threshold=0.1,
)

print(f"Pressure: {press_gpa} GPa ({press_ev_a3:.5f} eV/Å³)")
print(f"n_probes={config.n_probes}, max_gen={config.max_gen}, "
      f"patience={config.patience}")

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
path, barrier = pallas.run()

t_total = time.time() - t0
print(f"\nTotal runtime: {t_total:.0f}s ({t_total/60:.1f} min)")

if path:
    nat = len(graphite)
    print(f"Barrier: {barrier:.4f} eV ({barrier/nat:.4f} eV/atom)")
