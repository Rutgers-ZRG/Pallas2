#!/usr/bin/env python3
"""Test unified run() with NequIP CdSe model: drag-initialized dimer."""
import os, sys, time
sys.path.insert(0, '/Users/li/dev/torch-fplib')
sys.path.insert(0, '/Users/li/dev/Pallas2')

from ase.build import bulk
from ase.io import write
from pallas.nequip_calc import NequIPDualCalc
from pallas.optimize import set_calculator
from pallas import Pallas, PallasConfig

t0 = time.time()

# Set NequIP CdSe calculator
calc = NequIPDualCalc(
    ef_model_path='/Users/li/dev/Pallas2/tests/cdse/model/cdse_energy_force_model.pth',
    stress_model_path='/Users/li/dev/Pallas2/tests/cdse/model/cdse_stress_model.pth')
set_calculator(calc)
print("Using NequIP CdSe model")

rs = bulk('CdSe', 'rocksalt', a=6.05).repeat((2, 1, 1))
wz = bulk('CdSe', 'wurtzite', a=4.30, c=7.01)

print("=" * 60)
print("Unified run(): CdSe RS -> WZ (NequIP, 8 atoms)")
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
    n_probes=3,
    max_gen=10,
    patience=3,
    opt_steps=500, opt_fmax=0.005,
    saddle_steps=500, saddle_fmax=0.05,
    bias_steps=40,
    fp_step_scale=0.5, fp_push_scale=0.1,
    ediff=0.01, dist_threshold=0.1,
)

print(f"n_probes={config.n_probes}, max_gen={config.max_gen}, "
      f"patience={config.patience}")

pallas = Pallas(config)
pallas.init_run(['POSCAR1', 'POSCAR2'])
path, barrier = pallas.run()

t_total = time.time() - t0
print(f"\nTotal runtime: {t_total:.0f}s ({t_total/60:.1f} min)")

if path:
    nat = len(rs)
    print(f"Barrier: {barrier:.4f} eV ({barrier/nat:.4f} eV/atom, "
          f"{barrier/(nat//2):.4f} eV/f.u.)")
