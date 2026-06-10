#!/usr/bin/env python3
"""G-SSNEB baseline: carbon graphite -> HD at 15 GPa, Allegro r2SCAN (Amarel GPU).

Endpoints rebuilt exactly as in the PALLAS runs, then relaxed at 15 GPa with
the same calculator before the band starts.
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, '/scratch/lz432/tsase-793')

from ase.spacegroup import crystal
from ase.units import GPa

P_GPA = 15.0
press_ev = P_GPA * GPa

graphite = crystal(['C', 'C'], [(0, 0, 1 / 4), (1 / 3, 2 / 3, 1 / 4)],
                   spacegroup=194,
                   cellpar=[2.464, 2.464, 6.711, 90, 90, 120]).repeat((1, 1, 4))
hd = crystal('C', [(1 / 3, 2 / 3, 0.0625)], spacegroup=194,
             cellpar=[2.522, 2.522, 4.119, 90, 90, 120]).repeat((1, 1, 4))

from nequip.ase import NequIPCalculator
model_path = '/scratch/lz432/allegro_r2scan_finetune/allegro_r2scan_carbon.nequip.pth'
calc = NequIPCalculator.from_compiled_model(model_path, device='cuda')

COUNT = [0]
_orig = calc.get_forces


def counting_forces(atoms):
    COUNT[0] += 1
    return _orig(atoms)


calc.get_forces = counting_forces

from pallas import local_optimization

p1 = local_optimization(graphite, fmax=0.005, steps=1000, calc=calc, press=press_ev)
p2 = local_optimization(hd, fmax=0.005, steps=1000, calc=calc, press=press_ev)
print(f"relaxed: gra V={p1.get_volume():.1f} A^3, hd V={p2.get_volume():.1f} A^3")

p1.calc = calc
p2.calc = calc

from tsase import neb

t0 = time.time()
status = 'ok'
band = None
try:
    band = neb.ssneb(p1, p2, numImages=9, method='ci',
                     express=np.eye(3) * P_GPA)
    opt = neb.fire_ssneb(band, maxmove=0.1, dtmax=0.1, dt=0.1)
    opt.minimize(forceConverged=0.05, maxIterations=2000)
except Exception as e:
    status = f'failed: {type(e).__name__}: {e}'

result = {'method': 'G-SSNEB (TSASE, CI)', 'images': 9, 'pressure_gpa': P_GPA,
          'force_calls': COUNT[0], 'wallclock_s': round(time.time() - t0, 1),
          'status': status}
if band is not None:
    H = [float(im.get_potential_energy() + press_ev * im.get_volume())
         for im in band.path]
    result['enthalpies_ev'] = H
    result['barrier_ev'] = float(max(H) - H[0])

with open('result.json', 'w') as f:
    json.dump(result, f, indent=1)
print(json.dumps(result, indent=1))
