#!/usr/bin/env python3
"""G-SSNEB baseline runner (TSASE) for PALLAS benchmark comparisons.

Same endpoints, calculator, and pressure as the PALLAS runs. Barrier is
reported as max-image enthalpy minus first-image enthalpy (H = E + pV).

Usage:
  python run_ssneb.py A.vasp B.vasp --pressure 12 --calc mattersim \
      --images 9 --fmax 0.05 --maxiter 600 --workdir out/
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, '/Users/li/tsase-793')

from ase.io import read
from ase.units import GPa


class CountingCalc:
    """Transparent wrapper counting force evaluations."""

    def __init__(self, calc):
        self._calc = calc
        self.count = 0

    def get_forces(self, atoms):
        self.count += 1
        return self._calc.get_forces(atoms)

    def __getattr__(self, name):
        return getattr(self._calc, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('A')
    ap.add_argument('B')
    ap.add_argument('--pressure', type=float, default=0.0, help='GPa')
    ap.add_argument('--calc', default='mattersim', choices=['mattersim', 'emt'])
    ap.add_argument('--images', type=int, default=9)
    ap.add_argument('--fmax', type=float, default=0.05)
    ap.add_argument('--maxiter', type=int, default=600)
    ap.add_argument('--workdir', default='.')
    args = ap.parse_args()

    a_path, b_path = os.path.abspath(args.A), os.path.abspath(args.B)
    os.makedirs(args.workdir, exist_ok=True)
    os.chdir(args.workdir)

    if args.calc == 'emt':
        from ase.calculators.emt import EMT
        base = EMT()
    else:
        from mattersim.forcefield import MatterSimCalculator
        base = MatterSimCalculator(device='cpu')
    calc = CountingCalc(base)

    p1 = read(a_path, format='vasp')
    p2 = read(b_path, format='vasp')
    p1.calc = calc
    p2.calc = calc

    from tsase import neb
    express = np.eye(3) * args.pressure  # GPa; ssneb converts internally
    t0 = time.time()
    status = 'ok'
    try:
        band = neb.ssneb(p1, p2, numImages=args.images, method='ci',
                         express=express)
        opt = neb.fire_ssneb(band, maxmove=0.1, dtmax=0.1, dt=0.1)
        opt.minimize(forceConverged=args.fmax, maxIterations=args.maxiter)
    except Exception as e:
        status = f'failed: {type(e).__name__}: {e}'
        band = None
    wall = time.time() - t0

    result = {'method': 'G-SSNEB (TSASE, CI)', 'images': args.images,
              'pressure_gpa': args.pressure, 'fmax': args.fmax,
              'force_calls': calc.count, 'wallclock_s': round(wall, 1),
              'status': status}
    if band is not None:
        press_ev = args.pressure * GPa
        H = []
        for im in band.path:
            e = im.get_potential_energy()
            H.append(float(e + press_ev * im.get_volume()))
        result['enthalpies_ev'] = H
        result['barrier_ev'] = float(max(H) - H[0])
        result['barrier_image'] = int(np.argmax(H))

    with open('result.json', 'w') as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
