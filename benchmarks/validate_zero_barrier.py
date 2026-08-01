#!/usr/bin/env python3
"""Deep-check a near-zero-barrier saddle: real index-1 saddle, or an
endpoint duplicate / relaxation artifact?

A path saddle whose enthalpy is at or below an adjacent minimum cannot be a
true adjacent saddle. Distinguishes the three explanations:
  duplicate  — saddle ~= an endpoint, missed by the run's coarse dedup;
  loose min  — the flanking minimum is under-relaxed (its H too high);
  flat/inconsistent PES — real saddle within model noise (dual-model MLIPs).

Checks (post-hoc, read-only; no graph mutation):
  1. identity: tight FP distance + dE + spacegroup vs both endpoints
  2. tight re-relaxation of endpoints (fmax 1e-3) -> recomputed margins
  3. curvature at FIXED saddle geometry from stored + random + reaction modes
  4. interpolated single-point enthalpy profile M1 -> S -> M2

Usage (repo root):
  python benchmarks/validate_zero_barrier.py WORKDIR SADDLE_ID --calc nequip-cdse
"""
import argparse
import json
import os
import sys

import numpy as np

import ase.db

from pallas.dimer import SolidStateDimer
from pallas.optimize import (GeometryGuard, getx, local_optimization,
                             set_calculator, vrand, vunit)
from pallas.structure import PallasAtom, enthalpy, fp_distance, spacegroup_label


def make_calc(spec):
    if spec == 'mattersim':
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device='cpu')
    if spec == 'nequip-cdse':
        from pallas.nequip_calc import NequIPDualCalc
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return NequIPDualCalc(
            ef_model_path=root + '/tests/cdse/model/cdse_energy_force_model.pth',
            stress_model_path=root + '/tests/cdse/model/cdse_stress_model.pth')
    raise SystemExit(f'unknown calc {spec}')


def load(db, nid, cfg):
    row = db.get(id=nid)
    pa = PallasAtom(db.get_atoms(nid))
    pa.znucl, pa.natx, pa.fpcutoff = cfg['znucl'], cfg['natx'], cfg['fpcutoff']
    pa.fp = np.array(row.data['fp'])
    pa.id = nid
    return pa, row


def curvature_at(atoms_in, mode, calc, press, n_iter=6):
    """Converged dimer curvature at FIXED geometry (rotations only)."""
    atoms = atoms_in.copy()
    atoms.calc = calc
    d = SolidStateDimer(atoms, mode=mode, dimer_separation=0.01,
                        max_rotations=8, external_stress=np.eye(3) * press)
    for _ in range(n_iter):
        d.find_minimum_mode()
    return float(d.curvature)


def interp_image(a, b, f):
    """Fractional-coordinate + cell interpolation with minimum-image wrap."""
    cell = (1.0 - f) * a.cell.array + f * b.cell.array
    sa, sb = a.get_scaled_positions(), b.get_scaled_positions()
    df = sb - sa
    df -= np.round(df)
    at = a.copy()
    at.set_cell(cell, scale_atoms=False)
    at.set_scaled_positions(sa + f * df)
    return at


def profile(a, b, calc, press, n=8, label=''):
    out = []
    for i in range(1, n):
        f = i / n
        im = interp_image(a, b, f)
        im.calc = calc
        try:
            h = enthalpy(im.get_potential_energy(), im.get_volume(), press)
        except Exception as exc:
            out.append((f, None, type(exc).__name__))
            continue
        out.append((f, h, ''))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workdir')
    ap.add_argument('saddle_id', type=int)
    ap.add_argument('--calc', required=True)
    ap.add_argument('--tight-fmax', type=float, default=0.001)
    ap.add_argument('--dup-dfp', type=float, default=5e-3,
                    help='tight FP-distance threshold for duplicate call')
    ap.add_argument('--dup-de', type=float, default=5e-3)
    args = ap.parse_args()

    os.chdir(args.workdir)
    cfg = json.load(open('config.json'))
    press = float(cfg.get('press', 0.0))
    calc = GeometryGuard(make_calc(args.calc))
    set_calculator(calc)

    db = ase.db.connect('pallas.db')
    S, srow = load(db, args.saddle_id, cfg)
    M1, _ = load(db, 1, cfg)
    M2, _ = load(db, 2, cfg)
    types = np.array([cfg['znucl'].index(z) + 1
                      for z in S.get_atomic_numbers()])

    hS = enthalpy(srow.data['energy'], S.get_volume(), press)
    print(f"S{args.saddle_id}: H={hS:.6f} eV, spg={spacegroup_label(S)[0]}, "
          f"refined={srow.data.get('refined', False)}")

    # 1 — identity vs endpoints (tight thresholds)
    verdicts = []
    for name, M in (('M1', M1), ('M2', M2)):
        d = fp_distance(S.get_fp(), M.get_fp(), types)
        hM = enthalpy(db.get(id=M.id).data['energy'], M.get_volume(), press)
        dup = d < args.dup_dfp and abs(hS - hM) < args.dup_de
        print(f"  vs {name} ({spacegroup_label(M)[0]}): d_fp={d:.6f}, "
              f"dH={hS - hM:+.6f} eV -> {'DUPLICATE' if dup else 'distinct'}")
        if dup:
            verdicts.append(f'duplicate of {name}')

    # 2 — tight endpoint re-relaxation
    for name, M in (('M1', M1), ('M2', M2)):
        Mt = M.copy()
        Mt = PallasAtom(Mt)
        Mt.znucl, Mt.natx, Mt.fpcutoff = cfg['znucl'], cfg['natx'], cfg['fpcutoff']
        opt = local_optimization(Mt, fmax=args.tight_fmax, steps=3000,
                                 press=press)
        hMt = enthalpy(opt.get_potential_energy(), opt.get_volume(), press)
        print(f"  tight {name}: H={hMt:.6f} eV (margin H_S - H_{name} = "
              f"{hS - hMt:+.6f} eV, converged={opt.converged})")
        if hS - hMt < -args.dup_de:
            verdicts.append(f'H(S) below tight {name} — not its adjacent saddle')

    # 3 — curvature robustness at fixed geometry
    mode0 = srow.data.get('dimer_mode')
    curvs = []
    if mode0 is not None:
        curvs.append(('stored', curvature_at(S, np.array(mode0), calc, press)))
    curvs.append(('M1->M2', curvature_at(S, getx(M1, M2), calc, press)))
    rng = np.random.default_rng(7)
    for k in range(3):
        m = vunit(vrand(np.zeros((len(S) + 3, 3))))
        m[0] *= 0
        m[-3, 1:] *= 0
        m[-2, 2] *= 0
        curvs.append((f'random{k}', curvature_at(S, m, calc, press)))
    for tag, c in curvs:
        print(f"  curvature[{tag}] = {c:+.4f}")
    n_neg = sum(1 for _, c in curvs if c < 0)
    if n_neg == 0:
        verdicts.append('no negative curvature from any start — a MINIMUM, '
                        'not a saddle')

    # 4 — interpolated ridge scan
    for name, M in (('M1', M1), ('M2', M2)):
        hM = enthalpy(db.get(id=M.id).data['energy'], M.get_volume(), press)
        prof = profile(M, S, calc, press, n=8)
        line = ' '.join('%.4f' % h if h is not None else f'X({e})'
                        for _, h, e in prof)
        hs = [h for _, h, _ in prof if h is not None]
        ridge = max(hs) > max(hM, hS) + 1e-4 if hs else None
        print(f"  scan {name}->S: {hM:.4f} | {line} | {hS:.4f}"
              f"  (interior max above both ends: {ridge})")

    print('\nVERDICT:', '; '.join(verdicts) if verdicts
          else 'consistent with a real saddle on a flat PES '
               '(margins within model/relaxation noise)')


if __name__ == '__main__':
    main()
