#!/usr/bin/env python3
"""Audit the transition states on a saved PALLAS run's best path.

Checks per saddle on the kinetic-minimax path:
  A. topology  — H(saddle) strictly above BOTH flanking path minima (graph only)
  B. re-dimer  — restart a small-budget dimer from the stored saddle structure:
                 a true TS re-converges in place (small d_fp, small |dH|) with
                 curvature < 0; a shoulder/minimum walks away
  C. connectivity — push +/- the re-converged dimer mode, relax, and verify the
                 two basins match the flanking path minima (FP distance)

Usage:
  python benchmarks/audit_saddles.py WORKDIR --calc mattersim [--pressure GPa]
  python benchmarks/audit_saddles.py WORKDIR --calc nequip-cdse | allegro-carbon
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

import ase.db
from ase.units import GPa

from pallas import cal_saddle
from pallas.config import PallasConfig
from pallas.graph import minimax_path_kinetic
from pallas.optimize import GeometryGuard, set_calculator
from pallas.probes import vunit
from pallas.search import Pallas
from pallas.structure import PallasAtom, fp_distance


def load_run(workdir):
    with open(os.path.join(workdir, 'graph.pkl'), 'rb') as f:
        G = pickle.load(f)
    db = ase.db.connect(os.path.join(workdir, 'pallas.db'))
    cfg_file = os.path.join(workdir, 'config.json')
    cj = json.load(open(cfg_file)) if os.path.exists(cfg_file) else {}
    return G, db, cj


def make_calc(spec):
    if spec == 'mattersim':
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device='cpu')
    if spec == 'nequip-cdse':
        from pallas.nequip_calc import NequIPDualCalc
        return NequIPDualCalc(
            ef_model_path='tests/cdse/model/cdse_energy_force_model.pth',
            stress_model_path='tests/cdse/model/cdse_stress_model.pth')
    if spec == 'allegro-carbon':
        from nequip.ase import NequIPCalculator
        import os as _os
        path = (_os.environ.get('ALLEGRO_MODEL')
                or '/Users/li/dev/RA/mlip-active-learn/models/allegro_r2scan_carbon.nequip.pth')
        loader = getattr(NequIPCalculator, 'from_compiled_model',
                         getattr(NequIPCalculator, 'from_deployed_model', None))
        return loader(path, device='cpu')
    raise SystemExit(f'unknown calc {spec}')


def patoms(db, node_id, znucl, natx, fpcutoff):
    row = db.get(id=node_id)
    pa = PallasAtom(db.get_atoms(node_id))
    pa.znucl = znucl
    pa.natx = natx
    pa.fpcutoff = fpcutoff
    pa.fp = np.array(row.data['fp'])
    pa.id = node_id
    return pa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workdir')
    ap.add_argument('--calc', required=True)
    ap.add_argument('--pressure', type=float, default=None,
                    help='GPa; default from run config.json')
    ap.add_argument('--saddle-steps', type=int, default=300)
    args = ap.parse_args()

    G, db, cj = load_run(args.workdir)
    znucl = cj.get('znucl') or sorted({z for rid in (1,) for z in
                                       db.get_atoms(rid).get_atomic_numbers()})
    natx = cj.get('natx', 100)
    fpcutoff = cj.get('fpcutoff', 5.5)
    press = (args.pressure * GPa if args.pressure is not None
             else float(cj.get('press', 0.0)))

    calc = GeometryGuard(make_calc(args.calc))
    set_calculator(calc)

    # Endpoint ids are 1 and 2 by construction (init_run registers them first)
    path, bottleneck = minimax_path_kinetic(G, 1, 2)
    print(f"workdir: {args.workdir}")
    print(f"path: {' -> '.join(G.nodes[n]['xname'] for n in path)} | "
          f"kinetic bottleneck {bottleneck:.4f} eV | press {press:.5f} eV/A^3")

    cfg = PallasConfig(znucl=znucl, natx=natx, fpcutoff=fpcutoff, press=press,
                       opt_steps=800, opt_fmax=cj.get('opt_fmax', 0.005),
                       fp_push_scale=cj.get('fp_push_scale', 0.1))
    shim = Pallas(cfg)   # for _push_and_relax/_get_types reuse
    shim.G = G
    types = np.array([znucl.index(z) + 1 for z in
                      db.get_atoms(1).get_atomic_numbers()])

    report = []
    for i, n in enumerate(path):
        if not G.nodes[n]['xname'].startswith('S'):
            continue
        h_s = G.nodes[n]['e']
        h_prev = G.nodes[path[i - 1]]['e']
        h_next = G.nodes[path[i + 1]]['e']
        entry = {'saddle': G.nodes[n]['xname'], 'H': round(h_s, 4),
                 'above_prev': bool(h_s > h_prev), 'above_next': bool(h_s > h_next),
                 'local_barrier_fwd': round(h_s - h_prev, 4)}

        sad = patoms(db, n, znucl, natx, fpcutoff)
        sad.calc = calc
        e0 = sad.get_potential_energy()

        re = cal_saddle(sad, fmax=cj.get('saddle_fmax', 0.05),
                        steps=args.saddle_steps, calc=calc, press=press)
        e1 = re.get_potential_energy()
        d_move = fp_distance(_fp(re, znucl, natx, fpcutoff),
                             sad.get_fp(), types)
        entry.update({
            'redimer_converged': bool(getattr(re, 'converged', False)),
            'curvature': (round(float(re.dimer_curvature), 4)
                          if re.dimer_curvature is not None else None),
            'dH_reconverge': round(float(e1 - e0), 4),
            'dfp_reconverge': round(float(d_move), 5),
        })

        # C: connectivity from the re-converged saddle
        mode = vunit(re.dimer_mode)
        nat = len(re)
        jacob = (re.get_volume() / nat) ** (1 / 3) * nat ** 0.5
        rep = PallasAtom(re)
        rep.znucl, rep.natx, rep.fpcutoff = znucl, natx, fpcutoff
        mp = shim._push_and_relax(rep, mode, cfg.fp_push_scale * 3.0, jacob)
        mm = shim._push_and_relax(rep, -mode, cfg.fp_push_scale * 3.0, jacob)
        if mp is not None and mm is not None:
            prev_pa = patoms(db, path[i - 1], znucl, natx, fpcutoff)
            next_pa = patoms(db, path[i + 1], znucl, natx, fpcutoff)
            d_pp = fp_distance(_fp(mp, znucl, natx, fpcutoff), prev_pa.get_fp(), types)
            d_pn = fp_distance(_fp(mp, znucl, natx, fpcutoff), next_pa.get_fp(), types)
            d_mp_ = fp_distance(_fp(mm, znucl, natx, fpcutoff), prev_pa.get_fp(), types)
            d_mn = fp_distance(_fp(mm, znucl, natx, fpcutoff), next_pa.get_fp(), types)
            d_sides = fp_distance(_fp(mp, znucl, natx, fpcutoff),
                                  _fp(mm, znucl, natx, fpcutoff), types)
            entry.update({
                'sides_distinct': bool(d_sides > cj.get('dist_threshold', 0.05)),
                'd_sides': round(float(d_sides), 5),
                'matches_flanks': bool(min(d_pp, d_mp_) < 0.06 and
                                       min(d_pn, d_mn) < 0.06),
                'd_to_prev': round(float(min(d_pp, d_mp_)), 5),
                'd_to_next': round(float(min(d_pn, d_mn)), 5),
            })
        else:
            entry.update({'sides_distinct': None, 'matches_flanks': None})

        report.append(entry)
        print(json.dumps(entry))

    out = os.path.join(args.workdir, 'saddle_audit.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=1)
    print(f"-> {out}")


def _fp(atoms, znucl, natx, fpcutoff):
    pa = PallasAtom(atoms)
    pa.znucl, pa.natx, pa.fpcutoff = znucl, natx, fpcutoff
    return pa.get_fp()


if __name__ == '__main__':
    main()
