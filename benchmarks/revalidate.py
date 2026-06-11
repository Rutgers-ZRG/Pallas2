#!/usr/bin/env python3
"""Re-validate + stored-mode-refine a saved PALLAS run, then re-extract the path.

This is the production-quality post-processing pass (the CLI now does the
same in-run): validate_graph (escalating ±mode connectivity, prunes fakes)
-> refine_path_saddles (tight dimer restart WITH the stored mode, guarded
acceptance) -> kinetic-minimax re-extraction.

Usage:
  python benchmarks/revalidate.py WORKDIR --calc mattersim|nequip-cdse|allegro-carbon
"""
import argparse
import json
import os
import pickle

import numpy as np

import ase.db

from pallas import Pallas
from pallas.config import PallasConfig
from pallas.graph import minimax_path_kinetic
from pallas.optimize import GeometryGuard, set_calculator
from pallas.structure import PallasAtom


def make_calc(spec):
    if spec == 'mattersim':
        from mattersim.forcefield import MatterSimCalculator
        return MatterSimCalculator(device='cpu')
    if spec == 'nequip-cdse':
        from pallas.nequip_calc import NequIPDualCalc
        root = '/Users/li/dev/Pallas2'
        return NequIPDualCalc(
            ef_model_path=root + '/tests/cdse/model/cdse_energy_force_model.pth',
            stress_model_path=root + '/tests/cdse/model/cdse_stress_model.pth')
    if spec == 'allegro-carbon':
        from nequip.ase import NequIPCalculator
        path = (os.environ.get('ALLEGRO_MODEL')
                or '/Users/li/dev/RA/mlip-active-learn/models/allegro_r2scan_carbon.nequip.pth')
        loader = getattr(NequIPCalculator, 'from_compiled_model',
                         getattr(NequIPCalculator, 'from_deployed_model', None))
        device = os.environ.get('PALLAS_DEVICE', 'cpu')
        return loader(path, device=device)
    raise SystemExit(f'unknown calc {spec}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('workdir')
    ap.add_argument('--calc', required=True)
    ap.add_argument('--refine-fmax', type=float, default=0.01)
    args = ap.parse_args()

    os.chdir(args.workdir)
    cj = json.load(open('config.json'))
    cfg = PallasConfig(
        znucl=cj['znucl'], natx=cj.get('natx', 100),
        fpcutoff=cj.get('fpcutoff', 5.5), press=float(cj.get('press', 0.0)),
        opt_steps=800, opt_fmax=cj.get('opt_fmax', 0.005),
        saddle_steps=800, saddle_fmax=cj.get('saddle_fmax', 0.05),
        fp_push_scale=cj.get('fp_push_scale', 0.1),
        dist_threshold=cj.get('dist_threshold', 0.05),
        ediff=cj.get('ediff', 0.01))

    set_calculator(GeometryGuard(make_calc(args.calc)))

    p = Pallas(cfg)
    p.db = ase.db.connect('pallas.db')
    p._probe_stats = {}
    with open('graph.pkl', 'rb') as f:
        p.G = pickle.load(f)

    # init_minima shim (ids 1, 2 are the endpoints by construction)
    p.init_minima = []
    for nid in (1, 2):
        row = p.db.get(id=nid)
        pa = PallasAtom(p.db.get_atoms(nid))
        pa.znucl, pa.natx, pa.fpcutoff = cfg.znucl, cfg.natx, cfg.fpcutoff
        pa.fp = np.array(row.data['fp'])
        pa.id = nid
        p.init_minima.append(pa)

    path0, bn0 = minimax_path_kinetic(p.G, 1, 2)
    print(f"before: bottleneck {bn0:.4f} | "
          f"{' -> '.join(p.G.nodes[n]['xname'] for n in path0)}")

    vstats = p.validate_graph()
    try:
        path1, bn1 = minimax_path_kinetic(p.G, 1, 2)
    except Exception:
        print("after validation: DISCONNECTED — no surviving path")
        result = {'before': bn0, 'validation': vstats, 'after': None}
        json.dump(result, open('revalidate.json', 'w'), indent=1)
        return

    report = p.refine_path_saddles(path=path1, fmax=args.refine_fmax)
    path2, bn2 = minimax_path_kinetic(p.G, 1, 2)
    print(f"after:  bottleneck {bn2:.4f} | "
          f"{' -> '.join(p.G.nodes[n]['xname'] for n in path2)}")

    result = {'before_ev': float(bn0),
              'path_before': [p.G.nodes[n]['xname'] for n in path0],
              'validation': vstats,
              'refinement': report,
              'after_ev': float(bn2),
              'path_after': [p.G.nodes[n]['xname'] for n in path2]}
    json.dump(result, open('revalidate.json', 'w'), indent=1)
    print('-> revalidate.json')


if __name__ == '__main__':
    main()
