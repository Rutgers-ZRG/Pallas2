"""PALLAS command-line interface.

    pallas run A.vasp B.vasp [--calc mattersim|emt|nequip] [--pressure GPa] ...
    pallas plot WORKDIR
    pallas export WORKDIR
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time


def _commit_id():
    try:
        r = subprocess.run(['git', 'describe', '--tags', '--always', '--dirty'],
                           capture_output=True, text=True, timeout=10,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return 'unknown'


def _make_calculator(args):
    if args.calc == 'emt':
        from ase.calculators.emt import EMT
        return EMT()
    if args.calc == 'nequip':
        if not (args.model_ef and args.model_stress):
            sys.exit("--calc nequip requires --model-ef and --model-stress")
        from pallas.nequip_calc import NequIPDualCalc
        return NequIPDualCalc(ef_model_path=args.model_ef,
                              stress_model_path=args.model_stress)
    return None  # mattersim: lazy default in pallas.optimize


def cmd_run(args):
    import numpy as np
    from ase.io import read

    from pallas import Pallas, PallasConfig, set_calculator

    t0 = time.time()
    a_path = os.path.abspath(args.A)
    b_path = os.path.abspath(args.B)
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)

    random.seed(args.seed)
    np.random.seed(args.seed)

    calc = _make_calculator(args)
    if calc is not None:
        set_calculator(calc)

    atoms_a = read(a_path, format='vasp')
    znucl = sorted(set(atoms_a.get_atomic_numbers().tolist()))

    config = PallasConfig(
        znucl=znucl,
        pressure_gpa=args.pressure if args.pressure else None,
        n_probes=args.probes,
        max_gen=args.max_gen,
        patience=args.patience,
        opt_fmax=args.fmax,
        saddle_fmax=args.saddle_fmax,
        natx=args.natx,
    )
    with open('config.json', 'w') as f:
        json.dump({'seed': args.seed, 'commit': _commit_id(), 'calc': args.calc,
                   **{k: getattr(config, k) for k in config.__dataclass_fields__}},
                  f, indent=1)

    pallas = Pallas(config)
    pallas.init_run([a_path, b_path])
    path, barrier = pallas.run()
    if path and args.k_paths > 1:
        pallas.find_best_path(k=args.k_paths)

    nodes = pallas.G.nodes
    names = [nodes[n]['xname'] for n in path] if path else []
    hvals = [float(nodes[n]['e']) for n in path] if path else []
    spgs = [nodes[n].get('spg', '') for n in path] if path else []
    summary = {
        'barrier_ev': float(barrier) if path else None,
        'path': [int(n) for n in path] if path else [],
        'path_names': names,
        'path_enthalpies_ev': hvals,
        'spacegroups': spgs,
        'n_minima': sum(1 for _, d in nodes(data=True)
                        if d.get('xname', '').startswith('M')),
        'n_saddles': sum(1 for _, d in nodes(data=True)
                         if d.get('xname', '').startswith('S')),
        'runtime_s': round(time.time() - t0, 1),
        'commit': _commit_id(),
    }
    with open('summary.json', 'w') as f:
        json.dump(summary, f, indent=1)

    if path:
        from pallas.plotting import profile_png
        profile_png(names, hvals, spgs, 'profile.png')
        print(f"\nbarrier = {barrier:.4f} eV | results in {workdir}")
    else:
        print(f"\nNo path found in {args.max_gen} generations "
              f"(see {workdir}/summary.json)")
    return 0


def cmd_plot(args):
    wd = os.path.abspath(args.workdir)
    summary_file = os.path.join(wd, 'summary.json')
    if not os.path.exists(summary_file):
        sys.exit(f"no summary.json in {wd}")
    summary = json.load(open(summary_file))
    if not summary.get('path'):
        sys.exit("no path in summary.json — nothing to plot")
    from pallas.plotting import profile_png
    out = os.path.join(wd, 'profile.png')
    profile_png(summary['path_names'], summary['path_enthalpies_ev'],
                summary.get('spacegroups', []), out)
    print(out)
    return 0


def cmd_export(args):
    wd = os.path.abspath(args.workdir)
    os.chdir(wd)
    from pallas.export_path import main as export_main
    export_main()
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog='pallas', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    pr = sub.add_parser('run', help='search a transition pathway between two structures')
    pr.add_argument('A', help='reactant structure (VASP POSCAR format)')
    pr.add_argument('B', help='product structure (VASP POSCAR format)')
    pr.add_argument('--calc', default='mattersim',
                    choices=['mattersim', 'emt', 'nequip'])
    pr.add_argument('--model-ef', help='NequIP energy/force TorchScript model')
    pr.add_argument('--model-stress', help='NequIP stress TorchScript model')
    pr.add_argument('--pressure', type=float, default=0.0, help='pressure in GPa')
    pr.add_argument('--probes', type=int, default=5)
    pr.add_argument('--max-gen', type=int, default=50)
    pr.add_argument('--patience', type=int, default=5)
    pr.add_argument('--fmax', type=float, default=0.005)
    pr.add_argument('--saddle-fmax', type=float, default=0.05)
    pr.add_argument('--natx', type=int, default=100)
    pr.add_argument('--seed', type=int, default=42)
    pr.add_argument('--k-paths', type=int, default=1)
    pr.add_argument('--workdir', default='.')
    pr.set_defaults(func=cmd_run)

    pp = sub.add_parser('plot', help='regenerate profile.png from a run workdir')
    pp.add_argument('workdir')
    pp.set_defaults(func=cmd_plot)

    pe = sub.add_parser('export', help='export pathway structures as POSCARs')
    pe.add_argument('workdir')
    pe.set_defaults(func=cmd_export)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
