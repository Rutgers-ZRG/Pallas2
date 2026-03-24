#!/usr/bin/env python3
"""Export pathway structures from PALLAS database as VASP POSCAR files.

Usage:
    python -m pallas.export_path                    # defaults
    python -m pallas.export_path -d workdir         # specify directory
    python -m pallas.export_path -p 1 15 12 21 10 9 2  # explicit path
"""
import os
import argparse
import ase.db
import joblib
from ase.io import write


def export_pathway(db_file='pallas.db', graph_file='graph.pkl',
                   path=None, outdir='path_structures'):
    """Export structures along the best pathway as VASP POSCAR files.

    Parameters
    ----------
    db_file : str -- path to PALLAS ASE database.
    graph_file : str -- path to saved graph pickle.
    path : list of int, optional -- explicit node IDs. If None, finds
        the kinetically optimal path automatically.
    outdir : str -- output directory for POSCAR files.
    """
    db = ase.db.connect(db_file)
    G = joblib.load(graph_file)

    if path is None:
        from pallas.graph import minimax_path_kinetic
        # Find reactant and product (first and last minima by ID)
        minima_ids = sorted(n for n, d in G.nodes(data=True)
                            if d.get('xname', '').startswith('M'))
        start, end = minima_ids[0], minima_ids[1]
        path, barrier = minimax_path_kinetic(G, start, end)
        print(f"Best path: {' -> '.join(str(n) for n in path)}")
        print(f"Rate-limiting barrier: {barrier:.4f} eV")

    os.makedirs(outdir, exist_ok=True)

    for i, node_id in enumerate(path):
        nd = G.nodes[node_id]
        xname = nd['xname']
        h = nd['e']
        ntype = 'MIN' if xname.startswith('M') else 'SAD'

        row = db.get(id=node_id)
        atoms = row.toatoms()

        fname = f"{i:02d}_{xname}.vasp"
        fpath = os.path.join(outdir, fname)
        comment = f"{xname} H={h:.4f} eV V={atoms.get_volume():.1f}"
        write(fpath, atoms, format='vasp', direct=True, sort=True)

        print(f"  {fname}: {ntype} {xname}, H={h:.4f} eV, "
              f"V={atoms.get_volume():.1f} A^3")

    print(f"\n{len(path)} structures written to {outdir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export PALLAS pathway structures as VASP POSCAR files')
    parser.add_argument('-d', '--dir', default='.',
                        help='Working directory with pallas.db and graph.pkl')
    parser.add_argument('-p', '--path', nargs='+', type=int, default=None,
                        help='Explicit path node IDs (e.g. 1 15 12 21 10 9 2)')
    parser.add_argument('-o', '--outdir', default='path_structures',
                        help='Output directory (default: path_structures)')
    args = parser.parse_args()

    db_file = os.path.join(args.dir, 'pallas.db')
    graph_file = os.path.join(args.dir, 'graph.pkl')
    export_pathway(db_file, graph_file, path=args.path, outdir=args.outdir)
