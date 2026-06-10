"""Pathway analysis, trajectory output, and reporting."""


import os

import ase.db
import joblib
import networkx as nx
from ase.io import read, write

from pallas.config import PallasConfig
from pallas.graph import minimax_path_kinetic

# ── Main PALLAS class ────────────────────────────────────────────────


class AnalysisMixin:
    """Result reporting and trajectory export."""

    def _report_result(self, best_path, best_barrier, A, known):
        """Print final search results."""
        n_min = sum(1 for _, d in self.G.nodes(data=True)
                    if d.get('xname', '').startswith('M'))
        n_sad = sum(1 for _, d in self.G.nodes(data=True)
                    if d.get('xname', '').startswith('S'))

        print(f"\n{'='*60}")
        print(f"Search complete: {n_min} minima, {n_sad} saddles, "
              f"{self.G.number_of_edges()} edges")
        print(f"Known minima explored: {len(known)}")

        if best_path:
            print(f"\nBest path: {' -> '.join(str(n) for n in best_path)}")
            for node in best_path:
                nd = self.G.nodes[node]
                ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
                print(f"  {ntype} {node}: H={nd['e']:.4f} eV, "
                      f"V={nd['volume']:.1f} A^3")
            self.analyze_pathway(best_path)
            self.get_pathway_trajectory(best_path)
        else:
            print("\nNo complete path found")
        print(f"{'='*60}")

    # ── FP-guided step (probe core) ──────────────────────────────────

    def _save_to_traj(self, atoms, node_id, ctyp, enthalpy):
        """Append a structure to the exploration trajectory file."""
        from ase.io import write as ase_write
        atoms_copy = atoms.copy()
        atoms_copy.info['node_id'] = node_id
        atoms_copy.info['type'] = ctyp
        atoms_copy.info['enthalpy_eV'] = float(enthalpy)
        ase_write('pallas_traj.extxyz', atoms_copy, append=True)

    def get_pathway_trajectory(self, path, filename='pathway.extxyz'):
        """Extract smooth trajectory along a path by stitching per-edge files.

        For each consecutive pair of nodes on the path, looks for a saved
        per-probe trajectory file (traj_M*_S*_M*.extxyz) on the edge.
        If found, includes all intermediate frames for a smooth movie.
        Falls back to key-frame-only output if no trajectory files exist.

        Parameters
        ----------
        path : list of int — node IDs from minimax_path (e.g. [1, 5, 3, 8, 2]).
        filename : str — output file.

        Returns
        -------
        list of Atoms — ordered structures along the path.
        """
        from ase.io import read as ase_read
        from ase.io import write as ase_write

        smooth_traj = []
        used_files = set()

        # Try to build smooth trajectory from per-edge files
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            edge_data = self.G.edges.get((n1, n2), {})
            traj_file = edge_data.get('traj_file', None)

            if traj_file and os.path.exists(traj_file) \
                    and traj_file not in used_files:
                frames = ase_read(traj_file, index=':')
                # Determine direction: file is always saved as
                # M(source)→S→M(new), so if edge is reversed, flip frames
                if i == 0:
                    smooth_traj.extend(frames)
                else:
                    # Skip first frame to avoid duplicating the junction
                    smooth_traj.extend(frames[1:])
                used_files.add(traj_file)
            else:
                # Fallback: add key frame from database
                node_id = path[i]
                if not smooth_traj:  # only add if we haven't started
                    row = self.db.get(id=node_id)
                    atoms = row.toatoms()
                    nd = self.G.nodes[node_id]
                    atoms.info['xname'] = nd['xname']
                    atoms.info['enthalpy_eV'] = nd['e']
                    smooth_traj.append(atoms)

        # Always add final node
        if path:
            last_id = path[-1]
            if not used_files:
                row = self.db.get(id=last_id)
                atoms = row.toatoms()
                nd = self.G.nodes[last_id]
                atoms.info['xname'] = nd['xname']
                atoms.info['enthalpy_eV'] = nd['e']
                smooth_traj.append(atoms)

        if smooth_traj:
            ase_write(filename, smooth_traj)
            print(f"Pathway trajectory written to {filename} "
                  f"({len(smooth_traj)} frames, "
                  f"{len(used_files)} edge trajectories stitched)")
        return smooth_traj

    # ── I/O ──────────────────────────────────────────────────────────

    def analyze_pathway(self, path):
        """Decompose a pathway into per-step barriers and identify traps.

        For path M1→S1→M3→S2→M2, reports:
        - Per-step forward barriers (each saddle minus preceding minimum)
        - Overall forward barrier from start (max saddle - H_start)
        - Rate-limiting step (max local barrier)
        - Deep intermediates (minima significantly below adjacent saddles)

        Parameters
        ----------
        path : list of node IDs — alternating minima and saddles.

        Returns
        -------
        dict with keys: 'steps', 'forward_barrier', 'rate_limiting_barrier',
                        'rate_limiting_step', 'start_H', 'end_H'.
        """
        nodes = self.G.nodes
        h_start = nodes[path[0]]['e']
        h_end = nodes[path[-1]]['e']

        # Walk path and extract steps (handles min-saddle-min and min-min)
        steps = []
        i = 0
        while i < len(path) - 1:
            cur = path[i]
            cur_is_min = nodes[cur]['xname'].startswith('M')
            nxt = path[i + 1]
            nxt_is_sad = nodes[nxt]['xname'].startswith('S')

            if cur_is_min and nxt_is_sad and i + 2 < len(path):
                # Normal step: M → S → M
                sad_id = nxt
                min_after = path[i + 2]
                h_mb = nodes[cur]['e']
                h_sad = nodes[sad_id]['e']
                h_ma = nodes[min_after]['e']
                steps.append({
                    'min_before': cur, 'saddle': sad_id,
                    'min_after': min_after,
                    'H_min_before': h_mb, 'H_saddle': h_sad,
                    'H_min_after': h_ma,
                    'forward_barrier': h_sad - h_mb,
                    'reverse_barrier': h_sad - h_ma,
                })
                i += 2
            elif cur_is_min and not nxt_is_sad:
                # Direct min→min connection (same structure)
                steps.append({
                    'min_before': cur, 'saddle': None,
                    'min_after': nxt,
                    'H_min_before': nodes[cur]['e'], 'H_saddle': None,
                    'H_min_after': nodes[nxt]['e'],
                    'forward_barrier': 0.0, 'reverse_barrier': 0.0,
                })
                i += 1
            else:
                i += 1

        # Overall forward barrier = max saddle - start
        saddle_Hs = [s['H_saddle'] for s in steps if s['H_saddle'] is not None]
        max_sad_H = max(saddle_Hs) if saddle_Hs else h_start
        forward_barrier = max_sad_H - h_start

        # Rate-limiting step = max local forward barrier
        rate_limiting = max(steps, key=lambda s: s['forward_barrier']) \
            if steps else None
        rl_barrier = rate_limiting['forward_barrier'] if rate_limiting else 0.0

        # Print analysis
        nat = len(self.init_minima[0]) if self.init_minima else 1
        print(f"\n{'─'*60}")
        print(f"Pathway analysis ({len(steps)} steps)")
        print(f"{'─'*60}")
        print(f"Start M{path[0]}: H = {h_start:.4f} eV")

        for k, s in enumerate(steps):
            trap_flag = ""
            if s['forward_barrier'] > forward_barrier + 0.001:
                trap_flag = "  ** DEEP TRAP **"
            if s['saddle'] is not None:
                print(f"  Step {k+1}: M{s['min_before']} -> "
                      f"S{s['saddle']} -> M{s['min_after']}")
                print(f"    Forward barrier: {s['forward_barrier']:.4f} eV "
                      f"({s['forward_barrier']/nat:.4f} eV/atom){trap_flag}")
                print(f"    Reverse barrier: {s['reverse_barrier']:.4f} eV")
            else:
                print(f"  Step {k+1}: M{s['min_before']} == "
                      f"M{s['min_after']} (direct connection)")

        print(f"End M{path[-1]}: H = {h_end:.4f} eV")
        print(f"\nOverall forward barrier (max saddle - start): "
              f"{forward_barrier:.4f} eV ({forward_barrier/nat:.4f} eV/atom)")
        print("Rate-limiting step: ", end="")
        if rate_limiting and rate_limiting['saddle'] is not None:
            print(f"M{rate_limiting['min_before']} -> "
                  f"S{rate_limiting['saddle']} -> "
                  f"M{rate_limiting['min_after']}, "
                  f"barrier = {rl_barrier:.4f} eV "
                  f"({rl_barrier/nat:.4f} eV/atom)")
        else:
            print("(no barriers)")

        if rl_barrier > forward_barrier + 0.01:
            print(f"WARNING: Rate-limiting barrier ({rl_barrier:.4f}) > "
                  f"forward barrier ({forward_barrier:.4f}) — deep "
                  f"intermediate traps on pathway!")
        print(f"{'─'*60}")

        return {
            'steps': steps,
            'forward_barrier': forward_barrier,
            'rate_limiting_barrier': rl_barrier,
            'rate_limiting_step': rate_limiting,
            'start_H': h_start, 'end_H': h_end,
        }


# ── Standalone analysis utilities ─────────────────────────────────────



def listpath(graph_file='graph.pkl', db_file='pallas.db',
             start=1, end=None):
    """Load saved graph and list all paths between two nodes.

    Parameters
    ----------
    graph_file : str — path to saved graph pickle.
    db_file : str — path to ASE database.
    start, end : int — node IDs for reactant and product.
    """
    G = joblib.load(graph_file)
    db = ase.db.connect(db_file)

    if end is None:
        # Find the last minimum node
        minima_ids = [n for n, d in G.nodes(data=True)
                      if d.get('xname', '').startswith('M')]
        if len(minima_ids) < 2:
            print("Not enough minima nodes in graph")
            return
        end = max(minima_ids)

    try:
        path, bottleneck = minimax_path_kinetic(G, start, end)
    except nx.NetworkXNoPath:
        print(f"No path between {start} and {end}")
        return

    print(f"Best path (kinetic minimax): {path}")
    print(f"Rate-limiting barrier: {bottleneck:.4f}")

    # Create output directory
    os.makedirs("path_output", exist_ok=True)

    with open("path_output/path_info.txt", 'w') as f:
        f.write(f"Minimax path: {path}\n")
        f.write(f"Bottleneck energy: {bottleneck:.6f}\n")
        f.write(f"Number of nodes: {len(path)}\n\n")

        cumulative_dist = 0.0
        for j, node in enumerate(path):
            nd = G.nodes[node]
            ntype = 'Minimum' if nd['xname'].startswith('M') else 'Saddle'
            f.write(f"Node {node} ({ntype}): E={nd['e']:.6f}, "
                    f"V={nd['volume']:.4f}\n")
            print(f"  {ntype} {node}: E={nd['e']:.4f}")

            # Write POSCAR
            try:
                atoms = db.get_atoms(node)
                write(f"path_output/{node}_POSCAR", atoms,
                      format='vasp', direct=True)
            except Exception:
                pass

            # Edge info
            if j < len(path) - 1:
                ed = G.get_edge_data(path[j], path[j + 1])
                w = ed.get('weight', float('inf'))
                d = ed.get('dist', 0.0)
                cumulative_dist += d
                f.write(f"  → next: weight={w:.6f}, dist={d:.6f}\n")

    print("Output written to path_output/")


# ── Entry points ──────────────────────────────────────────────────────

def main():
    """Default run: read POSCAR1/POSCAR2, run FP-guided multi-probe search."""
    from pallas.search import Pallas  # local import: search imports this module

    config = PallasConfig()
    atoms = read('POSCAR1', format='vasp')
    config.znucl = sorted(set(atoms.get_atomic_numbers().tolist()))

    pallas = Pallas(config)
    pallas.init_run(['POSCAR1', 'POSCAR2'])
    pallas.run_fp_guided(n_probes=3)


if __name__ == "__main__":
    main()
