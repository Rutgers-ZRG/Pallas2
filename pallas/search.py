"""Pallas search engine: generational connect/refine orchestration."""



import ase.db
import joblib
import networkx as nx
import numpy as np
from ase.io import read
from ase.units import GPa

# ── Main PALLAS class ────────────────────────────────────────────────
from pallas.analysis import AnalysisMixin
from pallas.config import PallasConfig
from pallas.graph import minimax_path_kinetic
from pallas.optimize import local_optimization
from pallas.probes import ProbeMixin
from pallas.structure import PallasAtom, fp_distance


class Pallas(ProbeMixin, AnalysisMixin):
    """PALLAS pathway search engine.

    Primary method:
    - ``run()`` — unified generational search (connect + refine + converge)

    Advanced / legacy methods:
    - ``run_fp_guided()`` — FP-gradient-guided chain growing
    - ``refine_barrier()`` — iterative bottleneck refinement
    - ``validate_graph()`` — saddle connectivity validation

    Usage::

        config = PallasConfig(znucl=[6], press=10.0, n_probes=5)
        pallas = Pallas(config)
        pallas.init_run(['POSCAR_graphite', 'POSCAR_diamond'])
        path, barrier = pallas.run()
    """

    def __init__(self, config=None):
        self.config = config or PallasConfig()
        self.init_minima = []
        self.all_minima = []
        self.all_saddle = []
        self.dij = {}                # sparse: (id1, id2) → fp_dist
        self.baseenergy = 0.0
        self.G = nx.Graph()
        self.db = None

        # PSO state
        self.pbestx = []
        self.pbesty = []
        self.gbestx = None
        self.gbesty = None
        self.pdistx = []
        self.pdisty = []
        self.bestdist = float('inf')

    # ── Distance matrix (sparse dict) ────────────────────────────────

    def update_dij(self, id1, id2, fp_dist):
        """Store symmetric FP distance."""
        key = (min(id1, id2), max(id1, id2))
        self.dij[key] = fp_dist

    def get_dij(self, id1, id2):
        """Retrieve FP distance (0 for same ID, inf if unknown)."""
        if id1 == id2:
            return 0.0
        key = (min(id1, id2), max(id1, id2))
        return self.dij.get(key, float('inf'))

    # ── Initialization ───────────────────────────────────────────────

    def _get_types(self, atoms):
        """Get 1-indexed type array from atoms and config znucl."""
        numbers = atoms.get_atomic_numbers()
        return np.array([self.config.znucl.index(z) + 1 for z in numbers])

    def init_run(self, flist):
        """Initialize from POSCAR files.

        Parameters
        ----------
        flist : list of str
            At least 2 POSCAR paths: [reactant, product, ...optional intermediates].
        """
        if len(flist) < 2:
            raise ValueError("Need at least 2 structures (reactant and product)")

        self.db = ase.db.connect('pallas.db')

        self.init_minima = []
        for xf in flist:
            x = read(xf, format='vasp')
            pa = PallasAtom(x)
            pa.fpcutoff = self.config.fpcutoff
            pa.natx = self.config.natx
            pa.znucl = self.config.znucl
            self.init_minima.append(pa)

        self.reactant = self.init_minima[0]
        self.product = self.init_minima[1]

    # ── Unified generational search ──────────────────────────────────

    def run(self):
        """Unified pathway search: connect, then hybrid refine+explore.

        Generational loop with three modes:

        - **CONNECT**: When no A→B path exists, all probes extend chains
          from both sides toward each other.
        - **HYBRID**: Once connected, probes split between refining the
          bottleneck saddle and exploring from frontier intermediates to
          discover alternative pathways through new phases.

        Each generation launches ``n_probes`` saddle searches.  All
        discovered structures enter a shared graph; ``minimax_path``
        extracts the optimal route after each generation.

        Stops when no barrier improvement for ``patience`` consecutive
        generations, or ``max_gen`` generations reached.

        Returns
        -------
        tuple : (best_path, best_barrier) or (None, inf) if no path found.
        """
        cfg = self.config
        print(f"PALLAS search: n_probes={cfg.n_probes}, "
              f"max_gen={cfg.max_gen}, patience={cfg.patience}")

        # Optimize endpoints
        A = self._optimize_and_register(self.reactant, is_base=True)
        B = self._optimize_and_register(self.product, is_base=False)
        types = self._get_types(A)
        d0 = fp_distance(A.get_fp(), B.get_fp(), types)
        print(f"Endpoints: A (M{A.id}), B (M{B.id}), d_fp={d0:.5f}")

        # Track all known minima (id → PallasAtom)
        known = {A.id: A, B.id: B}

        best_barrier = float('inf')
        best_path = None
        stagnant = 0

        for gen in range(cfg.max_gen):
            # Check current path status
            path_exists = False
            try:
                cur_path, cur_barrier = minimax_path_kinetic(self.G, A.id, B.id)
                path_exists = True
            except nx.NetworkXNoPath:
                pass

            # Plan probes based on mode
            if not path_exists:
                probes = self._plan_connect_probes(A, B, known, types)
                mode = "CONNECT"
            else:
                probes = self._plan_hybrid_probes(
                    cur_path, A, B, known, types)
                mode = "HYBRID"

            bn_str = f", barrier={best_barrier:.4f}" if best_path else ""
            print(f"\n=== Gen {gen+1}/{cfg.max_gen} [{mode}]{bn_str} ===")

            # Execute probes
            for source, target_fp, alpha, label in probes:
                new_min = self._fp_guided_step(
                    source, target_fp, types,
                    step_scale=cfg.fp_step_scale,
                    alpha=alpha, side=label)
                if new_min is not source and new_min.id not in known:
                    known[new_min.id] = new_min

            # Cross-connect close minima
            self._cross_connect_all(known, types)

            # Evaluate best path
            try:
                path, barrier = minimax_path_kinetic(self.G, A.id, B.id)
                if barrier < best_barrier - cfg.min_barrier_change:
                    old = best_barrier
                    best_barrier = barrier
                    best_path = path
                    stagnant = 0
                    if old < float('inf'):
                        print(f"  NEW BEST: barrier={barrier:.4f} "
                              f"(improved by {old - barrier:.4f})")
                    else:
                        print(f"  FIRST PATH: barrier={barrier:.4f}")
                else:
                    if path_exists:
                        stagnant += 1
                    print(f"  barrier={barrier:.4f} "
                          f"(stagnant {stagnant}/{cfg.patience})")
            except nx.NetworkXNoPath:
                print("  No path yet")

            self._save_state()

            if path_exists and stagnant >= cfg.patience:
                print(f"\nConverged: no improvement for "
                      f"{cfg.patience} generations")
                break

        # Final report
        self._report_result(best_path, best_barrier, A, known)
        return best_path, best_barrier

    # ── PSO core ─────────────────────────────────────────────────────

    def _optimize_and_register(self, structure, is_base=False):
        """Optimize a structure and register it in the graph."""
        cfg = self.config
        opt = local_optimization(structure, fmax=cfg.opt_fmax, steps=cfg.opt_steps)
        idm, _ = self._update_minima(opt)
        opt.id = idm

        if is_base:
            self.baseenergy = (opt.get_volume() * self.config.press * GPa
                               + opt.get_potential_energy())
            h = 0.0
        else:
            h = (opt.get_volume() * self.config.press * GPa
                 + opt.get_potential_energy() - self.baseenergy)

        self.G.add_node(idm, xname=f'M{idm}', e=h, volume=opt.get_volume())
        self._save_to_traj(opt, idm, 'minima', h)
        print(f"Registered {'base' if is_base else 'target'}: "
              f"ID={idm}, H={h:.4f} eV")
        return opt

    def run_fp_guided(self, n_probes=1):
        """FP-gradient-guided bidirectional multi-probe pathway search.

        At each step, launches ``n_probes`` saddle searches per side with
        varying FP-gradient / random mixing ratios.  All discovered saddles
        and minima feed into a single graph; ``minimax_path`` automatically
        extracts the lowest-bottleneck route.

        Probe mixing schedule (per step, per side):
          probe 0 : pure FP gradient        (alpha = 1.0)
          probe 1 : 70 % FP + 30 % random   (alpha = 0.7)
          probe 2 : 40 % FP + 60 % random   (alpha = 0.4)
          probe 3+: 10 % FP + 90 % random   (alpha = 0.1)

        Parameters
        ----------
        n_probes : int
            Saddle searches per side per step (default 1 = pure FP guided).

        Returns
        -------
        nx.Graph — pathway network.
        """
        cfg = self.config
        print(f"Starting FP-guided search (n_probes={n_probes})")

        # Optimize endpoints
        A = self._optimize_and_register(self.reactant, is_base=True)
        B = self._optimize_and_register(self.product, is_base=False)

        types = self._get_types(A)
        d0 = fp_distance(A.get_fp(), B.get_fp(), types)
        print(f"Initial FP distance A↔B: {d0:.5f}")

        # Track all discovered minima per side (for multi-probe branching)
        minima_A = [A]      # all minima reachable from A
        minima_B = [B]      # all minima reachable from B

        best_bottleneck = float('inf')
        best_path = None

        for step in range(cfg.maxstep):
            # Pick the chain tip closest to the other side
            tip_A = self._closest_to(minima_A, B.get_fp(), types)
            tip_B = self._closest_to(minima_B, A.get_fp(), types)

            d_tips = fp_distance(tip_A.get_fp(), tip_B.get_fp(), types)
            ediff = abs(tip_A.get_potential_energy()
                        - tip_B.get_potential_energy())

            # Report current best path if exists
            bn_str = f", best barrier={best_bottleneck:.4f}" \
                if best_path else ""
            print(f"\n=== Step {step+1}/{cfg.maxstep} | "
                  f"d(tips)={d_tips:.5f}, ΔE={ediff:.5f}{bn_str} ===")

            # Check if tips connected
            if d_tips < cfg.dist_threshold and ediff < cfg.ediff:
                h_a = self.G.nodes[tip_A.id]['e']
                h_b = self.G.nodes[tip_B.id]['e']
                self.G.add_edge(tip_A.id, tip_B.id,
                                weight=max(h_a, h_b), dist=d_tips)
                print(f"  Tips connected: M{tip_A.id} ↔ M{tip_B.id}")

            # Adaptive step scale
            progress = d_tips / max(d0, 1e-10)
            step_scale = cfg.fp_step_scale * max(progress, 0.1)

            # Multi-probe: launch n_probes from each side
            for p in range(n_probes):
                alpha = max(1.0 - 0.3 * p, 0.1)
                label_A = f"A→B p{p}(α={alpha:.1f})"
                label_B = f"B→A p{p}(α={alpha:.1f})"

                # A-side probe
                new_A = self._fp_guided_step(
                    tip_A, B.get_fp(), types, step_scale,
                    alpha=alpha, side=label_A)
                if new_A is not tip_A:
                    minima_A.append(new_A)

                # B-side probe
                new_B = self._fp_guided_step(
                    tip_B, A.get_fp(), types, step_scale,
                    alpha=alpha, side=label_B)
                if new_B is not tip_B:
                    minima_B.append(new_B)

            # Cross-check: connect any close A/B minima pairs
            self._cross_connect(minima_A, minima_B, types)

            # Update best path
            try:
                path, bottleneck = minimax_path_kinetic(self.G, A.id, B.id)
                if bottleneck < best_bottleneck:
                    best_bottleneck = bottleneck
                    best_path = path
                    print(f"  New best path: {' → '.join(str(n) for n in path)}"
                          f" (barrier={bottleneck:.4f})")
            except nx.NetworkXNoPath:
                pass

            self._save_state()

        # Final report
        n_min = sum(1 for _, d in self.G.nodes(data=True)
                    if d['xname'].startswith('M'))
        n_sad = sum(1 for _, d in self.G.nodes(data=True)
                    if d['xname'].startswith('S'))
        print(f"\nGraph: {n_min} minima, {n_sad} saddles, "
              f"{self.G.number_of_edges()} edges")
        print(f"Discovered minima: {len(minima_A)} (A-side), "
              f"{len(minima_B)} (B-side)")

        if best_path:
            print(f"\nBest path: {' → '.join(str(n) for n in best_path)}")
            print(f"Bottleneck energy: {best_bottleneck:.4f} eV")
            for node in best_path:
                nd = self.G.nodes[node]
                ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
                print(f"  {ntype} {node}: E = {nd['e']:.4f} eV, "
                      f"V = {nd['volume']:.1f}")
        else:
            print("\nNo complete path found (increase maxstep or n_probes)")

        return self.G

    def refine_barrier(self, n_rounds=None, n_probes=None):
        """Iteratively attack the bottleneck saddle on the current best path.

        Each round:
        1. Find current best (minimax) path
        2. Identify the highest-energy saddle node on the path
        3. Get the minima on either side of that saddle
        4. Launch ``n_probes`` saddle searches from both adjacent minima,
           targeting each other (trying to find a lower alternative saddle)
        5. All new structures enter the graph; re-run minimax

        Repeats for ``n_rounds`` or until no improvement is found.

        Parameters
        ----------
        n_rounds : int, optional — override config.refine_rounds.
        n_probes : int, optional — override config.refine_probes.

        Returns
        -------
        tuple : (best_path, best_bottleneck) or (None, inf) if no path.
        """
        cfg = self.config
        n_rounds = n_rounds or cfg.refine_rounds
        n_probes = n_probes or cfg.refine_probes

        A_id = self.init_minima[0].id
        B_id = self.init_minima[1].id
        types = self._get_types(self.init_minima[0])

        try:
            best_path, best_bn = minimax_path_kinetic(self.G, A_id, B_id)
        except nx.NetworkXNoPath:
            print("No path exists to refine.")
            return None, float('inf')

        print(f"\n{'='*60}")
        print(f"Barrier refinement: {n_rounds} rounds x {n_probes} probes")
        print(f"Starting barrier: {best_bn:.4f} eV")
        print(f"Starting path: {' -> '.join(str(n) for n in best_path)}")
        print(f"{'='*60}")

        # Load all known minima from DB as PallasAtom objects
        all_minima = {}
        for x in self.db.select(ctyp='minima'):
            pa = PallasAtom(self.db.get_atoms(x.id))
            pa.znucl = cfg.znucl
            pa.fpcutoff = cfg.fpcutoff
            pa.natx = cfg.natx
            pa.id = x.id
            pa.fp = np.array(x.data['fp'])
            all_minima[x.id] = pa

        prev_bn = float('inf')

        for rnd in range(n_rounds):
            # Find the highest-energy saddle node on the current best path
            worst_saddle_id = None
            worst_saddle_e = -float('inf')
            worst_idx = -1
            for i, node in enumerate(best_path):
                nd = self.G.nodes[node]
                if nd['xname'].startswith('S') and nd['e'] > worst_saddle_e:
                    worst_saddle_e = nd['e']
                    worst_saddle_id = node
                    worst_idx = i

            if worst_saddle_id is None:
                print(f"  Round {rnd+1}: no saddle on path, nothing to refine")
                break

            # Find the minima on either side of the bottleneck saddle
            min_before_id = None
            for i in range(worst_idx - 1, -1, -1):
                if self.G.nodes[best_path[i]]['xname'].startswith('M'):
                    min_before_id = best_path[i]
                    break
            min_after_id = None
            for i in range(worst_idx + 1, len(best_path)):
                if self.G.nodes[best_path[i]]['xname'].startswith('M'):
                    min_after_id = best_path[i]
                    break

            if min_before_id is None or min_after_id is None:
                print(f"  Round {rnd+1}: cannot find minima flanking "
                      f"saddle S{worst_saddle_id}")
                break

            min_before = all_minima.get(min_before_id)
            min_after = all_minima.get(min_after_id)
            if min_before is None or min_after is None:
                print(f"  Round {rnd+1}: cannot load M{min_before_id} or "
                      f"M{min_after_id}")
                break

            e_before = self.G.nodes[min_before_id]['e']
            e_after = self.G.nodes[min_after_id]['e']

            print(f"\n  Round {rnd+1}/{n_rounds}: bottleneck S{worst_saddle_id} "
                  f"(E={worst_saddle_e:.4f})")
            print(f"    Between M{min_before_id} ({e_before:.4f}) and "
                  f"M{min_after_id} ({e_after:.4f})")

            # Launch probes from both sides of the bottleneck
            for p in range(n_probes):
                alpha = max(1.0 - 0.2 * p, 0.1)
                step_scale = cfg.fp_step_scale * (0.5 + 0.5 * np.random.rand())

                # From min_before toward min_after
                label = f"refine r{rnd+1} fwd p{p}(a={alpha:.1f})"
                new_min = self._fp_guided_step(
                    min_before, min_after.get_fp(), types, step_scale,
                    alpha=alpha, side=label)
                if new_min is not min_before and new_min.id not in all_minima:
                    all_minima[new_min.id] = new_min

                # From min_after toward min_before
                label = f"refine r{rnd+1} rev p{p}(a={alpha:.1f})"
                new_min = self._fp_guided_step(
                    min_after, min_before.get_fp(), types, step_scale,
                    alpha=alpha, side=label)
                if new_min is not min_after and new_min.id not in all_minima:
                    all_minima[new_min.id] = new_min

            # Cross-connect new minima
            min_list = [m for m in all_minima.values() if m.id is not None]
            for ma in min_list:
                for mb in min_list:
                    if ma.id >= mb.id or self.G.has_edge(ma.id, mb.id):
                        continue
                    d = fp_distance(ma.get_fp(), mb.get_fp(), types)
                    ediff = abs(ma.get_potential_energy()
                                - mb.get_potential_energy())
                    if d < cfg.dist_threshold and ediff < cfg.ediff:
                        h_a = self.G.nodes.get(ma.id, {}).get('e', 0)
                        h_b = self.G.nodes.get(mb.id, {}).get('e', 0)
                        self.G.add_edge(ma.id, mb.id,
                                        weight=max(h_a, h_b), dist=d)

            # Re-evaluate best path
            try:
                path, bn = minimax_path_kinetic(self.G, A_id, B_id)
                if bn < best_bn:
                    print(f"    IMPROVED: {best_bn:.4f} -> {bn:.4f} eV")
                    print(f"    New path: {' -> '.join(str(n) for n in path)}")
                    best_bn = bn
                    best_path = path
                else:
                    print(f"    No improvement (barrier still {best_bn:.4f})")
            except nx.NetworkXNoPath:
                print("    Path lost during refinement")

            self._save_state()

            # Early stop if no improvement for 2 consecutive rounds
            if best_bn >= prev_bn and rnd > 0:
                print("    Stopping: no improvement for 2 rounds")
                break
            prev_bn = best_bn

        # Final report
        print("\nRefinement complete.")
        print(f"Final barrier: {best_bn:.4f} eV")
        print(f"Final path: {' -> '.join(str(n) for n in best_path)}")
        for node in best_path:
            nd = self.G.nodes[node]
            ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
            print(f"  {ntype} {node}: E = {nd['e']:.4f} eV, "
                  f"V = {nd['volume']:.1f}")

        n_min = sum(1 for _, d in self.G.nodes(data=True)
                    if d['xname'].startswith('M'))
        n_sad = sum(1 for _, d in self.G.nodes(data=True)
                    if d['xname'].startswith('S'))
        print(f"Graph: {n_min} minima, {n_sad} saddles, "
              f"{self.G.number_of_edges()} edges")

        return best_path, best_bn

    # ── Saddle validation ──────────────────────────────────────────────

    def validate_graph(self):
        """Validate all saddle nodes in the graph and prune invalid ones.

        For each saddle:
        1. Check curvature < 0
        2. Push ±mode, relax, verify distinct connected minima
        3. Update edges: connect saddle to its verified minima
        4. Remove saddle nodes that fail validation

        Returns
        -------
        dict : summary with counts of valid/invalid/pruned saddles.
        """
        cfg = self.config
        types = self._get_types(self.init_minima[0])

        saddle_nodes = [(n, d) for n, d in self.G.nodes(data=True)
                        if d['xname'].startswith('S')]

        print(f"\nValidating {len(saddle_nodes)} saddle points...")

        stats = {'total': len(saddle_nodes), 'valid': 0,
                 'invalid': 0, 'pruned': 0, 'new_edges': 0}

        for node_id, node_data in saddle_nodes:
            # Load saddle from DB
            try:
                saddle_row = self.db.get(id=node_id)
                saddle = PallasAtom(self.db.get_atoms(node_id))
                saddle.znucl = cfg.znucl
                saddle.fpcutoff = cfg.fpcutoff
                saddle.natx = cfg.natx
                saddle.id = node_id
                saddle.fp = np.array(saddle_row.data['fp'])
            except Exception:
                print(f"  S{node_id}: cannot load from DB, pruning")
                self.G.remove_node(node_id)
                stats['pruned'] += 1
                continue

            # Retrieve dimer_mode/curvature if stored in DB
            saddle.dimer_curvature = saddle_row.data.get('curvature', None)
            dimer_mode_list = saddle_row.data.get('dimer_mode', None)
            if dimer_mode_list is not None:
                saddle.dimer_mode = np.array(dimer_mode_list)
            else:
                saddle.dimer_mode = None

            # If no dimer info stored, we can only do energy-based checks
            if saddle.dimer_mode is None:
                # Check: saddle should be higher than at least one neighbor
                neighbors = list(self.G.neighbors(node_id))
                neighbor_es = [self.G.nodes[n]['e'] for n in neighbors
                               if n in self.G.nodes]
                if neighbor_es and node_data['e'] <= min(neighbor_es):
                    print(f"  S{node_id} ({node_data['e']:.4f}): "
                          f"INVALID — below all neighbors, pruning")
                    self.G.remove_node(node_id)
                    stats['invalid'] += 1
                    stats['pruned'] += 1
                else:
                    print(f"  S{node_id} ({node_data['e']:.4f}): "
                          f"no dimer data, keeping (energy OK)")
                    stats['valid'] += 1
                continue

            # Full validation with connectivity
            result = self._validate_saddle(saddle, types)
            curv_str = f"κ={result['curvature']:.4f}" \
                if result['curvature'] is not None else "κ=?"

            if result['valid']:
                print(f"  S{node_id} ({node_data['e']:.4f}): "
                      f"VALID — connects distinct minima ({curv_str})")
                stats['valid'] += 1

                # Register the verified minima and add edges
                for m in [result['min_plus'], result['min_minus']]:
                    if m is None:
                        continue
                    mid, _ = self._update_minima(m)
                    m.id = mid
                    h_m = (m.get_volume() * cfg.press * GPa
                           + m.get_potential_energy() - self.baseenergy)
                    if mid not in self.G:
                        self.G.add_node(mid, xname=f'M{mid}', e=h_m,
                                        volume=m.get_volume())
                        self._save_to_traj(m, mid, 'minima', h_m)
                    else:
                        h_m = self.G.nodes[mid]['e']  # use stored enthalpy
                    if not self.G.has_edge(node_id, mid):
                        fp_s = saddle.get_fp()
                        fp_m = m.get_fp()
                        d = fp_distance(fp_s, fp_m, types)
                        self.G.add_edge(node_id, mid,
                                        weight=max(node_data['e'], h_m),
                                        dist=d)
                        stats['new_edges'] += 1
            else:
                print(f"  S{node_id} ({node_data['e']:.4f}): "
                      f"INVALID — {result['reason']} ({curv_str}), pruning")
                self.G.remove_node(node_id)
                stats['invalid'] += 1
                stats['pruned'] += 1

        self._save_state()

        print(f"\nValidation complete: {stats['valid']} valid, "
              f"{stats['invalid']} invalid, {stats['pruned']} pruned, "
              f"{stats['new_edges']} new edges")
        return stats

    def _closest_to(self, minima_list, target_fp, types):
        """Return the minimum from the list closest to target in FP space."""
        best = minima_list[0]
        best_d = float('inf')
        for m in minima_list:
            fp = m.get_fp()
            if fp is None:
                continue
            d = fp_distance(fp, target_fp, types)
            if d < best_d:
                best_d = d
                best = m
        return best

    def _cross_connect(self, minima_A, minima_B, types):
        """Check all A-B minima pairs and add graph edges for close matches."""
        cfg = self.config
        for ma in minima_A:
            fp_a = ma.get_fp()
            if fp_a is None or ma.id is None:
                continue
            for mb in minima_B:
                fp_b = mb.get_fp()
                if fp_b is None or mb.id is None:
                    continue
                if self.G.has_edge(ma.id, mb.id):
                    continue
                d = fp_distance(fp_a, fp_b, types)
                ediff = abs(ma.get_potential_energy()
                            - mb.get_potential_energy())
                if d < cfg.dist_threshold and ediff < cfg.ediff:
                    h_a = self.G.nodes.get(ma.id, {}).get('e', 0)
                    h_b = self.G.nodes.get(mb.id, {}).get('e', 0)
                    self.G.add_edge(ma.id, mb.id,
                                    weight=max(h_a, h_b), dist=d)
                    print(f"  CONNECT: M{ma.id} ↔ M{mb.id} "
                          f"(d={d:.5f})")

    # ── Generational search helpers ──────────────────────────────────

    def _plan_connect_probes(self, A, B, known, types):
        """Plan probes to extend chains from both sides toward each other.

        Classifies known minima into A-reachable and B-reachable sets,
        ranks each side's frontier by closeness to the other side,
        and distributes probes round-robin across the top frontier tips.

        Returns list of (source, target_fp, alpha, label) tuples.
        """
        cfg = self.config

        # Classify minima by graph connectivity
        A_ids, B_ids = self._classify_sides(A.id, B.id)

        # Rank each side's frontier by closeness to other side
        A_frontier = self._rank_frontier(A_ids, B_ids, known, types)
        B_frontier = self._rank_frontier(B_ids, A_ids, known, types)

        if not A_frontier or not B_frontier:
            return []

        probes = []
        for i in range(cfg.n_probes):
            alpha = max(1.0 - 0.3 * (i // 2), 0.1)
            if i % 2 == 0:
                # A-side probe — round-robin across frontier tips
                idx = (i // 2) % len(A_frontier)
                src_id, tgt_fp, d = A_frontier[idx]
                probes.append((
                    known[src_id], tgt_fp, alpha,
                    f'A→B p{i}(α={alpha:.1f},d={d:.3f})'))
            else:
                idx = (i // 2) % len(B_frontier)
                src_id, tgt_fp, d = B_frontier[idx]
                probes.append((
                    known[src_id], tgt_fp, alpha,
                    f'B→A p{i}(α={alpha:.1f},d={d:.3f})'))

        return probes

    def _plan_hybrid_probes(self, path, A, B, known, types):
        """Plan probes: half refine bottleneck, half explore from intermediates.

        Splits n_probes between:
        - **Refine**: Attack the bottleneck saddle from both flanking minima.
        - **Explore**: Probe from frontier intermediates (not on the
          bottleneck) toward the other side, discovering new phases and
          alternative pathways.

        Falls back to connect-style probes if no saddle found on path.

        Returns list of (source, target_fp, alpha, label) tuples.
        """
        cfg = self.config

        # Find the worst saddle on the path
        worst_id, worst_idx = None, -1
        worst_e = -float('inf')
        for i, node in enumerate(path):
            nd = self.G.nodes[node]
            if nd['xname'].startswith('S') and nd['e'] > worst_e:
                worst_e = nd['e']
                worst_id = node
                worst_idx = i

        if worst_id is None:
            return self._plan_connect_probes(A, B, known, types)

        # Find flanking minima for refine
        min_left = self._find_flanking_min(path, worst_idx, -1, known)
        min_right = self._find_flanking_min(path, worst_idx, +1, known)

        if min_left is None or min_right is None:
            return self._plan_connect_probes(A, B, known, types)

        h_left = self.G.nodes[min_left.id]['e']
        h_right = self.G.nodes[min_right.id]['e']
        print(f"  Bottleneck: S{worst_id} ({worst_e:.4f}), "
              f"flanked by M{min_left.id} ({h_left:.4f}) "
              f"↔ M{min_right.id} ({h_right:.4f})")

        # Split probes: half refine, half explore
        n_refine = max(cfg.n_probes // 2, 2)
        n_explore = cfg.n_probes - n_refine

        probes = []

        # --- Refine probes: attack bottleneck ---
        for i in range(n_refine):
            alpha = max(1.0 - 0.2 * i, 0.2)
            if i % 2 == 0:
                probes.append((
                    min_left, min_right.get_fp(), alpha,
                    f'refine fwd p{i}(α={alpha:.1f})'))
            else:
                probes.append((
                    min_right, min_left.get_fp(), alpha,
                    f'refine rev p{i}(α={alpha:.1f})'))

        # --- Explore probes: from frontier intermediates ---
        # Collect all known minima that are NOT the bottleneck flanks
        # Rank by closeness to the opposite endpoint
        flanks = {min_left.id, min_right.id}
        explore_candidates = []
        fp_A = A.get_fp()
        fp_B = B.get_fp()

        for mid, m in known.items():
            if mid in flanks:
                continue
            fp_m = m.get_fp()
            if fp_m is None:
                continue
            d_to_A = fp_distance(fp_m, fp_A, types)
            d_to_B = fp_distance(fp_m, fp_B, types)
            # Interesting intermediates: not too close to either endpoint
            # Probe toward whichever endpoint is farther
            if d_to_A > d_to_B:
                explore_candidates.append(
                    (mid, fp_A, d_to_A, f'd_A={d_to_A:.3f}'))
            else:
                explore_candidates.append(
                    (mid, fp_B, d_to_B, f'd_B={d_to_B:.3f}'))

        # Sort by distance to target (farthest first — most unexplored)
        explore_candidates.sort(key=lambda x: -x[2])

        for i in range(n_explore):
            if explore_candidates:
                idx = i % len(explore_candidates)
                mid, tgt_fp, d, dstr = explore_candidates[idx]
                alpha = max(0.8 - 0.2 * i, 0.2)
                probes.append((
                    known[mid], tgt_fp, alpha,
                    f'explore M{mid} p{i}(α={alpha:.1f},{dstr})'))
            else:
                # No intermediates yet — extra refine probe
                alpha = max(0.5 - 0.1 * i, 0.1)
                if i % 2 == 0:
                    probes.append((
                        min_left, min_right.get_fp(), alpha,
                        f'refine+ fwd p{i}(α={alpha:.1f})'))
                else:
                    probes.append((
                        min_right, min_left.get_fp(), alpha,
                        f'refine+ rev p{i}(α={alpha:.1f})'))

        print(f"  Probes: {n_refine} refine + {n_explore} explore "
              f"({len(explore_candidates)} intermediate candidates)")
        return probes

    def _classify_sides(self, A_id, B_id):
        """Classify minimum nodes into A-reachable and B-reachable sets."""
        if A_id in self.G:
            A_comp = nx.node_connected_component(self.G, A_id)
        else:
            A_comp = {A_id}
        if B_id in self.G:
            B_comp = nx.node_connected_component(self.G, B_id)
        else:
            B_comp = {B_id}

        A_mins = {n for n in A_comp
                  if self.G.nodes.get(n, {}).get('xname', '').startswith('M')}
        B_mins = {n for n in B_comp
                  if self.G.nodes.get(n, {}).get('xname', '').startswith('M')}
        return A_mins, B_mins

    def _rank_frontier(self, source_ids, target_ids, known, types):
        """Rank source minima by closeness to nearest target minimum.

        Returns list of (source_id, nearest_target_fp, distance)
        sorted by distance (closest first).
        """
        ranked = []
        for sid in source_ids:
            if sid not in known:
                continue
            fp_s = known[sid].get_fp()
            best_d = float('inf')
            best_tgt_fp = None
            for tid in target_ids:
                if tid not in known:
                    continue
                fp_t = known[tid].get_fp()
                d = fp_distance(fp_s, fp_t, types)
                if d < best_d:
                    best_d = d
                    best_tgt_fp = fp_t
            if best_tgt_fp is not None:
                ranked.append((sid, best_tgt_fp, best_d))
        ranked.sort(key=lambda x: x[2])
        return ranked

    def _find_flanking_min(self, path, saddle_idx, direction, known):
        """Find the nearest minimum in the given direction from a saddle.

        Parameters
        ----------
        path : list of node IDs.
        saddle_idx : int — index of the saddle node in the path.
        direction : int — -1 for left, +1 for right.
        known : dict — id → PallasAtom cache.

        Returns PallasAtom or None.
        """
        i = saddle_idx + direction
        while 0 <= i < len(path):
            if self.G.nodes[path[i]]['xname'].startswith('M'):
                return self._ensure_known(path[i], known)
            i += direction
        return None

    def _ensure_known(self, mid, known):
        """Ensure a minimum is in the known dict; load from DB if needed."""
        if mid in known:
            return known[mid]

        cfg = self.config
        try:
            row = self.db.get(id=mid)
            pa = PallasAtom(self.db.get_atoms(mid))
            pa.znucl = cfg.znucl
            pa.fpcutoff = cfg.fpcutoff
            pa.natx = cfg.natx
            pa.id = mid
            pa.fp = np.array(row.data['fp'])
            known[mid] = pa
            return pa
        except Exception:
            return None

    def _cross_connect_all(self, known, types):
        """Check all pairs of known minima for close FP distance."""
        cfg = self.config
        ids = [mid for mid in known
               if mid in self.G
               and self.G.nodes[mid].get('xname', '').startswith('M')]

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id, b_id = ids[i], ids[j]
                if self.G.has_edge(a_id, b_id):
                    continue
                d = fp_distance(
                    known[a_id].get_fp(), known[b_id].get_fp(), types)
                ediff = abs(known[a_id].get_potential_energy()
                            - known[b_id].get_potential_energy())
                if d < cfg.dist_threshold and ediff < cfg.ediff:
                    h_a = self.G.nodes[a_id]['e']
                    h_b = self.G.nodes[b_id]['e']
                    self.G.add_edge(a_id, b_id,
                                    weight=max(h_a, h_b), dist=d)
                    print(f"  CONNECT: M{a_id} ↔ M{b_id} (d={d:.5f})")

    def _update_minima(self, minima):
        """Register minimum in DB, dedup by FP distance + energy."""
        fpm = minima.get_fp()
        em = minima.get_potential_energy()
        types = self._get_types(minima)

        isnew = True
        idm = None
        for x in self.db.select(ctyp='minima'):
            fpx = np.array(x.data['fp'])
            d = fp_distance(fpm, fpx, types)
            ediff = abs(em - x.data['energy'])
            if d < self.config.dist_threshold and ediff < self.config.ediff:
                idm = x.id
                isnew = False
                break

        if isnew:
            idm = self.db.write(
                minima, ctyp='minima',
                data={'fp': fpm.tolist(), 'energy': float(em)}
            )

        # Update distance matrix
        for x in self.db.select(ctyp='minima'):
            if x.id != idm:
                fpx = np.array(x.data['fp'])
                self.update_dij(idm, x.id, fp_distance(fpm, fpx, types))
        for x in self.db.select(ctyp='saddle'):
            fpx = np.array(x.data['fp'])
            self.update_dij(idm, x.id, fp_distance(fpm, fpx, types))

        return idm, isnew

    def _update_saddle(self, saddle):
        """Register saddle in DB, dedup by FP distance + energy."""
        fps = saddle.get_fp()
        es = saddle.get_potential_energy()
        types = self._get_types(saddle)

        isnew = True
        ids = None
        for x in self.db.select(ctyp='saddle'):
            fpx = np.array(x.data['fp'])
            d = fp_distance(fps, fpx, types)
            ediff = abs(es - x.data['energy'])
            if d < self.config.dist_threshold and ediff < self.config.ediff:
                ids = x.id
                isnew = False
                break

        if isnew:
            # Store dimer mode and curvature for later validation
            data = {'fp': fps.tolist(), 'energy': float(es)}
            dimer_mode = getattr(saddle, 'dimer_mode', None)
            if dimer_mode is not None:
                data['dimer_mode'] = dimer_mode.tolist()
            curvature = getattr(saddle, 'dimer_curvature', None)
            if curvature is not None:
                data['curvature'] = float(curvature)
            ids = self.db.write(saddle, ctyp='saddle', data=data)

        for x in self.db.select(ctyp='saddle'):
            if x.id != ids:
                fpx = np.array(x.data['fp'])
                self.update_dij(ids, x.id, fp_distance(fps, fpx, types))
        for x in self.db.select(ctyp='minima'):
            fpx = np.array(x.data['fp'])
            self.update_dij(ids, x.id, fp_distance(fps, fpx, types))

        return ids, isnew

    # ── Trajectory ─────────────────────────────────────────────────

    def _save_state(self):
        """Save graph and distance matrix to disk."""
        joblib.dump(self.G, 'graph.pkl')
        nx.write_gml(self.G, 'graph.gml')
        nx.write_gexf(self.G, 'graph.gexf')
        joblib.dump(self.dij, 'dij.pkl')

    def find_best_path(self):
        """Find kinetically optimal path between reactant and product.

        Uses directed minimax to minimize the rate-limiting local barrier
        (max of H_saddle − H_preceding_min over all steps).

        Returns
        -------
        path : list of node IDs
        bottleneck : float — rate-limiting local barrier (eV).
        """
        react_id = self.init_minima[0].id
        prod_id = self.init_minima[1].id
        return minimax_path_kinetic(self.G, react_id, prod_id)

