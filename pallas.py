"""PALLAS — Phase Transition Pathway Sampling via Swarm Intelligence and Graph Theory.

Automated method for finding transition pathways between crystal phases using:
- FP-gradient-guided chain-growing search (primary method)
- Bidirectional PSO search (legacy method)
- Solid-state dimer method for saddle point location
- NetworkX graph with minimax bottleneck path finding
- torch_fplib GOM fingerprints for structural distance and gradients

Dependencies: torch_fplib, MatterSim (via zfunc), ASE, NetworkX.
"""

import os
import sys
from copy import deepcopy as cp
from dataclasses import dataclass, field

import numpy as np
import networkx as nx
import joblib
from ase import Atoms
from ase.io import read, write
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from ase.units import GPa
import ase.db

import torch
import torch_fplib
from xcal import XCalculator, atoms_to_cell, fp_dist_with_assignment
from zfunc import local_optimization, cal_saddle, vunit, vrand
from barrier import minimax_path


# ── Configuration ─────────────────────────────────────────────────────

@dataclass
class PallasConfig:
    """Configuration for PALLAS pathway search."""
    # Fingerprint parameters
    fpcutoff: float = 5.5
    natx: int = 200
    lmax: int = 0       # 0 = s-only, 1 = s+p

    # System
    znucl: list = field(default_factory=list)   # atomic numbers in type order
    press: float = 0.0                          # external pressure (eV/Å³)

    # PSO parameters
    maxstep: int = 50
    popsize: int = 10
    velocity_weight: float = 0.9
    c1: float = 2.0     # personal best weight
    c2: float = 1.5     # global best weight

    # Optimization step limits
    opt_steps: int = 2000           # max FIRE steps for local optimization
    opt_fmax: float = 0.001         # force convergence for optimization
    saddle_steps: int = 2000        # max FIRE steps for dimer saddle search
    saddle_fmax: float = 0.01       # force convergence for saddle
    bias_steps: int = 60            # max FIRE steps for FP bias relaxation

    # FP-guided search parameters
    fp_step_scale: float = 0.05     # perturbation scale along FP gradient (small to stay near basin)
    fp_push_scale: float = 0.05     # post-saddle push scale toward target
    max_retries: int = 2            # retries with smaller step on saddle failure

    # Barrier refinement parameters
    refine_rounds: int = 3          # number of refinement iterations
    refine_probes: int = 5          # saddle searches per refinement round

    # Convergence
    ediff: float = 0.001            # energy diff threshold for same structure
    dist_threshold: float = 0.01    # FP distance threshold for connection


# ── PallasAtom: Atoms with fingerprint caching ────────────────────────

class PallasAtom(Atoms):
    """ASE Atoms subclass with cached fingerprints and metadata.

    Fingerprints are computed via torch_fplib and cached until invalidated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.natx = 200
        self.fpcutoff = 5.5
        self.fp = None
        self.converged = False
        self.id = None
        self._znucl = None

    @property
    def znucl(self):
        return self._znucl

    @znucl.setter
    def znucl(self, val):
        self._znucl = list(val) if val is not None else None

    def get_fp(self):
        """Return cached fingerprints, computing if needed."""
        if self.fp is None:
            self.fp = self.cal_fp()
        return self.fp

    def cal_fp(self):
        """Compute GOM fingerprints via torch_fplib."""
        if self._znucl is None or len(self._znucl) == 0:
            raise ValueError("znucl not set on PallasAtom")
        lat_np, rxyz_np, types, znucl = atoms_to_cell(self, self._znucl)
        with torch.no_grad():
            fp = torch_fplib.get_lfp(
                (lat_np, rxyz_np, types, znucl),
                cutoff=self.fpcutoff, natx=self.natx, orbital='s'
            )
        return fp.numpy()

    def invalidate_fp(self):
        """Clear cached fingerprint (call after position/cell changes)."""
        self.fp = None


# ── Fingerprint distance helper ───────────────────────────────────────

def fp_distance(fp1, fp2, types):
    """Hungarian-matched fingerprint distance between two structures.

    Parameters
    ----------
    fp1, fp2 : np.ndarray, shape (nat, fp_dim)
    types : array-like, shape (nat,)  — 1-indexed atom types.

    Returns
    -------
    float — averaged FP distance.
    """
    d, _ = fp_dist_with_assignment(fp1, fp2, types)
    return d


# ── Main PALLAS class ────────────────────────────────────────────────

class Pallas:
    """PALLAS pathway search engine.

    Two search modes:
    - ``run_fp_guided()`` — FP-gradient-guided chain growing (recommended)
    - ``run_pso()`` — bidirectional PSO (legacy)

    Usage::

        config = PallasConfig(znucl=[6], press=10.0)
        pallas = Pallas(config)
        pallas.init_run(['POSCAR_graphite', 'POSCAR_diamond'])
        graph = pallas.run_fp_guided()
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

    # ── PSO core ─────────────────────────────────────────────────────

    def run_pso(self):
        """Bidirectional PSO search for transition pathway.

        Returns
        -------
        nx.Graph — pathway network with minima/saddle nodes and weighted edges.
        """
        cfg = self.config
        print("Starting bidirectional PSO pathway search")

        # Optimize reactant and product
        react_opt = self._optimize_and_register(self.reactant, is_base=True)
        prod_opt = self._optimize_and_register(self.product, is_base=False)

        react_id, prod_id = react_opt.id, prod_opt.id
        types = self._get_types(react_opt)

        # Initialize particles (perturbed copies + random velocities)
        r_particles, r_velocities = [], []
        p_particles, p_velocities = [], []

        for _ in range(cfg.popsize):
            r_particles.append(self._add_perturbation(react_opt))
            r_velocities.append(self._gen_random_velocity(react_opt))
            p_particles.append(self._add_perturbation(prod_opt))
            p_velocities.append(self._gen_random_velocity(prod_opt))

        self.pdistx = [float('inf')] * cfg.popsize
        self.pdisty = [float('inf')] * cfg.popsize
        self.pbestx = cp(r_particles)
        self.pbesty = cp(p_particles)

        # Main PSO loop
        for step in range(cfg.maxstep):
            print(f"\n=== PSO step {step+1}/{cfg.maxstep} ===")

            # Process all particles on both sides
            for i in range(cfg.popsize):
                # Reactant side: saddle → minimum biased toward product
                r_particles[i] = self._process_particle(
                    r_particles[i], r_velocities[i],
                    react_opt, prod_opt, react_id, types,
                    side="reactant", particle_idx=i
                )

                # Product side: saddle → minimum biased toward reactant
                p_particles[i] = self._process_particle(
                    p_particles[i], p_velocities[i],
                    prod_opt, react_opt, prod_id, types,
                    side="product", particle_idx=i
                )

            # Check for connections between sides
            connection_found = self._check_connections(
                r_particles, p_particles, types
            )

            # Update personal and global bests
            self._update_bests(r_particles, p_particles, types)

            # Update velocities
            self._update_velocities(
                r_particles, r_velocities, p_particles, p_velocities, step
            )

            # Check if path exists
            if connection_found:
                print("Connection found! Searching for path...")
                try:
                    path, bottleneck = minimax_path(self.G, react_id, prod_id)
                    print(f"Path found: {path}")
                    print(f"Bottleneck energy: {bottleneck:.4f}")
                except nx.NetworkXNoPath:
                    print("Connected but no path yet (graph not fully linked)")

            self._save_state()

        print("\nPSO search complete")
        return self.G

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
        print(f"Registered {'base' if is_base else 'target'}: "
              f"ID={idm}, H={h:.4f} eV")
        return opt

    def _process_particle(self, particle, velocity, anchor_opt, target_opt,
                          anchor_id, types, side="", particle_idx=0):
        """Process one PSO particle: find saddle → relax to new minimum.

        Used identically for both reactant and product sides.

        Parameters
        ----------
        particle : PallasAtom — current particle position.
        velocity : np.ndarray — current velocity mode.
        anchor_opt : PallasAtom — optimized anchor (reactant or product).
        target_opt : PallasAtom — optimized target to bias toward.
        anchor_id : int — graph node ID for the anchor.
        types : np.ndarray — atom type array.
        side : str — label for logging.
        particle_idx : int — particle index for logging.

        Returns
        -------
        PallasAtom — new minimum found from saddle descent.
        """
        try:
            # Find saddle point
            saddle = self._calculate_saddle_with_velocity(particle, velocity)
            sad_id, _ = self._update_saddle(saddle)
            saddle.id = sad_id

            h_sad = (saddle.get_volume() * self.config.press * GPa
                     + saddle.get_potential_energy() - self.baseenergy)
            self.G.add_node(sad_id, xname=f'S{sad_id}', e=h_sad,
                            volume=saddle.get_volume())

            # Edge: anchor → saddle
            fp_a = anchor_opt.get_fp()
            fp_s = saddle.get_fp()
            fp_dist = fp_distance(fp_a, fp_s, types)
            edge_w = max(self.G.nodes[anchor_id]['e'], h_sad)
            self.G.add_edge(anchor_id, sad_id, weight=edge_w, dist=fp_dist)

            # Bias saddle toward target, then optimize
            cfg = self.config
            target_fp = target_opt.get_fp()
            biased = self._xcal_bias(saddle, target_fp)
            new_min = local_optimization(biased, fmax=cfg.opt_fmax, steps=cfg.opt_steps)
            min_id, _ = self._update_minima(new_min)
            new_min.id = min_id

            h_min = (new_min.get_volume() * self.config.press * GPa
                     + new_min.get_potential_energy() - self.baseenergy)
            self.G.add_node(min_id, xname=f'M{min_id}', e=h_min,
                            volume=new_min.get_volume())

            # Edge: saddle → new minimum
            fp_m = new_min.get_fp()
            fp_dist2 = fp_distance(fp_s, fp_m, types)
            edge_w2 = max(h_sad, h_min)
            self.G.add_edge(sad_id, min_id, weight=edge_w2, dist=fp_dist2)

            print(f"  [{side} p{particle_idx}] saddle S{sad_id} "
                  f"({h_sad:.3f}) → min M{min_id} ({h_min:.3f})")
            return new_min

        except Exception as e:
            print(f"  [{side} p{particle_idx}] FAILED: {e}")
            return particle

    def _check_connections(self, r_particles, p_particles, types):
        """Check if any reactant-side minimum is close to a product-side one."""
        found = False
        min_dist = float('inf')
        best_pair = None

        for i, mr in enumerate(r_particles):
            fp_r = mr.get_fp()
            if fp_r is None:
                continue
            for j, mp in enumerate(p_particles):
                fp_p = mp.get_fp()
                if fp_p is None:
                    continue
                try:
                    d = fp_distance(fp_r, fp_p, types)
                    ediff = abs(mr.get_potential_energy()
                                - mp.get_potential_energy())

                    if d < self.config.dist_threshold and ediff < self.config.ediff:
                        # Close enough — add connecting edge
                        h_r = self.G.nodes.get(mr.id, {}).get('e', 0)
                        h_p = self.G.nodes.get(mp.id, {}).get('e', 0)
                        self.G.add_edge(mr.id, mp.id,
                                        weight=max(h_r, h_p), dist=d)
                        print(f"  CONNECTION: M{mr.id} ↔ M{mp.id} "
                              f"(d={d:.5f}, ΔE={ediff:.5f})")
                        found = True

                    if d < min_dist:
                        min_dist = d
                        best_pair = (i, j)
                except Exception:
                    continue

        if best_pair and min_dist < self.bestdist:
            self.bestdist = min_dist
            self.gbestx = cp(r_particles[best_pair[0]])
            self.gbesty = cp(p_particles[best_pair[1]])
            print(f"  Global best distance: {min_dist:.5f}")

        return found

    def _update_bests(self, r_particles, p_particles, types):
        """Update personal bests for all particles."""
        for i, mr in enumerate(r_particles):
            fp_r = mr.get_fp()
            if fp_r is None:
                continue
            best_d = float('inf')
            for mp in p_particles:
                fp_p = mp.get_fp()
                if fp_p is None:
                    continue
                try:
                    d = fp_distance(fp_r, fp_p, types)
                    best_d = min(best_d, d)
                except Exception:
                    continue
            if best_d < self.pdistx[i]:
                self.pdistx[i] = best_d
                self.pbestx[i] = cp(mr)

        for j, mp in enumerate(p_particles):
            fp_p = mp.get_fp()
            if fp_p is None:
                continue
            best_d = float('inf')
            for mr in r_particles:
                fp_r = mr.get_fp()
                if fp_r is None:
                    continue
                try:
                    d = fp_distance(fp_r, fp_p, types)
                    best_d = min(best_d, d)
                except Exception:
                    continue
            if best_d < self.pdisty[j]:
                self.pdisty[j] = best_d
                self.pbesty[j] = cp(mp)

    def _update_velocities(self, r_particles, r_velocities,
                           p_particles, p_velocities, step):
        """Update PSO velocities for next iteration."""
        cfg = self.config
        w = cfg.velocity_weight - 0.5 * step / cfg.maxstep  # linear decay

        for i in range(cfg.popsize):
            r1, r2 = np.random.rand(2)
            v_pb = self._velocity_toward(r_particles[i], self.pbestx[i])
            v_gb = self._velocity_toward(r_particles[i], self.gbestx)
            r_velocities[i] = (w * r_velocities[i]
                               + cfg.c1 * r1 * v_pb
                               + cfg.c2 * r2 * v_gb)

            r1, r2 = np.random.rand(2)
            v_pb = self._velocity_toward(p_particles[i], self.pbesty[i])
            v_gb = self._velocity_toward(p_particles[i], self.gbesty)
            p_velocities[i] = (w * p_velocities[i]
                               + cfg.c1 * r1 * v_pb
                               + cfg.c2 * r2 * v_gb)

    # ── Structure perturbation & velocity helpers ─────────────────────

    def _add_perturbation(self, structure):
        """Add random perturbation to structure (positions + cell)."""
        atoms = cp(structure)
        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        mode = vrand(np.zeros((natom + 3, 3)))
        # Constrain redundant freedoms (translation + rotation)
        mode[0] *= 0
        mode[-3, 1:] *= 0
        mode[-2, 2] *= 0
        mode = vunit(mode)

        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), mode[-3:] / jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + mode[:-3])
        atoms.invalidate_fp()
        return atoms

    def _gen_random_velocity(self, structure):
        """Generate random normalized velocity mode."""
        natom = len(structure)
        mode = vrand(np.zeros((natom + 3, 3)))
        mode[0] *= 0
        mode[-3, 1:] *= 0
        mode[-2, 2] *= 0
        return vunit(mode)

    def _velocity_toward(self, current, target):
        """Compute velocity component pointing from current to target."""
        if current is None or target is None:
            src = current if current is not None else target
            return self._gen_random_velocity(src)

        natom = len(current)
        vol = current.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        velocity = np.zeros((natom + 3, 3))
        velocity[:natom] = target.get_positions() - current.get_positions()
        velocity[-3:] = (target.get_cell() - current.get_cell()) / jacob
        return vunit(velocity)

    def _calculate_saddle_with_velocity(self, structure, velocity):
        """Displace along velocity direction, then find saddle point."""
        atoms = cp(structure)
        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        velocity = vunit(velocity)
        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), velocity[-3:] / jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + velocity[:-3])
        atoms.invalidate_fp()

        cfg = self.config
        return cal_saddle(atoms, fmax=cfg.saddle_fmax, steps=cfg.saddle_steps)

    def _xcal_bias(self, structure, target_fp):
        """Bias structure toward target fingerprint using XCalculator."""
        atoms = cp(structure)
        calc = XCalculator(
            fp0=target_fp,
            znucl=self.config.znucl,
            cutoff=self.config.fpcutoff,
            natx=self.config.natx,
        )
        atoms.calc = calc
        af = FrechetCellFilter(atoms)
        opt = FIRE(af, maxstep=0.1, logfile='xcal.log')
        opt.run(fmax=0.01, steps=self.config.bias_steps)
        return atoms

    # ── FP-guided chain-growing search ────────────────────────────────

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
                path, bottleneck = minimax_path(self.G, A.id, B.id)
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
            best_path, best_bn = minimax_path(self.G, A_id, B_id)
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
                path, bn = minimax_path(self.G, A_id, B_id)
                if bn < best_bn:
                    print(f"    IMPROVED: {best_bn:.4f} -> {bn:.4f} eV")
                    print(f"    New path: {' -> '.join(str(n) for n in path)}")
                    best_bn = bn
                    best_path = path
                else:
                    print(f"    No improvement (barrier still {best_bn:.4f})")
            except nx.NetworkXNoPath:
                print(f"    Path lost during refinement")

            self._save_state()

            # Early stop if no improvement for 2 consecutive rounds
            if best_bn >= prev_bn and rnd > 0:
                print(f"    Stopping: no improvement for 2 rounds")
                break
            prev_bn = best_bn

        # Final report
        print(f"\nRefinement complete.")
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

    def _validate_saddle(self, saddle, types):
        """Validate a saddle point: curvature check + connectivity test.

        1. Curvature check: dimer curvature must be negative.
        2. Connectivity: push along ±dimer_mode, relax on real PES,
           verify the two endpoints are distinct minima.

        Parameters
        ----------
        saddle : PallasAtom — saddle with .dimer_mode and .dimer_curvature.
        types : np.ndarray — atom type array.

        Returns
        -------
        dict with keys:
            valid : bool — True if saddle passes both checks.
            curvature : float — dimer curvature value.
            min_plus : PallasAtom or None — minimum from +mode descent.
            min_minus : PallasAtom or None — minimum from -mode descent.
            reason : str — failure reason if not valid.
        """
        cfg = self.config
        result = {'valid': False, 'curvature': None,
                  'min_plus': None, 'min_minus': None, 'reason': ''}

        # Check 1: Curvature must be negative
        curvature = getattr(saddle, 'dimer_curvature', None)
        result['curvature'] = curvature

        if curvature is None:
            result['reason'] = 'no curvature stored'
            return result
        if curvature >= 0:
            result['reason'] = f'positive curvature ({curvature:.4f})'
            return result

        # Check 2: Push ±mode, relax, verify distinct minima
        dimer_mode = getattr(saddle, 'dimer_mode', None)
        if dimer_mode is None:
            result['reason'] = 'no dimer_mode stored'
            return result

        mode = vunit(dimer_mode)
        natom = len(saddle)
        vol = saddle.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5
        push = cfg.fp_push_scale * 3.0  # same scale as _saddle_escape

        min_plus = self._push_and_relax(saddle, mode, push, jacob)
        min_minus = self._push_and_relax(saddle, -mode, push, jacob)

        result['min_plus'] = min_plus
        result['min_minus'] = min_minus

        if min_plus is None or min_minus is None:
            result['reason'] = 'relaxation failed'
            return result

        # Check the two minima are distinct
        d = fp_distance(min_plus.get_fp(), min_minus.get_fp(), types)
        ediff = abs(min_plus.get_potential_energy()
                    - min_minus.get_potential_energy())

        if d < cfg.dist_threshold and ediff < cfg.ediff:
            result['reason'] = (f'same minimum on both sides '
                                f'(d={d:.5f}, dE={ediff:.5f})')
            return result

        result['valid'] = True
        return result

    def _push_and_relax(self, saddle, mode, push_scale, jacob):
        """Push saddle along mode and relax on real PES.

        Returns optimized PallasAtom or None on failure.
        """
        try:
            atoms = cp(saddle)
            natom = len(atoms)
            scaled = mode * push_scale
            cellt = atoms.get_cell() + np.dot(atoms.get_cell(),
                                              scaled[-3:] / jacob)
            atoms.set_cell(cellt, scale_atoms=True)
            atoms.set_positions(atoms.get_positions() + scaled[:-3])
            if hasattr(atoms, 'invalidate_fp'):
                atoms.invalidate_fp()

            cfg = self.config
            opt = local_optimization(atoms, fmax=cfg.opt_fmax,
                                     steps=cfg.opt_steps)
            return opt
        except Exception:
            return None

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
                      f"VALID ({curv_str})")
                stats['valid'] += 1

                # Register the verified minima and add edges
                for m in [result['min_plus'], result['min_minus']]:
                    if m is None:
                        continue
                    mid, _ = self._update_minima(m)
                    m.id = mid
                    h_m = (m.get_volume() * cfg.press * GPa
                           + m.get_potential_energy() - self.baseenergy)
                    self.G.add_node(mid, xname=f'M{mid}', e=h_m,
                                    volume=m.get_volume())
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

    def _fp_guided_step(self, current, target_fp, types,
                        step_scale, alpha=1.0, side=""):
        """One step of FP-guided chain growing.

        Parameters
        ----------
        current : PallasAtom — current chain tip (optimized minimum).
        target_fp : np.ndarray — target fingerprint to grow toward.
        types : np.ndarray — atom type array.
        step_scale : float — perturbation magnitude.
        alpha : float — FP/random mixing ratio (1.0 = pure FP, 0.0 = pure random).
        side : str — label for logging.

        Returns
        -------
        PallasAtom — new minimum (or current if step failed).
        """
        cfg = self.config

        # Step 1: Build search mode from FP gradient + random component
        fp_mode = self._fp_gradient_mode(current, target_fp)
        rand_mode = self._gen_random_velocity(current)
        mode = vunit(alpha * fp_mode + (1.0 - alpha) * rand_mode)

        # Step 2+3: Perturb along mode, run dimer with this mode
        saddle = None
        for attempt in range(cfg.max_retries + 1):
            try:
                scale = step_scale * (0.5 ** attempt)  # halve on retry
                saddle = self._fp_guided_saddle(current, mode, scale)
                break
            except Exception as e:
                if attempt < cfg.max_retries:
                    print(f"  [{side}] Saddle attempt {attempt+1} failed "
                          f"(scale={scale:.3f}): {e}, retrying...")
                else:
                    print(f"  [{side}] All saddle attempts failed: {e}")
                    return current

        # Register saddle
        sad_id, _ = self._update_saddle(saddle)
        saddle.id = sad_id
        h_sad = (saddle.get_volume() * cfg.press * GPa
                 + saddle.get_potential_energy() - self.baseenergy)
        self.G.add_node(sad_id, xname=f'S{sad_id}', e=h_sad,
                        volume=saddle.get_volume())

        # Edge: current → saddle
        fp_c = current.get_fp()
        fp_s = saddle.get_fp()
        d_cs = fp_distance(fp_c, fp_s, types)
        h_cur = self.G.nodes[current.id]['e']
        self.G.add_edge(current.id, sad_id,
                        weight=max(h_cur, h_sad), dist=d_cs)

        # Step 4: Validate saddle
        curv = getattr(saddle, 'dimer_curvature', None)
        curv_str = f", κ={curv:.3f}" if curv is not None else ""

        if curv is not None and curv >= 0:
            print(f"  [{side}] SKIP: S{sad_id} has positive curvature "
                  f"({curv:.4f}) — not a saddle")
            return current

        if h_sad < h_cur:
            print(f"  [{side}] WARNING: saddle S{sad_id} ({h_sad:.3f}) "
                  f"< current M{current.id} ({h_cur:.3f}){curv_str}")

        # Step 5: Escape saddle along dimer mode toward target, then bias
        escaped = self._saddle_escape(saddle, target_fp)

        # Step 6: Optimize on real PES
        new_min = local_optimization(escaped, fmax=cfg.opt_fmax,
                                     steps=cfg.opt_steps)
        min_id, _ = self._update_minima(new_min)
        new_min.id = min_id

        h_min = (new_min.get_volume() * cfg.press * GPa
                 + new_min.get_potential_energy() - self.baseenergy)
        self.G.add_node(min_id, xname=f'M{min_id}', e=h_min,
                        volume=new_min.get_volume())

        # Edge: saddle → new minimum
        fp_m = new_min.get_fp()
        d_sm = fp_distance(fp_s, fp_m, types)
        self.G.add_edge(sad_id, min_id,
                        weight=max(h_sad, h_min), dist=d_sm)

        # Progress report
        d_target = fp_distance(fp_m, target_fp, types)
        print(f"  [{side}] S{sad_id} ({h_sad:.3f}) → M{min_id} ({h_min:.3f}) "
              f"| d(→target)={d_target:.5f}")
        return new_min

    def _fp_gradient_mode(self, structure, target_fp):
        """Compute normalized mode pointing toward target in FP space.

        Uses XCalculator to get forces (position gradient) and stress
        (cell gradient) of the FP distance, then assembles them into
        a (natom+3, 3) dimer-compatible mode vector.

        Parameters
        ----------
        structure : PallasAtom
        target_fp : np.ndarray, shape (nat, fp_dim)

        Returns
        -------
        np.ndarray, shape (natom+3, 3) — normalized mode.
        """
        atoms = structure.copy()
        calc = XCalculator(
            fp0=target_fp, znucl=self.config.znucl,
            cutoff=self.config.fpcutoff, natx=self.config.natx)
        atoms.calc = calc

        forces = atoms.get_forces()      # (natom, 3)
        stress = atoms.get_stress()      # (6,) Voigt

        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        # Voigt → 3x3 strain gradient
        # stress_voigt = (1/V) * dE/d(epsilon)
        # Cell descent direction: -dE/d(epsilon) = -V * stress_3x3
        stress_3x3 = np.zeros((3, 3))
        stress_3x3[0, 0] = stress[0]
        stress_3x3[1, 1] = stress[1]
        stress_3x3[2, 2] = stress[2]
        stress_3x3[1, 2] = stress_3x3[2, 1] = stress[3]
        stress_3x3[0, 2] = stress_3x3[2, 0] = stress[4]
        stress_3x3[0, 1] = stress_3x3[1, 0] = stress[5]

        mode = np.zeros((natom + 3, 3))
        mode[:natom] = forces                           # position direction
        mode[-3:] = -jacob * vol * stress_3x3           # cell direction

        # Constrain redundant freedoms
        mode[0] *= 0
        mode[-3, 1:] *= 0
        mode[-2, 2] *= 0

        return vunit(mode)

    def _fp_guided_saddle(self, structure, mode, step_scale):
        """Perturb along FP gradient, then find saddle with aligned dimer.

        Parameters
        ----------
        structure : PallasAtom — starting minimum.
        mode : np.ndarray — FP gradient mode (natom+3, 3).
        step_scale : float — perturbation magnitude.

        Returns
        -------
        PallasAtom — saddle point.
        """
        atoms = cp(structure)
        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        # Perturb along FP gradient direction
        scaled = mode * step_scale
        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), scaled[-3:] / jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + scaled[:-3])
        atoms.invalidate_fp()

        # Run dimer with FP gradient as initial mode (structure already perturbed)
        cfg = self.config
        return cal_saddle(atoms, fmax=cfg.saddle_fmax, steps=cfg.saddle_steps,
                          mode=mode)

    def _saddle_escape(self, saddle, target_fp):
        """Escape saddle toward target using dimer mode + XCalculator bias.

        Two-step descent:
        1. Push along the dimer's unstable mode in the direction toward target
           (this crosses the saddle barrier into the next basin)
        2. Run XCalculator bias to drive further toward target in FP space

        Parameters
        ----------
        saddle : PallasAtom — saddle point with .dimer_mode attribute.
        target_fp : np.ndarray — target fingerprints.

        Returns
        -------
        PallasAtom — structure on the target side of the saddle.
        """
        atoms = cp(saddle)
        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        # Get dimer's unstable mode
        dimer_mode = getattr(saddle, 'dimer_mode', None)
        if dimer_mode is None:
            # Fallback: use FP gradient
            dimer_mode = self._fp_gradient_mode(saddle, target_fp)

        # Determine which direction along the dimer mode leads toward target:
        # project FP gradient onto dimer mode
        fp_mode = self._fp_gradient_mode(saddle, target_fp)
        dot = np.vdot(dimer_mode, fp_mode)
        push_dir = vunit(dimer_mode) * np.sign(dot) if abs(dot) > 1e-12 \
            else vunit(fp_mode)

        # Step 1: Push along dimer mode (larger step to cross the saddle)
        push_scale = self.config.fp_push_scale * 3.0  # stronger than FP push
        scaled = push_dir * push_scale
        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), scaled[-3:] / jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + scaled[:-3])

        # Step 2: XCalculator bias toward target (proven to work)
        calc = XCalculator(
            fp0=target_fp, znucl=self.config.znucl,
            cutoff=self.config.fpcutoff, natx=self.config.natx)
        atoms.calc = calc
        af = FrechetCellFilter(atoms)
        opt = FIRE(af, maxstep=0.1, logfile='xcal_escape.log')
        opt.run(fmax=0.01, steps=self.config.bias_steps)

        if hasattr(atoms, 'invalidate_fp'):
            atoms.invalidate_fp()
        return atoms

    def _fp_guided_push(self, saddle, target_fp):
        """Small FP-guided push on saddle toward target.

        Parameters
        ----------
        saddle : PallasAtom — saddle point structure.
        target_fp : np.ndarray — target fingerprint.

        Returns
        -------
        PallasAtom — slightly displaced structure.
        """
        mode = self._fp_gradient_mode(saddle, target_fp)
        atoms = cp(saddle)
        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        scaled = mode * self.config.fp_push_scale
        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), scaled[-3:] / jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + scaled[:-3])
        atoms.invalidate_fp()
        return atoms

    # ── Database / deduplication ──────────────────────────────────────

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
            if d < 0.005 and ediff < 0.001:
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
            if d < 0.005 and ediff < 0.001:
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

    # ── I/O ──────────────────────────────────────────────────────────

    def _save_state(self):
        """Save graph and distance matrix to disk."""
        joblib.dump(self.G, 'graph.pkl')
        nx.write_gml(self.G, 'graph.gml')
        nx.write_gexf(self.G, 'graph.gexf')
        joblib.dump(self.dij, 'dij.pkl')

    def find_best_path(self):
        """Find minimax-bottleneck path between reactant and product.

        Returns
        -------
        path : list of node IDs
        bottleneck : float — max edge weight along the path.
        """
        react_id = self.init_minima[0].id
        prod_id = self.init_minima[1].id
        return minimax_path(self.G, react_id, prod_id)


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
        path, bottleneck = minimax_path(G, start, end)
    except nx.NetworkXNoPath:
        print(f"No path between {start} and {end}")
        return

    print(f"Best path (minimax): {path}")
    print(f"Bottleneck energy: {bottleneck:.4f}")

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

    print(f"Output written to path_output/")


# ── Entry points ──────────────────────────────────────────────────────

def main():
    """Default run: read POSCAR1/POSCAR2, run FP-guided multi-probe search."""
    config = PallasConfig()
    atoms = read('POSCAR1', format='vasp')
    config.znucl = sorted(set(atoms.get_atomic_numbers().tolist()))

    pallas = Pallas(config)
    pallas.init_run(['POSCAR1', 'POSCAR2'])
    pallas.run_fp_guided(n_probes=3)


if __name__ == "__main__":
    main()
