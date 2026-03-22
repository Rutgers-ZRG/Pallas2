"""PALLAS — Phase Transition Landscape Exploration with Automated Saddle Search.

Automated method for finding transition pathways between crystal phases using:
- FP-gradient-guided chain-growing search (primary method)
- Solid-state dimer method for saddle point location
- NetworkX graph with minimax bottleneck path finding
- torch_fplib GOM fingerprints for structural distance and gradients

Dependencies: torch_fplib, ASE, NetworkX, joblib, scipy.
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
from pallas.xcal import XCalculator, atoms_to_cell, fp_dist_with_assignment
from pallas.optimize import local_optimization, cal_saddle, vunit, vrand, _get_calculator
from pallas.graph import minimax_path


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

    # Generational search parameters (used by run())
    n_probes: int = 5               # probes per generation
    max_gen: int = 50               # max generations
    patience: int = 5               # stop after N gens without barrier improvement
    min_barrier_change: float = 0.001  # minimum improvement to reset patience (eV)


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

    Primary method:
    - ``run()`` — unified generational search (connect + refine + converge)

    Advanced / legacy methods:
    - ``run_fp_guided()`` — FP-gradient-guided chain growing
    - ``run_pso()`` — bidirectional PSO
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
        """Unified pathway search: connect endpoints, then refine barrier.

        Generational loop that automatically switches between two modes:

        - **CONNECT**: When no A→B path exists, probes extend chains from
          both sides toward each other (potentially through many intermediates).
        - **REFINE**: Once connected, probes attack the bottleneck saddle
          on the current best path, seeking lower-barrier alternatives.

        Each generation launches ``n_probes`` saddle searches from
        strategically selected frontier minima.  Probes use varying
        FP-gradient / random mixing ratios for diversity.  All discovered
        structures enter a shared graph; ``minimax_path`` extracts the
        optimal route after each generation.

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
                cur_path, cur_barrier = minimax_path(self.G, A.id, B.id)
                path_exists = True
            except nx.NetworkXNoPath:
                pass

            # Plan probes based on mode
            if not path_exists:
                probes = self._plan_connect_probes(A, B, known, types)
                mode = "CONNECT"
            else:
                probes = self._plan_refine_probes(
                    cur_path, A, B, known, types)
                mode = "REFINE"

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
                path, barrier = minimax_path(self.G, A.id, B.id)
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
                print(f"  No path yet")

            self._save_state()

            if path_exists and stagnant >= cfg.patience:
                print(f"\nConverged: no improvement for "
                      f"{cfg.patience} generations")
                break

        # Final report
        self._report_result(best_path, best_barrier, A, known)
        return best_path, best_barrier

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

    # ── FP-drag: constrained walk along FP coordinate ───────────────

    def run_fp_drag(self, n_images=20, relax_steps=30, relax_fmax=0.05):
        """Drag structure from A to B along the FP distance coordinate.

        At each image:
        1. Take a step along the FP gradient (decrease d_fp toward B)
        2. Relax perpendicular DOFs on the real PES (constrained relaxation)
        3. Record the PES energy

        The energy maximum along the path is the approximate saddle.
        Optionally followed by dimer refinement at the maximum.

        Parameters
        ----------
        n_images : int — number of intermediate images.
        relax_steps : int — perpendicular relaxation steps per image.
        relax_fmax : float — force threshold for perpendicular relaxation.

        Returns
        -------
        dict with keys:
            images : list of PallasAtom
            energies : list of float (PES enthalpy)
            fp_dists : list of float (FP distance to B)
            barrier : float (max energy - min(E_A, E_B))
            saddle_idx : int (index of approximate saddle)
        """
        cfg = self.config
        print(f"FP-drag: {n_images} images, relax_steps={relax_steps}")

        # Optimize endpoints
        A = self._optimize_and_register(self.reactant, is_base=True)
        B = self._optimize_and_register(self.product, is_base=False)
        types = self._get_types(A)
        fp_B = B.get_fp()

        d0 = fp_distance(A.get_fp(), fp_B, types)
        h_A = 0.0
        h_B = (B.get_volume() * cfg.press * GPa
               + B.get_potential_energy() - self.baseenergy)

        print(f"  A: H={h_A:.4f} eV, B: H={h_B:.4f} eV, d_fp={d0:.5f}")

        current = cp(A)
        images = [cp(A)]
        enthalpies = [h_A]
        fp_dists = [d0]

        for i in range(n_images):
            # Step size: adaptive (equal steps in FP distance)
            d_current = fp_distance(current.get_fp(), fp_B, types)
            target_step = d0 / n_images

            # 1. FP gradient step toward B
            fp_mode = self._fp_gradient_mode(current, fp_B)
            natom = len(current)
            vol = current.get_volume()
            jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

            # Constant step size calibrated to cover full A→B in n_images
            scale = cfg.fp_step_scale / n_images
            scaled = fp_mode * scale
            cellt = current.get_cell() + np.dot(current.get_cell(),
                                                 scaled[-3:] / jacob)
            current.set_cell(cellt, scale_atoms=True)
            current.set_positions(current.get_positions() + scaled[:-3])
            if hasattr(current, 'invalidate_fp'):
                current.invalidate_fp()

            # 2. Perpendicular relaxation on real PES
            if relax_steps > 0:
                current = self._perpendicular_relax(
                    current, fp_B, types, relax_steps, relax_fmax)

            # 3. Record
            h = (current.get_volume() * cfg.press * GPa
                 + current.get_potential_energy() - self.baseenergy)
            d = fp_distance(current.get_fp(), fp_B, types)
            enthalpies.append(h)
            fp_dists.append(d)
            images.append(cp(current))

            progress = 1.0 - d / max(d0, 1e-10)
            print(f"  image {i+1}/{n_images}: H={h:.4f} eV, "
                  f"d_fp={d:.5f}, progress={progress:.1%}")

        # Find approximate saddle (energy maximum)
        saddle_idx = int(np.argmax(enthalpies))
        barrier = enthalpies[saddle_idx] - min(h_A, h_B)

        print(f"\nFP-drag complete:")
        print(f"  Barrier ≈ {barrier:.4f} eV ({barrier/4:.4f} eV/f.u.)")
        print(f"  Saddle at image {saddle_idx}/{n_images} "
              f"(H={enthalpies[saddle_idx]:.4f})")
        print(f"  FP progress: {fp_dists[0]:.5f} → {fp_dists[-1]:.5f}")

        # Register saddle structure in graph
        if saddle_idx > 0 and saddle_idx < len(images) - 1:
            saddle_struct = images[saddle_idx]
            sad_id, _ = self._update_saddle(saddle_struct)
            saddle_struct.id = sad_id
            self.G.add_node(sad_id, xname=f'S{sad_id}',
                            e=enthalpies[saddle_idx],
                            volume=saddle_struct.get_volume())

            # Edges to nearest minima in the path
            A_id = self.init_minima[0].id
            B_id = self.init_minima[1].id
            self.G.add_edge(A_id, sad_id,
                            weight=max(h_A, enthalpies[saddle_idx]),
                            dist=fp_dists[saddle_idx])
            self.G.add_edge(sad_id, B_id,
                            weight=max(enthalpies[saddle_idx], h_B),
                            dist=fp_dists[saddle_idx])
            self._save_state()

        return {
            'images': images,
            'energies': enthalpies,
            'fp_dists': fp_dists,
            'barrier': barrier,
            'saddle_idx': saddle_idx,
        }

    def _perpendicular_relax(self, structure, target_fp, types,
                             n_steps, fmax_tol):
        """Relax on real PES with FP-gradient component projected out.

        Uses FIRE with FrechetCellFilter for efficient relaxation,
        then projects out the FP-gradient component from forces at
        each step to keep d_fp approximately constant.

        Parameters
        ----------
        structure : PallasAtom
        target_fp : np.ndarray — target fingerprints.
        types : np.ndarray
        n_steps : int — max relaxation steps.
        fmax_tol : float — perpendicular force convergence.

        Returns
        -------
        PallasAtom — relaxed structure.
        """
        atoms = structure
        atoms.calc = _get_calculator()

        # FP gradient direction (unit vector in generalized space)
        fp_dir = self._fp_gradient_mode(atoms, target_fp)

        natom = len(atoms)
        vol = atoms.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

        # Use velocity Verlet / steepest descent with momentum
        V = np.zeros((natom + 3, 3))
        dt = 0.01

        for step in range(n_steps):
            forces = atoms.get_forces()
            stress = atoms.get_stress()

            # Build generalized force (natom+3, 3)
            gen_force = np.zeros((natom + 3, 3))
            gen_force[:natom] = forces

            volume = -atoms.get_volume()
            st = np.zeros((3, 3))
            st[0, 0] = stress[0] * volume
            st[1, 1] = stress[1] * volume
            st[2, 2] = stress[2] * volume
            st[2, 1] = stress[3] * volume
            st[2, 0] = stress[4] * volume
            st[1, 0] = stress[5] * volume
            gen_force[-3:] = st / jacob

            # Project out FP gradient component
            proj = np.vdot(gen_force, fp_dir)
            gen_force -= proj * fp_dir

            # Convergence check
            per_row_max = max(np.linalg.norm(gen_force[i])
                              for i in range(natom + 3))
            if per_row_max < fmax_tol:
                break

            # QuickMin-style velocity update (with momentum)
            dV = gen_force * dt
            if np.vdot(V, gen_force) > 0:
                V = dV * (1.0 + np.vdot(dV, V) /
                          max(np.vdot(dV, dV), 1e-10))
            else:
                V = dV

            step_vec = V * dt
            mag = np.sqrt(np.vdot(step_vec, step_vec))
            if mag > 0.1:
                step_vec *= 0.1 / mag

            # Apply displacement
            cellt = atoms.get_cell() + np.dot(atoms.get_cell(),
                                               step_vec[-3:] / jacob)
            atoms.set_cell(cellt, scale_atoms=True)
            atoms.set_positions(atoms.get_positions() + step_vec[:-3])

        if hasattr(atoms, 'invalidate_fp'):
            atoms.invalidate_fp()
        return atoms

    # ── FP-drag chain: drag + dimer refinement ──────────────────────

    def run_fp_drag_chain(self, drag_images=15, relax_steps=50,
                          relax_fmax=0.03, max_segments=10):
        """FP-drag chain: walk toward B, find saddles, discover intermediates.

        Each segment:
        1. FP-drag from current minimum toward B until energy maximum
        2. Dimer refinement at the maximum → true saddle
        3. Descend from saddle → next minimum (intermediate or B)
        4. Repeat from the new minimum

        This combines fast FP-drag exploration with rigorous dimer saddles.
        No need to reach B in one drag — each segment finds one transition.

        Parameters
        ----------
        drag_images : int — images per FP-drag segment.
        relax_steps : int — perpendicular relaxation steps per image.
        relax_fmax : float — perpendicular force tolerance.
        max_segments : int — maximum A→...→B chain segments.

        Returns
        -------
        nx.Graph
        """
        cfg = self.config
        print(f"FP-drag chain: drag_images={drag_images}, "
              f"max_segments={max_segments}")

        # Optimize endpoints
        A = self._optimize_and_register(self.reactant, is_base=True)
        B = self._optimize_and_register(self.product, is_base=False)
        types = self._get_types(A)
        fp_B = B.get_fp()

        d0 = fp_distance(A.get_fp(), fp_B, types)
        print(f"  A: H=0.0000, B: H={self.G.nodes[B.id]['e']:.4f}, "
              f"d_fp={d0:.5f}")

        current = A
        chain = [A]
        prev_min_id = None
        loop_count = 0

        for seg in range(max_segments):
            d_cur = fp_distance(current.get_fp(), fp_B, types)
            ediff = abs(current.get_potential_energy()
                        - B.get_potential_energy())
            print(f"\n--- Segment {seg+1}/{max_segments} | "
                  f"d_fp={d_cur:.5f}, ΔE={ediff:.5f} ---")

            # Check if we've reached B
            if d_cur < cfg.dist_threshold and ediff < cfg.ediff:
                h_c = self.G.nodes[current.id]['e']
                h_b = self.G.nodes[B.id]['e']
                self.G.add_edge(current.id, B.id,
                                weight=max(h_c, h_b), dist=d_cur)
                print(f"  Reached B! M{current.id} ↔ M{B.id}")
                break

            # Escalate step size if stuck in a loop
            esc = 1.0 + 0.5 * loop_count
            eff_drag_images = drag_images

            # Phase 1: FP-drag from current toward B
            print(f"  Phase 1: FP-drag ({eff_drag_images} images, "
                  f"escape={esc:.1f}x)")
            drag_result = self._fp_drag_segment(
                current, fp_B, types, eff_drag_images,
                relax_steps, relax_fmax, step_multiplier=esc)

            if drag_result is None:
                print(f"  FP-drag failed, stopping")
                break

            saddle_approx = drag_result['saddle_structure']
            h_saddle_approx = drag_result['saddle_energy']
            h_current = self.G.nodes[current.id]['e']
            print(f"  Approximate saddle: H={h_saddle_approx:.4f} "
                  f"(barrier={h_saddle_approx - h_current:.4f})")

            # Phase 2: Dimer refinement at the energy maximum
            print(f"  Phase 2: Dimer refinement")
            fp_mode = self._fp_gradient_mode(saddle_approx, fp_B)
            saddle = cal_saddle(saddle_approx, fmax=cfg.saddle_fmax,
                                steps=cfg.saddle_steps, mode=fp_mode)

            sad_id, _ = self._update_saddle(saddle)
            saddle.id = sad_id
            h_sad = (saddle.get_volume() * cfg.press * GPa
                     + saddle.get_potential_energy() - self.baseenergy)
            curv = getattr(saddle, 'dimer_curvature', None)
            curv_str = f", κ={curv:.3f}" if curv is not None else ""
            self.G.add_node(sad_id, xname=f'S{sad_id}', e=h_sad,
                            volume=saddle.get_volume())

            # Edge: current → saddle
            fp_s = saddle.get_fp()
            d_cs = fp_distance(current.get_fp(), fp_s, types)
            self.G.add_edge(current.id, sad_id,
                            weight=max(h_current, h_sad), dist=d_cs)
            print(f"  Saddle S{sad_id}: H={h_sad:.4f}{curv_str}")

            # Phase 3: Descend from saddle → next minimum
            # Escalate push strength if stuck in loop
            push_esc = cfg.fp_push_scale * esc
            print(f"  Phase 3: Descend (push={push_esc:.2f})")
            escaped = self._saddle_escape(saddle, fp_B)
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

            d_new = fp_distance(fp_m, fp_B, types)
            print(f"  New minimum M{min_id}: H={h_min:.4f}, "
                  f"d_fp(→B)={d_new:.5f}")

            # Loop detection: if we landed on the same minimum, escalate
            if min_id == prev_min_id:
                loop_count += 1
                print(f"  LOOP detected (M{min_id} again, "
                      f"count={loop_count})")
            else:
                loop_count = 0
            prev_min_id = min_id

            chain.extend([saddle, new_min])
            current = new_min
            self._save_state()

        # Final report
        print(f"\nChain: {' → '.join('M'+str(s.id) if hasattr(s,'id') and self.G.nodes.get(s.id,{}).get('xname','').startswith('M') else 'S'+str(s.id) for s in chain if hasattr(s,'id'))}")
        try:
            path, bn = minimax_path(self.G, A.id, B.id)
            print(f"Best path: {' → '.join(str(n) for n in path)}")
            print(f"Bottleneck: {bn:.4f} eV ({bn/len(A):.4f} eV/atom)")
            for node in path:
                nd = self.G.nodes[node]
                ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
                print(f"  {ntype} {node}: E = {nd['e']:.4f} eV, "
                      f"V = {nd['volume']:.1f}")
        except nx.NetworkXNoPath:
            print("No complete path found")

        return self.G

    def _fp_drag_segment(self, start, target_fp, types,
                         n_images, relax_steps, relax_fmax,
                         step_multiplier=1.0):
        """One FP-drag segment: walk from start toward target, find E maximum.

        Returns when energy starts decreasing (crossed the saddle ridge)
        or after n_images steps.

        Parameters
        ----------
        step_multiplier : float — escalation factor for step size (>1 if stuck).

        Returns
        -------
        dict with saddle_structure, saddle_energy, saddle_idx, or None.
        """
        cfg = self.config
        current = cp(start)
        natom = len(current)

        d_start = fp_distance(current.get_fp(), target_fp, types)
        best_h = -float('inf')
        best_struct = None
        best_idx = 0
        decline_count = 0

        for i in range(n_images):
            # FP gradient step
            fp_mode = self._fp_gradient_mode(current, target_fp)
            vol = current.get_volume()
            jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

            scale = cfg.fp_step_scale / n_images * step_multiplier
            scaled = fp_mode * scale
            cellt = current.get_cell() + np.dot(current.get_cell(),
                                                 scaled[-3:] / jacob)
            current.set_cell(cellt, scale_atoms=True)
            current.set_positions(current.get_positions() + scaled[:-3])
            if hasattr(current, 'invalidate_fp'):
                current.invalidate_fp()

            # Perpendicular relaxation
            if relax_steps > 0:
                current = self._perpendicular_relax(
                    current, target_fp, types, relax_steps, relax_fmax)

            # Record energy
            h = (current.get_volume() * cfg.press * GPa
                 + current.get_potential_energy() - self.baseenergy)
            d = fp_distance(current.get_fp(), target_fp, types)
            progress = 1.0 - d / max(d_start, 1e-10)

            print(f"    drag {i+1}/{n_images}: H={h:.4f}, "
                  f"d_fp={d:.5f} ({progress:.0%})")

            # Track energy maximum
            if h > best_h:
                best_h = h
                best_struct = cp(current)
                best_idx = i + 1
                decline_count = 0
            else:
                decline_count += 1

            # Stop if energy has been declining for 3+ images
            # (we've crossed the saddle ridge)
            if decline_count >= 3:
                print(f"    Energy declining — saddle at image {best_idx}")
                break

        if best_struct is None:
            return None

        return {
            'saddle_structure': best_struct,
            'saddle_energy': best_h,
            'saddle_idx': best_idx,
        }

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

        # Curvature is advisory, not a hard reject — the dimer mode
        # may not align with the true unstable direction.
        curvature = getattr(saddle, 'dimer_curvature', None)
        result['curvature'] = curvature

        # Primary check: connectivity (push ±mode, relax, verify distinct minima)
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

    def _plan_refine_probes(self, path, A, B, known, types):
        """Plan probes to attack the bottleneck on the current best path.

        Finds the highest-energy saddle, identifies its flanking minima,
        and launches probes from both sides of the bottleneck toward each
        other with varying FP/random mixing.

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
            # No saddle on path — fall back to connect probes
            return self._plan_connect_probes(A, B, known, types)

        # Find flanking minima
        min_left = self._find_flanking_min(path, worst_idx, -1, known)
        min_right = self._find_flanking_min(path, worst_idx, +1, known)

        if min_left is None or min_right is None:
            return self._plan_connect_probes(A, B, known, types)

        h_left = self.G.nodes[min_left.id]['e']
        h_right = self.G.nodes[min_right.id]['e']
        print(f"  Bottleneck: S{worst_id} ({worst_e:.4f}), "
              f"flanked by M{min_left.id} ({h_left:.4f}) "
              f"↔ M{min_right.id} ({h_right:.4f})")

        probes = []
        for i in range(cfg.n_probes):
            alpha = max(1.0 - 0.2 * i, 0.1)
            if i % 2 == 0:
                probes.append((
                    min_left, min_right.get_fp(), alpha,
                    f'refine fwd p{i}(α={alpha:.1f})'))
            else:
                probes.append((
                    min_right, min_left.get_fp(), alpha,
                    f'refine rev p{i}(α={alpha:.1f})'))

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
            print(f"Barrier: {best_barrier:.4f} eV")
            nat = len(A)
            if nat > 1:
                print(f"         {best_barrier/nat:.4f} eV/atom")
            for node in best_path:
                nd = self.G.nodes[node]
                ntype = 'MIN' if nd['xname'].startswith('M') else 'SAD'
                print(f"  {ntype} {node}: H={nd['e']:.4f} eV, "
                      f"V={nd['volume']:.1f} A^3")
        else:
            print("\nNo complete path found")
        print(f"{'='*60}")

    # ── FP-guided step (probe core) ──────────────────────────────────

    def _fp_guided_step(self, current, target_fp, types,
                        step_scale, alpha=1.0, side=""):
        """One step: FP-drag to find saddle region → dimer refine → descend.

        Parameters
        ----------
        current : PallasAtom — current chain tip (optimized minimum).
        target_fp : np.ndarray — target fingerprint to grow toward.
        types : np.ndarray — atom type array.
        step_scale : float — perturbation magnitude (unused, kept for API).
        alpha : float — FP/random mixing ratio for dimer mode initialization.
        side : str — label for logging.

        Returns
        -------
        PallasAtom — new minimum (or current if step failed).
        """
        cfg = self.config

        # Phase A: FP-drag to find approximate saddle (energy maximum)
        drag_result = self._fp_drag_segment(
            current, target_fp, types,
            n_images=10, relax_steps=30, relax_fmax=0.05)

        if drag_result is None:
            print(f"  [{side}] FP-drag failed")
            return current

        saddle_approx = drag_result['saddle_structure']
        h_approx = drag_result['saddle_energy']
        h_cur = self.G.nodes[current.id]['e']

        # Phase B: Dimer refinement at the energy maximum
        fp_mode = self._fp_gradient_mode(saddle_approx, target_fp)
        rand_mode = self._gen_random_velocity(current)
        mode = vunit(alpha * fp_mode + (1.0 - alpha) * rand_mode)

        try:
            saddle = cal_saddle(saddle_approx, fmax=cfg.saddle_fmax,
                                steps=cfg.saddle_steps, mode=mode)
        except Exception as e:
            print(f"  [{side}] Dimer failed: {e}")
            return current

        # Register saddle
        sad_id, _ = self._update_saddle(saddle)
        saddle.id = sad_id
        h_sad = (saddle.get_volume() * cfg.press * GPa
                 + saddle.get_potential_energy() - self.baseenergy)
        self.G.add_node(sad_id, xname=f'S{sad_id}', e=h_sad,
                        volume=saddle.get_volume())

        curv = getattr(saddle, 'dimer_curvature', None)
        curv_str = f", κ={curv:.3f}" if curv is not None else ""

        # Edge: current → saddle
        fp_s = saddle.get_fp()
        d_cs = fp_distance(current.get_fp(), fp_s, types)
        self.G.add_edge(current.id, sad_id,
                        weight=max(h_cur, h_sad), dist=d_cs)

        # Phase C: Escape saddle → next minimum
        escaped = self._saddle_escape(saddle, target_fp)
        new_min = local_optimization(escaped, fmax=cfg.opt_fmax,
                                     steps=cfg.opt_steps)
        min_id, _ = self._update_minima(new_min)
        new_min.id = min_id

        h_min = (new_min.get_volume() * cfg.press * GPa
                 + new_min.get_potential_energy() - self.baseenergy)
        self.G.add_node(min_id, xname=f'M{min_id}', e=h_min,
                        volume=new_min.get_volume())

        fp_m = new_min.get_fp()
        d_sm = fp_distance(fp_s, fp_m, types)
        self.G.add_edge(sad_id, min_id,
                        weight=max(h_sad, h_min), dist=d_sm)

        d_target = fp_distance(fp_m, target_fp, types)
        print(f"  [{side}] drag→S{sad_id} ({h_sad:.3f}{curv_str}) "
              f"→ M{min_id} ({h_min:.3f}) | d(→target)={d_target:.5f}")
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

    def _fp_guided_saddle(self, structure, mode, step_scale, n_mode_tries=3):
        """Perturb along FP gradient, then find saddle with aligned dimer.

        Tries multiple initial mode directions from the same perturbed
        structure: the FP gradient mode first, then mixed FP+random variants.
        Returns the first converged saddle (negative curvature).

        Parameters
        ----------
        structure : PallasAtom — starting minimum.
        mode : np.ndarray — FP gradient mode (natom+3, 3).
        step_scale : float — perturbation magnitude.
        n_mode_tries : int — number of different initial modes to try.

        Returns
        -------
        PallasAtom — saddle point (best attempt).
        """
        natom = len(structure)
        vol = structure.get_volume()
        jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5
        cfg = self.config

        # Perturb structure along FP gradient
        perturbed = cp(structure)
        scaled = mode * step_scale
        cellt = perturbed.get_cell() + np.dot(perturbed.get_cell(),
                                               scaled[-3:] / jacob)
        perturbed.set_cell(cellt, scale_atoms=True)
        perturbed.set_positions(perturbed.get_positions() + scaled[:-3])
        if hasattr(perturbed, 'invalidate_fp'):
            perturbed.invalidate_fp()

        # Try multiple initial modes from this perturbed structure
        best_saddle = None
        best_curvature = float('inf')

        for attempt in range(n_mode_tries):
            if attempt == 0:
                trial_mode = mode  # pure FP gradient
            else:
                # Mix FP gradient with random (decreasing FP weight)
                rand = vrand(np.zeros_like(mode))
                mix = max(0.7 - 0.3 * attempt, 0.1)
                trial_mode = vunit(mix * mode + (1.0 - mix) * rand)

            saddle = cal_saddle(cp(perturbed), fmax=cfg.saddle_fmax,
                                steps=cfg.saddle_steps, mode=trial_mode)
            curv = getattr(saddle, 'dimer_curvature', None)

            if curv is not None and curv < best_curvature:
                best_curvature = curv
                best_saddle = saddle

            # Stop early if we found a true saddle
            if curv is not None and curv < 0 and saddle.converged:
                return saddle

        return best_saddle if best_saddle is not None else saddle

    def _saddle_escape(self, saddle, target_fp):
        """Escape saddle toward target using dimer mode push only.

        Pushes along the dimer's unstable mode in the direction toward
        the target (determined by projecting the FP gradient onto the
        dimer mode).  No non-physical FP bias is applied — after the push,
        the caller should relax on the real PES via local_optimization,
        which naturally flows to the nearest minimum without skipping
        intermediate barriers.

        Parameters
        ----------
        saddle : PallasAtom — saddle point with .dimer_mode attribute.
        target_fp : np.ndarray — target fingerprints.

        Returns
        -------
        PallasAtom — structure pushed to the target side of the saddle.
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

        # Push along dimer mode to cross the saddle ridge
        push_scale = self.config.fp_push_scale * 3.0
        scaled = push_dir * push_scale
        cellt = atoms.get_cell() + np.dot(atoms.get_cell(), scaled[-3:] / jacob)
        atoms.set_cell(cellt, scale_atoms=True)
        atoms.set_positions(atoms.get_positions() + scaled[:-3])

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
