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
    fp_step_scale: float = 0.5      # perturbation scale along FP gradient
    fp_push_scale: float = 0.1      # post-saddle push scale toward target
    max_retries: int = 2            # retries with smaller step on saddle failure

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

    def run_fp_guided(self):
        """FP-gradient-guided bidirectional chain-growing pathway search.

        Grows chains from both reactant (A) and product (B) simultaneously.
        At each step:
        1. Compute FP-gradient direction toward the opposite endpoint
        2. Perturb along this direction (directed, not random)
        3. Run dimer with FP-gradient as initial mode
        4. Validate saddle (energy must exceed the departing minimum)
        5. Push saddle toward target with small FP-gradient step
        6. Optimize on real PES to find next minimum
        7. Check if chains have connected

        Returns
        -------
        nx.Graph — pathway network.
        """
        cfg = self.config
        print("Starting FP-guided chain-growing search")

        # Optimize endpoints
        A = self._optimize_and_register(self.reactant, is_base=True)
        B = self._optimize_and_register(self.product, is_base=False)

        types = self._get_types(A)
        d0 = fp_distance(A.get_fp(), B.get_fp(), types)
        print(f"Initial FP distance A↔B: {d0:.5f}")

        # Bidirectional chains
        chain_A = [A]       # grows from A toward B
        chain_B = [B]       # grows from B toward A

        for step in range(cfg.maxstep):
            tip_A = chain_A[-1]
            tip_B = chain_B[-1]

            # Current distance between chain tips
            d_tips = fp_distance(tip_A.get_fp(), tip_B.get_fp(), types)
            ediff = abs(tip_A.get_potential_energy()
                        - tip_B.get_potential_energy())
            print(f"\n=== Step {step+1}/{cfg.maxstep} | "
                  f"d(tips)={d_tips:.5f}, ΔE={ediff:.5f} ===")

            # Check if chains connected
            if d_tips < cfg.dist_threshold and ediff < cfg.ediff:
                h_a = self.G.nodes[tip_A.id]['e']
                h_b = self.G.nodes[tip_B.id]['e']
                self.G.add_edge(tip_A.id, tip_B.id,
                                weight=max(h_a, h_b), dist=d_tips)
                print(f"Chains connected! M{tip_A.id} ↔ M{tip_B.id}")
                break

            # Adaptive step scale: larger when far, smaller when close
            progress = d_tips / max(d0, 1e-10)
            step_scale = cfg.fp_step_scale * max(progress, 0.1)

            # A-side: grow toward B
            new_A = self._fp_guided_step(
                tip_A, B.get_fp(), types, step_scale, side="A→B")
            if new_A is not tip_A:
                chain_A.append(new_A)

            # B-side: grow toward A
            new_B = self._fp_guided_step(
                tip_B, A.get_fp(), types, step_scale, side="B→A")
            if new_B is not tip_B:
                chain_B.append(new_B)

            self._save_state()

        # Final path search
        try:
            path, bottleneck = minimax_path(self.G, A.id, B.id)
            print(f"\nPath found: {' → '.join(str(n) for n in path)}")
            print(f"Bottleneck energy: {bottleneck:.4f} eV")
        except nx.NetworkXNoPath:
            print("\nNo complete path found (increase maxstep)")

        print(f"\nGraph: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")
        print(f"A-chain length: {len(chain_A)}, B-chain length: {len(chain_B)}")
        return self.G

    def _fp_guided_step(self, current, target_fp, types,
                        step_scale, side=""):
        """One step of FP-guided chain growing.

        Parameters
        ----------
        current : PallasAtom — current chain tip (optimized minimum).
        target_fp : np.ndarray — target fingerprint to grow toward.
        types : np.ndarray — atom type array.
        step_scale : float — perturbation magnitude.
        side : str — label for logging.

        Returns
        -------
        PallasAtom — new minimum (or current if step failed).
        """
        cfg = self.config

        # Step 1: FP gradient mode (direction toward target in FP space)
        mode = self._fp_gradient_mode(current, target_fp)

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

        # Step 4: Validate saddle energy
        if h_sad < h_cur:
            print(f"  [{side}] WARNING: saddle S{sad_id} ({h_sad:.3f}) "
                  f"< current M{current.id} ({h_cur:.3f})")

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
            ids = self.db.write(
                saddle, ctyp='saddle',
                data={'fp': fps.tolist(), 'energy': float(es)}
            )

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
    """Default run: read POSCAR1/POSCAR2, run FP-guided search."""
    config = PallasConfig()
    atoms = read('POSCAR1', format='vasp')
    config.znucl = sorted(set(atoms.get_atomic_numbers().tolist()))

    pallas = Pallas(config)
    pallas.init_run(['POSCAR1', 'POSCAR2'])
    pallas.run_fp_guided()


if __name__ == "__main__":
    main()
