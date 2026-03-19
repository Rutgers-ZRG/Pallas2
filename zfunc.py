"""Optimization utilities for PALLAS.

Provides local structure optimization (MatterSim) and saddle point search
(solid-state dimer method). MatterSim calculator is lazily initialized on
first use.
"""

import numpy as np
from copy import deepcopy as cp

from ase.io import read, write, Trajectory
from ase.optimize import FIRE, BFGS
from ase.filters import FrechetCellFilter

from dimer import SolidStateDimer


# ── Vector utilities ──────────────────────────────────────────────────

def vunit(v):
    """Normalize vector to unit length."""
    mag = np.sqrt(np.vdot(v, v))
    if mag == 0:
        return v
    return v / mag


def vrand(v):
    """Generate random vector with same shape as v."""
    return np.random.randn(*v.shape)


# ── Lazy MatterSim calculator ─────────────────────────────────────────

_calc = None


def _get_calculator():
    """Lazily initialize and return MatterSim calculator."""
    global _calc
    if _calc is None:
        from mattersim.forcefield import MatterSimCalculator
        _calc = MatterSimCalculator(device="cpu")
    return _calc


# ── Structure optimization ────────────────────────────────────────────

def local_optimization(patoms, fmax=0.001, steps=2000, calc=None):
    """Optimize structure (positions + cell) with MatterSim.

    Parameters
    ----------
    patoms : Atoms — input structure (modified in-place).
    fmax : float — force convergence threshold (eV/Å).
    steps : int — max optimizer steps.
    calc : Calculator, optional — override default MatterSim.

    Returns
    -------
    Atoms — optimized structure with .converged attribute set.
    """
    atoms = patoms
    atoms.calc = calc or _get_calculator()
    ecf = FrechetCellFilter(atoms)
    opt = FIRE(ecf, maxstep=0.1, logfile='opt.log')
    opt.run(fmax=fmax, steps=steps)

    actual_fmax = np.max(np.abs(ecf.get_forces()))
    atoms.converged = actual_fmax <= fmax
    if not atoms.converged:
        print(f"Warning: optimization did not converge (fmax={actual_fmax:.4f})")

    new_cell = lower_triangular_cell(atoms)
    atoms.set_cell(new_cell, scale_atoms=True)

    if hasattr(atoms, 'invalidate_fp'):
        atoms.invalidate_fp()
    return atoms


def cal_saddle(patoms, fmax=0.01, steps=2000, calc=None):
    """Find saddle point using solid-state dimer method.

    Parameters
    ----------
    patoms : Atoms — starting structure.
    fmax : float — force convergence threshold.
    steps : int — max dimer steps.
    calc : Calculator, optional — override default MatterSim.

    Returns
    -------
    Atoms — saddle point structure with .converged attribute set.
    """
    atoms = cp(patoms)
    atoms.calc = calc or _get_calculator()

    natom = len(atoms)
    vol = atoms.get_volume()
    jacob = (vol / natom) ** (1.0 / 3.0) * natom ** 0.5

    # Random initial mode
    mode = vrand(np.zeros((natom + 3, 3)))
    # Constrain redundant freedoms
    mode[0] *= 0
    mode[-3, 1:] *= 0
    mode[-2, 2] *= 0
    mode = vunit(mode)

    # Displace along mode
    cellt = atoms.get_cell() + np.dot(atoms.get_cell(), mode[-3:] / jacob)
    atoms.set_cell(cellt, scale_atoms=True)
    atoms.set_positions(atoms.get_positions() + mode[:-3])

    # Run dimer search
    d = SolidStateDimer(atoms, mode=mode)
    dyn = FIRE(d, maxstep=0.1, logfile='ssdimer.log')
    dyn.run(fmax=fmax, steps=steps)

    actual_fmax = np.max(np.abs(atoms.get_forces()))
    atoms.converged = actual_fmax <= fmax
    if not atoms.converged:
        print(f"Warning: dimer did not converge (fmax={actual_fmax:.4f})")

    new_cell = lower_triangular_cell(atoms)
    atoms.set_cell(new_cell, scale_atoms=True)

    if hasattr(atoms, 'invalidate_fp'):
        atoms.invalidate_fp()
    return atoms


def getx(cell1, cell2):
    """Compute normalized displacement vector between two ASE Atoms.

    Parameters
    ----------
    cell1, cell2 : ase.Atoms

    Returns
    -------
    np.ndarray, shape (nat+3, 3) — normalized displacement mode.
    """
    nat = len(cell1)
    mode = np.zeros((nat + 3, 3))

    # Cell displacement (in reduced coordinates, scaled by Jacobian)
    lat1 = np.array(cell1.get_cell())
    lat2 = np.array(cell2.get_cell())
    ilat = np.linalg.inv(lat1)
    vol = cell1.get_volume()
    jacob = (vol / nat) ** (1.0 / 3.0) * nat ** 0.5

    mode[-3:] = np.dot(ilat, lat2 - lat1) * jacob

    # Position displacement
    pos1 = cell1.get_positions()
    pos2 = cell2.get_positions()
    mode[:nat] = pos2 - pos1

    try:
        mode = vunit(mode)
    except Exception:
        mode = np.zeros((nat + 3, 3))
    return mode


def lower_triangular_cell(atoms):
    """Convert cell to lower-triangular (Niggli-like) form.

    Preserves cell volume and angles. Returns the new 3×3 cell matrix.
    """
    old_cell = atoms.cell.array
    a1 = old_cell[0]

    u1 = a1 / np.linalg.norm(a1)
    v2 = old_cell[1] - np.dot(old_cell[1], u1) * u1
    u2 = v2 / np.linalg.norm(v2)
    u3 = np.cross(u1, u2)

    a = np.linalg.norm(a1)
    bx = np.dot(old_cell[1], u1)
    by = np.dot(old_cell[1], u2)
    cx = np.dot(old_cell[2], u1)
    cy = np.dot(old_cell[2], u2)
    cz = np.dot(old_cell[2], u3)

    return np.array([[a, 0.0, 0.0],
                     [bx, by, 0.0],
                     [cx, cy, cz]])
