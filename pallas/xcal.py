"""ASE calculator for fingerprint-distance energy with autograd forces/stress.

Drives a structure toward a target fingerprint configuration (fp0) by
defining energy = FP distance and computing forces/stress via torch_fplib
autograd (VJP-based, following CRISP's strain parametrization).

Replaces the old libfp + mpi4py implementation with pure PyTorch autograd.
"""

import numpy as np
import torch
import torch_fplib
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, CalculatorSetupError, all_changes
from scipy.optimize import linear_sum_assignment

# Voigt index mapping: (row, col) for xx, yy, zz, yz, xz, xy
_VOIGT_IDX = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def atoms_to_cell(atoms, znucl):
    """Convert ASE Atoms to cell tuple ``(lat, rxyz, types, znucl)``.

    Parameters
    ----------
    atoms : ase.Atoms
    znucl : list of int
        Atomic numbers in type order (e.g. [6] for carbon, [11, 17] for NaCl).

    Returns
    -------
    tuple : (lat, rxyz, types, znucl)
        lat: (3,3) lattice, rxyz: (nat,3) Cartesian, types: (nat,) 1-indexed.
    """
    lat = np.array(atoms.cell)
    rxyz = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    znucl_list = list(znucl)
    types = np.array([znucl_list.index(z) + 1 for z in numbers])
    return lat, rxyz, types, znucl_list


def fp_dist_with_assignment(fp, fp0, types):
    """Hungarian-matched fingerprint distance with atom assignment.

    Parameters
    ----------
    fp, fp0 : array-like, shape (nat, fp_dim)
    types : array-like, shape (nat,)  — 1-indexed atom types.

    Returns
    -------
    dist : float — averaged FP distance.
    assignment : np.ndarray, shape (nat,)  — assignment[k] = index in fp0.
    """
    fp = np.asarray(fp, dtype=np.float64)
    fp0 = np.asarray(fp0, dtype=np.float64)
    types = np.asarray(types)

    nat, lenfp = fp.shape
    unique_types = sorted(set(types.tolist()))
    fpd = 0.0
    assignment = np.arange(nat, dtype=np.int32)

    for ityp in unique_types:
        idx = np.where(types == ityp)[0]
        n = len(idx)
        if n == 0:
            continue
        cost = np.zeros((n, n))
        for a in range(n):
            for b in range(n):
                diff = fp[idx[a]] - fp0[idx[b]]
                cost[a, b] = np.sqrt(np.dot(diff, diff) / lenfp)
        row, col = linear_sum_assignment(cost)
        fpd += cost[row, col].sum()
        for a, b in zip(row, col):
            assignment[idx[a]] = idx[b]

    return fpd / nat, assignment


class XCalculator(Calculator):
    """ASE calculator that minimizes fingerprint distance to a target.

    Energy is the fingerprint distance d(fp, fp0) with Hungarian matching.
    Forces and stress are computed via torch_fplib autograd using CRISP's
    strain-parametrization approach.

    Parameters
    ----------
    fp0 : array-like, shape (nat, fp_dim)
        Target fingerprints to drive toward.
    znucl : list of int
        Atomic numbers in type order.
    cutoff : float
        Fingerprint cutoff radius (Angstrom).
    natx : int
        Maximum neighbors per atom (fingerprint dimension).
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, fp0, znucl, cutoff=5.5, natx=200, **kwargs):
        super().__init__(**kwargs)
        self.fp0 = np.asarray(fp0, dtype=np.float64)
        self.znucl = list(znucl)
        self.cutoff = float(cutoff)
        self.natx = int(natx)
        self._assignment = None

    def calculate(self, atoms=None, properties=None,
                  system_changes=tuple(all_changes)):
        if properties is None:
            properties = self.implemented_properties
        super().calculate(atoms, properties, system_changes)

        atoms = self.atoms
        _check_atoms(atoms)

        lat_np, rxyz_np, types, znucl = atoms_to_cell(atoms, self.znucl)
        vol = atoms.get_volume()

        # --- Step 1: Hungarian assignment (no grad) ---
        with torch.no_grad():
            fp_np = torch_fplib.get_lfp(
                (lat_np, rxyz_np, types, znucl),
                cutoff=self.cutoff, natx=self.natx
            ).numpy()

        dist, assignment = fp_dist_with_assignment(fp_np, self.fp0, types)
        self._assignment = assignment
        fp0_matched = self.fp0[assignment]

        # --- Step 2: Forward pass with autograd ---
        lat_t = torch.tensor(lat_np, dtype=torch.float64)
        spos = torch.tensor(atoms.get_scaled_positions(), dtype=torch.float64)

        # Strain variable (evaluated at ε = 0) for stress
        strain = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        lat_s = (torch.eye(3, dtype=torch.float64) + strain) @ lat_t

        # Fractional-coordinate variable for atomic forces
        spos_var = spos.clone().detach().requires_grad_(True)
        rxyz_t = spos_var @ lat_s

        fp = torch_fplib.get_lfp(
            (lat_s, rxyz_t, types, znucl),
            cutoff=self.cutoff, natx=self.natx
        )

        # Energy = sum of per-atom L2 distances to matched targets
        fp0_t = torch.tensor(fp0_matched, dtype=torch.float64)
        diff = fp - fp0_t
        per_atom_dist = torch.norm(diff, dim=1)
        energy = per_atom_dist.sum()

        # --- Step 3: Backward pass ---
        grad_spos, grad_strain = torch.autograd.grad(
            energy, [spos_var, strain]
        )

        # Forces: fractional gradient → Cartesian
        lat_inv_T = torch.linalg.inv(lat_t.T)
        forces = -(grad_spos @ lat_inv_T).detach().numpy()

        # Stress: Voigt components from strain gradient
        gs = grad_strain.detach().numpy()
        stress_voigt = np.array([gs[a, b] for a, b in _VOIGT_IDX]) / vol

        self.results = {
            'energy': energy.item(),
            'forces': forces,
            'stress': stress_voigt,
        }


# ── Validation helpers ────────────────────────────────────────────────

def _check_atoms(atoms):
    """Validate that atoms can be used for periodic fingerprint calculation."""
    if not isinstance(atoms, Atoms):
        raise CalculatorSetupError(
            f"Expected Atoms object, got {type(atoms)}")
    if atoms.cell.rank < 3:
        raise CalculatorSetupError(
            "Lattice vectors are zero — please specify a unit cell.")
    if not atoms.pbc.all():
        raise CalculatorSetupError(
            "Fingerprint calculator requires periodic boundary conditions.")
