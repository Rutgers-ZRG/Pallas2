"""PallasAtom (Atoms with fingerprint caching) and FP distance helpers."""

import torch
import torch_fplib
from ase import Atoms

from pallas.xcal import atoms_to_cell, fp_dist_with_assignment


class PallasAtom(Atoms):
    """ASE Atoms subclass with cached fingerprints and metadata.

    Fingerprints are computed via torch_fplib and cached until invalidated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.natx = 200
        self.fpcutoff = 5.5
        self.fp = None
        self.converged = None   # None = never optimized/dimered (unknown);
                                # local_optimization/cal_saddle set True/False
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


def enthalpy(energy, volume, press):
    """H = E + P*V. ``press`` is in eV/A^3 (PALLAS internal pressure unit)."""
    return energy + press * volume


def spacegroup_label(atoms, symprecs=(1e-5, 1e-3, 1e-2, 1e-1)):
    """Space-group label of a structure, robust to slight distortion.

    Tries symprec from tight to loose and returns the first result that is
    confirmed by the next-looser tolerance (a "stable" assignment). If no
    two adjacent tolerances agree, returns the tightest result rather than
    overclaiming symmetry. Never raises: worst case is ('P1', 1).

    Returns
    -------
    (international_symbol, number) : (str, int)
    """
    import spglib

    cell = (atoms.get_cell()[:], atoms.get_scaled_positions(),
            atoms.get_atomic_numbers())
    results = []
    for sp in sorted(symprecs):
        try:
            ds = spglib.get_symmetry_dataset(cell, symprec=sp)
        except Exception:
            ds = None
        if ds is None:
            continue
        try:
            results.append((ds.international, int(ds.number)))
        except AttributeError:  # spglib < 2.5 returns a dict
            results.append((ds['international'], int(ds['number'])))
    if not results:
        return ('P1', 1)
    for i in range(len(results) - 1):
        if results[i] == results[i + 1]:
            return results[i]
    return results[0]


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
