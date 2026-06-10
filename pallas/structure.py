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
