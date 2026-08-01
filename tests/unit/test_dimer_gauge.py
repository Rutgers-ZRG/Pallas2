"""Dimer gauge consistency (D2): re-kick constraints + mode frame rotation.

The Voigt->3x3 stress mapping fills only the lower triangle, which assumes
the cell stays lower-triangular (LT). A kick with unconstrained cell
components breaks that gauge permanently; the final lower_triangular_cell()
projection then rotates the structure away from the frame the stored
dimer_mode was captured in.
"""
import numpy as np
import pytest
from ase import Atoms

from pallas.dimer import SolidStateDimer
from pallas.optimize import lower_triangular_cell, rotate_mode_with_cell

pytestmark = pytest.mark.unit


def _dimer(natom=2):
    a = Atoms('Cu%d' % natom,
              positions=[[1.8 * i, 0, 0] for i in range(natom)],
              cell=[[4.0, 0, 0], [0.3, 4.0, 0], [0.2, 0.1, 4.0]], pbc=True)
    return SolidStateDimer(a, solid_state=True)


def test_kick_respects_gauge():
    d = _dimer()
    np.random.seed(0)
    for _ in range(20):
        k = d._generate_kick()
        assert np.all(k[0] == 0.0), "atom-0 pin lost"
        assert np.all(k[-3, 1:] == 0.0), "cell row a not LT"
        assert k[-2, 2] == 0.0, "cell row b not LT"
        assert np.isclose(np.sqrt(np.vdot(k, k)), 1.0)


def test_kick_single_atom_keeps_atomic_component():
    """natom=1 (Mueller-Brown-style): pinning the only atom would make the
    kick cell-only and useless on a position-dependent PES."""
    a = Atoms('H', positions=[[25.0, 25.0, 25.0]], cell=[50.0] * 3, pbc=True)
    d = SolidStateDimer(a, solid_state=True)
    np.random.seed(0)
    k = d._generate_kick()
    assert np.any(k[0] != 0.0)
    assert np.all(k[-3, 1:] == 0.0) and k[-2, 2] == 0.0


def test_rotate_mode_preserves_displacements():
    rng = np.random.default_rng(1)
    old_cell = np.eye(3) * 5.0 + rng.normal(0, 0.4, (3, 3))  # generic non-LT
    natom = 4
    mode = rng.normal(size=(natom + 3, 3))
    atoms = Atoms('H%d' % natom, cell=old_cell, pbc=True)
    new_cell = lower_triangular_cell(atoms)
    R = np.linalg.solve(old_cell, new_cell)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)  # pure rotation

    rot = rotate_mode_with_cell(mode, old_cell, new_cell, natom)
    # atomic displacement rows co-rotate with the structure
    assert np.allclose(rot[:natom], mode[:natom] @ R)
    # cell-block displacements co-rotate: new_cell @ M_new == (old_cell @ M_old) @ R
    assert np.allclose(new_cell @ rot[-3:], (old_cell @ mode[-3:]) @ R,
                       atol=1e-10)


def test_rotate_mode_identity_when_already_lt():
    natom = 3
    lt_cell = np.array([[4.0, 0, 0], [0.5, 4.0, 0], [0.2, 0.3, 4.0]])
    atoms = Atoms('H%d' % natom, cell=lt_cell, pbc=True)
    mode = np.arange((natom + 3) * 3, dtype=float).reshape(natom + 3, 3)
    out = rotate_mode_with_cell(mode, lt_cell, lower_triangular_cell(atoms),
                                natom)
    assert np.allclose(out, mode, atol=1e-12)


def test_kicked_search_keeps_cell_lower_triangular(tmp_path, monkeypatch):
    """Start a solid-state dimer at a relaxed minimum: forces ~ 0 with
    curvature > 0 forces re-kicks. The cell must stay LT throughout."""
    monkeypatch.chdir(tmp_path)
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.calc = EMT()
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.005, steps=200)

    np.random.seed(3)
    d = SolidStateDimer(atoms, solid_state=True)
    d.search(fmax=0.02, max_force_calls=300, quiet=True,
             logfile=str(tmp_path / 'kick.log'))

    log = (tmp_path / 'kick.log').read_text()
    assert 'Re-kick' in log, "scenario did not exercise the kick path"
    cell = atoms.get_cell()[:]
    assert np.allclose(np.triu(cell, 1), 0.0, atol=1e-8), \
        f"cell left LT gauge:\n{cell}"
