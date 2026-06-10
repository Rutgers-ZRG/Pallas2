import numpy as np
import pytest
from ase.build import bulk

from pallas import PallasAtom, XCalculator

pytestmark = pytest.mark.unit


def _fd_force(src, fp0, i, c, h=1e-4):
    ap, am = src.copy(), src.copy()
    ap.positions[i, c] += h
    am.positions[i, c] -= h
    ap.calc = XCalculator(fp0, [14])
    am.calc = XCalculator(fp0, [14])
    return -(ap.get_potential_energy() - am.get_potential_energy()) / (2 * h)


def test_fp_force_matches_finite_difference():
    """XCalculator autograd forces vs central finite differences.

    The source must NOT be a perfect crystal: at a symmetry-degenerate
    configuration (e.g. pristine diamond, site symmetry Td) displacements
    +h and -h along a cell axis give congruent structures, so the
    Hungarian-matched FP distance is exactly even (FD = 0) and non-smooth
    (autograd returns a subgradient). PALLAS always perturbs before using
    FP gradients, so the physically relevant check is at a generic point.
    """
    src = bulk('Si', 'diamond', a=5.43, cubic=True)
    src.rattle(0.02, seed=7)  # break site symmetry -> generic point
    tgt = src.copy()
    tgt.positions[0] += [0.3, 0.1, -0.2]
    tgt.rattle(0.05, seed=1)
    pt = PallasAtom(tgt)
    pt.znucl = [14]
    fp0 = pt.get_fp()

    a0 = src.copy()
    a0.calc = XCalculator(fp0, [14])
    forces = a0.get_forces()

    for i, c in [(0, 0), (2, 1), (5, 2)]:
        assert forces[i, c] == pytest.approx(_fd_force(src, fp0, i, c), abs=1e-5), \
            f"atom {i} component {c}"


def test_fp_distance_even_at_symmetric_point():
    """Document the degenerate-point behavior: in pristine diamond the S4
    site operation maps +x to -x displacements, so E(+h) == E(-h) exactly."""
    src = bulk('Si', 'diamond', a=5.43, cubic=True)
    tgt = src.copy()
    tgt.positions[0] += [0.3, 0.1, -0.2]
    tgt.rattle(0.05, seed=1)
    pt = PallasAtom(tgt)
    pt.znucl = [14]
    fp0 = pt.get_fp()

    h = 1e-4
    ap, am = src.copy(), src.copy()
    ap.positions[0, 0] += h
    am.positions[0, 0] -= h
    ap.calc = XCalculator(fp0, [14])
    am.calc = XCalculator(fp0, [14])
    assert ap.get_potential_energy() == pytest.approx(
        am.get_potential_energy(), abs=1e-10)
