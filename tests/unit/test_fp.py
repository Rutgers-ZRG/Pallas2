import numpy as np
import pytest
from ase.build import bulk

from pallas import PallasAtom
from pallas.core import fp_distance

pytestmark = pytest.mark.unit


def _pa(atoms, znucl):
    p = PallasAtom(atoms)
    p.znucl = znucl
    return p


@pytest.fixture
def si_pair():
    a = _pa(bulk('Si', 'diamond', a=5.43, cubic=True), [14])           # 8 atoms
    b = _pa(bulk('Si', 'fcc', a=3.9, cubic=True).repeat((2, 1, 1)), [14])  # 8 atoms
    return a, b


def test_self_distance_zero(si_pair):
    a, _ = si_pair
    types = np.ones(len(a), dtype=int)
    assert fp_distance(a.get_fp(), a.get_fp(), types) == pytest.approx(0.0, abs=1e-10)


def test_symmetric_and_positive(si_pair):
    a, b = si_pair
    types = np.ones(len(a), dtype=int)
    d1 = fp_distance(a.get_fp(), b.get_fp(), types)
    d2 = fp_distance(b.get_fp(), a.get_fp(), types)
    assert d1 == pytest.approx(d2, rel=1e-8)
    assert d1 > 0.01


def test_permutation_invariant(si_pair):
    a, _ = si_pair
    perm = a[[1, 0, 3, 2, 5, 4, 7, 6]]
    types = np.ones(len(a), dtype=int)
    d = fp_distance(_pa(perm, [14]).get_fp(), a.get_fp(), types)
    assert d == pytest.approx(0.0, abs=1e-8)


def test_fp_dist_workaround_in_place():
    # torch_fplib.get_fp_dist returns 0.0 for all inputs (known upstream bug);
    # PALLAS must route distances through xcal.fp_dist_with_assignment.
    from pallas import xcal
    assert hasattr(xcal, 'fp_dist_with_assignment')
