import pytest
from ase.build import bulk
from ase.spacegroup import crystal

from pallas.structure import spacegroup_label

pytestmark = pytest.mark.unit


def test_fcc_label():
    assert spacegroup_label(bulk('Cu', 'fcc', a=3.6)) == ('Fm-3m', 225)


def test_graphite_label():
    g = crystal(['C', 'C'], [(0, 0, 1 / 4), (1 / 3, 2 / 3, 1 / 4)], spacegroup=194,
                cellpar=[2.464, 2.464, 6.711, 90, 90, 120])
    assert spacegroup_label(g)[1] == 194


def test_rocksalt_label():
    assert spacegroup_label(bulk('NaCl', 'rocksalt', a=5.64))[1] == 225


def test_rattled_falls_back_gracefully():
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    a.rattle(0.3, seed=2)
    sym, num = spacegroup_label(a)
    assert num >= 1  # never raises; P1 in the worst case
