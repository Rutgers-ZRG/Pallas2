import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from pallas.optimize import GeometryError, GeometryGuard

pytestmark = pytest.mark.unit


def test_passes_through_on_sane_structure():
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    a.calc = GeometryGuard(EMT())
    e_guarded = a.get_potential_energy()
    b = bulk('Cu', 'fcc', a=3.6, cubic=True)
    b.calc = EMT()
    assert e_guarded == pytest.approx(b.get_potential_energy(), abs=1e-10)


def test_raises_on_squashed_cell():
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    cell = a.get_cell()[:]
    cell[2] *= 0.2  # c-height 0.72 A: neighbor-list explosion territory
    a.set_cell(cell, scale_atoms=True)
    a.calc = GeometryGuard(EMT())
    with pytest.raises(GeometryError):
        a.get_potential_energy()


def test_raises_on_fused_atoms():
    a = Atoms('Cu2', positions=[[0, 0, 0], [0.2, 0, 0]],
              cell=[8, 8, 8], pbc=True)
    a.calc = GeometryGuard(EMT())
    with pytest.raises(GeometryError):
        a.get_potential_energy()


def test_raises_on_runaway_forces():
    from ase.calculators.calculator import Calculator, all_changes

    class HugeForce(Calculator):
        implemented_properties = ['energy', 'forces', 'stress']

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            f = np.zeros((len(self.atoms), 3))
            f[0, 0] = 120.0
            self.results = {'energy': 5.0, 'forces': f, 'stress': np.zeros(6)}

    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    a.calc = GeometryGuard(HugeForce())
    with pytest.raises(GeometryError):
        a.get_forces()
