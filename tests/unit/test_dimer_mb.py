import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pallas import cal_saddle

pytestmark = pytest.mark.unit

# Müller-Brown saddle between minimum A (-0.558, 1.442) and the
# intermediate minimum C (-0.050, 0.467): known position and energy.
#
# SCALE brings MB forces (natively O(100)) to the O(1) eV/Å regime the
# dimer's step sizes are tuned for; geometry is unchanged.
#
# OFFSET centers the MB frame in the periodic box: the solid-state dimer
# works in scaled coordinates and wraps positions, so the landscape must
# sit far from cell boundaries (a non-periodic test potential at x<0
# would see the atom teleport to x+L under wrapping).
SCALE = 0.01
OFFSET = 25.0
MB_SADDLE_XY = (-0.8220, 0.6243)
MB_SADDLE_E = -40.6644 * SCALE


class MuellerBrown(Calculator):
    """Müller-Brown PES (scaled) on (x, y) of atom 0, centered at OFFSET."""

    implemented_properties = ['energy', 'forces', 'stress']
    A = [-200.0, -100.0, -170.0, 15.0]
    a = [-1.0, -1.0, -6.5, 0.7]
    b = [0.0, 0.0, 11.0, 0.6]
    c = [-10.0, -10.0, -6.5, 0.7]
    x0 = [1.0, 0.0, -0.5, -1.0]
    y0 = [0.0, 0.5, 1.5, 1.0]

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = self.atoms.positions[0, 0] - OFFSET
        y = self.atoms.positions[0, 1] - OFFSET
        energy, fx, fy = 0.0, 0.0, 0.0
        for A, a, b, c, x0, y0 in zip(self.A, self.a, self.b, self.c, self.x0, self.y0):
            t = A * np.exp(a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + c * (y - y0) ** 2)
            energy += t
            fx -= t * (2 * a * (x - x0) + b * (y - y0))
            fy -= t * (b * (x - x0) + 2 * c * (y - y0))
        forces = np.zeros((len(self.atoms), 3))
        forces[0, 0], forces[0, 1] = fx, fy
        self.results = {'energy': energy * SCALE, 'forces': forces * SCALE,
                        'stress': np.zeros(6)}


def test_dimer_finds_mb_saddle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep ssdimer.log out of the repo

    # Start displaced from minimum A toward the saddle, mode pointing at it.
    start = Atoms('H', positions=[[OFFSET - 0.70, OFFSET + 1.00, 0.0]],
                  cell=[50.0, 50.0, 50.0], pbc=True)
    mode = np.zeros((4, 3))  # natom + 3 cell rows
    direction = np.array([MB_SADDLE_XY[0] + 0.70, MB_SADDLE_XY[1] - 1.00, 0.0])
    mode[0] = direction / np.linalg.norm(direction)

    saddle = cal_saddle(start, fmax=0.01, steps=2000,
                        calc=MuellerBrown(), mode=mode)

    x = saddle.positions[0, 0] - OFFSET
    y = saddle.positions[0, 1] - OFFSET
    assert saddle.converged, "dimer did not converge"
    assert saddle.dimer_curvature < 0, "converged with non-negative curvature"
    assert (x, y) == pytest.approx(MB_SADDLE_XY, abs=0.05)
    saddle.calc = MuellerBrown()
    assert saddle.get_potential_energy() == pytest.approx(MB_SADDLE_E, abs=abs(MB_SADDLE_E) * 0.01)
