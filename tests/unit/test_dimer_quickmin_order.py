"""D1: QuickMin must return the geometry whose forces passed the test.

Pre-fix ordering was evaluate -> step -> check(pre-step forces) -> break,
leaving the returned saddle one momentum step past the point that
satisfied fmax (its energy quoted at a force-unchecked geometry).
"""
import numpy as np
import pytest
from ase import Atoms

from pallas.dimer import SolidStateDimer

from test_dimer_mb import MB_SADDLE_XY, OFFSET, MuellerBrown

pytestmark = pytest.mark.unit


class RecordingDimer(SolidStateDimer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.call_log = []

    def get_forces(self):
        self.call_log.append('force_eval')
        return super().get_forces()

    def set_positions(self, x):
        self.call_log.append('move')
        super().set_positions(x)


def _mb_dimer():
    start = Atoms('H', positions=[[OFFSET - 0.70, OFFSET + 1.00, 0.0]],
                  cell=[50.0] * 3, pbc=True)
    start.calc = MuellerBrown()
    mode = np.zeros((4, 3))
    d = np.array([MB_SADDLE_XY[0] + 0.70, MB_SADDLE_XY[1] - 1.00, 0.0])
    mode[0] = d / np.linalg.norm(d)
    return RecordingDimer(start, mode=mode, dimer_separation=0.01,
                          max_rotations=8, external_stress=np.zeros((3, 3)))


def test_no_move_after_final_force_eval(tmp_path):
    d = _mb_dimer()
    d.search(fmax=0.05, max_force_calls=6000, quiet=True,
             logfile=str(tmp_path / 'q.log'))
    assert d.converged_flag
    assert d.call_log[-1] == 'force_eval', \
        "geometry moved after the final (convergence-passing) force eval"


def test_returned_geometry_satisfies_fmax(tmp_path):
    fmax = 0.05
    d = _mb_dimer()
    d.search(fmax=fmax, max_force_calls=6000, quiet=True,
             logfile=str(tmp_path / 'q.log'))
    assert d.converged_flag
    # re-evaluate at the returned geometry (mode re-rotation may shift the
    # effective force slightly -> 20% margin)
    f = d.get_forces()
    assert d.gradient_norm(f.ravel()) < fmax * 1.2
    assert d.curvature < 0
