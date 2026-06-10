"""Pressure must reach the optimizer, not just the bookkeeping.

EMT Cu (B ~ 134 GPa): at 8 GPa (0.05 eV/A^3) the relaxed cell must be
measurably compressed relative to the zero-pressure cell.
"""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pallas import local_optimization

pytestmark = pytest.mark.integration


def test_local_optimization_applies_pressure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # opt.log
    a0 = bulk('Cu', 'fcc', a=3.6, cubic=True)

    r0 = local_optimization(a0.copy(), fmax=0.01, steps=500, calc=EMT(), press=0.0)
    v0 = r0.get_volume()

    r1 = local_optimization(a0.copy(), fmax=0.01, steps=500, calc=EMT(), press=0.05)
    v1 = r1.get_volume()

    assert v1 < 0.97 * v0, f"no compression under pressure: V(P)={v1:.2f} vs V(0)={v0:.2f}"
    np.testing.assert_allclose(r0.cell.angles(), [90, 90, 90], atol=1.0)
